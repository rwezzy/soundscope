"""Bounded local evidence storage for recorded events; no capture, playback or network.

Audio is the normalized 16 kHz event clip, not a calibrated loudness measurement.
Scene timestamps describe proximity to that clip, not hardware synchronization or
proof that a pictured object produced the sound. Files stay under the evidence root.
"""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
import wave

import cv2
import numpy as np

from sound_memory import DATA_DIR

SCHEMA = 'soundscope.event-evidence.v1'
MAX_AUDIO_SECONDS = 10.0
MAX_WAV_BYTES = 16000 * 2 * 10 + 65536
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_METADATA_BYTES = 65536
MAX_INPUT_PIXELS = 3840 * 2160
MAX_SCENE_GAP_S = 1.5
ID_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')


def _record_id(value):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError('record_id must be a simple event identifier, not a path')
    return value


def _number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite timestamp')
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite timestamp') from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative timestamp')
    return value


def _wav_info(wav):
    if not isinstance(wav, (bytes, bytearray)) or len(wav) > MAX_WAV_BYTES:
        raise ValueError('event audio must be WAV bytes, at most 10 seconds')
    try:
        with wave.open(io.BytesIO(wav), 'rb') as stream:
            channels, width, rate, count = (stream.getnchannels(), stream.getsampwidth(),
                                             stream.getframerate(), stream.getnframes())
            if channels != 1 or width != 2 or rate != 16000 or stream.getcomptype() != 'NONE':
                raise ValueError('event WAV must be uncompressed 16-bit, 16 kHz, mono audio')
            if not 0 < count <= 16000 * MAX_AUDIO_SECONDS:
                raise ValueError('event WAV must contain more than zero and at most 10 seconds')
            if len(stream.readframes(count)) != count * width:
                raise ValueError('event WAV is truncated')
    except (wave.Error, EOFError) as exc:
        raise ValueError('event audio is not a valid WAV file') from exc
    return {'audio_duration_s': count / rate, 'audio_sample_rate_hz': rate,
            'audio_channels': channels, 'audio_sample_width_bytes': width,
            'audio_normalized': True,
            'audio_level_note': 'Normalized event audio; use stored raw-microphone levels for dBFS, not this clip.'}


def _frame(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or
            frame.shape[2] != 3 or not frame.shape[0] or not frame.shape[1] or
            frame.shape[0] * frame.shape[1] > MAX_INPUT_PIXELS):
        raise ValueError('event image must be a bounded uint8 BGR camera frame')
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width, 960 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok or encoded.nbytes > MAX_IMAGE_BYTES:
        raise ValueError('could not encode a bounded JPEG event frame')
    return encoded.tobytes(), [width, height], list(size)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'unsupported metadata value {type(value).__name__}')


class EvidenceStore:
    def __init__(self, root=None):
        self.root = Path(root if root is not None else DATA_DIR / 'evidence').resolve()

    def _resolve(self, relative):
        if not isinstance(relative, str) or not relative:
            raise ValueError('missing evidence file path')
        path = Path(relative)
        if path.is_absolute() or path.drive or '..' in path.parts:
            raise ValueError('evidence paths must stay inside the evidence folder')
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root) or resolved == self.root:
            raise ValueError('evidence path escapes the evidence folder')
        return resolved

    @staticmethod
    def _read(path, limit):
        try:
            with path.open('rb') as stream:
                data = stream.read(limit + 1)
        except OSError as exc:
            raise ValueError(f'evidence file is missing or unreadable: {path.name}') from exc
        if len(data) > limit:
            raise ValueError(f'evidence file is unexpectedly large: {path.name}')
        return data

    def save(self, record_id, wav, frame, *, frame_at, event_observed_at,
             event_onset_at=None, event_end_at=None, geometry=None):
        record_id = _record_id(record_id)
        audio = bytes(wav) if isinstance(wav, (bytes, bytearray)) else wav
        info = _wav_info(audio)
        image, original_size, stored_size = _frame(frame)
        frame_at = _number(frame_at, 'frame_at')
        observed = _number(event_observed_at, 'event_observed_at')
        onset = _number(event_onset_at, 'event_onset_at') if event_onset_at is not None else observed
        end = _number(event_end_at, 'event_end_at') if event_end_at is not None else observed
        if end < onset:
            raise ValueError('event audio end precedes its onset')
        gap = max(onset - frame_at, frame_at - end, 0.0)
        scene_valid = gap <= MAX_SCENE_GAP_S
        offset = frame_at - observed
        bundle = f'{record_id}-{uuid.uuid4().hex[:12]}'
        artifact = {'schema': SCHEMA, 'record_id': record_id,
                    'audio_path': f'{bundle}/audio.wav', 'image_path': f'{bundle}/scene.jpg',
                    'metadata_path': f'{bundle}/metadata.json',
                    'original_image_size': original_size, 'stored_image_size': stored_size,
                    'frame_offset_s': offset, 'scene_valid': scene_valid,
                    'scene_reason': (f'Frame is {offset:+.3f}s from the audio observation; proximity only, not synchronized source proof.'
                                     if scene_valid else
                                     f'Frame is {gap:.3f}s outside the audio window; do not use it to locate this sound.'),
                    **info}
        metadata = {**artifact, 'frame_at_monotonic': frame_at, 'event_observed_at_monotonic': observed,
                    'event_onset_at_monotonic': onset, 'event_end_at_monotonic': end,
                    'frame_during_audio': onset <= frame_at <= end, 'distance_to_audio_s': gap,
                    'source_verified': False, 'hardware_synchronized': False, 'geometry': geometry,
                    'audio_sha256': hashlib.sha256(audio).hexdigest(),
                    'image_sha256': hashlib.sha256(image).hexdigest()}
        try:
            metadata_bytes = json.dumps(metadata, allow_nan=False, default=_json_default,
                                        separators=(',', ':')).encode('utf-8')
        except (TypeError, ValueError) as exc:
            raise ValueError('evidence geometry must be finite JSON metadata') from exc
        if len(metadata_bytes) > MAX_METADATA_BYTES:
            raise ValueError('evidence metadata is too large')
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f'.{record_id}-', dir=self.root)).resolve()
        destination = self._resolve(bundle)
        try:
            for name, data in (('audio.wav', audio), ('scene.jpg', image), ('metadata.json', metadata_bytes)):
                with (staging / name).open('wb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            # Publishing the complete directory makes all three files visible together.
            os.replace(staging, destination)
        finally:
            # Only this newly created staging directory can be removed; never caller-supplied paths.
            if staging.exists() and staging.is_relative_to(self.root) and staging.parent == self.root:
                shutil.rmtree(staging)
        artifact['metadata_sha256'] = hashlib.sha256(metadata_bytes).hexdigest()
        return artifact

    def load(self, artifact):
        if not isinstance(artifact, dict) or artifact.get('schema') != SCHEMA:
            raise ValueError('this event has no supported saved evidence')
        record_id = _record_id(artifact.get('record_id'))
        paths = {key: self._resolve(artifact.get(key)) for key in ('audio_path', 'image_path', 'metadata_path')}
        if len({path.parent for path in paths.values()}) != 1:
            raise ValueError('evidence files do not belong to the same event bundle')
        metadata_bytes = self._read(paths['metadata_path'], MAX_METADATA_BYTES)
        if hashlib.sha256(metadata_bytes).hexdigest() != artifact.get('metadata_sha256'):
            raise ValueError('saved event metadata has changed or is incomplete')
        try:
            metadata = json.loads(metadata_bytes)
        except (ValueError, UnicodeError) as exc:
            raise ValueError('saved event metadata is missing or invalid') from exc
        if not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA or metadata.get('record_id') != record_id:
            raise ValueError('saved evidence does not match this event')
        if any(metadata.get(key) != artifact[key] for key in paths):
            raise ValueError('saved evidence paths do not match this event')
        audio = self._read(paths['audio_path'], MAX_WAV_BYTES)
        image_bytes = self._read(paths['image_path'], MAX_IMAGE_BYTES)
        if (hashlib.sha256(audio).hexdigest() != metadata.get('audio_sha256') or
                hashlib.sha256(image_bytes).hexdigest() != metadata.get('image_sha256')):
            raise ValueError('saved evidence is incomplete or has changed')
        _wav_info(audio)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None or [image.shape[1], image.shape[0]] != metadata.get('stored_image_size'):
            raise ValueError('saved event image is invalid')
        for key in ('frame_at_monotonic', 'event_observed_at_monotonic',
                    'event_onset_at_monotonic', 'event_end_at_monotonic'):
            _number(metadata.get(key), key)
        return image, audio, metadata


class EvidenceWriter:
    """One background disk/JPEG worker with bounded input and completion queues.

    submit(...) copies the frame and returns False if busy/closing; it never waits for disk.
    drain() returns (record_id, artifact_or_None, error_or_None) tuples for Memory.update.
    close() drains outstanding results while waiting briefly and returns those same tuples.
    """
    def __init__(self, store, max_pending=4):
        if not isinstance(max_pending, int) or isinstance(max_pending, bool) or max_pending < 1:
            raise ValueError('max_pending must be a positive integer')
        self.store = store
        self.pending = queue.Queue(maxsize=max_pending)
        self.results = queue.Queue(maxsize=max_pending + 1)
        self.closing = threading.Event()
        self.thread = threading.Thread(target=self._run, name='event-evidence-writer', daemon=True)
        self.thread.start()

    def submit(self, record_id, wav, frame, **kwargs):
        if self.closing.is_set() or self.pending.full():
            return False
        _record_id(record_id)
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or
                frame.shape[2] != 3 or frame.shape[0] * frame.shape[1] > MAX_INPUT_PIXELS):
            raise ValueError('event image must be a bounded uint8 BGR camera frame')
        if not isinstance(wav, (bytes, bytearray)) or len(wav) > MAX_WAV_BYTES:
            raise ValueError('event audio must be bounded WAV bytes')
        try:
            self.pending.put_nowait((record_id, bytes(wav), frame.copy(), copy.deepcopy(kwargs)))
        except queue.Full:
            return False
        return True

    def _run(self):
        while not self.closing.is_set() or not self.pending.empty():
            try:
                record_id, wav, frame, kwargs = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                artifact = self.store.save(record_id, wav, frame, **kwargs)
                result = (record_id, artifact, None)
            except Exception as exc:
                result = (record_id, None, str(exc))
            self.results.put(result)  # bounded; UI drain() releases back-pressure
            self.pending.task_done()

    def drain(self):
        completed = []
        while True:
            try:
                completed.append(self.results.get_nowait())
            except queue.Empty:
                return completed

    def close(self, timeout=2.0):
        self.closing.set()
        deadline, completed = time.monotonic() + max(0.0, timeout), []
        while self.thread.is_alive() and time.monotonic() < deadline:
            completed.extend(self.drain())
            self.thread.join(min(0.05, max(0.0, deadline - time.monotonic())))
        completed.extend(self.drain())
        return completed
