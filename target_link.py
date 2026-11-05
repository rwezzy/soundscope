"""Atomic, local-only handoff of an acoustic/depth candidate. No devices or network.

Packets use this machine's monotonic clock, so they are not portable between computers.
The reader is stateless: a consumer must process each (session_id, event_id) at most once.
``not_before`` additionally rejects observations made before that consumer was armed.
A candidate is NOT proof that the visible surface produced the sound; this bridge is
for pointing at candidates, not authorizing a grasp or declaring a verified source.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np

import aim

SCHEMA = 'soundscope.target.v1'
MAX_PACKET_BYTES = 65536


def geometry_fingerprint(cfg):
    """Stable hash of camera and array geometry only; no credentials or safety switches."""
    try:
        data = json.dumps({'camera': cfg['camera'], 'array': cfg['array']},
                          sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid camera/microphone geometry') from exc
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{label} must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a finite number') from exc
    if not math.isfinite(number):
        raise ValueError(f'{label} must be a finite number')
    return number


def _point(value, label):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must contain three finite coordinates') from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f'{label} must contain three finite coordinates')
    return vector


def _fix(value):
    if not isinstance(value, dict):
        raise ValueError('sound evidence is missing')
    if value.get('directional_ambiguous', False):
        raise ValueError('sound direction is ambiguous')
    try:
        direction = _point(value['direction'], 'sound direction')
        length = math.hypot(*direction)
        if not math.isfinite(length) or abs(length - 1.0) > 0.01:
            raise ValueError('sound direction must be a normalized unit vector')
        result = {'direction': direction.tolist(), 'z': _number(value['z'], 'sound confidence'),
                  'snr_db': _number(value['snr_db'], 'sound SNR'), 'directional_ambiguous': False}
    except KeyError as exc:
        raise ValueError(f'sound evidence is missing {exc.args[0]}') from exc
    # Never serialize recorded audio, FFT arrays or the large evidence tuple.
    for key in ('coherence', 'azimuth', 'elevation', 'seconds'):
        if key in value:
            result[key] = _number(value[key], f'sound {key}')
    return result


def _base_point(camera_point, cfg):
    if cfg.get('camera', {}).get('mount') != 'base' or cfg.get('array', {}).get('on_arm', False):
        raise ValueError('target bridge requires a stationary calibrated camera and microphone')
    minimum = _number(cfg['camera'].get('min_depth_mm', 0), 'camera minimum depth') / 1000.0
    if camera_point[2] <= 0 or camera_point[2] < minimum:
        raise ValueError('target has no usable positive camera depth')
    with np.errstate(over='ignore', invalid='ignore'):
        return _point(aim.camera_to_base(camera_point * 1000.0, cfg), 'base point')


class TargetWriter:
    """Publish final measured candidates, replacing old targets with explicit invalid states.

    ``cfg`` is retained by reference so a live calibration changes the fingerprint immediately.
    ``publish`` returns the packet written; inspect its ``ready`` / ``reason`` fields if desired.
    """
    def __init__(self, path, cfg):
        self.path = Path(path)
        self.cfg = cfg
        self.session_id = uuid.uuid4().hex
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.invalidate('waiting for a final measured acoustic candidate')

    def _packet(self, now):
        return {'schema': SCHEMA, 'session_id': self.session_id, 'event_id': '',
                'observed_at': None, 'published_at': now,
                'geometry_fingerprint': geometry_fingerprint(self.cfg),
                'ready': False, 'final': False, 'measured': False, 'quality': 'not-ready',
                'source_verified': False}

    def _write(self, packet):
        content = json.dumps(packet, allow_nan=False, separators=(',', ':'))
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=self.path.parent,
                                             prefix=f'.{self.path.name}.{self.session_id}.',
                                             suffix='.tmp', delete=False) as stream:
                temp = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        return packet

    def invalidate(self, reason):
        packet = self._packet(time.monotonic())
        packet['reason'] = str(reason)[:500] or 'target unavailable'
        return self._write(packet)

    def publish(self, view, now=None):
        now = _number(time.monotonic() if now is None else now, 'publication time')
        packet = self._packet(now)
        try:
            if not isinstance(view, dict):
                raise ValueError('target view is missing')
            if view.get('final') is not True:
                raise ValueError('waiting for final sound evidence')
            quality = view.get('quality')
            state = quality.get('state') if isinstance(quality, dict) else quality
            if state != 'candidate':
                raise ValueError(f'acoustic quality is {state or "unknown"}, not a candidate')
            if view.get('off_screen'):
                raise ValueError('sound lies outside the camera view')
            if view.get('measured') is not True:
                raise ValueError('target depth was not measured')
            event = view.get('event_id')
            if (isinstance(event, bool) or not isinstance(event, (str, int, np.integer)) or
                    not str(event) or len(str(event)) > 256):
                raise ValueError('sound event identifier is missing')
            observed = _number(view.get('time'), 'observation time')
            if observed < 0 or observed > now:
                raise ValueError('observation time is invalid or in the future')
            point = _point(view.get('point'), 'camera point')
            base = _base_point(point, self.cfg)
            fix = _fix(view.get('fix'))
            packet.update(ready=True, final=True, measured=True, quality='candidate',
                          event_id=str(event), observed_at=observed, camera_point_m=point.tolist(),
                          base_point_mm=base.tolist(), fix=fix,
                          reason='Measured candidate surface; sound source identity is unverified.')
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            packet['reason'] = str(exc)[:500]
        return self._write(packet)


def read_target(path, cfg, now=None, max_age_s=1.5, not_before=None):
    """Read one actionable candidate or raise ValueError; does not consume/deduplicate it.

    Age is measured from the sound observation, NOT the file-write time. A consumer
    must retain used (session_id, event_id) pairs and must re-read before acting.
    """
    now = _number(time.monotonic() if now is None else now, 'current time')
    max_age_s = _number(max_age_s, 'maximum sound age')
    if max_age_s <= 0:
        raise ValueError('maximum sound age must be positive')
    if not_before is not None:
        not_before = _number(not_before, 'consumer start time')
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_PACKET_BYTES + 1)
        if len(raw) > MAX_PACKET_BYTES:
            raise ValueError('target packet is too large')
        def reject_constant(value):
            raise ValueError(f'non-finite JSON value {value}')
        packet = json.loads(raw, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('target packet is missing, unreadable or incomplete') from exc
    if not isinstance(packet, dict) or packet.get('schema') != SCHEMA:
        raise ValueError('unrecognized target packet schema')
    if packet.get('geometry_fingerprint') != geometry_fingerprint(cfg):
        raise ValueError('camera/microphone geometry differs; use the same calibrated mount profile')
    if packet.get('ready') is not True:
        raise ValueError(f'target not ready: {packet.get("reason", "no measured candidate")}')
    if (packet.get('final') is not True or packet.get('measured') is not True or
            packet.get('quality') != 'candidate' or packet.get('source_verified') is not False):
        raise ValueError('target is not a final measured candidate with unverified source identity')
    for key in ('session_id', 'event_id'):
        if not isinstance(packet.get(key), str) or not packet[key] or len(packet[key]) > 256:
            raise ValueError(f'target {key} is missing or invalid')
    observed = _number(packet.get('observed_at'), 'observation time')
    published = _number(packet.get('published_at'), 'publication time')
    if observed < 0 or observed > now or published > now or published < observed:
        raise ValueError('target timestamp is invalid or in the future')
    if not_before is not None and observed <= not_before:
        raise ValueError('target observation predates this consumer session')
    if now - observed > max_age_s:
        raise ValueError(f'target is stale ({now - observed:.2f}s old)')
    fix = _fix(packet.get('fix'))
    point = _point(packet.get('camera_point_m'), 'camera point')
    base = _point(packet.get('base_point_mm'), 'base point')
    expected_base = _base_point(point, cfg)
    if not np.allclose(base, expected_base, atol=1e-4, rtol=1e-9):
        raise ValueError('target base coordinates do not match camera calibration')
    packet['fix'] = fix
    return packet
