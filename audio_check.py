"""Local six-channel reSpeaker level/cancellation check. No camera, arm or API.

python audio_check.py --seconds 300
SPACE/R records 12 seconds: keep quiet for 2 seconds, then play the ringtone.
1/2 replay the selected raw microphone / old unaligned mean at the SAME gain.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import time
import wave

import cv2
import numpy as np
import sounddevice as sd

from mic_test import find_array, raw_channels, FS, CHANNELS
from arm_follow import select_event_audio

ROOT = Path(__file__).resolve().parent / 'recordings' / 'audio_checks'
WINDOW = 'SoundScope - microphone audio check'
RECORD_SECONDS = 12
BLOCK = 512


def channel_stats(samples):
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or not len(data) or data.shape[1] != CHANNELS:
        raise ValueError('Expected nonempty six-channel audio.')
    rows = []
    for channel in range(CHANNELS):
        x = data[:, channel]
        valid = np.isfinite(x)
        x = x[valid]
        if not len(x):
            rows.append({'channel': channel, 'rms_dbfs': None, 'peak_dbfs': None,
                         'clipped_fraction': None, 'nonfinite_samples': int(len(valid))})
            continue
        rms = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        rows.append({'channel': channel, 'rms_dbfs': max(-120., float(20 * np.log10(max(rms, 1e-12)))),
                     'peak_dbfs': max(-120., float(20 * np.log10(max(peak, 1e-12)))),
                     'clipped_fraction': float(np.mean(np.abs(x) >= .999)),
                     'nonfinite_samples': int(np.sum(~valid))})
    return rows


def prepare_comparison(samples, channels):
    """Keep relative selected/mean levels intact with one common replay gain."""
    data = np.asarray(samples, dtype=np.float32)
    stats = channel_stats(data)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite captured samples: comparison withheld; check capture/driver.')
    selected, selection = select_event_audio(data, channels)
    mean = data[:, channels].mean(axis=1)
    selected_rms = float(np.sqrt(np.mean(selected.astype(float) ** 2)))
    mean_rms = float(np.sqrt(np.mean(mean.astype(float) ** 2)))
    ratio_db = (float(20 * np.log10(max(mean_rms, 1e-12) / selected_rms))
                if selected_rms > 1e-9 else None)
    peak = max(float(np.max(np.abs(selected))), float(np.max(np.abs(mean))))
    gain = min(1000., .8 / peak) if peak > 1e-9 else 1.
    metadata = {'sample_rate': FS, 'device_channels': CHANNELS, 'raw_channel_map': list(channels),
                'duration_s': len(data) / FS, 'selection': selection,
                'per_channel_levels': stats, 'selected_rms': selected_rms, 'old_mean_rms': mean_rms,
                'old_mean_relative_rms_db': ratio_db, 'shared_replay_gain': gain,
                'comparison': 'Both mono files use one shared gain. Relative levels are preserved.',
                'level_unit': 'dBFS; not calibrated acoustic sound pressure',
                'classification': 'none'}
    return (np.asarray(selected * gain, np.float32), np.asarray(mean * gain, np.float32), metadata)


def _wav(path, samples, bits=16):
    data = np.asarray(samples)
    channel_count = 1 if data.ndim == 1 else data.shape[1]
    if bits == 32:
        pcm = np.rint(np.clip(data.astype(float), -1, 1) * 2147483647).astype('<i4')
    else:
        pcm = np.rint(np.clip(data.astype(float), -1, 1) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(channel_count)
        output.setsampwidth(bits // 8)
        output.setframerate(FS)
        output.writeframes(pcm.tobytes())


def save_capture(samples, channels, warnings=(), folder=ROOT):
    selected, mean, metadata = prepare_comparison(samples, channels)
    destination = Path(folder) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    destination.mkdir(parents=True, exist_ok=False)
    _wav(destination / 'all_six_channels_original_gain.wav', samples, bits=32)
    _wav(destination / '1_selected_raw_shared_gain.wav', selected)
    _wav(destination / '2_old_mean_shared_gain.wav', mean)
    metadata['capture_warnings'] = list(warnings)
    metadata['saved_at'] = datetime.now().astimezone().isoformat()
    if any(row['clipped_fraction'] for row in metadata['per_channel_levels']):
        metadata['capture_warnings'].append('At least one channel reached the clipping threshold.')
    (destination / 'levels.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    return selected, mean, metadata, destination


def _draw(levels, channels, status, detail, recording_samples, playing, previous, seconds_left):
    canvas = np.full((700, 1120, 3), (18, 15, 12), np.uint8)
    def text(message, x, y, scale=.58, color=(230, 233, 235)):
        cv2.putText(canvas, str(message), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    text('SoundScope / microphone audio check', 28, 44, .85)
    text('Local only. No Meta calls. No robot or camera connection.', 28, 78, .53, (160, 150, 135))
    text(status[:110], 28, 122, .64, (220, 207, 75))
    text(detail[:136], 28, 152, .48, (180, 174, 163))
    for row in levels:
        channel = row['channel']
        y = 203 + channel * 49
        role = f'RAW mic {channels.index(channel) + 1}' if channel in channels else 'PROCESSED'
        text(f'CH {channel}   {role}', 28, y, .57)
        cv2.rectangle(canvas, (275, y - 18), (900, y + 3), (45, 39, 30), -1)
        value = row['rms_dbfs']
        amount = np.clip((value + 90) / 90, 0, 1) if value is not None else 0
        color = (90, 95, 245) if row['clipped_fraction'] else (220, 207, 75) if channel in channels else (160, 143, 120)
        cv2.rectangle(canvas, (275, y - 18), (275 + round(625 * amount), y + 3), color, -1)
        text(f'{value:5.1f} dBFS' if value is not None else 'invalid', 918, y, .52, color)
        if row['clipped_fraction']:
            text('CLIP', 1045, y, .45, color)
    text('Scale: -90 to 0 dBFS. Clipping flag means samples >= 0.999 full scale.', 275, 499, .43, (160, 150, 135))
    if previous:
        selected = previous['selection']['selected_device_channel']
        ratio = previous['old_mean_relative_rms_db']
        attenuation = f'{ratio:+.1f} dB' if ratio is not None else 'undefined (silent)'
        text(f'Saved: selected raw CH {selected} | old mean / selected RMS: {attenuation}', 28, 534, .58)
        text(f'Both playback files have the SAME gain ({previous["shared_replay_gain"]:.1f}x). No sound label is inferred.', 28, 561, .45, (160, 150, 135))
    else:
        text('Press SPACE/R. Keep quiet for 2 seconds, then play the ring for 10 seconds.', 28, 537, .55)
    buttons = [('record', (28, 591, 324, 57), 'SPACE / R   RECORD 12s'),
               ('selected', (371, 591, 345, 57), '1   PLAY SELECTED MIC'),
               ('mean', (735, 591, 357, 57), '2   PLAY OLD MEAN')]
    for action, (x, y, w, h), label in buttons:
        enabled = not playing and recording_samples is None and (action == 'record' or previous is not None)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (54, 48, 36) if enabled else (27, 25, 23), -1)
        text(label, x + 15, y + 36, .53, (220, 207, 75) if enabled else (100, 100, 100))
    text(f'Q / ESC quit  |  playback uses your default speakers  |  closes in {seconds_left:.0f}s', 28, 682, .46, (160, 150, 135))
    return canvas, buttons


def run(args):
    channels = list(raw_channels())
    if len(channels) != 4 or len(set(channels)) != 4 or any(c not in range(CHANNELS) for c in channels):
        raise SystemExit('The raw microphone mapping must contain four unique channels from 0 to 5.')
    device = find_array()
    samples_queue = queue.Queue(maxsize=64)
    counters = {'dropped': 0, 'status': ''}
    def callback(indata, frames, time_info, flags):
        if flags:
            counters['status'] = str(flags)
        try:
            samples_queue.put_nowait(indata.copy())
        except queue.Full:
            counters['dropped'] += 1

    need = RECORD_SECONDS * FS
    capture = np.empty((need, CHANNELS), np.float32)
    position, started, capture_started = None, time.monotonic(), None
    baseline_drops, warnings = 0, []
    selected_audio = mean_audio = metadata = None
    levels = channel_stats(np.zeros((1, CHANNELS), np.float32))
    status, detail = 'Ready. Start a recording before playing the test sound.', 'Keep other microphone/camera programs closed for this check.'
    action_queue = queue.SimpleQueue()
    playing_until = cooldown_until = 0.
    buttons = []
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, (bx, by, bw, bh), _ in buttons:
                if bx <= x < bx + bw and by <= y < by + bh:
                    action_queue.put(action)
                    break
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, mouse)
    try:
        with sd.InputStream(device=device, channels=CHANNELS, samplerate=FS, dtype='float32',
                            blocksize=BLOCK, callback=callback):
            while time.monotonic() - started < args.seconds:
                now = time.monotonic()
                playing = now < playing_until
                if playing_until and not playing:
                    sd.stop()
                    playing_until = 0.
                    status = 'Playback finished. Recording is available after a short settling pause.'
                while not samples_queue.empty():
                    block = samples_queue.get_nowait()
                    if now < cooldown_until:
                        continue  # output audio must not become another captured comparison
                    levels = channel_stats(block)
                    if position is not None:
                        count = min(len(block), need - position)
                        capture[position:position + count] = block[:count]
                        position += count
                if position is not None:
                    elapsed = position / FS
                    status = (f'RECORDING {elapsed:.1f}/{RECORD_SECONDS}s - KEEP QUIET' if elapsed < 2 else
                              f'RECORDING {elapsed:.1f}/{RECORD_SECONDS}s - PLAY YOUR RING NOW')
                    detail = 'Do not move or cover the mic board. This comparison does not estimate source direction.'
                    if counters['status'] and counters['status'] not in warnings:
                        warnings.append(counters['status'])
                    finished = position >= need
                    timed_out = now - capture_started > RECORD_SECONDS + 5
                    if finished or timed_out:
                        if timed_out:
                            warnings.append('Capture ended before all expected samples arrived.')
                        dropped = counters['dropped'] - baseline_drops
                        if dropped:
                            warnings.append(f'{dropped} input blocks were dropped; capture is incomplete.')
                        try:
                            selected_audio, mean_audio, metadata, folder = save_capture(capture[:position].copy(), channels, warnings, args.output)
                            status = 'Saved. Press 1, then 2, to compare at the same gain.'
                            detail = str(folder)
                            print(f'Saved audio check: {folder}', flush=True)
                            print(json.dumps(metadata, indent=2), flush=True)
                        except Exception as exc:
                            status, detail = 'Capture could not be saved.', str(exc)
                        position = None
                canvas, buttons = _draw(levels, channels, status, detail, position, playing or now < cooldown_until,
                                        metadata, args.seconds - (now - started))
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(20) & 0xff
                if key in (ord('q'), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                action = ('record' if key in (32, ord('r'), ord('R')) else 'selected' if key == ord('1') else
                          'mean' if key == ord('2') else None)
                if not action_queue.empty():
                    action = action_queue.get()
                if position is None and now >= cooldown_until:
                    if action == 'record':
                        while not samples_queue.empty():
                            samples_queue.get_nowait()
                        position, capture_started = 0, time.monotonic()
                        baseline_drops, warnings = counters['dropped'], []
                        counters['status'] = ''
                    elif action in ('selected', 'mean') and metadata is not None:
                        audio = selected_audio if action == 'selected' else mean_audio
                        try:
                            sd.play(audio, FS, blocking=False)
                            playing_until = time.monotonic() + len(audio) / FS
                            cooldown_until = playing_until + .6
                            status = 'PLAYBACK: selected raw microphone' if action == 'selected' else 'PLAYBACK: old unaligned raw mean'
                            detail = 'Recording is suspended. Both files use identical gain; listening is the test.'
                        except Exception as exc:
                            status, detail = 'Playback failed.', str(exc)
    finally:
        sd.stop()
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass  # Windows may already have destroyed it when the user clicks X.
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=300, help='window lifetime in seconds (default300; max3600)')
    parser.add_argument('--output', type=Path, default=ROOT, help='local recordings folder')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 1 <= args.seconds <= 3600:
        parser.error('--seconds must be between1 and3600')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
