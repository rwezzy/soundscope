"""SoundScope mic-array go/no-go test (reSpeaker XVF3800 on the 16k6ch firmware).

  python mic_test.py devices     is it on the 6-channel firmware, and are the 4 raw mics alive?
  python mic_test.py tap         tap each mic once so we know which channel is which
  python mic_test.py ping        play a test sound from the laptop and point at it
  python mic_test.py live        live direction readout; try quiet and far-away sounds
  python mic_test.py trial       the go/no-go: 5 known angles -> error table, PASS/FAIL
  python mic_test.py analyze recordings/<session>     re-score a saved trial
  python mic_test.py selftest    check the math on simulated quiet/distant sounds (no hardware)

Angles: 0 = the board's front edge (between the front-right and front-left mics),
positive = right, negative = left. Lay the board flat with the mic holes uncovered.

Flashing the 6-channel firmware (once):
  1. dfu-util 0.11 from dfu-util.sourceforge.net, and
     xmos_firmwares/usb/respeaker_xvf3800_usb_dfu_firmware_v2.1.1_16k6ch.bin from
     github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
  2. dfu-util -R -e -a 1 -D respeaker_xvf3800_usb_dfu_firmware_v2.1.1_16k6ch.bin
  3. Unplug and replug. If Windows still shows 2 channels, uninstall the device in
     Device Manager and replug.
"""
import argparse
import json
import queue
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from doa import FS, HOP, MIC_XYZ, N_FFT, SPEED_OF_SOUND, Localizer, Tracker

HERE = Path(__file__).resolve().parent
MAP_FILE = HERE / 'mic_map.json'
RECORDINGS = HERE / 'recordings'
CHANNELS = 6
DEFAULT_RAW = [2, 3, 4, 5]  # Seeed's 6-channel map: 0-1 processed, 2-5 raw mics 0-3
POSITIONS = ['front-left', 'front-right', 'back-right', 'back-left']  # same order as MIC_XYZ
API_ORDER = ['Windows WDM-KS', 'Windows WASAPI', 'MME', 'Windows DirectSound']
Z_OK, Z_SOLID = 6.0, 12.0  # peak height in noise standard deviations


# ---------- display ----------

def bearing(az):
    """Board azimuth (counterclockwise from the front edge) -> degrees, positive to the right."""
    return (-az + 180) % 360 - 180


def side(deg):
    deg = (deg + 180) % 360 - 180
    if abs(deg) < 0.5:
        return 'straight ahead'
    return f"{abs(deg):.0f}° {'right' if deg > 0 else 'left'}" + (' (behind)' if abs(deg) > 90 else '')


def sureness(z):
    return 'solid' if z >= Z_SOLID else 'ok' if z >= Z_OK else 'weak'


def describe(r):
    if r is None:
        return 'nothing above the noise floor'
    return (f"{side(bearing(r['azimuth'])):>18}  elev {r['elevation']:3.0f}°  "
            f"{sureness(r['z']):>5} (z {r['z']:5.1f})  SNR {r['snr_db']:+5.1f} dB")


# ---------- audio I/O ----------

def find_array():
    apis = sd.query_hostapis()
    devices = list(sd.query_devices())
    found = [(i, d, apis[d['hostapi']]['name']) for i, d in enumerate(devices)
             if 'respeaker' in d['name'].lower() and d['max_input_channels'] > 0]
    rank = {api: k for k, api in enumerate(API_ORDER)}
    rejected = []
    for i, d, api in sorted(found, key=lambda f: rank.get(f[2], len(API_ORDER))):
        if d['max_input_channels'] >= CHANNELS:
            try:
                sd.check_input_settings(device=i, channels=CHANNELS, samplerate=FS, dtype='int32')
                return i
            except sd.PortAudioError as exc:
                detail = ' '.join(str(exc).split())[:180] or 'PortAudio rejected the capture settings'
                rejected.append(f'  [{i}] {api}: {detail}')
                continue
    # PortAudio's view depends on the host API, driver and process permissions.
    # A two-channel WASAPI endpoint does not establish the installed firmware.
    inventory = []
    for api_index, api in enumerate(apis):
        visible = [d for d in devices if d['hostapi'] == api_index]
        inputs = [d for d in visible if d['max_input_channels'] > 0]
        array_channels = sorted({int(d['max_input_channels']) for d in inputs
                                 if 'respeaker' in d['name'].lower()})
        channels = ', '.join(map(str, array_channels)) if array_channels else 'none'
        inventory.append(f"  {api['name']}: {len(visible)} devices, {len(inputs)} inputs; "
                         f'reSpeaker input channels: {channels}')
    if not found:
        reason = 'No reSpeaker input is visible to this process.'
    elif rejected:
        reason = (f'Six-channel reSpeaker inputs are visible, but none accepted {CHANNELS}-channel '
                  f'{FS} Hz int32 capture settings.\nRejected settings:\n' + '\n'.join(rejected))
    else:
        most = max(d['max_input_channels'] for _, d, _ in found)
        reason = (f'No {CHANNELS}-channel reSpeaker input is exposed to this process; '
                  f'the largest visible reSpeaker input has {most} channels.')
    raise SystemExit(reason + '\nHost API discovery:\n' +
                     ('\n'.join(inventory) if inventory else '  No host APIs were reported.') +
                     '\nFirmware version is unknown from this check; do not flash based on channel count alone. '
                     'Retry `python mic_test.py devices` using the same Python in a normal, unrestricted '
                     'Windows Terminal or PowerShell session. Windows WDM-KS is preferred for raw multichannel '
                     'capture; WASAPI/MME may expose only a processed two-channel endpoint. '
                     'If capture remains unavailable, close other audio apps and check the USB data connection '
                     '(the array USB-C port next to the 3.5 mm jack), driver and host-API availability.')


def raw_channels():
    if MAP_FILE.exists():
        return json.loads(MAP_FILE.read_text())['channels']
    return DEFAULT_RAW


def record(dev, seconds, during=None):
    """Record all 6 channels as int32, (samples, 6). `during` runs once the stream is open."""
    need, chunks = int(seconds * FS), []
    with sd.InputStream(device=dev, channels=CHANNELS, samplerate=FS, dtype='int32',
                        callback=lambda indata, *_: chunks.append(indata.copy())):
        if during:
            during()
        while sum(map(len, chunks)) < need:
            sd.sleep(20)
    return np.concatenate(chunks)[:need]


def as_float(x, chans):
    return x[:, chans].astype(np.float64) / 2 ** 31


def save_wav(path, x):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(x.shape[1])
        f.setsampwidth(4)
        f.setframerate(FS)
        f.writeframes(np.ascontiguousarray(x, '<i4').tobytes())


def load_wav(path):
    with wave.open(str(path), 'rb') as f:
        if f.getsampwidth() != 4 or f.getframerate() != FS:
            raise SystemExit(f'{path}: expected 32-bit audio at {FS} Hz')
        return np.frombuffer(f.readframes(f.getnframes()), '<i4').reshape(-1, f.getnchannels())


def band_limit(x, lo, hi, rate=FS):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / rate)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


# ---------- commands ----------

def cmd_devices(args):
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        if 'respeaker' in d['name'].lower() and d['max_input_channels'] > 0:
            print(f"  [{i}] {apis[d['hostapi']]['name']}: {d['max_input_channels']} in, "
                  f"default {d['default_samplerate']:.0f} Hz")
    dev, chans = find_array(), raw_channels()
    print(f'Using device {dev}. Recording 2 s of room sound...')
    x = record(dev, 2.0).astype(np.float64) / 2 ** 31
    x -= x.mean(axis=0)
    for ch in range(CHANNELS):
        role = f'raw mic, {POSITIONS[chans.index(ch)]}' if ch in chans else 'processed'
        print(f'  ch {ch}: {20 * np.log10(x[:, ch].std() + 1e-12):6.1f} dBFS  ({role})')
    raw = x[:, chans]
    if np.any(raw.std(axis=0) == 0):
        print('Problem: a raw channel is silent. Reinstall the device in Device Manager and replug.')
    elif np.any(np.abs(np.corrcoef(raw.T)[np.triu_indices(4, 1)]) > 0.9999):
        print('Problem: two raw channels are identical copies, so these are not the raw mics.')
    else:
        print('Raw mics look alive. Next: python mic_test.py tap')


def wait_for_tap(q, timeout=20):
    """(channel, margin in dB) of the raw mic with the biggest short spike."""
    while not q.empty():
        q.get()  # ignore anything from before the prompt
    floor, t_end = None, time.time() + timeout
    while time.time() < t_end:
        e = (q.get() ** 2).mean(axis=0)
        if floor is not None and e.max() > 30 * floor.max():
            for _ in range(4):  # include ~64 ms of ring-down
                e = e + (q.get() ** 2).mean(axis=0)
            time.sleep(0.6)
            top = np.sort(e)
            return DEFAULT_RAW[int(np.argmax(e))], 10 * np.log10(top[-1] / top[-2])
        floor = e if floor is None else 0.9 * floor + 0.1 * e
    raise SystemExit('No tap heard. Check `python mic_test.py devices` first.')


def cmd_tap(args):
    dev, q = find_array(), queue.Queue()
    print('Decide which edge of the board is the FRONT (it will face the sounds / camera).')
    order = []
    with sd.InputStream(device=dev, channels=CHANNELS, samplerate=FS, dtype='float32', blocksize=HOP,
                        callback=lambda indata, *_: q.put(indata[:, DEFAULT_RAW].copy())):
        for pos in POSITIONS:
            while True:
                print(f'  Tap the {pos.upper()} mic once with a fingertip...', flush=True)
                ch, margin = wait_for_tap(q)
                if margin >= 3 and ch not in order:
                    break
                print(f'    unclear (channel {ch}, only {margin:.1f} dB above the next mic). Tap again.')
            order.append(ch)
            print(f'    -> channel {ch}, {margin:.0f} dB louder than the others')
    MAP_FILE.write_text(json.dumps({'channels': order, 'order': POSITIONS}, indent=2))
    note = '' if order == DEFAULT_RAW else ' (differs from the documented order, so keep this file)'
    print(f'Saved {MAP_FILE.name}: channels {order}{note}')


def cmd_ping(args):
    dev, chans = find_array(), raw_channels()
    out = sd.query_devices(kind='output') if args.out is None else \
        sd.query_devices(int(args.out) if args.out.isdigit() else args.out, 'output')
    rate = int(out['default_samplerate'])
    burst = band_limit(np.random.default_rng().standard_normal(rate // 4), 300, 7000, rate)
    sig = np.tile(np.concatenate([args.volume * burst / np.abs(burst).max(), np.zeros(rate // 4)]), 4)
    print(f"Test sound plays from: {out['name']}. Measuring the room for 1.5 s...")
    loc = Localizer()
    loc.calibrate(as_float(record(dev, 1.5), chans))
    for i in range(args.repeat):
        x = record(dev, len(sig) / rate + 0.5,
                   during=lambda: sd.play(sig.astype(np.float32), rate, device=out['index']))
        sd.wait()
        print(f'  ping {i + 1}: ' + describe(loc.locate_clip(as_float(x, chans))))


def cmd_live(args):
    dev, chans = find_array(), raw_channels()
    loc = Localizer()
    print('Measuring the room for 2 s, stay quiet...')
    loc.calibrate(as_float(record(dev, 2.0), chans))
    tracker, q = Tracker(loc, threshold_db=args.threshold), queue.Queue()
    print('Listening. Make sounds around the board, quiet and far ones too. Ctrl+C to stop.\n')
    frame, pending = np.zeros((N_FFT, len(chans))), np.zeros((0, len(chans)))
    try:
        with sd.InputStream(device=dev, channels=CHANNELS, samplerate=FS, dtype='float32',
                            callback=lambda indata, *_: q.put(indata[:, chans].copy())):
            while True:
                pending = np.concatenate([pending, q.get()])
                while len(pending) >= HOP:
                    frame = np.concatenate([frame[HOP:], pending[:HOP]])
                    pending = pending[HOP:]
                    out = tracker.push(loc.stft(frame)[0])
                    if out and out[1]:
                        kind, r = out
                        if kind == 'update':
                            print('\r  hearing: ' + describe(r) + '   ', end='', flush=True)
                        else:
                            print(f"\r{time.strftime('%H:%M:%S')}  {r['seconds']:4.1f} s sound: {describe(r)}   ")
    except KeyboardInterrupt:
        print('\nStopped.')


def score(loc, x, angle):
    r = loc.locate_clip(x)
    error = None if r is None else (bearing(r['azimuth']) - angle + 180) % 360 - 180
    return {'placed': angle, 'result': r, 'error': error}


def report(rows, pass_deg=15.0):
    print(f"\n{'placed':>14} | {'measured':>14} | {'error':>6} | {'sure':>13} | {'SNR':>8}")
    for row in rows:
        r = row['result']
        if r is None:
            print(f"{side(row['placed']):>14} | {'(not heard)':>14} |")
            continue
        print(f"{side(row['placed']):>14} | {side(bearing(r['azimuth'])):>14} | {row['error']:+5.0f}° | "
              f"{sureness(r['z']):>5} (z {r['z']:4.0f}) | {r['snr_db']:+5.1f} dB")
    errors = [abs(row['error']) for row in rows if row['error'] is not None]
    if not errors or len(errors) < len(rows):
        print('\nFAIL: some positions were not heard. Go louder or closer, or run `devices`.')
        return
    mean = sum(errors) / len(errors)
    print(f"\nMean error {mean:.1f}°, worst {max(errors):.1f}°  ->  "
          f"{'PASS' if mean <= pass_deg else 'FAIL'} (pass = mean under {pass_deg:.0f}°)")


def cmd_trial(args):
    dev, chans = find_array(), raw_channels()
    session = RECORDINGS / time.strftime('%Y%m%d-%H%M%S')
    session.mkdir(parents=True)
    print(f'Board flat, front edge = 0°. Use a phone playing something continuous (keys video, music), '
          f'{args.distance:g} m from the board, at about table height.')
    input('\nPause the sound and stay quiet, then press Enter (3 s room-noise measurement)...')
    noise = record(dev, 3.0)
    save_wav(session / 'noise.wav', noise)
    loc = Localizer()
    loc.calibrate(as_float(noise, chans))
    print('Now start the sound on loop and leave it playing.')
    rows = []
    for angle in args.angles:
        input(f'\nPut the phone {side(angle)}, {args.distance:g} m away. Step back, press Enter, stay still...')
        x = record(dev, args.seconds)
        save_wav(session / f'angle_{angle:+04d}.wav', x)
        rows.append(score(loc, as_float(x, chans), angle))
        print('  ' + describe(rows[-1]['result']))
    report(rows)
    summary = [{'placed': r['placed'], 'error': r['error'],
                **({k: v for k, v in r['result'].items() if k != 'direction'} if r['result'] else {})}
               for r in rows]
    (session / 'results.json').write_text(json.dumps({'distance_m': args.distance, 'channels': chans,
                                                      'rows': summary}, indent=2))
    print(f'Recordings saved in {session}')


def cmd_analyze(args):
    session, chans = Path(args.session), raw_channels()
    loc = Localizer()
    loc.calibrate(as_float(load_wav(session / 'noise.wav'), chans))
    files = sorted(session.glob('angle_*.wav'), key=lambda p: int(p.stem.split('_')[1]))
    report([score(loc, as_float(load_wav(p), chans), int(p.stem.split('_')[1])) for p in files])


# ---------- simulation ----------

def unit(angle, elevation):
    """Board-frame direction for a bearing (degrees, positive right) and an elevation."""
    az, el = np.radians(-angle), np.radians(elevation)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def test_source(kind, n, rng):
    t = np.arange(n) / FS
    if kind == 'keys':  # short bright jingles with gaps
        env, i = np.zeros(n), 0
        while i < n:
            k = min(int(rng.uniform(0.03, 0.12) * FS), n - i)
            env[i:i + k] = np.exp(-np.arange(k) / (0.02 * FS))
            i += k + int(rng.uniform(0.02, 0.15) * FS)
        return band_limit(rng.standard_normal(n), 2000, 7500) * env
    if kind == 'voice':  # speech-like: bass-heavy spectrum, 4 Hz syllable rhythm
        pink = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.arange(n // 2 + 1) + 1), n)
        return band_limit(pink, 200, 4000) * np.sin(np.pi * 4 * t) ** 2
    if kind == 'beep':  # a pure 3 kHz alarm tone, 0.2 s on / 0.2 s off
        return np.sin(2 * np.pi * 3000 * t) * ((t % 0.4) < 0.2)
    raise ValueError(kind)


def simulate(kind, snr, angle, rng, elevation=10.0, drr=None, fan=None, seconds=3.0, tail=0.0):
    """Fake 4-mic recording (samples, 4): 1 s of room noise, then `kind` from `angle`.

    snr: source vs. mic self-noise (dB). drr: direct-to-echo ratio (dB); lower = farther away.
    fan: a steady noise from 120° away, this many dB louder than the source, running throughout.
    """
    n, lead, end = int(seconds * FS), FS, int((seconds - tail) * FS)
    m = n + FS  # padding so echoes don't wrap around
    f = np.fft.rfftfreq(m, 1 / FS)

    def arrival(spectrum, d, delay=0.0):  # a plane wave from direction d at each mic, (mics, bins)
        return spectrum[None] * np.exp(2j * np.pi * f[None] * (MIC_XYZ @ d / SPEED_OF_SOUND - delay)[:, None])

    def random_dirs(k):
        d = rng.standard_normal((k, 3))
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    X = np.zeros((len(MIC_XYZ), len(f)), complex)
    power = 1.0
    if kind:
        src = np.zeros(n)
        src[lead:end] = test_source(kind, end - lead, rng)
        power = np.mean(src[lead:end] ** 2)
        S = np.fft.rfft(src, m)
        X += arrival(S, unit(angle, elevation))
        if drr is not None:  # 6 strong early echoes (floor, walls), then a dense tail with 0.5 s reverb time
            for d in random_dirs(6):
                X += rng.choice([-0.5, 0.5]) * arrival(S, d, rng.uniform(0.002, 0.015))
            delays = 0.002 + 0.5 * rng.random(2000) ** (1 / 3)  # echo density grows like t^2
            gains = rng.standard_normal(2000) * np.exp(-6.9 * delays / 0.5)
            gains *= np.sqrt(max(10 ** (-drr / 10) - 1.5, 0) / np.sum(gains ** 2))
            dirs, h = random_dirs(2000), np.zeros((len(MIC_XYZ), m))
            for i, p in enumerate(MIC_XYZ):
                np.add.at(h[i], np.round((delays - dirs @ p / SPEED_OF_SOUND) * FS).astype(int), gains)
            X += S[None] * np.fft.rfft(h, m)
    if fan is not None:
        hum = band_limit(rng.standard_normal(n), 100, 3000)
        hum *= np.sqrt(power * 10 ** (fan / 10) / np.mean(hum ** 2))
        X += arrival(np.fft.rfft(hum, m), unit(angle + 120, 0.0))
    x = np.fft.irfft(X, m)[:, :n].T
    return x + rng.standard_normal(x.shape) * np.sqrt(power / 10 ** (snr / 10))


def cmd_selftest(args):
    rng = np.random.default_rng(7)
    cases = [
        ('keys, normal (+10 dB)', dict(kind='keys', snr=10)),
        ('keys, quiet (-5 dB)', dict(kind='keys', snr=-5)),
        ('keys, very quiet (-12 dB)', dict(kind='keys', snr=-12)),
        ('voice, quiet (-5 dB)', dict(kind='voice', snr=-5)),
        ('keys, distant (echo -6 dB)', dict(kind='keys', snr=0, drr=-6)),
        ('keys, quiet + distant', dict(kind='keys', snr=-5, drr=-9)),
        ('voice, distant', dict(kind='voice', snr=0, drr=-6)),
        ('keys + fan 10 dB louder', dict(kind='keys', snr=0, fan=10)),
        ('pure 3 kHz beep (+10 dB)', dict(kind='beep', snr=10)),
        ('3 kHz beep, distant', dict(kind='beep', snr=10, drr=-6)),
        ('limit: keys very far (-12)', dict(kind='keys', snr=-5, drr=-12)),
        ('nothing playing', dict(kind=None, snr=0)),
    ]
    print(f"{'simulated case':<28} {'worst error':>11} {'z':>12}")
    for name, kw in cases:
        errors, zs = [], []
        for angle in rng.uniform(-150, 150, 3):
            x = simulate(angle=angle, rng=rng, **kw)
            loc = Localizer()
            loc.calibrate(x[:FS])
            r = loc.locate_clip(x[FS:])
            if r:
                errors.append(abs((bearing(r['azimuth']) - angle + 180) % 360 - 180))
                zs.append(r['z'])
        if kw['kind'] is None:
            print(f"{name:<28} {'-':>11} {'highest':>7} {max(zs, default=0):4.1f}")
        else:
            print(f"{name:<28} {max(errors):10.1f}° {'lowest':>7} {min(zs):4.1f}")

    x = simulate('keys', -5, 40.0, rng, seconds=4.0, tail=1.0)
    loc = Localizer()
    loc.calibrate(x[:FS])
    tracker = Tracker(loc)
    ends = [out[1] for out in map(tracker.push, loc.stft(x[FS:])) if out and out[0] == 'end' and out[1]]
    heard = ', '.join(f"{side(bearing(r['azimuth']))} (z {r['z']:.0f})" for r in ends) or 'nothing'
    print(f'live tracker, quiet keys at 40° right: heard {heard}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('devices', help='check firmware and raw mics').set_defaults(fn=cmd_devices)
    sub.add_parser('tap', help='map channels to physical mics').set_defaults(fn=cmd_tap)
    s = sub.add_parser('ping', help='localize a test sound played by the laptop')
    s.add_argument('--out', help='output device index or part of its name (default: system output)')
    s.add_argument('--repeat', type=int, default=3)
    s.add_argument('--volume', type=float, default=0.3, help='0-1; try 0.01 to test faint sounds')
    s.set_defaults(fn=cmd_ping)
    s = sub.add_parser('live', help='live direction readout')
    s.add_argument('--threshold', type=float, default=0.5,
                   help='dB above the noise floor that counts as a sound (lower = more sensitive)')
    s.set_defaults(fn=cmd_live)
    s = sub.add_parser('trial', help='go/no-go accuracy test at known angles')
    s.add_argument('--angles', type=int, nargs='+', default=[-60, -30, 0, 30, 60])
    s.add_argument('--distance', type=float, default=1.0, help='meters, for the prompts and the log')
    s.add_argument('--seconds', type=float, default=4.0, help='recording length per position')
    s.set_defaults(fn=cmd_trial)
    s = sub.add_parser('analyze', help='re-score a saved trial')
    s.add_argument('session')
    s.set_defaults(fn=cmd_analyze)
    sub.add_parser('selftest', help='check the math on simulated sounds').set_defaults(fn=cmd_selftest)
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
