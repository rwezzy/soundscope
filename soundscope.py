"""SoundScope sound memory from the command line: search, log, examples, budget. No camera needed.

  python soundscope.py search "when did the phone ring?"   Muse Spark answers from the memory
  python soundscope.py search "loud sounds in the last 10 minutes" --no-meta   free, local search
  python soundscope.py digest --minutes 30                 Muse Spark summarizes what happened
  python soundscope.py map                                 picture: where every sound came from
  python soundscope.py log --last 20                       what was heard, one sentence each
  python soundscope.py enroll "phone ringing"              record your own example (3 takes)
  python soundscope.py examples                            list your examples
  python soundscope.py listen [--meta] [--mount tripod]    mics only: log sounds, no camera
  python soundscope.py budget                              today's estimated Meta spend

The viewer (scope.py) and the arm app (arm_follow.py --memory) write the same memory, in
recordings/memory/ (git-ignored). Examples live in references/ (git-ignored): the free label
for later sounds, and the labeled rows Muse Spark compares a new sound with.
"""
import argparse
import json
import queue
import time

import numpy as np

import aim
import sound_memory
from meta_api import ENV_FILE, Budget, Meta, MetaError, load_env, wav_bytes


def get_meta(args):
    if getattr(args, 'no_meta', False):
        return None
    try:
        return Meta()
    except MetaError as exc:
        print(f'(Meta unavailable, answering locally: {exc})')
        return None


def cmd_search(args):
    memory = sound_memory.Memory(days=args.days)
    meta = get_meta(args)
    result = sound_memory.search(args.question, memory, meta)
    print(result['answer'])
    if result['matches']:
        print()
        for rec in result['matches']:
            print(' - ' + sound_memory.phrase(rec))
    if meta is not None:
        print(f'\n({result["source"]}; {meta.budget.short()})')


def cmd_digest(args):
    """Muse Spark summarizes a stretch of the memory: what happened, where, how often."""
    memory = sound_memory.Memory(days=args.days)
    meta = get_meta(args)
    span = f'in the last {args.minutes:g} minutes' if args.minutes else 'today'
    question = (f'Summarize the sounds {span} in 2-4 sentences: group similar sounds with counts, say where '
                'they came from and when, mention who talked and any alerts.')
    result = sound_memory.search(question, memory, meta, limit=8)
    print(result['answer'])
    if meta is None and result['matches']:
        for rec in result['matches']:
            print(' - ' + sound_memory.phrase(rec))


def cmd_map(args):
    """A top-down picture of where the remembered sounds came from (free, no Meta)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = sound_memory.Memory(days=args.days).recent(args.last)
    if not records:
        raise SystemExit('No sounds in memory yet.')
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # clockwise: + azimuth = right of the camera
    labels = {}
    for rec in records:
        w = rec.get('where') or {}
        az = np.radians(w.get('azimuth_deg') or 0.0)
        known = w.get('distance_m') is not None
        r = w['distance_m'] if known else 2.5
        what = sound_memory.what_it_was(rec)
        key = what.replace('probably ', '')[:24]
        color = labels.setdefault(key, f'C{len(labels) % 10}')
        snr = (rec.get('loudness') or {}).get('snr_db') or 5
        ax.scatter(az, r, s=20 + 6 * max(0, snr), color=color, alpha=0.75,
                   edgecolors='black' if known else color, facecolors=color if known else 'none')
    for key, color in labels.items():
        ax.scatter([], [], color=color, label=key)
    ax.set_rmax(3.0)
    ax.set_title('Where sounds came from (camera at the center, facing up)\n'
                 'filled = depth-measured distance, hollow = direction only', fontsize=9)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.15), fontsize=8, ncol=2)
    out = sound_memory.DATA_DIR / 'sound_map.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved {out}')


def cmd_log(args):
    records = sound_memory.Memory(days=args.days).recent(args.last)
    if not records:
        print('No sounds in memory yet. Run scope.py or `soundscope.py listen` first.')
    for rec in records:
        print(sound_memory.phrase(rec))


def record_take(seconds):
    """Record `seconds` from the array; returns the mean of the raw mics as WAV (like a heard sound)."""
    from arm_follow import record
    from mic_test import find_array, raw_channels
    raw = record(find_array(), seconds)
    chans = raw_channels()
    x = raw[:, chans].astype(float).mean(axis=1)
    total = np.concatenate([[0.0], np.cumsum(np.abs(x))])
    env = np.pad((total[800:] - total[:-800]) / 800, (400, 399), mode='edge')
    active = np.nonzero(env > env.max() * 0.08)[0]
    if len(active):
        x = x[max(0, active[0] - 1600):active[-1] + 3200]  # keep the sound with a little margin
    return wav_bytes(x)


def cmd_enroll(args):
    bank = sound_memory.ReferenceBank()
    print(f'Recording {args.takes} example(s) of "{args.label}", {args.seconds:g} s each.')
    for take in range(1, args.takes + 1):
        input(f'  Take {take}: press Enter, then make the sound...')
        path = bank.add(args.label, record_take(args.seconds))
        print(f'    saved {path.relative_to(bank.folder.parent)}')
    print(f'Done. Examples now: {", ".join(bank.labels())}')


def cmd_examples(args):
    bank = sound_memory.ReferenceBank()
    if not bank.items:
        print('No examples yet. Record one: python soundscope.py enroll "phone ringing"')
    for label in bank.labels():
        print(f'{label}: {sum(1 for lab, _, _ in bank.items if lab == label)} take(s)')


def cmd_budget(args):
    print(json.dumps(Budget().summary(), indent=1))


def cmd_listen(args):
    """Mic-only logging: every sound goes to memory with its direction (relative to the camera's
    position in the mount profile, or to the array when no camera is set up)."""
    from arm_follow import Listener
    from insight import FEATURES, Enricher
    cfg = aim.load_config(profile=args.mount)
    memory = sound_memory.Memory()
    meta = get_meta(args) if args.meta else None
    references = sound_memory.ReferenceBank()
    enricher = None
    if meta is not None or references.items:
        def show(rid, kind, payload):
            rec = memory.get(rid)
            if kind == 'notice':
                print('  ' + str(payload))
            elif kind == 'captions':
                for turn in payload:
                    print(f"  {turn['name']} ({turn['where']}): {turn['text']}")
            elif kind == 'alert':
                print(f"  ALERT: {payload['watch']} ({payload['how']})")
            elif rec is not None and kind in ('identify', 'label'):
                print('  -> ' + sound_memory.phrase(rec))
        features = set(FEATURES) - {'context', 'outline'} if meta is not None else set()
        enricher = Enricher(memory, meta, features, on_result=show, references=references)
        enricher.watch = list(args.watch or [])
        enricher.start()
    inbox = queue.Queue()
    listener = Listener(inbox.put, updates=True, threshold_db=cfg.get('listen', {}).get('threshold_db', 0.5))
    listener.start()
    if not listener.wait_ready() or listener.error:
        raise SystemExit(f'Microphone problem: {listener.error or "did not start"}')
    print('Listening (mics only). Ctrl+C to stop.')
    tracks = {}
    end = time.monotonic() + args.seconds if args.seconds else float('inf')
    try:
        while time.monotonic() < end:
            try:
                ev = inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            if ev['kind'] == 'error':
                raise SystemExit(f"Microphone problem: {ev['message']}")
            if ev['kind'] not in ('update', 'sound') or 'fix' not in ev:
                continue
            direction = aim.array_to_camera(ev['fix']['direction'], cfg)
            az = float(np.degrees(np.arctan2(direction[0], direction[2])))
            tracks.setdefault(ev.get('event_id'), []).append((ev.get('observed_at', 0.0), az, 0.0))
            if ev['kind'] != 'sound' or not sound_memory.worth_remembering(ev['fix']):
                continue
            wav = ev.get('wav')
            profile = sound_memory.acoustic_profile(sound_memory.wav_samples(wav)) if wav else None
            onset = ev.get('onset_at') or ev.get('observed_at')
            started = time.time() - (time.monotonic() - onset) if onset else None
            rec = memory.add(sound_memory.make_record(ev['fix'], direction, profile=profile, level=ev.get('level'),
                                                      mode='mics-only', now=started))
            print(sound_memory.phrase(rec))
            if enricher is not None:
                enricher.submit({'record': rec, 'wav': wav, 'speech_wav': ev.get('speech_wav'),
                                 'track': tracks.pop(ev.get('event_id'), []),
                                 'clip_end_at': ev.get('clip_end_at', 0.0), 'clip_seconds': ev.get('clip_seconds', 0.0)})
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        if enricher is not None:
            enricher.stop()
            time.sleep(0.3)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env-file', help='also load keys from this .env file')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search', help='ask about past sounds')
    s.add_argument('question')
    s.add_argument('--days', type=int, default=2, help='how many days of memory to search')
    s.add_argument('--no-meta', action='store_true', help='local search only (no Meta request)')
    s = sub.add_parser('digest', help='Muse Spark summarizes what was heard')
    s.add_argument('--minutes', type=float, help='only the last N minutes (default: today)')
    s.add_argument('--days', type=int, default=1)
    s.add_argument('--no-meta', action='store_true', help='local list instead of a summary')
    s = sub.add_parser('map', help='picture of where sounds came from')
    s.add_argument('--days', type=int, default=1)
    s.add_argument('--last', type=int, default=200)
    s = sub.add_parser('log', help='recent sounds, one sentence each')
    s.add_argument('--days', type=int, default=1)
    s.add_argument('--last', type=int, default=20)
    s = sub.add_parser('enroll', help='record your own example of a sound')
    s.add_argument('label')
    s.add_argument('--takes', type=int, default=3)
    s.add_argument('--seconds', type=float, default=3.0)
    sub.add_parser('examples', help='list your recorded examples')
    sub.add_parser('budget', help="today's estimated Meta spend")
    s = sub.add_parser('listen', help='mics only: log every sound to memory')
    s.add_argument('--mount', choices=sorted(aim.profiles()))
    s.add_argument('--meta', action='store_true', help='label sounds and caption speech with Meta models')
    s.add_argument('--watch', action='append', help='alert when this sound happens (repeatable)')
    s.add_argument('--seconds', type=float)
    args = p.parse_args()
    load_env(ENV_FILE)
    if args.env_file:
        load_env(args.env_file)
    {'search': cmd_search, 'digest': cmd_digest, 'map': cmd_map, 'log': cmd_log, 'enroll': cmd_enroll,
     'examples': cmd_examples, 'budget': cmd_budget, 'listen': cmd_listen}[args.cmd](args)


if __name__ == '__main__':
    main()
