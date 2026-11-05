"""Point one bounded step toward scope's measured depth candidate. Never grasp.

Run scope.py separately with target publishing enabled. This consumer opens no
camera or microphone and does not call Meta. Dry run is the default:

  python arm_target.py --seconds 30
  python arm_target.py --live

Live motion additionally requires safety.live=true and typing MOVE. A candidate
is an acoustic/depth hypothesis, not confirmation that an object made the sound.
"""
import argparse
import asyncio
from collections import deque
import copy
import math
from pathlib import Path
import time

import aim
import arm_follow
from meta_api import ENV_FILE, load_env
from target_link import read_target


DEFAULT_TARGET_FILE = Path(__file__).with_name('recordings') / 'latest_target.json'


class Notices:
    """Keep waiting/rejection messages readable while polling the same file."""
    def __init__(self, interval=2.0):
        self.interval, self.key, self.last = interval, None, -math.inf

    def report(self, key, message):
        now = time.monotonic()
        if key != self.key or now - self.last >= self.interval:
            print(message, flush=True)
            self.key, self.last = key, now


def diagnostic_config(cfg):
    """This bridge only permits slow, single-step yaw pointing with a fixed rig."""
    cfg = copy.deepcopy(cfg)
    if cfg['camera']['mount'] != 'base' or cfg['array'].get('on_arm', False):
        raise SystemExit('The depth-target bridge requires a stationary camera and microphone beside the arm.')
    safety = cfg['safety']
    safety['max_step_deg'] = min(float(safety['max_step_deg']), 5.0)
    safety['speed_deg_s'] = min(float(safety['speed_deg_s']), 5.0)
    safety['accel_deg_s2'] = min(float(safety['accel_deg_s2']), 10.0)
    safety['single_step_per_sound'] = True
    cfg['aim']['use_pitch'] = False
    return cfg


class TargetFollower:
    """Consume validated final packets once, using measured points rather than assumed range."""
    def __init__(self, cfg, arm, gate, pointer, started_at, notices=None, target_file=None):
        self.cfg, self.arm, self.gate = cfg, arm, gate
        self.follower = arm_follow.Follower(cfg, arm, gate, pointer, None,
                                            asyncio.Queue(), None, None, True)
        self.not_before = started_at
        self.seen = deque(maxlen=256)
        self.notices = notices or Notices()
        self.target_file = target_file

    async def consume(self, packet):
        """packet must have passed target_link.read_target for this exact configuration."""
        identity = (packet['session_id'], packet['event_id'])
        if identity in self.seen:
            self.notices.report('duplicate', 'Waiting for a new final sound event; this candidate was already handled.')
            return False
        ev = {'event_id': identity, 'observed_at': packet['observed_at'], 'fix': packet['fix']}
        why = self.follower._sound_rejection(ev) or self.gate.accept(packet['fix'])
        if why:
            self.notices.report('weak', f'Not moving: {why}')
            return False
        point = packet['base_point_mm']
        wanted = aim.j1_to_face(point, self.follower.pointer)
        if wanted is None:
            self.notices.report('unpointable', 'Not moving: candidate is too close or behind the pointer geometry.')
            return False
        # A valid final event gets one decision, including when already facing it
        # or rate limited. Never replay an old packet until it eventually moves.
        self.seen.append(identity)
        def still_current():
            if self.target_file is None:
                return None
            try:
                current = read_target(self.target_file, self.cfg, now=time.monotonic(),
                                      max_age_s=self.cfg['safety'].get('max_sound_age_s', 1.5),
                                      not_before=self.not_before)
            except (ValueError, OSError) as exc:
                return f'candidate withdrawn before motion: {exc}'
            if ((current['session_id'], current['event_id']) != identity or
                    current['observed_at'] != packet['observed_at'] or
                    current['base_point_mm'] != packet['base_point_mm']):
                return 'candidate changed before motion; waiting for the new final event'
            return None
        line = (f'{arm_follow.stamp()} depth-supported acoustic candidate '
                f'({point[0]:.0f}, {point[1]:.0f}, {point[2]:.0f}) mm in arm coordinates; '
                'pointing only, sound source unverified')
        moved = await self.follower.turn_to(wanted, line, sound_event=ev, before_move=still_current)
        if moved:
            # scope runs separately, so its listener cannot be muted here. Reject
            # anything captured during this move or its motor-noise settling time.
            self.not_before = time.monotonic() + self.cfg['safety']['mute_after_move_s']
        return moved


async def _stop_and_close(arm):
    try:
        try:
            await asyncio.wait_for(arm.stop(), timeout=2.0)
        except (Exception, asyncio.CancelledError) as exc:
            print(f'Arm stop could not be confirmed: {exc}. Use the physical E-stop.', flush=True)
    finally:
        try:
            await asyncio.wait_for(arm.close(), timeout=2.0)
        except (Exception, asyncio.CancelledError) as exc:
            print(f'Arm connection close could not be confirmed: {exc}', flush=True)


async def run(args, cfg):
    cfg = diagnostic_config(cfg)
    started_at = time.monotonic()
    arm = await arm_follow.open_arm(args, cfg)
    try:
        gate = aim.Gate(cfg, await arm.joints())
        pointer = await arm.pointer(cfg)
        notices = Notices()
        follower = TargetFollower(cfg, arm, gate, pointer, started_at, notices, args.target_file)
        duration = getattr(args, 'seconds', None)
        end = time.monotonic() + duration if duration is not None else math.inf
        mode = 'DRY RUN: nothing will move.' if arm.dry else 'LIVE: slow candidate pointing only.'
        print(f'{mode} One yaw step of at most {cfg["safety"]["max_step_deg"]:g} degrees per fresh final event.\n'
              f'Reading {args.target_file}. Camera and mic remain owned by scope. Ctrl+C stops and closes.', flush=True)
        while time.monotonic() < end:
            try:
                packet = read_target(args.target_file, cfg, now=time.monotonic(),
                                     max_age_s=cfg['safety'].get('max_sound_age_s', 1.5),
                                     not_before=follower.not_before)
            except (ValueError, OSError) as exc:
                notices.report('unavailable', f'Waiting; not moving: {exc}')
            else:
                await follower.consume(packet)
            await asyncio.sleep(min(0.1, max(0.0, end - time.monotonic())))
    finally:
        await _stop_and_close(arm)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mount', choices=sorted(aim.profiles()), default='carrier_v3_desk')
    parser.add_argument('--env-file', help='also load keys from this .env file')
    parser.add_argument('--target-file', type=Path, default=DEFAULT_TARGET_FILE)
    parser.add_argument('--live', action='store_true', help='enable real motion; also requires config and MOVE confirmation')
    parser.add_argument('--seconds', type=float, help='stop after this many seconds')
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be a finite number above zero')
    return args


def main():
    args = parse_args()
    load_env(ENV_FILE)
    if args.env_file:
        load_env(args.env_file)
    cfg = aim.load_config(profile=args.mount)
    try:
        asyncio.run(run(args, cfg))
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
