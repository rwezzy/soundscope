"""Turn the Lite 6, and the camera on it, toward sounds. Dry run unless you arm it on purpose.

  python arm_follow.py simulate 40 -30 170      fake sounds at those bearings (+ = right); no mic, no arm
  python arm_follow.py listen                   real mic; prints what the arm would do
  python arm_follow.py listen --target "phone ringing"   only turn for sounds Muse Spark confirms
  python arm_follow.py listen --voice           press Enter, then speak ("find my phone", "stop")
  python arm_follow.py where 0 0 1000           camera point (mm) -> arm base frame, reach check
  python arm_follow.py viam-config              JSON to paste into the Viam app
  python arm_follow.py check                    connect through Viam and read the arm; no motion
  python arm_follow.py listen --live            real motion: also needs live = true in arm_config.toml

Where: the mic array (physics). What: Muse Spark, reading a spectrogram of the sound.
Spoken commands: Muse Voice Transcribe + Muse Spark.
No image detection anywhere. Settings and offsets: arm_config.toml. Setup: ARM_SETUP.md.
"""
import argparse
import asyncio
import json
import math
import os
import queue
import threading
import time

import numpy as np
import sounddevice as sd

import aim
from doa import FS, HOP, N_FFT, Localizer, Tracker
from meta_api import ENV_FILE, Meta, MetaError, load_env, wav_bytes
from mic_test import CHANNELS, as_float, bearing, find_array, raw_channels, record, side

ASR_CHANNEL = 1  # the XVF3800's processed output tuned for speech recognition

LIVE_CHECKLIST = """Before the arm moves:
  - The E-stop button is in your hand or within reach, and nobody is inside the arm's reach.
  - The arm is parked in a safe pose (set it in UFACTORY Studio). Only J1{j5} will move from it.
  - Nothing within 50 cm of the arm's swing.
  - Limits: {speed}°/s, at most {step}° per move, J1 within {lo:+.0f}°..{hi:+.0f}° of where it is now."""


def stamp():
    return time.strftime('%H:%M:%S')


class Listener(threading.Thread):
    """Mic thread: finds sounds with the direction tracker and keeps recent audio for Meta."""

    def __init__(self, post, keep_s=8.0, updates=False):
        super().__init__(daemon=True)
        self.dev, self.chans, self.post, self.updates = find_array(), raw_channels(), post, updates
        self.buf = np.zeros((int(keep_s * FS), CHANNELS), np.float32)
        self.pos = 0
        self.mute_until = 0.0
        self.voice = None  # [samples still needed, chunks, done event] while a command is recorded
        self.ready = threading.Event()
        self.error = None

    def run(self):
        try:
            loc = Localizer()
            print('Measuring the room for 2 s, stay quiet...')
            loc.calibrate(as_float(record(self.dev, 2.0), self.chans))
            tracker, q = Tracker(loc), queue.Queue()
            frame, pending = np.zeros((N_FFT, len(self.chans))), np.zeros((0, len(self.chans)))
            with sd.InputStream(device=self.dev, channels=CHANNELS, samplerate=FS, dtype='float32',
                                callback=lambda indata, *_: q.put(indata.copy())):
                self.ready.set()
                while True:
                    block = q.get()
                    self._keep(block)
                    pending = np.concatenate([pending, block[:, self.chans]])
                    while len(pending) >= HOP:
                        frame = np.concatenate([frame[HOP:], pending[:HOP]])
                        pending = pending[HOP:]
                        out = tracker.push(loc.stft(frame)[0])
                        if not (out and out[1]) or time.monotonic() < self.mute_until:
                            continue
                        if out[0] == 'end':
                            clip = self._recent(min(out[1]['seconds'] + 0.9, 6.0))  # the sound + its quiet tail
                            self.post({'kind': 'sound', 'fix': out[1],
                                       'wav': wav_bytes(clip[:, self.chans].mean(axis=1))})
                        elif self.updates:  # running estimate while the sound is still playing
                            self.post({'kind': 'update', 'fix': out[1]})
        except Exception as exc:  # report mic trouble in the main thread
            self.error = exc
            self.ready.set()

    def _keep(self, block):
        self.buf[(self.pos + np.arange(len(block))) % len(self.buf)] = block
        self.pos += len(block)
        if self.voice:
            need, chunks, done = self.voice
            chunks.append(block[:need, ASR_CHANNEL].copy())
            self.voice[0] -= len(chunks[-1])
            if self.voice[0] <= 0:
                self.voice = None
                done.set()

    def _recent(self, seconds):
        n = min(int(seconds * FS), self.pos, len(self.buf))
        return self.buf[(self.pos - n + np.arange(n)) % len(self.buf)]

    def record_command(self, seconds=4.0):
        """Record a spoken command from the speech channel; sounds are ignored meanwhile."""
        chunks, done = [], threading.Event()
        self.mute_until = time.monotonic() + seconds + 1.0  # don't turn toward your own voice
        self.voice = [int(seconds * FS), chunks, done]
        done.wait(seconds + 3)
        return wav_bytes(np.concatenate(chunks)) if chunks else None


def voice_loop(listener, post):
    while True:
        input()
        print('  Listening for your command (4 s)...', flush=True)
        wav = listener.record_command()
        if wav:
            post({'kind': 'voice', 'wav': wav})


class DryRunArm:
    """Stands in for the arm: never touches hardware, just remembers where it 'is'."""
    dry = True

    def __init__(self, cfg):
        self.q = [float(v) for v in cfg['look']['start_joints_deg']]

    async def joints(self):
        return list(self.q)

    async def pointer(self, cfg):
        return cfg['look']

    async def move(self, target):
        self.q = list(target)

    async def stop(self):
        pass

    async def close(self):
        pass


class ViamArm:
    """The real Lite 6 through Viam. Only created after every safety check has passed."""
    dry = False

    def __init__(self, robot, arm, cfg):
        self.robot, self.arm, self.cfg = robot, arm, cfg
        self.direction_checked = False

    @classmethod
    async def connect(cls, cfg):
        from viam.components.arm import Arm
        from viam.robot.client import RobotClient
        missing = [k for k in ('VIAM_ADDRESS', 'VIAM_API_KEY', 'VIAM_API_KEY_ID') if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"Set {', '.join(missing)} in meta-soundscope/.env (Viam app > CONNECT > API keys).")
        options = RobotClient.Options.with_api_key(api_key=os.environ['VIAM_API_KEY'],
                                                   api_key_id=os.environ['VIAM_API_KEY_ID'])
        robot = await RobotClient.at_address(os.environ['VIAM_ADDRESS'], options)
        return cls(robot, Arm.from_robot(robot, cfg['viam']['arm_name']), cfg)

    async def prepare(self):
        """Refuse to take over a moving arm, and slow it down before anything moves."""
        if await self.arm.is_moving():
            raise SystemExit('The arm is already moving; not taking control.')
        s = self.cfg['safety']
        await self.arm.do_command({'set_speed': float(s['speed_deg_s']),
                                   'set_acceleration': float(s['accel_deg_s2'])})

    async def joints(self):
        return list((await self.arm.get_joint_positions()).values)

    async def pointer(self, cfg):
        """Measure what should face the sound through Viam's frame system, as if J1 were 0."""
        from viam.proto.common import Pose, PoseInFrame
        j1 = (await self.joints())[0]
        if cfg['camera']['mount'] == 'arm':
            query = PoseInFrame(reference_frame=cfg['viam']['camera_name'], pose=Pose(o_z=1))
            pose = (await self.robot.transform_pose(query, 'world')).pose
        else:
            pose = await self.arm.get_end_position()
        x, y = aim.rot_z(-j1)[:2, :2] @ np.array([pose.x, pose.y])
        if math.hypot(pose.o_x, pose.o_y) > 0.3:  # looks sideways: use its real heading
            heading = math.degrees(math.atan2(pose.o_y, pose.o_x)) - j1
        else:  # points mostly up or down: treat it as facing straight out along the arm
            heading = math.degrees(math.atan2(y, x))
        return {'reach_mm': float(x), 'lateral_mm': float(y), 'heading_deg': (heading + 180) % 360 - 180,
                'height_mm': float(pose.z), 'pitch_deg': math.degrees(math.asin(max(-1.0, min(1.0, pose.o_z))))}

    async def move(self, target):
        from viam.components.arm import JointPositions
        s = self.cfg['safety']
        before, end_before = await self.joints(), await self.arm.get_end_position()
        started = time.monotonic()
        await self.arm.move_to_joint_positions(JointPositions(values=list(target)))
        while await self.arm.is_moving():
            if time.monotonic() - started > 30:
                await self.stop()
                raise SystemExit('A move took over 30 s. Stopped the arm.')
            await asyncio.sleep(0.05)
        elapsed = time.monotonic() - started
        after = await self.joints()
        turned = max(abs(a - b) for a, b in zip(after, before))
        if turned > 3 and turned / max(elapsed, 1e-3) > 1.3 * s['speed_deg_s']:
            await self.stop()
            raise SystemExit(f"The arm turned {turned:.0f}° in {elapsed:.1f} s, faster than "
                             f"{s['speed_deg_s']}°/s. Stopped. Check the speed settings in Viam.")
        # First real turn: make sure J1 swings the arm the way this program's base frame assumes.
        end_after = await self.arm.get_end_position()
        if (not self.direction_checked and abs(after[0] - before[0]) > 3
                and min(math.hypot(end_before.x, end_before.y), math.hypot(end_after.x, end_after.y)) > 50):
            swing = math.degrees(math.atan2(end_after.y, end_after.x) - math.atan2(end_before.y, end_before.x))
            if ((swing + 180) % 360 - 180) * (after[0] - before[0]) < 0:
                await self.stop()
                raise SystemExit('J1 swung the arm the opposite way from what this program assumes. '
                                 'Stopped; see "Base frame check" in ARM_SETUP.md.')
            self.direction_checked = True

    async def stop(self):
        await self.arm.stop()

    async def close(self):
        await self.robot.close()


class Follower:
    def __init__(self, cfg, arm, gate, pointer, meta, inbox, listener, target, following):
        self.cfg, self.arm, self.gate, self.pointer, self.meta = cfg, arm, gate, pointer, meta
        self.inbox, self.listener = inbox, listener
        self.target, self.following, self.once = target, following, False

    def _mute(self, seconds):
        if self.listener:
            self.listener.mute_until = time.monotonic() + seconds

    def _where(self, point):
        """Short note on where the sound probably is: in the base camera's view? within grasp?"""
        notes = []
        if self.cfg['camera']['mount'] == 'base':
            x = aim.view_position(point, self.cfg)
            notes.append('behind the camera' if x is None else
                         f'in camera view {x:.0%} across' if 0 <= x <= 1 else
                         'off camera ' + ('left' if x < 0 else 'right'))
        ok, radius = aim.reach_check(point, self.cfg)
        notes.append(f"{'within' if ok else 'out of'} reach ({radius / 1000:.1f} m from base, assumed distance)")
        return ' | ' + ', '.join(notes)

    async def turn_to(self, want_j1, line, want_j5=None):
        joints = await self.arm.joints()
        new, note = self.gate.next_joints(joints, want_j1, want_j5, now=time.monotonic())
        if new is None:
            print(f'{line} - {note}')
            return False
        tag = '  [dry run]' if self.arm.dry else ''
        while new is not None:
            tilt = f', J5 {joints[4]:.0f}° -> {new[4]:.0f}°' if new[4] != joints[4] else ''
            print(f'{line}\n          J1 {joints[0]:.0f}° -> {new[0]:.0f}°{tilt}{note}{tag}')
            self._mute(math.inf)  # motor noise is not a sound to chase
            try:
                await self.arm.move(new)
            finally:
                self.gate.moved(time.monotonic())
                self._mute(self.cfg['safety']['mute_after_move_s'])
            if not self.inbox.empty():  # something newer to handle first
                break
            joints = await self.arm.joints()
            new, note = self.gate.next_joints(joints, want_j1, want_j5, continuing=True)
            line = f'{stamp()} continuing'
        return True

    async def on_sound(self, ev):
        fix = ev['fix']
        line = (f"{stamp()} heard {side(aim.base_bearing(fix['direction'], self.cfg)[0])} "
                f"(z {fix['z']:.0f}, SNR {fix['snr_db']:+.0f} dB)")
        if not self.following:
            print(line + ' - not following; press Enter and say "follow"')
            return
        why = self.gate.accept(fix)
        if why:
            print(f'{line} - {why}')
            return
        if self.target and ev.get('wav') is not None:
            try:
                verdict = await asyncio.get_running_loop().run_in_executor(
                    None, self.meta.sound_matches, ev['wav'], self.target)
            except MetaError as exc:
                print(f'{line} - Muse Spark check failed, not moving: {exc}')
                return
            line += f" | Muse Spark: {verdict['heard']}"
            if not verdict['match'] or verdict['confidence'] < 0.5:
                print(f'{line} - not "{self.target}", staying put')
                return
        point = aim.sound_point(fix['direction'], self.cfg)
        want = aim.j1_to_face(point, self.pointer)
        if want is None:
            print(f'{line} - too close to the arm to aim at')
            return
        want_j5 = None
        if self.cfg['aim']['use_pitch']:
            want_j5 = self.gate.start[4] + self.cfg['aim']['pitch_sign'] * aim.pitch_to_face(point, self.pointer, want)
        if await self.turn_to(want, line + self._where(point), want_j5) and self.once:
            self.following = False
            print('          found it; holding still. Say "follow" to keep tracking.')

    async def on_voice(self, ev):
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self.meta.transcribe, ev['wav'])
            if not text.strip():
                print(f"{stamp()} didn't catch that")
                return
            cmd = await loop.run_in_executor(None, self.meta.parse_command, text)
        except MetaError as exc:
            print(f'{stamp()} voice command failed: {exc}')
            return
        sound = f' "{cmd["sound"]}"' if cmd['sound'] else ''
        print(f'{stamp()} you said "{text}" -> {cmd["action"]}{sound}  ({cmd["reply"]})')
        if cmd['action'] == 'stop':
            self.following = False
            await self.arm.stop()
        elif cmd['action'] in ('follow', 'find'):
            self.following, self.once, self.target = True, cmd['action'] == 'find', cmd['sound'] or None
        elif cmd['action'] == 'home':
            self.following = False
            j5 = self.gate.start[4] if self.cfg['aim']['use_pitch'] else None
            await self.turn_to(self.gate.start[0], f'{stamp()} going back to the start', j5)


async def feed_simulated(bearings, post, gap):
    for b in bearings:
        az = math.radians(-b)
        post({'kind': 'sound', 'wav': None,
              'fix': {'azimuth': -b % 360, 'elevation': 0.0, 'direction': np.array([math.cos(az), math.sin(az), 0.0]),
                      'z': 50.0, 'snr_db': 10.0, 'seconds': 1.0}})
        await asyncio.sleep(gap)
    post({'kind': 'done'})


async def open_arm(args, cfg):
    if not getattr(args, 'live', False):
        return DryRunArm(cfg)
    if not cfg['safety']['live']:
        raise SystemExit('Not moving: set live = true under [safety] in arm_config.toml first.')
    arm = await ViamArm.connect(cfg)
    try:
        gate = aim.Gate(cfg, await arm.joints())
        lo, hi = (r - gate.start[0] for r in gate.ranges[0])
        print(LIVE_CHECKLIST.format(j5=' and J5' if cfg['aim']['use_pitch'] else '', lo=lo, hi=hi,
                                    speed=cfg['safety']['speed_deg_s'], step=cfg['safety']['max_step_deg']))
        if input('Type MOVE to let the arm move: ').strip() != 'MOVE':
            raise SystemExit('Not armed. Nothing moved.')
        await arm.prepare()
    except BaseException:
        await arm.close()
        raise
    return arm


async def follow(args, cfg):
    loop = asyncio.get_running_loop()
    inbox = asyncio.Queue()

    def post(ev):
        loop.call_soon_threadsafe(inbox.put_nowait, ev)

    for w in aim.warnings(cfg):
        print('Warning: ' + w)
    target, voice = getattr(args, 'target', None), getattr(args, 'voice', False)
    try:
        meta = Meta() if (target or voice) else None
    except MetaError as exc:
        raise SystemExit(f'Meta: {exc}')
    arm = await open_arm(args, cfg)
    try:
        gate = aim.Gate(cfg, await arm.joints())
        pointer = await arm.pointer(cfg)
        what = 'camera' if cfg['camera']['mount'] == 'arm' else 'gripper (camera sits by the base)'
        print(f"{'DRY RUN, nothing will move. ' if arm.dry else ''}Pointing the {what}. Start J1 {gate.start[0]:.0f}°, "
              f"allowed {gate.ranges[0][0]:.0f}°..{gate.ranges[0][1]:.0f}°." +
              (f' Only sounds Muse Spark hears as "{target}".' if target else ''))
        listener = None
        if args.cmd == 'simulate':
            asyncio.create_task(feed_simulated(args.bearings, post, cfg['safety']['min_interval_s'] + 0.3))
        else:
            listener = Listener(post)
            listener.start()
            await loop.run_in_executor(None, listener.ready.wait)
            if listener.error:
                raise SystemExit(f'Microphone problem: {listener.error}')
            if voice:
                threading.Thread(target=voice_loop, args=(listener, post), daemon=True).start()
            print('Listening. ' + ('Press Enter, then say a command. ' if voice else '') + 'Ctrl+C to stop.\n')
        follower = Follower(cfg, arm, gate, pointer, meta, inbox, listener, target, following=not voice or bool(target))
        end = time.monotonic() + args.seconds if getattr(args, 'seconds', None) else math.inf
        while time.monotonic() < end:
            try:
                ev = await asyncio.wait_for(inbox.get(), timeout=min(1.0, max(0.05, end - time.monotonic())))
            except asyncio.TimeoutError:
                continue
            if ev['kind'] == 'done':
                break
            await (follower.on_sound(ev) if ev['kind'] == 'sound' else follower.on_voice(ev))
    finally:
        await arm.stop()
        await arm.close()


def cmd_where(args, cfg):
    if cfg['camera']['mount'] != 'base':
        raise SystemExit('The camera is set as arm-mounted; `check` gets its pose from Viam instead.')
    p = aim.camera_to_base(np.array(args.xyz, float), cfg)
    ok, radius = aim.reach_check(p, cfg)
    want = aim.j1_to_face(p, cfg['look'])
    print(f'camera point {args.xyz} mm -> arm base frame: x {p[0]:.0f}, y {p[1]:.0f}, z {p[2]:.0f} mm')
    print(f"{radius:.0f} mm from the base axis: {'within reach' if ok else 'out of reach'}")
    print('J1 to point the gripper at it: ' + (f'{want:.0f}°' if want is not None else 'too close to aim'))


def cmd_viam_config(args, cfg):
    frames = aim.viam_frames(cfg)
    print('1) Module: in the Viam app, + > Configuration block > search "lite6" (module viam:ufactory).')
    print(f"   Name it \"{cfg['viam']['arm_name']}\" and paste these attributes (put your arm's IP in host):")
    print(json.dumps(aim.viam_arm_attributes(cfg), indent=2))
    print('\n2) Arm frame (base at the world origin):')
    print(json.dumps(frames['arm'], indent=2))
    print(f"\n3) Camera frame, on the camera component \"{cfg['viam']['camera_name']}\". Use ONE of these:")
    print('   a) Camera beside the base (now):')
    print(json.dumps(frames['camera_beside_base'], indent=2))
    print('   b) Camera on the 3D-printed mount (later):')
    print(json.dumps(frames['camera_on_arm'], indent=2))
    gap = aim.fingertip_clearance(cfg)
    ok = gap >= cfg['camera']['min_depth_mm']
    print(f"\nMount check: fingertips {gap:.0f} mm in front of the lens "
          f"({'OK' if ok else 'too close'}; depth starts at {cfg['camera']['min_depth_mm']} mm).")


async def cmd_check(args, cfg):
    arm = await ViamArm.connect(cfg)
    try:
        joints = await arm.joints()
        print('Joints (deg): ' + ', '.join(f'J{i + 1} {v:.1f}' for i, v in enumerate(joints)))
        measured = await arm.pointer(cfg)
        print('Pointer measured through Viam (as if J1 = 0):')
        for key in ('reach_mm', 'lateral_mm', 'heading_deg', 'height_mm', 'pitch_deg'):
            print(f"  {key:12} {measured[key]:8.1f}   (arm_config.toml [look]: {cfg['look'][key]})")
        print('Nothing moved.')
    finally:
        await arm.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env-file', help='also load keys from this .env file')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('simulate', help='fake sounds, no mic, no arm')
    s.add_argument('bearings', type=float, nargs='+', help='degrees, + = right of the array front edge')
    s = sub.add_parser('listen', help='real mic')
    s.add_argument('--target', help='only turn for this sound, as judged by Muse Spark')
    s.add_argument('--voice', action='store_true', help='spoken commands via Muse Voice Transcribe + Muse Spark')
    s.add_argument('--live', action='store_true', help='really move the arm (needs live = true in the config)')
    s.add_argument('--seconds', type=float, help='stop after this long')
    s = sub.add_parser('where', help='camera point -> arm base frame')
    s.add_argument('xyz', type=float, nargs=3, help='mm: x right, y down, z out of the lens')
    sub.add_parser('viam-config', help='print the Viam configuration to paste')
    sub.add_parser('check', help='read the arm through Viam; no motion')
    args = p.parse_args()

    load_env(ENV_FILE)
    if args.env_file:
        load_env(args.env_file)
    cfg = aim.load_config()
    try:
        if args.cmd == 'where':
            cmd_where(args, cfg)
        elif args.cmd == 'viam-config':
            cmd_viam_config(args, cfg)
        elif args.cmd == 'check':
            asyncio.run(cmd_check(args, cfg))
        else:
            asyncio.run(follow(args, cfg))
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
