"""Offline tests for the arm code: python -m unittest test_arm (no arm, mic or network needed)."""
import asyncio
import copy
import importlib.util
import math
import tomllib
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import aim

CFG = aim.load_config()
# Fixed geometry, so the tests don't depend on what you measured into arm_config.toml.
CFG['array'].update(mount='flat', position_mm=[200, -250, 0], front_yaw_deg=0)
CFG['camera'].update(mount='base')
CFG['camera']['base'].update(position_mm=[150, 250, 60], yaw_deg=0, pitch_deg=0)


def cfg_with(**sections):
    cfg = copy.deepcopy(CFG)
    for section, values in sections.items():
        cfg[section].update(values)
    return cfg


class Geometry(unittest.TestCase):
    def test_centered_pointer_faces_the_target_bearing(self):
        p = {'reach_mm': 250, 'lateral_mm': 0, 'heading_deg': 0}
        self.assertAlmostEqual(aim.j1_to_face([0, 1000, 0], p), 90)
        self.assertAlmostEqual(aim.j1_to_face([1000, -1000, 0], p), -45)

    def test_offset_camera_line_passes_through_target(self):
        p = {'reach_mm': 250, 'lateral_mm': 60, 'heading_deg': 10}
        target = np.array([600.0, 400.0])
        j1 = math.radians(aim.j1_to_face([*target, 0], p))
        pos = aim.rot_z(math.degrees(j1))[:2, :2] @ np.array([250, 60])
        look = np.array([math.cos(j1 + math.radians(10)), math.sin(j1 + math.radians(10))])
        v = target - pos
        self.assertAlmostEqual(v[0] * look[1] - v[1] * look[0], 0, places=6)  # on the line of sight
        self.assertGreater(v @ look, 0)                                       # in front, not behind

    def test_target_behind_the_pointer_is_refused(self):
        self.assertIsNone(aim.j1_to_face([100, 0, 0], {'reach_mm': 400, 'lateral_mm': 0, 'heading_deg': 0}))

    def test_sound_point_uses_array_position_and_yaw(self):
        cfg = cfg_with(array={'position_mm': [100, 0, 0], 'front_yaw_deg': 90}, aim={'assumed_distance_mm': 1000})
        np.testing.assert_allclose(aim.sound_point([1, 0, 0], cfg), [100, 1000, 0], atol=1e-9)

    def test_base_camera_point_straight_ahead(self):
        cfg = cfg_with()
        cfg['camera']['base'].update(position_mm=[0, 300, 50], yaw_deg=-30, pitch_deg=0)
        p = aim.camera_to_base([0, 0, 1000], cfg)
        np.testing.assert_allclose(p, [1000 * math.cos(math.radians(-30)), 300 + 1000 * math.sin(math.radians(-30)), 50], atol=1e-6)
        self.assertAlmostEqual(aim.view_position(p, cfg), 0.5)
        right = aim.camera_to_base([200, 0, 1000], cfg)
        self.assertGreater(aim.view_position(right, cfg), 0.5)  # +x in the image is to the right

    def test_fingertip_clearance_matches_the_20cm_rule(self):
        cfg = cfg_with()
        cfg['camera']['arm'].update(offset_mm=[0, 70, -130], lens_direction=[0, 0, 1], gripper_tip_mm=75)
        self.assertAlmostEqual(aim.fingertip_clearance(cfg), 205)
        cfg['camera']['mount'] = 'arm'
        self.assertEqual(aim.warnings(cfg), [])
        cfg['camera']['arm']['offset_mm'] = [0, 70, -50]
        self.assertTrue(any('fingertips' in w for w in aim.warnings(cfg)))


class MountAxes(unittest.TestCase):
    def test_mirrored_board_swaps_left_and_right_only(self):
        board = {'mount': 'facing', 'cable_edge': 'up', 'front_yaw_deg': 0}
        plain = aim.array_axes(cfg_with(array=board)) @ np.array([0.3, 0.5, 0.81])
        mirrored = aim.array_axes(cfg_with(array={**board, 'mirror': True})) @ np.array([0.3, 0.5, 0.81])
        np.testing.assert_allclose(mirrored, plain * [1, -1, 1])  # ahead and up unchanged


class Safety(unittest.TestCase):
    def gate(self, start=(0, 0, 90, 0, 90, 0), **safety):
        return aim.Gate(cfg_with(safety=safety), start)

    def test_big_turns_are_split_into_capped_steps(self):
        g = self.gate(yaw_window_deg=[-90, 90], max_step_deg=30)
        new, note = g.next_joints([0, 0, 90, 0, 90, 0], 80, now=10)
        self.assertEqual(new[0], 30)
        self.assertIn('50° more', note)
        self.assertEqual(new[1:], [0, 90, 0, 90, 0])  # nothing but J1 moves

    def test_small_errors_are_ignored(self):
        self.assertEqual(self.gate().next_joints([0] * 6, 4, now=10), (None, 'already facing it'))

    def test_turns_stop_at_the_window(self):
        g = self.gate(yaw_window_deg=[-90, 90])
        q = [0.0, 0, 90, 0, 90, 0]
        for _ in range(10):
            new, note = g.next_joints(q, 150, continuing=True)
            if new is None:
                break
            q = new
        self.assertEqual(q[0], 90)

    def test_wraparound_takes_the_short_way(self):
        g = self.gate(start=(170, 0, 90, 0, 90, 0))
        new, _ = g.next_joints([170, 0, 90, 0, 90, 0], -170, now=10)  # -170 is 190: 20° away, not 340°
        self.assertEqual(new[0], 190)

    def test_waits_between_new_turns(self):
        g = self.gate()
        g.moved(100.0)
        self.assertEqual(g.next_joints([0] * 6, 40, now=100.5)[0], None)
        self.assertIsNotNone(g.next_joints([0] * 6, 40, now=100.5, continuing=True)[0])

    def test_unsure_or_faint_fixes_are_rejected(self):
        g = self.gate()
        self.assertIn('not sure', g.accept({'z': 5, 'snr_db': 0}))
        self.assertIn('faint', g.accept({'z': 50, 'snr_db': -40}))
        self.assertIsNone(g.accept({'z': 50, 'snr_db': -10}))

    def test_start_near_a_limit_only_allows_moving_back(self):
        g = self.gate(start=(0, 0, 90, 0, 120, 0))  # J5 limit is 124; the margin keeps us under 114
        self.assertEqual(g.ranges[4], (105, 114))

    def test_impossible_joint_reading_is_refused(self):
        with self.assertRaises(SystemExit):
            self.gate(start=(500, 0, 90, 0, 90, 0))

    def test_config_caps_cannot_be_exceeded(self):
        text = aim.CONFIG_FILE.read_text().replace(f"speed_deg_s = {CFG['safety']['speed_deg_s']}", 'speed_deg_s = 90')
        with mock.patch('builtins.open', mock.mock_open(read_data=text.encode())):
            with self.assertRaises(SystemExit):
                aim.load_config()
        self.assertEqual(tomllib.loads(text)['safety']['speed_deg_s'], 90)


class ListenPipeline(unittest.TestCase):
    """The real `listen` code path, fed a simulated recording instead of the mic."""

    def test_quiet_keys_turn_the_dry_run_arm_toward_them(self):
        import contextlib
        import io
        import threading
        import time
        from argparse import Namespace

        import arm_follow
        from mic_test import simulate

        raw = simulate('keys', -5, 40.0, np.random.default_rng(3), seconds=5.0, tail=1.5) * 1e-3
        six = np.zeros((len(raw), 6), np.float32)
        six[:, 2:6] = raw
        six[:, :2] = raw.mean(axis=1, keepdims=True)
        calibration = np.round(six[:16000].astype(np.float64) * 2 ** 31).astype(np.int32)

        class FakeStream:
            def __init__(self, callback, **_):
                self.callback, self.running = callback, True

            def __enter__(self):
                def feed():
                    for i in range(16000, len(six), 256):
                        if not self.running:
                            break
                        self.callback(six[i:i + 256], 256, None, None)
                        time.sleep(0.002)
                threading.Thread(target=feed, daemon=True).start()
                return self

            def __exit__(self, *_):
                self.running = False

        out = io.StringIO()
        with mock.patch.object(arm_follow, 'find_array', return_value=0), \
                mock.patch.object(arm_follow, 'raw_channels', return_value=[2, 3, 4, 5]), \
                mock.patch.object(arm_follow, 'record', return_value=calibration), \
                mock.patch.object(arm_follow.sd, 'InputStream', FakeStream), \
                contextlib.redirect_stdout(out):
            args = Namespace(cmd='listen', target=None, voice=False, live=False, seconds=3.0)
            asyncio.run(arm_follow.follow(args, CFG))
        log = out.getvalue()
        self.assertRegex(log, r'heard (3[6-9]|4[0-4])° right')
        point = aim.sound_point([math.cos(math.radians(-40)), math.sin(math.radians(-40)), 0], CFG)
        expected = round(math.degrees(math.atan2(point[1], point[0])))
        self.assertIn('[dry run]', log)
        self.assertRegex(log, rf'-> {expected - 3}°|-> {expected - 2}°|-> {expected - 1}°|-> {expected}°|'
                              rf'-> {expected + 1}°|-> {expected + 2}°|-> {expected + 3}°')


class TargetGate(unittest.IsolatedAsyncioTestCase):
    """With --target, the arm only turns when Muse Spark says the sound matches."""

    async def run_event(self, verdict):
        import arm_follow
        meta = SimpleNamespace(sound_matches=mock.Mock(return_value=verdict))
        arm = arm_follow.DryRunArm(CFG)
        follower = arm_follow.Follower(CFG, arm, aim.Gate(CFG, await arm.joints()), CFG['look'], meta,
                                       asyncio.Queue(), None, 'phone ringing', following=True)
        fix = {'azimuth': 320.0, 'elevation': 0.0, 'z': 50.0, 'snr_db': 5.0, 'seconds': 1.0,
               'direction': np.array([math.cos(math.radians(-40)), math.sin(math.radians(-40)), 0.0])}
        with mock.patch('builtins.print'):
            await follower.on_sound({'kind': 'sound', 'fix': fix, 'wav': b'clip'})
        meta.sound_matches.assert_called_once_with(b'clip', 'phone ringing')
        return (await arm.joints())[0]

    async def test_no_match_no_move(self):
        self.assertEqual(await self.run_event({'match': False, 'heard': 'speech', 'confidence': 0.9}), 0)

    async def test_low_confidence_no_move(self):
        self.assertEqual(await self.run_event({'match': True, 'heard': 'maybe a ring', 'confidence': 0.3}), 0)

    async def test_match_turns_toward_it(self):
        self.assertLess(await self.run_event({'match': True, 'heard': 'phone ringing', 'confidence': 0.9}), -30)


class FakeArm:
    """Viam Arm stand-in: J1 swings the end effector around the base, at a chosen speed."""

    def __init__(self, deg_per_s=15.0, mirrored=False):
        self.q, self.deg_per_s, self.mirrored, self.stopped, self.commands = [0.0] * 6, deg_per_s, mirrored, False, []

    async def get_joint_positions(self):
        return SimpleNamespace(values=list(self.q))

    async def move_to_joint_positions(self, positions):
        await asyncio.sleep(max(abs(a - b) for a, b in zip(positions.values, self.q)) / self.deg_per_s)
        self.q = list(positions.values)

    async def is_moving(self):
        return False

    async def get_end_position(self):
        a = math.radians(-self.q[0] if self.mirrored else self.q[0])
        return SimpleNamespace(x=300 * math.cos(a), y=300 * math.sin(a), z=200, o_x=0, o_y=0, o_z=-1)

    async def stop(self):
        self.stopped = True

    async def do_command(self, command):
        self.commands.append(command)
        return {}


@unittest.skipUnless(importlib.util.find_spec('viam'), 'viam-sdk not installed')
class LiveArmSafety(unittest.IsolatedAsyncioTestCase):
    def arm(self, fake):
        from arm_follow import ViamArm
        return ViamArm(robot=None, arm=fake, cfg=CFG)

    async def test_prepare_slows_the_arm_first(self):
        fake = FakeArm()
        await self.arm(fake).prepare()
        self.assertEqual(fake.commands, [{'set_speed': float(CFG['safety']['speed_deg_s']),
                                          'set_acceleration': float(CFG['safety']['accel_deg_s2'])}])

    async def test_normal_speed_move_passes(self):
        fake = FakeArm(deg_per_s=15)
        await self.arm(fake).move([6, 0, 0, 0, 0, 0])
        self.assertFalse(fake.stopped)

    async def test_too_fast_move_stops_the_arm(self):
        fake = FakeArm(deg_per_s=200)
        with self.assertRaises(SystemExit):
            await self.arm(fake).move([20, 0, 0, 0, 0, 0])
        self.assertTrue(fake.stopped)

    async def test_mirrored_j1_stops_the_arm(self):
        fake = FakeArm(deg_per_s=15, mirrored=True)
        with self.assertRaises(SystemExit):
            await self.arm(fake).move([6, 0, 0, 0, 0, 0])
        self.assertTrue(fake.stopped)


if __name__ == '__main__':
    unittest.main()
