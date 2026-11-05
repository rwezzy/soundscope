"""Offline diagnostic motion tests. No arm, sensors, or model API are opened."""
import asyncio
import copy
import math
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import aim
import arm_follow


def config():
    cfg = aim.load_config()
    cfg['safety'].update(max_step_deg=5, deadband_deg=1, min_interval_s=0,
                         single_step_per_sound=True, max_sound_age_s=1.5,
                         move_timeout_s=0.03)
    cfg['array'].update(on_arm=False, mount='facing', position_mm=[0, 0, 0], front_yaw_deg=0)
    cfg['camera'].update(mount='base')
    cfg['camera']['base'].update(position_mm=[0, 0, 0], yaw_deg=0, pitch_deg=0)
    cfg['aim']['use_pitch'] = False
    cfg['look'].update(start_joints_deg=[0, 0, 90, 0, 90, 0], reach_mm=250,
                       lateral_mm=0, heading_deg=0)
    return cfg


class RecordedArm(arm_follow.DryRunArm):
    def __init__(self, cfg, dry=False):
        super().__init__(cfg)
        self.dry, self.moves = dry, []

    async def move(self, target):
        self.moves.append(list(target))
        await super().move(target)


class SoundMotion(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = config()
        self.arm = RecordedArm(self.cfg)
        self.meta = SimpleNamespace(sound_matches=mock.Mock(return_value={
            'match': True, 'heard': 'ringing', 'confidence': 0.9}))
        self.follower = arm_follow.Follower(
            self.cfg, self.arm, aim.Gate(self.cfg, self.arm.q), self.cfg['look'], self.meta,
            asyncio.Queue(), None, None, True)
        self.print_patch = mock.patch('builtins.print')
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def event(self, event_id=1):
        a = math.radians(45)
        return {'kind': 'sound', 'event_id': event_id, 'observed_at': time.monotonic(), 'wav': b'clip',
                'fix': {'direction': aim.camera_to_array([math.sin(a), 0, math.cos(a)], self.cfg),
                        'z': 50, 'snr_db': 10, 'seconds': 1}}

    async def test_diagnostic_turns_one_step_then_rejects_same_event(self):
        ev = self.event()
        await self.follower.on_sound(ev)
        self.assertEqual(len(self.arm.moves), 1)
        self.assertAlmostEqual(self.arm.q[0], -5)
        repeated = copy.deepcopy(ev)
        repeated['observed_at'] = time.monotonic()
        await self.follower.on_sound(repeated)
        self.assertEqual(len(self.arm.moves), 1)
        await self.follower.on_sound(self.event(2))
        self.assertEqual(len(self.arm.moves), 2)
        self.assertAlmostEqual(self.arm.q[0], -10)

    async def test_wrist_also_takes_only_one_step(self):
        self.follower.wrist = True
        await self.follower.on_sound(self.event())
        self.assertEqual(len(self.arm.moves), 1)
        self.assertAlmostEqual(abs(self.arm.q[0]), 5)

    async def test_stale_sound_does_not_spend_meta_or_move(self):
        self.follower.target = 'ringing'
        ev = self.event()
        ev['observed_at'] -= 3
        await self.follower.on_sound(ev)
        self.meta.sound_matches.assert_not_called()
        self.assertEqual(self.arm.moves, [])

    async def test_sound_aging_during_meta_does_not_move(self):
        self.follower.target = 'ringing'
        ev = self.event()
        def slow_verdict(*_):
            ev['observed_at'] -= 3
            return {'match': True, 'heard': 'ringing', 'confidence': 0.9}
        self.meta.sound_matches.side_effect = slow_verdict
        await self.follower.on_sound(ev)
        self.meta.sound_matches.assert_called_once()
        self.assertEqual(self.arm.moves, [])

    async def test_sound_aging_during_joint_read_does_not_move(self):
        ev = self.event()
        async def delayed_joints():
            ev['observed_at'] -= 3
            return list(self.arm.q)
        self.arm.joints = delayed_joints
        await self.follower.on_sound(ev)
        self.assertEqual(self.arm.moves, [])

    async def test_ambiguous_sound_does_not_move(self):
        ev = self.event()
        ev['fix']['directional_ambiguous'] = True
        await self.follower.on_sound(ev)
        self.assertEqual(self.arm.moves, [])

    async def test_missing_timestamp_blocks_real_but_allows_dry_synthetic(self):
        ev = self.event()
        del ev['observed_at']
        await self.follower.on_sound(ev)
        self.assertEqual(self.arm.moves, [])
        self.arm.dry = True
        await self.follower.on_sound(ev)
        self.assertEqual(len(self.arm.moves), 1)

    async def test_invalid_and_future_capture_times_do_not_move(self):
        for observed in ('bad', float('nan'), float('inf'), time.monotonic() + 5):
            ev = self.event()
            ev['observed_at'] = observed
            await self.follower.on_sound(ev)
        self.assertEqual(self.arm.moves, [])


class MoveDeadline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = SimpleNamespace(
            get_joint_positions=mock.AsyncMock(return_value=SimpleNamespace(values=[0] * 6)),
            get_end_position=mock.AsyncMock(return_value=SimpleNamespace(x=300, y=0, z=200)),
            move_to_joint_positions=mock.AsyncMock(),
            is_moving=mock.AsyncMock(return_value=False), stop=mock.AsyncMock())
        self.arm = arm_follow.ViamArm(None, self.device, config())

    async def hang(self, *_):
        await asyncio.Future()

    async def test_blocking_move_rpc_times_out_and_stops(self):
        self.device.move_to_joint_positions.side_effect = self.hang
        with self.assertRaisesRegex(SystemExit, 'exceeded'):
            await self.arm.move([5, 0, 0, 0, 0, 0])
        self.device.stop.assert_awaited_once()

    async def test_blocking_poll_times_out_and_stops(self):
        self.device.is_moving.side_effect = self.hang
        with self.assertRaisesRegex(SystemExit, 'exceeded'):
            await self.arm.move([5, 0, 0, 0, 0, 0])
        self.device.stop.assert_awaited_once()

    async def test_rpc_error_stops_and_preserves_original_when_stop_fails(self):
        self.device.move_to_joint_positions.side_effect = RuntimeError('original RPC failure')
        self.device.stop.side_effect = RuntimeError('stop failure')
        with mock.patch('builtins.print'), self.assertRaisesRegex(RuntimeError, 'original RPC failure'):
            await self.arm.move([5, 0, 0, 0, 0, 0])
        self.device.stop.assert_awaited_once()

    async def test_cancelled_move_stops(self):
        started = asyncio.Event()
        async def waiting(*_):
            started.set()
            await asyncio.Future()
        self.device.move_to_joint_positions.side_effect = waiting
        task = asyncio.create_task(self.arm.move([5, 0, 0, 0, 0, 0]))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.device.stop.assert_awaited_once()


class SimulationFrames(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_and_wrist_bearings_round_trip_to_camera(self):
        for on_arm in (False, True):
            cfg = config()
            cfg['array'].update(on_arm=on_arm, front_yaw_deg=21)
            cfg['camera']['base'].update(yaw_deg=-17, pitch_deg=8)
            posted = []
            await arm_follow.feed_simulated([0, -30, 45], posted.append, 0, cfg)
            for ev, bearing in zip(posted, [0, -30, 45]):
                got = aim.array_to_camera(ev['fix']['direction'], cfg)
                angle = math.radians(bearing)
                np.testing.assert_allclose(got, [math.sin(angle), 0, math.cos(angle)], atol=1e-12)
                self.assertLess(time.monotonic() - ev['observed_at'], 1.5)
            self.assertEqual(len({ev['event_id'] for ev in posted[:-1]}), 3)


if __name__ == '__main__':
    unittest.main()
