"""Depth-target consumer tests, using local temporary packets and a fake arm only."""
import asyncio
import copy
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import aim
import arm_follow
import arm_target
from target_link import TargetWriter, read_target
from test_arm_diagnostics import RecordedArm, config


class Configuration(unittest.TestCase):
    def test_cli_defaults_are_dry_and_use_desk_profile(self):
        args = arm_target.parse_args([])
        self.assertFalse(args.live)
        self.assertEqual(args.mount, 'carrier_v3_desk')
        self.assertEqual(args.target_file.name, 'latest_target.json')

    def test_diagnostic_caps_preserve_original_config_and_do_not_enable_live(self):
        original = config()
        original['safety'].update(max_step_deg=30, speed_deg_s=15, accel_deg_s2=60, live=False)
        original['aim']['use_pitch'] = True
        cfg = arm_target.diagnostic_config(original)
        self.assertEqual(cfg['safety']['max_step_deg'], 5)
        self.assertEqual(cfg['safety']['speed_deg_s'], 5)
        self.assertEqual(cfg['safety']['accel_deg_s2'], 10)
        self.assertTrue(cfg['safety']['single_step_per_sound'])
        self.assertFalse(cfg['safety']['live'])
        self.assertFalse(cfg['aim']['use_pitch'])
        self.assertEqual(original['safety']['max_step_deg'], 30)

    def test_wrist_geometry_is_rejected_without_connecting(self):
        cfg = config()
        cfg['camera']['mount'] = 'arm'
        with self.assertRaises(SystemExit):
            arm_target.diagnostic_config(cfg)


class DepthConsumer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'target.json'
        self.cfg = arm_target.diagnostic_config(config())
        self.cfg['safety']['mute_after_move_s'] = 0
        self.arm = RecordedArm(self.cfg)
        self.arm.stop = mock.AsyncMock()
        self.arm.close = mock.AsyncMock()
        self.started = time.monotonic()
        self.writer = TargetWriter(self.path, self.cfg)
        self.consumer = arm_target.TargetFollower(
            self.cfg, self.arm, aim.Gate(self.cfg, self.arm.q), self.cfg['look'], self.started,
            target_file=self.path)
        self.print_patch = mock.patch('builtins.print')
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def publish(self, event_id=1, **changes):
        view = {'event_id': event_id, 'time': time.monotonic(), 'final': True,
                'quality': 'candidate', 'measured': True, 'point': [0.6, 0.0, 1.0],
                'fix': {'direction': aim.camera_to_array([0, 0, 1], self.cfg),
                        'z': 50, 'snr_db': 10, 'directional_ambiguous': False}}
        view.update(changes)
        self.writer.publish(view)
        return read_target(self.path, self.cfg, not_before=self.started)

    async def test_measured_depth_point_drives_exactly_one_step_without_assumed_range(self):
        packet = self.publish()
        self.cfg['aim']['assumed_distance_mm'] = 999999
        with mock.patch.object(aim, 'sound_point', side_effect=AssertionError('assumed range used')), \
                mock.patch.object(aim, 'j1_to_face', wraps=aim.j1_to_face) as target:
            self.assertTrue(await self.consumer.consume(packet))
        self.assertEqual(target.call_args.args[0], packet['base_point_mm'])
        self.assertEqual(len(self.arm.moves), 1)
        self.assertEqual(self.arm.q[0], -5)
        self.assertFalse(await self.consumer.consume(packet))
        self.assertEqual(len(self.arm.moves), 1)

    async def test_new_publisher_session_can_reuse_event_number(self):
        await self.consumer.consume(self.publish())
        self.writer = TargetWriter(self.path, self.cfg)
        self.assertTrue(await self.consumer.consume(self.publish()))
        self.assertEqual(len(self.arm.moves), 2)

    async def test_weak_packet_never_moves(self):
        packet = self.publish()
        packet['fix']['z'] = 0
        self.assertFalse(await self.consumer.consume(packet))
        self.assertEqual(self.arm.moves, [])

    async def test_stale_packet_never_moves(self):
        packet = self.publish()
        packet['observed_at'] -= 3
        self.assertFalse(await self.consumer.consume(packet))
        self.assertEqual(self.arm.moves, [])

    async def test_publisher_invalidation_during_joint_read_prevents_motion(self):
        packet = self.publish()
        async def joints():
            self.writer.invalidate('calibrating')
            return list(self.arm.q)
        self.arm.joints = joints
        self.assertFalse(await self.consumer.consume(packet))
        self.assertEqual(self.arm.moves, [])

    async def test_new_candidate_during_joint_read_prevents_motion_for_old_one(self):
        packet = self.publish()
        async def joints():
            self.publish(2)
            return list(self.arm.q)
        self.arm.joints = joints
        self.assertFalse(await self.consumer.consume(packet))
        self.assertEqual(self.arm.moves, [])

    async def test_post_move_capture_gate_excludes_motor_noise(self):
        self.cfg['safety']['mute_after_move_s'] = 0.8
        before = time.monotonic()
        await self.consumer.consume(self.publish())
        self.assertGreaterEqual(self.consumer.not_before, before + 0.8)
        self.publish(2)
        with self.assertRaisesRegex(ValueError, 'predates'):
            read_target(self.path, self.cfg, not_before=self.consumer.not_before)

    async def test_run_ignores_preexisting_packet_then_stops_and_closes(self):
        self.publish()
        args = SimpleNamespace(live=False, target_file=self.path, seconds=0.02)
        with mock.patch.object(arm_follow, 'open_arm', new=mock.AsyncMock(return_value=self.arm)):
            await arm_target.run(args, self.cfg)
        self.assertEqual(self.arm.moves, [])
        self.arm.stop.assert_awaited_once()
        self.arm.close.assert_awaited_once()

    async def test_unexpected_reader_error_stops_and_closes(self):
        args = SimpleNamespace(live=False, target_file=self.path, seconds=0.02)
        with mock.patch.object(arm_follow, 'open_arm', new=mock.AsyncMock(return_value=self.arm)), \
                mock.patch.object(arm_target, 'read_target', side_effect=RuntimeError('reader failure')):
            with self.assertRaisesRegex(RuntimeError, 'reader failure'):
                await arm_target.run(args, self.cfg)
        self.arm.stop.assert_awaited_once()
        self.arm.close.assert_awaited_once()

    async def test_cancelled_consumer_stops_and_closes(self):
        reading = asyncio.Event()
        def unavailable(*_, **__):
            reading.set()
            raise ValueError('not ready')
        args = SimpleNamespace(live=False, target_file=self.path, seconds=None)
        with mock.patch.object(arm_follow, 'open_arm', new=mock.AsyncMock(return_value=self.arm)), \
                mock.patch.object(arm_target, 'read_target', side_effect=unavailable):
            task = asyncio.create_task(arm_target.run(args, self.cfg))
            await reading.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.arm.stop.assert_awaited_once()
        self.arm.close.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
