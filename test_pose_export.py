"""Offline tests: fake read-only SDK; no hardware or network."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from pose_export import PoseHistory, PoseSampler, atomic_publish, export_connected, pose_to_transform
from rig_geometry import RigGeometry


def pose(**kwargs):
    values = dict(x=100, y=200, z=300, o_x=0, o_y=0, o_z=1, theta=90)
    values.update(kwargs)
    return SimpleNamespace(**values)


class FakeQuaternion:
    seen = None

    @classmethod
    def from_pose(cls, value):
        cls.seen = value
        # SDK fixture for +Z pointing, +90-degree roll; intentionally asymmetric.
        return SimpleNamespace(to_rotation_matrix=lambda: SimpleNamespace(elements=[0, -1, 0, 1, 0, 0, 0, 0, 1]))


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now


class FakeArm:
    def __init__(self, clock):
        self.clock, self.calls, self.motions = clock, [], []
        self.value, self.slow, self.error = pose(), 0, None

    async def is_moving(self):
        self.calls.append('is_moving')
        return self.motions.pop(0) if self.motions else False

    async def get_end_position(self):
        self.calls.append('get_end_position')
        self.clock.now += self.slow
        if self.error: raise self.error
        return self.value


class PoseExportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = Clock()
        self.arm = FakeArm(self.clock)
        self.convert = lambda p: pose_to_transform(p, FakeQuaternion)
        self.sampler = PoseSampler(self.arm, self.convert, clock=self.clock)

    async def ready(self):
        first = await self.sampler.sample()
        self.assertIsNone(first['T_base_from_tool'])
        for _ in range(4):
            self.clock.now += .1
            packet = await self.sampler.sample()
        self.assertEqual(packet['status'], 'ready')
        return packet

    def test_sdk_bridge_preserves_degrees_and_maps_row_major_mm_to_metres(self):
        p = pose()
        t = self.convert(p)
        self.assertIs(FakeQuaternion.seen, p)
        self.assertEqual(FakeQuaternion.seen.theta, 90)
        np.testing.assert_allclose(t[:3, 3], [.1, .2, .3])
        np.testing.assert_allclose(t[:3, :3] @ [1, 0, 0], [0, 1, 0])
        for bad in (pose(x=float('nan')), pose(o_z=0)):
            with self.assertRaises(ValueError): self.convert(bad)

    async def test_read_only_sequence_and_preview_contract(self):
        packet = await self.ready()
        self.assertEqual(self.arm.calls, ['is_moving', 'get_end_position', 'is_moving'] * 5)
        rig = RigGeometry(np.eye(4), mode='wrist', T_tool_from_camera=np.eye(4),
                          robot_calibrated=True, calibration_id='fixture')
        result = rig.camera_point_to_base([0, 0, .4], packet, observed_at=self.clock(), now=self.clock())
        self.assertTrue(result['ok'], result)
        np.testing.assert_allclose(result['point_base_m'], [.1, .2, .7])
        self.assertIs(packet['motion_enabled'], False)

    async def test_movement_before_or_after_read_invalidates_previous_pose(self):
        await self.ready()
        for states in ([True], [False, True], [None]):
            self.arm.motions = list(states)
            packet = await self.sampler.sample()
            self.assertIsNone(packet['T_base_from_tool'])
            self.assertIsNone(self.sampler.settled_since)

    async def test_slow_read_and_error_omit_matrix(self):
        await self.ready()
        self.arm.slow = .2
        packet = await self.sampler.sample()
        self.assertIn('timing budget', packet['reason'])
        self.assertIsNone(packet['T_base_from_tool'])
        self.arm.slow, self.arm.error = 0, RuntimeError('secret detail must not be published')
        packet = await self.sampler.sample()
        self.assertNotIn('secret', json.dumps(packet))
        self.assertIsNone(packet['T_base_from_tool'])

    async def test_manual_pose_change_or_polling_gap_restarts_settling(self):
        await self.ready()
        self.arm.value = pose(x=105)
        packet = await self.sampler.sample()
        self.assertIsNone(packet['T_base_from_tool'])
        self.clock.now += 1
        packet = await self.sampler.sample()
        self.assertIsNone(packet['T_base_from_tool'])
        self.assertEqual(packet['settled_since'], self.clock())

    def test_atomic_json_replacement_leaves_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pose.json'
            atomic_publish(path, {'status': 'ready', 'T_base_from_tool': np.eye(4).tolist()})
            atomic_publish(path, {'status': 'blocked', 'T_base_from_tool': None})
            self.assertIsNone(json.loads(path.read_text())['T_base_from_tool'])
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_history_bounded_by_time_and_count_and_cleared_on_block(self):
        history = PoseHistory(max_age_s=2., max_samples=4)
        for timestamp in [100., 100.1, 100.2, 100.3, 100.4]:
            packet = history.attach({'status': 'ready', 'moving': False, 'captured_at': timestamp})
        self.assertEqual(len(packet['samples']), 4)
        self.assertEqual(packet['samples'][0]['captured_at'], 100.1)
        packet = history.attach({'status': 'ready', 'moving': False, 'captured_at': 102.3})
        self.assertTrue(all(s['captured_at'] >= 100.3 for s in packet['samples']))
        self.assertTrue(all('samples' not in s for s in packet['samples']))
        packet = history.attach({'status': 'blocked', 'moving': None, 'captured_at': 102.4})
        self.assertEqual(packet['samples'], [])
        packet = history.attach({'status': 'ready', 'moving': False, 'captured_at': 103.})
        self.assertEqual(len(packet['samples']), 1)

    async def test_export_exit_closes_client_without_control_methods(self):
        closed = []
        async def close(): closed.append(True)
        connection = SimpleNamespace(arm=self.arm, robot=SimpleNamespace(close=close))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pose.json'
            await export_connected(connection, self.convert, path, interval_s=.001, max_samples=1)
            self.assertEqual(json.loads(path.read_text())['reason'], 'pose exporter closed')
        self.assertEqual(closed, [True])
        self.assertEqual(self.arm.calls, ['is_moving', 'get_end_position', 'is_moving'])


if __name__ == '__main__':
    unittest.main()
