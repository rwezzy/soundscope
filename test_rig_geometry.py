"""Offline tests of frame conventions, timing gates and non-actuating previews."""
import copy
import unittest
from pathlib import Path

import numpy as np

import aim
from rig_geometry import RigGeometry, RobotPreview, validate_transform


def pose(yaw=0, position=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = aim.rot_z(yaw)
    t[:3, 3] = position
    return t


class RigGeometryTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {'camera': {'mount':'base', 'base':{'position_mm':[0,0,0], 'yaw_deg':0, 'pitch_deg':0}},
                    'array': {'mount':'facing','position_mm':[0,0,-80],'front_yaw_deg':0}}

    def test_legacy_projection_preserves_existing_scoring_coordinates(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['camera']['base'].update(position_mm=[150, -220, 300], yaw_deg=21, pitch_deg=-13)
        cfg['array'].update(position_mm=[160, -180, 370], front_yaw_deg=32)
        points = np.random.default_rng(3).uniform(-0.3, 0.8, (5, 4, 3))
        # Old camera_to_base expects millimetres; supply those explicitly.
        expected = aim.base_to_array(aim.camera_to_base(points * 1000, cfg), cfg)
        rig = RigGeometry.from_legacy_config(cfg)
        np.testing.assert_allclose(rig.camera_points_to_array(points), expected, atol=1e-14)
        self.assertFalse(rig.robot_calibrated)

    def test_legacy_default_array_below_camera(self):
        rig = RigGeometry.from_legacy_config(self.cfg)
        np.testing.assert_allclose(rig.array_origin_camera_m, [0, .08, 0], atol=1e-12)
        np.testing.assert_allclose(rig.array_direction_to_camera([1, 0, 0]), [0, -1, 0])
        np.testing.assert_allclose(rig.array_direction_to_camera([0, 0, 2]), [0, 0, 1])

    def test_rigid_pair_does_not_need_robot_to_score_acoustics(self):
        fixed = RigGeometry.from_legacy_config(self.cfg)
        wrist = RigGeometry(fixed.T_camera_from_array, mode='wrist')
        points = np.array([[.1, -.2, .8], [-.2, .3, 1.4]])
        np.testing.assert_allclose(wrist.camera_points_to_array(points), fixed.camera_points_to_array(points))
        self.assertFalse(wrist.camera_point_to_base(points[0], observed_at=10, now=10)['ok'])

    def test_joint_rigid_motion_invariance(self):
        rig = RigGeometry.from_legacy_config(self.cfg)
        camera_point = np.array([.1, -.05, .8])
        array_point = rig.camera_points_to_array(camera_point)
        for base_camera in [pose(0, (.2, .1, .3)), pose(90, (.3, -.1, .2)), pose(-50, (-.1, .4, .2))]:
            base_array = base_camera @ rig.T_camera_from_array
            p_from_camera = base_camera[:3, :3] @ camera_point + base_camera[:3, 3]
            p_from_array = base_array[:3, :3] @ array_point + base_array[:3, 3]
            np.testing.assert_allclose(p_from_camera, p_from_array, atol=1e-14)

    def test_invalid_se3_and_unit_errors_rejected(self):
        bad = [np.eye(3), np.full((4, 4), np.nan)]
        for indices, value in [((0, 0), 2), ((0, 0), -1), ((3, 0), 1), ((0, 3), 80)]:
            t = np.eye(4); t[indices] = value; bad.append(t)
        for t in bad:
            with self.subTest(t=t), self.assertRaises(ValueError):
                validate_transform(t)

    def test_matrix_is_copied_read_only(self):
        t = np.eye(4)
        rig = RigGeometry(t)
        t[0, 3] = .5
        self.assertEqual(rig.array_origin_camera_m[0], 0)
        with self.assertRaises(ValueError):
            rig.T_camera_from_array[0, 3] = .6

    def test_invalid_direction(self):
        rig = RigGeometry(np.eye(4))
        for direction in [[0,0,0], [1,2], [0,float('nan'),1], [[1,2,3]]]:
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                rig.array_direction_to_camera(direction)

    def test_config_requires_explicit_metres_and_no_fake_robot_calibration(self):
        example = Path(__file__).with_name('rigs.example.json')
        rig = RigGeometry.load(example)
        self.assertFalse(rig.status()['robot_calibrated'])
        self.assertFalse(rig.status()['motion_enabled'])
        with self.assertRaises(ValueError):
            RigGeometry.from_dict({'T_camera_from_array':np.eye(4).tolist()})
        with self.assertRaises(ValueError):
            RigGeometry(np.eye(4), robot_calibrated='false')

    def test_fixed_calibration_conversion(self):
        rig = RigGeometry(np.eye(4), T_base_from_camera=pose(90, (.3, .1, .2)),
                          robot_calibrated=True, calibration_id='offline-test')
        result = rig.camera_point_to_base([.1, .2, .5], observed_at=10, now=10.2)
        self.assertTrue(result['ok'])
        np.testing.assert_allclose(result['point_base_m'], [.1, .2, .7])

    def test_fixed_refuses_missing_timestamp_stale_future_and_unknown_calibration(self):
        rig = RigGeometry(np.eye(4), T_base_from_camera=np.eye(4), robot_calibrated=True, calibration_id='test')
        for observed in [None, 8, 11, float('nan')]:
            self.assertFalse(rig.camera_point_to_base([0, 0, .3], observed_at=observed, now=10)['ok'])
        no_id = RigGeometry(np.eye(4), T_base_from_camera=np.eye(4), robot_calibrated=True)
        self.assertFalse(no_id.camera_point_to_base([0, 0, .3], observed_at=10, now=10)['ok'])

    def wrist(self):
        return RigGeometry(np.eye(4), mode='wrist', T_tool_from_camera=pose(0, (0, 0, .1)),
                           robot_calibrated=True, calibration_id='measured-test-fixture')

    def packet(self):
        return {'T_base_from_tool':pose(90, (.2,.1,.3)), 'captured_at':10.,
                'moving':False, 'settled_since':9.}

    def test_wrist_uses_capture_pose_and_metres(self):
        result = self.wrist().camera_point_to_base([.1, 0, .3], self.packet(), observed_at=10, now=10.1)
        self.assertTrue(result['ok'])
        np.testing.assert_allclose(result['point_base_m'], [.2,.2,.7])

    def test_wrist_refuses_missing_moving_unsynced_stale_unsettled_poses(self):
        bad = [None]
        for changes in [{'moving':True}, {'moving':None}, {'settled_since':9.9},
                        {'captured_at':8.}, {'captured_at':10.2},
                        {'T_base_from_tool':None}, {'settled_since':float('nan')}]:
            p=self.packet();p.update(changes);bad.append(p)
        for p in bad:
            with self.subTest(packet=p):
                self.assertFalse(self.wrist().camera_point_to_base([0,0,.3],p,observed_at=10,now=10.3)['ok'])

    def test_preview_never_authorizes_motion_or_grasp(self):
        planner=RobotPreview(self.wrist())
        event={'quality':'candidate','point_camera_m':[0,0,.3],'observed_at':10,'measured':True}
        result=planner.plan(event,self.packet(),now=10.1)
        self.assertEqual(result['status'],'preview')
        self.assertTrue(result['dry_run'])
        self.assertFalse(result['motion_enabled'])
        self.assertFalse(result['source_verified'])
        self.assertEqual(planner.plan(event,self.packet(),now=10.1,action='grasp')['status'],'blocked')
        for changes in [{'quality':'weak'}, {'quality':'ambiguous'}, {'point_camera_m':None}, {'measured':False}]:
            modified=dict(event,**changes)
            self.assertEqual(planner.plan(modified,self.packet(),now=10.1)['status'],'blocked')


if __name__ == '__main__':
    unittest.main()
