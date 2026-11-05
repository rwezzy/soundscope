"""Offline geometry and sound-gate regressions; no hardware, audio capture or API calls."""
import copy
import math
import unittest
from unittest import mock

import numpy as np

import aim


class DeskGeometry(unittest.TestCase):
    def setUp(self):
        self.standalone = aim.load_config(profile='carrier_v3')
        self.desk = aim.load_config(profile='carrier_v3_desk')

    def test_desk_positions_share_the_arm_base_frame(self):
        np.testing.assert_allclose(self.desk['camera']['base']['position_mm'], [-380, 740, 160])
        np.testing.assert_allclose(self.desk['array']['position_mm'], [-376, 729, 288])
        offset = np.asarray(self.desk['array']['position_mm']) - self.desk['camera']['base']['position_mm']
        np.testing.assert_allclose(offset, [4, -11, 128])

    def test_same_camera_points_have_same_acoustic_geometry_in_both_profiles(self):
        # Camera axes are right/down/forward. Translating the whole rig must not change
        # the microphone-to-scene distances used to compute acoustic time differences.
        camera_points = np.array([[0, 0, 1000], [180, -120, 450], [-280, 90, 1700],
                                  [11, -128, 4]], dtype=float)
        relative = []
        for cfg in (self.standalone, self.desk):
            base_points = aim.camera_to_base(camera_points, cfg)
            relative.append(aim.base_to_array(base_points, cfg))
        np.testing.assert_allclose(relative[0], relative[1], atol=1e-12)
        np.testing.assert_allclose(relative[1][0], [-0.128, 0.011, 0.996], atol=1e-12)
        np.testing.assert_allclose(relative[1][-1], [0, 0, 0], atol=1e-12)

    def test_projection_of_sound_ray_is_invariant_under_rig_translation(self):
        for direction in ([0, 0, 1], [0, 0.5, math.sqrt(0.75)],
                          [0.3, -0.4, math.sqrt(0.75)]):
            a = aim.sound_point(direction, self.standalone, 1000)
            b = aim.sound_point(direction, self.desk, 1000)
            np.testing.assert_allclose(b - a, [-380, 740, 160], atol=1e-9)
            self.assertAlmostEqual(aim.view_position(a, self.standalone),
                                   aim.view_position(b, self.desk), places=12)

    def test_usb_up_and_led_forward_axes(self):
        np.testing.assert_allclose(aim.array_to_camera([1, 0, 0], self.desk), [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(aim.array_to_camera([0, 0, 1], self.desk), [0, 0, 1], atol=1e-12)

    def test_mirrored_carrier_flips_only_array_y_and_keeps_direction_round_trip(self):
        self.assertTrue(self.standalone['array']['mirror'])
        self.assertTrue(self.desk['array']['mirror'])
        normal = copy.deepcopy(self.desk)
        normal['array']['mirror'] = False
        direction_cam = np.array([0.3, -0.2, math.sqrt(0.87)])
        a = aim.camera_to_array(direction_cam, normal)
        b = aim.camera_to_array(direction_cam, self.desk)
        np.testing.assert_allclose(b, a * [1, -1, 1], atol=1e-12)
        np.testing.assert_allclose(aim.array_to_camera(b, self.desk), direction_cam, atol=1e-12)


class SoundGate(unittest.TestCase):
    def setUp(self):
        self.cfg = aim.load_config(profile='carrier_v3_desk')
        self.gate = aim.Gate(self.cfg, self.cfg['look']['start_joints_deg'])
        self.fix = {'z': 50.0, 'snr_db': 10.0, 'direction': [0, 0, 1]}

    def test_finite_unambiguous_fix_passes(self):
        self.assertIsNone(self.gate.accept(self.fix))

    def test_nonfinite_quality_never_passes_thresholds(self):
        for field in ('z', 'snr_db'):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(field=field, value=value):
                    self.assertIsNotNone(self.gate.accept({**self.fix, field: value}))

    def test_bad_or_missing_direction_cannot_reach_motion(self):
        for value in ([float('nan'), 0, 1], [0, float('inf'), 1], [0, 0, 0],
                      [0, 1], [[0, 0, 1]], None, 'invalid'):
            with self.subTest(value=value):
                self.assertIsNotNone(self.gate.accept({**self.fix, 'direction': value}))
        missing = dict(self.fix)
        del missing['direction']
        self.assertIsNotNone(self.gate.accept(missing))

    def test_ambiguous_fix_is_rejected_even_with_high_confidence(self):
        self.assertIn('ambiguous', self.gate.accept({**self.fix, 'directional_ambiguous': True}))

    def load_safety(self, **settings):
        cfg = copy.deepcopy(self.cfg)
        cfg['safety'].update(settings)
        with mock.patch('builtins.open', mock.mock_open(read_data=b'')), \
                mock.patch.object(aim.tomllib, 'load', return_value=cfg):
            return aim.load_config(profile='')

    def test_sound_age_limit_requires_finite_positive_number(self):
        for value in (0, -1, float('nan'), float('inf'), True, '1.5'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.load_safety(max_sound_age_s=value)
        self.assertEqual(self.load_safety(max_sound_age_s=1.5)['safety']['max_sound_age_s'], 1.5)

    def test_single_step_setting_must_be_boolean(self):
        for value in ('false', 1, 0, None):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.load_safety(single_step_per_sound=value)
        self.assertIs(self.load_safety(single_step_per_sound=True)['safety']['single_step_per_sound'], True)


if __name__ == '__main__':
    unittest.main()
