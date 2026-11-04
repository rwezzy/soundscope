"""Offline calibration tests: python -m unittest test_calibration."""
import math
import unittest

from calibration import CalibrationSession, wrap_deg


class YawCalibration(unittest.TestCase):
    def add(self, session, expected, heard, **kwargs):
        quality = dict(timestamp=float(len(session.samples)), z=18.0, coherence=0.3)
        quality.update(kwargs)
        return session.add(expected, heard, **quality)

    def fit(self, expected, heard):
        session = CalibrationSession()
        for target, observed in zip(expected, heard):
            result = self.add(session, target, observed)
        return session, result

    def test_known_offset_requires_three_positions(self):
        session = CalibrationSession()
        self.assertFalse(self.add(session, -25, 0).ready)
        self.assertFalse(self.add(session, 0, 25).ready)
        result = self.add(session, 25, 50)
        self.assertTrue(result.ready)
        self.assertEqual(result.count, 3)
        self.assertAlmostEqual(result.correction_deg, -25)
        self.assertAlmostEqual(result.residual_deg, 0)
        self.assertIn('Yaw-only', result.message)
        self.assertIn('new source position', result.message)

    def test_large_consistent_correction_is_allowed(self):
        _, result = self.fit([-30, 0, 30], [140, 170, -160])
        self.assertTrue(result.ready)
        self.assertAlmostEqual(result.correction_deg, -170)

    def test_wraparound_positions_and_corrections(self):
        _, result = self.fit([160, 180, -160], [-20, 1, 19])
        self.assertTrue(result.ready)
        self.assertLess(abs(wrap_deg(result.correction_deg - 180)), 0.01)
        self.assertAlmostEqual(result.residual_deg, math.sqrt(2 / 3), places=6)
        self.assertAlmostEqual(result.expected_span_deg, 40)

    def test_repeat_of_same_position_is_not_calibration(self):
        _, result = self.fit([10, 10, 10], [30, 30, 30])
        self.assertFalse(result.ready)
        self.assertIn('three distinct', result.message)

    def test_two_positions_plus_repeat_are_not_three_positions(self):
        _, result = self.fit([-20, 20, 20], [-5, 35, 35])
        self.assertFalse(result.ready)

    def test_narrow_span_is_rejected_across_circular_seam(self):
        _, result = self.fit([176, -179, -174], [151, 156, 161])
        self.assertFalse(result.ready)
        self.assertAlmostEqual(result.expected_span_deg, 10)

    def test_same_heard_source_cannot_fit_multiple_clicked_sources(self):
        _, result = self.fit([-7.5, 0, 7.5], [20, 20, 20])
        self.assertFalse(result.ready)
        self.assertEqual(result.heard_span_deg, 0)

    def test_inconsistent_reflection_or_reversed_mapping_is_rejected(self):
        for heard in ([20, 0, -20], [-20, 20, 90]):
            with self.subTest(heard=heard):
                _, result = self.fit([-20, 0, 20], heard)
                self.assertFalse(result.ready)
                self.assertIn('disagree', result.message)

    def test_noise_plus_one_outlier_needs_three_consistent_positions(self):
        session, result = self.fit([-30, 0, 30], [-11, 21, -50])
        self.assertFalse(result.ready)
        result = self.add(session, 55, 75)
        self.assertTrue(result.ready)
        self.assertEqual(result.count, 3)
        self.assertEqual(result.outlier_count, 1)
        self.assertFalse(result.samples[2].inlier)
        self.assertAlmostEqual(result.correction_deg, -20)
        self.assertLess(result.residual_deg, 1)

    def test_large_inlier_rms_cannot_hide_behind_small_median(self):
        _, result = self.fit([-30, 0, 30], [-19, 0, 19])
        self.assertFalse(result.ready)
        self.assertGreater(result.residual_deg, 8)

    def test_half_consensus_does_not_validate_bad_geometry(self):
        _, result = self.fit([-30, 0, 30, 60, 90, 120],
                             [-10, 20, 50, 0, 30, 60])
        self.assertFalse(result.ready)

    def test_quality_weights_are_bounded(self):
        session = CalibrationSession()
        self.add(session, -30, -10, z=12, coherence=0.1)
        self.add(session, 0, 20, z=12, coherence=0.1)
        result = self.add(session, 30, 56, z=1e20, coherence=1e20)
        self.assertTrue(result.ready)
        self.assertAlmostEqual(session.samples[-1].weight, 3)
        self.assertGreater(result.correction_deg, -24)

    def test_invalid_inputs_do_not_mutate_session(self):
        session = CalibrationSession()
        self.add(session, -20, 0)
        for field in ('expected_deg', 'heard_deg', 'timestamp', 'z', 'coherence'):
            for invalid in (float('nan'), float('inf'), -float('inf'), None, 'bad'):
                with self.subTest(field=field, invalid=invalid):
                    values = dict(expected_deg=0, heard_deg=20, timestamp=1,
                                  z=18, coherence=0.3)
                    values[field] = invalid
                    result = session.add(**values)
                    self.assertFalse(result.accepted)
                    self.assertEqual(len(session.samples), 1)
                    self.assertEqual(result.count, 1)

    def test_weak_observation_does_not_mutate_session(self):
        session, previous = self.fit([-30, 0, 30], [-10, 20, 50])
        for quality in (dict(z=11.9), dict(coherence=0.099)):
            result = self.add(session, 60, 80, **quality)
            self.assertFalse(result.accepted)
            self.assertEqual(result.count, previous.count)
            self.assertEqual(session.evaluate(), previous)

    def test_threshold_quality_is_accepted(self):
        result = self.add(CalibrationSession(), 0, 20, z=12, coherence=0.1)
        self.assertTrue(result.accepted)


if __name__ == '__main__':
    unittest.main()
