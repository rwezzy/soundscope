"""Pure display-quality tests: python -m unittest test_localization_quality."""
import copy
import math
import unittest

import numpy as np

from localization_quality import assess


class LocalizationQuality(unittest.TestCase):
    def setUp(self):
        self.fix = {'z': 20.0, 'coherence': 0.7}

    @staticmethod
    def gaussian(shape=(60, 80), center=(30, 40), sigma=4.0):
        y, x = np.indices(shape)
        return 0.7 * np.exp(-((y - center[0]) ** 2 + (x - center[1]) ** 2) /
                            (2 * sigma ** 2))

    def test_compact_strong_peak_is_candidate(self):
        scores = self.gaussian()
        quality, support = assess(scores, self.fix)
        self.assertEqual(quality['state'], 'candidate')
        self.assertAlmostEqual(quality['peak_score'], 0.7)
        self.assertGreater(quality['support_fraction'], 0)
        self.assertLess(quality['support_fraction'], 0.15)
        np.testing.assert_array_equal(support, scores >= 0.8 * scores.max())

    def test_broad_plateau_is_ambiguous(self):
        scores = np.zeros((40, 50))
        scores[10:30, 10:40] = 0.7
        quality, support = assess(scores, self.fix)
        self.assertEqual(quality['state'], 'ambiguous')
        self.assertGreater(quality['support_fraction'], 0.15)
        self.assertTrue(support[20, 20])

    def test_tall_thin_ridge_does_not_claim_resolved_height(self):
        scores = np.zeros((90, 160))
        scores[:, 79:81] = .7
        quality, support = assess(scores, self.fix)
        self.assertEqual(quality['state'], 'ambiguous')
        self.assertIn('height', quality['reason'])
        self.assertEqual(quality['vertical_span_fraction'], 1.)
        self.assertLess(quality['support_fraction'], .02)
        self.assertTrue(support.any())

    def test_wide_thin_ridge_does_not_claim_resolved_bearing(self):
        scores = np.zeros((90, 160))
        scores[44:46, :] = .7
        quality, _ = assess(scores, self.fix)
        self.assertEqual(quality['state'], 'ambiguous')
        self.assertIn('bearing', quality['reason'])
        self.assertEqual(quality['horizontal_span_fraction'], 1.)

    def test_camera_crop_cannot_hide_whole_hemisphere_ambiguity(self):
        fix = dict(self.fix, directional_ambiguous=True,
                   directional_reason='Comparable directions outside the crop.')
        quality, _ = assess(self.gaussian(), fix)
        self.assertEqual(quality['state'], 'ambiguous')
        self.assertIn('outside', quality['reason'])

    def test_axis_fraction_must_be_valid(self):
        for fraction in (0, -1, 1.1, float('nan')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                assess(self.gaussian(), self.fix, max_axis_fraction=fraction)

    def test_broad_dome_is_not_sharpened_by_median_subtraction(self):
        quality, _ = assess(self.gaussian(sigma=30), self.fix)
        self.assertEqual(quality['state'], 'ambiguous')
        self.assertGreater(quality['support_fraction'], 0.15)

    def test_separated_comparable_peaks_are_ambiguous(self):
        scores = self.gaussian(center=(30, 15)) + 0.9 * self.gaussian(center=(30, 65))
        quality, _ = assess(scores, self.fix)
        self.assertEqual(quality['state'], 'ambiguous')
        self.assertIn('Multiple', quality['reason'])
        self.assertLess(quality['support_fraction'], 0.15)

    def test_small_secondary_peak_is_not_comparable(self):
        scores = self.gaussian(center=(30, 15)) + 0.5 * self.gaussian(center=(30, 65))
        quality, _ = assess(scores, self.fix)
        self.assertEqual(quality['state'], 'candidate')

    def test_absolute_sound_evidence_precedes_map_sharpness(self):
        for field, value in (('z', 11.9), ('coherence', 0.099)):
            with self.subTest(field=field):
                fix = dict(self.fix, **{field: value})
                quality, support = assess(self.gaussian(), fix)
                self.assertEqual(quality['state'], 'weak')
                self.assertFalse(support.any())

    def test_absolute_map_peak_precedes_relative_normalization(self):
        quality, support = assess(self.gaussian() * 0.01, self.fix)
        self.assertEqual(quality['state'], 'weak')
        self.assertFalse(support.any())

    def test_quality_threshold_is_inclusive(self):
        quality, _ = assess(self.gaussian(), {'z': 12, 'coherence': 0.1})
        self.assertEqual(quality['state'], 'candidate')

    def test_flat_positive_zero_and_negative_noise_maps_are_weak(self):
        for scores in (np.full((30, 40), 0.7), np.zeros((30, 40)),
                       self.gaussian() * 0.01 - 0.1):
            with self.subTest(peak=scores.max()):
                quality, support = assess(scores, self.fix)
                self.assertEqual(quality['state'], 'weak')
                self.assertFalse(support.any())

    def test_tiny_contrast_cannot_make_a_bright_spot(self):
        quality, support = assess(0.7 + self.gaussian() * 0.001, self.fix)
        self.assertEqual(quality['state'], 'weak')
        self.assertFalse(support.any())

    def test_nonfinite_map_is_weak_without_hiding_bad_cells(self):
        for invalid in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(invalid=invalid):
                scores = self.gaussian()
                scores[0, 0] = invalid
                quality, support = assess(scores, self.fix)
                self.assertEqual(quality['state'], 'weak')
                self.assertFalse(support.any())

    def test_missing_and_invalid_quality_are_weak(self):
        for fix in ({}, None, {'z': 20}, {'z': math.nan, 'coherence': 0.7},
                    {'z': 20, 'coherence': math.inf}, {'z': 'bad', 'coherence': 0.7}):
            with self.subTest(fix=fix):
                quality, support = assess(self.gaussian(), fix)
                self.assertEqual(quality['state'], 'weak')
                self.assertFalse(support.any())

    def test_shapes_and_boolean_mask(self):
        for shape in ((1, 1), (1, 80), (60, 1), (0, 10), (10, 0), (3,), (2, 3, 4), ()):
            with self.subTest(shape=shape):
                values = np.zeros(shape)
                quality, support = assess(values, self.fix)
                self.assertEqual(support.shape, values.shape)
                self.assertEqual(support.dtype, np.bool_)
                self.assertEqual(quality['state'], 'weak')

    def test_inputs_are_not_mutated_and_mask_is_independent(self):
        scores = self.gaussian()
        original, fix = scores.copy(), copy.deepcopy(self.fix)
        scores.flags.writeable = False
        quality, support = assess(scores, self.fix)
        support[:] = False
        np.testing.assert_array_equal(scores, original)
        self.assertEqual(self.fix, fix)
        self.assertIsInstance(quality['peak_score'], float)
        self.assertIsInstance(quality['support_fraction'], float)
        self.assertIsInstance(quality['reason'], str)


if __name__ == '__main__':
    unittest.main()
