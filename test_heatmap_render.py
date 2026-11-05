"""Hardware-free display regressions: styles cannot invent localization evidence."""
import unittest

import cv2
import numpy as np

from heatmap_render import HEATMAP_PALETTES, HEATMAP_STYLES, blend_heatmap, palette_lut


class HeatmapRender(unittest.TestCase):
    def setUp(self):
        self.image = np.full((120, 160, 3), (82, 104, 128), np.uint8)
        y, x = np.indices(self.image.shape[:2], dtype=np.float32)
        self.strength = np.maximum(np.clip(1 - ((x - 43) / 32) ** 2 - ((y - 60) / 40) ** 2, 0, 1),
                                   np.clip(1 - ((x - 116) / 25) ** 2 - ((y - 62) / 28) ** 2, 0, 1))

    def test_all_styles_keep_inputs_and_mutate_only_output_image(self):
        original = self.strength.copy()
        outputs = []
        for style in HEATMAP_STYLES:
            out = self.image.copy()
            self.assertIs(blend_heatmap(out, self.strength, style=style, palette='neon'), out)
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(out.shape, self.image.shape)
            self.assertFalse(np.array_equal(out, self.image))
            np.testing.assert_array_equal(self.strength, original)
            outputs.append(out)
        for first in range(len(outputs)):
            for second in range(first):
                self.assertFalse(np.array_equal(outputs[first], outputs[second]))

    def test_missing_zero_flat_and_invalid_fields_have_no_overlay(self):
        for strength in (None, np.zeros((120, 160)), np.ones((120, 160)),
                         np.full((120, 160), .4), np.full((120, 160), np.nan),
                         np.full((120, 160), np.inf)):
            for style in HEATMAP_STYLES:
                with self.subTest(style=style):
                    out = self.image.copy()
                    blend_heatmap(out, strength, style=style, palette='neon')
                    np.testing.assert_array_equal(out, self.image)

    def test_invalid_pixels_and_zero_opacity_leave_camera_intact(self):
        strength = self.strength.copy()
        strength[40:80, 30:80] = np.nan
        strength[20:30, 110:120] = np.inf
        for style in HEATMAP_STYLES:
            out = self.image.copy()
            blend_heatmap(out, strength, style=style)
            np.testing.assert_array_equal(out[~np.isfinite(strength)], self.image[~np.isfinite(strength)])
            out = self.image.copy()
            blend_heatmap(out, self.strength, style=style, opacity=0)
            np.testing.assert_array_equal(out, self.image)

    def test_smooth_default_matches_previous_byte_blend_within_one_level(self):
        level = np.linspace(0, 1, 256, dtype=np.float32)
        alpha = .44 * level ** 1.2
        mix = np.clip((level - .25) / .75, 0, 1)[:, None]
        colors = np.array((220, 207, 75)) * (1 - mix) + np.array((91, 187, 244)) * mix
        retain = np.rint((1 - alpha) * 255).astype(np.uint8)
        tint = np.rint(colors * alpha[:, None]).astype(np.uint8).reshape(256, 1, 3)
        indices = cv2.convertScaleAbs(self.strength, alpha=255)
        old = cv2.multiply(self.image, cv2.cvtColor(cv2.LUT(indices, retain), cv2.COLOR_GRAY2BGR), scale=1/255)
        old = cv2.add(old, cv2.applyColorMap(indices, tint))
        out = blend_heatmap(self.image.copy(), self.strength, style='smooth')
        self.assertLessEqual(int(np.max(np.abs(out.astype(int) - old.astype(int)))), 1)

    def test_weak_styles_all_show_the_same_reduced_smooth_evidence(self):
        expected = blend_heatmap(self.image.copy(), self.strength, weak=True, palette='neon')
        self.assertFalse(np.array_equal(expected, self.image))
        for style in HEATMAP_STYLES:
            out = blend_heatmap(self.image.copy(), self.strength, style=style, weak=True, palette='neon')
            np.testing.assert_array_equal(out, expected)

    def test_two_broad_peaks_and_the_gap_are_preserved(self):
        for style in ('classic', 'smooth', 'bands'):
            out = blend_heatmap(np.zeros_like(self.image), self.strength, style=style, palette='neon')
            self.assertTrue(np.any(out[60, 43]))
            self.assertTrue(np.any(out[62, 116]))
            self.assertFalse(np.any(out[60, 82]))
            self.assertFalse(np.any(out[self.strength == 0]))
            # Color changes span a broad region rather than a generated peak dot.
            self.assertGreater(np.count_nonzero(np.any(out != 0, axis=2)), 3000)

    def test_classic_adds_only_actual_supplied_support_boundary_to_gradient(self):
        # Deliberately use a supplied support threshold different from .8. The
        # boundary must follow that measured mask, not silently invent a region.
        support = self.strength >= .5
        before = support.copy()
        smooth = blend_heatmap(self.image.copy(), self.strength, style='smooth')
        classic = blend_heatmap(self.image.copy(), self.strength, style='classic', support_mask=support)
        boundary = np.any(classic != smooth, axis=2)
        self.assertTrue(boundary[:, :80].any())
        self.assertTrue(boundary[:, 90:].any())
        self.assertFalse(boundary[~support].any())
        self.assertFalse(boundary[60, 43])
        self.assertFalse(boundary[62, 116])
        interior = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        self.assertFalse(boundary[interior].any())
        np.testing.assert_array_equal(support, before)
        no_outline = blend_heatmap(self.image.copy(), self.strength, style='classic', outline=False)
        np.testing.assert_array_equal(no_outline, smooth)
        empty_support = blend_heatmap(self.image.copy(), self.strength, style='classic',
                                      support_mask=np.zeros_like(support))
        np.testing.assert_array_equal(empty_support, smooth)

    def test_classic_preserves_one_pixel_support_gaps_and_invalid_holes(self):
        field = np.full((120, 160), .95, np.float32)
        field[:10] = .2
        field[50:70, 100:115] = np.nan
        support = np.ones(field.shape, bool)
        support[:, 79] = False
        smooth = blend_heatmap(self.image.copy(), field, style='smooth')
        classic = blend_heatmap(self.image.copy(), field, style='classic', support_mask=support)
        boundary = np.any(classic != smooth, axis=2)
        self.assertFalse(boundary[:, 79].any())
        self.assertTrue(boundary[20:100, 78].any())
        self.assertTrue(boundary[20:100, 80].any())
        np.testing.assert_array_equal(classic[~np.isfinite(field)], self.image[~np.isfinite(field)])

    def test_ui_classic_weak_stale_and_offscreen_keep_existing_evidence_gates(self):
        from scope_ui import _camera_overlay
        view = {'scores': self.strength, 'support_mask': self.strength >= .5,
                'spot': (42, 60), 'contour': [(1, 1), (150, 1), (150, 110)]}
        for state in ('WEAK', 'STALE', 'LISTENING'):
            classic, smooth = self.image.copy(), self.image.copy()
            _camera_overlay(classic, view, state, True, False, 1, heatmap_style='classic')
            _camera_overlay(smooth, view, state, True, False, 1, heatmap_style='smooth')
            np.testing.assert_array_equal(classic, smooth)
            if state != 'WEAK':
                np.testing.assert_array_equal(classic, self.image)
        offscreen = {**view, 'off_screen': ('left', 0)}
        classic, smooth = self.image.copy(), self.image.copy()
        _camera_overlay(classic, offscreen, 'AMBIGUOUS', True, False, 1, heatmap_style='classic')
        _camera_overlay(smooth, offscreen, 'AMBIGUOUS', True, False, 1, heatmap_style='smooth')
        np.testing.assert_array_equal(classic, smooth)

    def test_ui_classic_ambiguous_outline_keeps_all_regions_and_ignores_peak_marker(self):
        from scope_ui import _camera_overlay
        view = {'scores': self.strength, 'support_mask': self.strength >= .5}
        output = self.image.copy()
        _camera_overlay(output, view, 'AMBIGUOUS', True, False, 1, heatmap_style='classic')
        different_spot = self.image.copy()
        _camera_overlay(different_spot, {**view, 'spot': (1, 1)}, 'AMBIGUOUS',
                        True, False, 1, heatmap_style='classic')
        np.testing.assert_array_equal(output, different_spot)
        smooth = self.image.copy()
        _camera_overlay(smooth, view, 'AMBIGUOUS', True, False, 1, heatmap_style='smooth')
        boundary = np.any(output != smooth, axis=2)
        self.assertTrue(boundary[:, :80].any())
        self.assertTrue(boundary[:, 90:].any())
        self.assertFalse(boundary[60, 43])
        self.assertFalse(boundary[62, 116])

    def test_contours_follow_both_fields_and_do_not_fill_peak_centres(self):
        out = blend_heatmap(self.image.copy(), self.strength, style='contours', palette='neon')
        changed = np.any(out != self.image, axis=2)
        self.assertFalse(changed[60, 43])
        self.assertFalse(changed[62, 116])
        self.assertTrue(changed[:, :80].any())
        self.assertTrue(changed[:, 90:].any())
        self.assertFalse(changed[self.strength == 0].any())

    def test_full_resolution_contours_preserve_one_pixel_gaps_and_invalid_holes(self):
        image = np.full((720, 1280, 3), 100, np.uint8)
        strength = np.zeros((720, 1280), np.float32)
        strength[100:620, 100:1180] = .95
        strength[:, 639] = 0  # must not disappear through display downsampling
        strength[340:380, 800:850] = np.nan
        strength[50, 50] = .95  # retain a genuine isolated grid cell too
        original = strength.copy()
        out = blend_heatmap(image.copy(), strength, style='contours', palette='neon')
        changed = np.any(out != image, axis=2)
        self.assertFalse(changed[:, 639].any())
        self.assertFalse(changed[340:380, 800:850].any())
        self.assertTrue(changed[350:370, 636:639].any())
        self.assertTrue(changed[350:370, 640:643].any())
        self.assertTrue(changed[50, 50])
        np.testing.assert_array_equal(strength, original)

    def test_spotlight_reveals_actual_peak_scene_without_a_generated_circle(self):
        neutral = np.full_like(self.image, 120)
        out = blend_heatmap(neutral.copy(), self.strength, style='spotlight', palette='neon', opacity=.5)
        self.assertGreater(float(out[60, 43].mean()), float(out[60, 82].mean()))
        self.assertGreater(float(out[62, 116].mean()), float(out[60, 82].mean()))
        self.assertGreater(float(out[60, 53].mean()), float(out[60, 70].mean()))

    def test_palettes_opacity_bounds_and_shape_validation(self):
        for name in HEATMAP_PALETTES:
            table = palette_lut(name)
            self.assertEqual(table.shape, (256, 1, 3))
            self.assertFalse(table.flags.writeable)
            out = blend_heatmap(self.image.copy(), self.strength, palette=name, opacity=.85)
            self.assertFalse(np.array_equal(out, self.image))
        np.testing.assert_array_equal(palette_lut('neon'), palette_lut('pink-yellow'))
        for kwargs in ({'palette': 'missing'}, {'style': 'missing'}, {'opacity': -.1},
                       {'opacity': .86}, {'opacity': float('nan')}, {'opacity': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                blend_heatmap(self.image.copy(), self.strength, **kwargs)
        with self.assertRaisesRegex(ValueError, 'match'):
            blend_heatmap(self.image.copy(), np.zeros((10, 10)))
        with self.assertRaisesRegex(ValueError, 'uint8'):
            blend_heatmap(self.image.astype(float), self.strength)


if __name__ == '__main__':
    unittest.main()
