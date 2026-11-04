"""Display-only palette checks; no camera, microphones or paid calls."""
import copy
import unittest

import numpy as np

import scope_ui


class HeatmapPalettes(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((80, 120, 3), 100, np.uint8)
        self.view = {'time': 10., 'scores': np.tile(np.linspace(0, 1, 15), (10, 1)),
                     'quality': {'state': 'weak'}, 'spot': (110, 40)}

    def overlay(self, palette='amber', opacity=.44):
        image = self.frame.copy()
        scope_ui._camera_overlay(image, self.view, 'WEAK', True, False, 1,
                                 palette, opacity)
        return image

    def test_preserves_original_default_tables(self):
        level = np.linspace(0, 1, 256, dtype=np.float32)
        blend = np.clip((level - .25) / .75, 0, 1)[:, None]
        colors = (np.asarray(scope_ui.CYAN, np.float32) * (1-blend)
                  + np.asarray(scope_ui.AMBER, np.float32) * blend)
        for weak in (False, True):
            alpha = (.40 if weak else .44) * level ** 1.2
            retention, contribution = scope_ui._heat_tables(weak)
            np.testing.assert_array_equal(retention, np.rint((1-alpha)*255).astype(np.uint8))
            np.testing.assert_array_equal(contribution,
                np.rint(colors * alpha[:, None]).astype(np.uint8).reshape(256, 1, 3))

    def test_palettes_change_only_tint_and_preserve_low_score_pixels(self):
        before = copy.deepcopy(self.view)
        outputs = [self.overlay(name) for name in scope_ui.HEATMAP_PALETTES]
        for out in outputs:
            self.assertEqual(out.shape, self.frame.shape)
            self.assertEqual(out.dtype, np.uint8)
            np.testing.assert_array_equal(out[:, 0], self.frame[:, 0])
        self.assertEqual(len({out.tobytes() for out in outputs}), len(outputs))
        np.testing.assert_array_equal(self.view['scores'], before['scores'])
        self.assertEqual(self.view['spot'], before['spot'])
        np.testing.assert_array_equal(self.frame, np.full_like(self.frame, 100))

    def test_opacity_zero_is_transparent_and_strength_increases(self):
        for name in scope_ui.HEATMAP_PALETTES:
            np.testing.assert_array_equal(self.overlay(name, 0), self.frame)
            low = self.overlay(name, .2).astype(float)
            high = self.overlay(name, .8).astype(float)
            self.assertGreater(np.abs(high-self.frame).sum(), np.abs(low-self.frame).sum())

    def test_render_accepts_all_presets_and_keeps_geometry(self):
        shape = None
        for name in scope_ui.HEATMAP_PALETTES:
            out = scope_ui.render(self.frame, self.view, '', '', now=10,
                                  heatmap_palette=name, heatmap_opacity=.6)
            shape = shape or out.shape
            self.assertEqual(out.shape, shape)

    def test_invalid_settings_rejected_even_without_event(self):
        for palette, opacity in [('unknown', .44), ('amber', -.1), ('amber', .86),
                                 ('amber', float('nan')), ('amber', float('inf')),
                                 ('amber', None), ('amber', 'bad')]:
            with self.subTest(palette=palette, opacity=opacity), self.assertRaises(ValueError):
                scope_ui.render(self.frame, None, '', '', heatmap_palette=palette,
                                heatmap_opacity=opacity)


if __name__ == '__main__':
    unittest.main()
