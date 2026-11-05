"""Offline UI checks; no hardware, network, credentials, or model calls.

Run: python -B -m unittest test_scope_ui
Synthetic preview: python -B test_scope_ui.py --preview PATH.png
"""
import copy
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

import scope_ui


def synthetic_scene(width=960, height=540):
    """A deliberately synthetic tabletop; no private camera capture is used."""
    image = np.empty((height, width, 3), np.uint8)
    image[:] = (62, 65, 67)
    horizon = int(height * 0.45)
    image[horizon:] = (108, 116, 124)
    for xx in range(0, width, max(1, width // 12)):
        cv2.line(image, (width // 2 + (xx - width // 2) // 3, horizon), (xx, height), (119, 127, 135), 1)
    for yy in np.linspace(horizon, height - 1, 7).astype(int):
        cv2.line(image, (0, yy), (width - 1, yy), (119, 127, 135), 1)
    cv2.rectangle(image, (int(width * 0.08), int(height * 0.1)), (int(width * 0.28), int(height * 0.37)), (87, 82, 73), -1)
    cv2.rectangle(image, (int(width * 0.11), int(height * 0.13)), (int(width * 0.25), int(height * 0.34)), (114, 108, 96), -1)
    cv2.ellipse(image, (int(width * 0.61), int(height * 0.73)), (int(width * 0.09), int(height * 0.025)), 0, 0, 360, (69, 76, 80), -1)
    cv2.rectangle(image, (int(width * 0.54), int(height * 0.53)), (int(width * 0.67), int(height * 0.72)), (61, 70, 78), -1)
    cv2.rectangle(image, (int(width * 0.55), int(height * 0.54)), (int(width * 0.66), int(height * 0.65)), (91, 104, 116), -1)
    cv2.circle(image, (int(width * 0.27), int(height * 0.71)), int(height * 0.064), (113, 131, 142), -1, cv2.LINE_AA)
    cv2.ellipse(image, (int(width * 0.27), int(height * 0.65)), (int(height * 0.064), int(height * 0.025)), 0, 0, 360, (147, 161, 168), -1)
    cv2.putText(image, 'SYNTHETIC CAMERA SCENE', (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (193, 193, 193), 1, cv2.LINE_AA)
    return image


def synthetic_view(state='candidate', now=10.0, age=0.25):
    yy, xx = np.mgrid[0:54, 0:96]
    if state == 'ambiguous':
        scores = np.exp(-((xx - 49) ** 2 / 850 + (yy - 31) ** 2 / 250))
        reason = 'Broad support across several surfaces. Keep listening.'
    else:
        scores = np.exp(-((xx - 58) ** 2 / 50 + (yy - 34) ** 2 / 42))
        reason = 'A compact region agrees with the microphone array.'
    support = scores > 0.48
    return {'time': now - age, 'final': True, 'expires_after': 1.2,
            'fix': {'z': 8.3, 'coherence': 0.67, 'snr_db': 14.2},
            'scores': scores, 'spot': (580, 340), 'point': np.array([0.12, 0.08, 1.14]), 'measured': True,
            'support_mask': support, 'quality': {'state': state, 'reason': reason, 'peak_score': 0.76,
                                                'support_fraction': float(support.mean())}}


class Dashboard(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_scene()
        self.view = synthetic_view()

    def camera(self, canvas):
        x, y, w, h = scope_ui.camera_rect(self.frame.shape)
        return canvas[y:y + h, x:x + w]

    def test_native_camera_and_rgb_preview_crop(self):
        canvas = scope_ui.render(self.frame, None, '', '', now=10)
        np.testing.assert_array_equal(self.camera(canvas), self.frame)
        np.testing.assert_array_equal(scope_ui.camera_rgb_crop(canvas, self.frame.shape), self.frame[..., ::-1])
        self.assertEqual(canvas.dtype, np.uint8)

    def test_mouse_mapping_excludes_dashboard_and_all_image_edges(self):
        for shape in ((24, 32, 3), (540, 960, 3), (1080, 1920, 3)):
            with self.subTest(shape=shape):
                x, y, w, h = scope_ui.camera_rect(shape)
                self.assertEqual((w, h), (shape[1], shape[0]))
                self.assertEqual(scope_ui.canvas_to_frame(x, y, shape), (0, 0))
                self.assertEqual(scope_ui.canvas_to_frame(x + w - 1, y + h - 1, shape), (w - 1, h - 1))
                for point in ((x - 1, y), (x, y - 1), (x + w, y), (x, y + h)):
                    self.assertIsNone(scope_ui.canvas_to_frame(*point, shape))

    def test_live_and_final_estimates_expire_without_any_camera_marker(self):
        for final in (False, True):
            for field in ('scores', 'off_screen'):
                with self.subTest(final=final, field=field):
                    view = synthetic_view(age=1.21)
                    view['final'] = final
                    if field == 'off_screen':
                        view['off_screen'] = ('left', 39)
                    out = scope_ui.render(self.frame, view, '', '', now=10)
                    np.testing.assert_array_equal(self.camera(out), self.frame)
                    self.assertEqual(scope_ui._state(view, 10)[0], 'STALE')

    def test_default_expiration_and_custom_expiration(self):
        self.view.pop('expires_after')
        self.assertEqual(scope_ui._state(self.view, 11.0)[0], 'STALE')
        self.view['expires_after'] = 3.0
        self.assertEqual(scope_ui._state(self.view, 11.0)[0], 'CANDIDATE')

    def test_heatmap_keeps_low_score_scene_visible(self):
        out = scope_ui.render(self.frame, self.view, '', '', now=10)
        camera = self.camera(out)
        np.testing.assert_array_equal(camera[:160, :400], self.frame[:160, :400])
        self.assertGreater(np.count_nonzero(camera != self.frame), 0)
        self.assertLess(np.count_nonzero(np.any(camera != self.frame, axis=2)) / (540 * 960), 0.20)

    def test_broad_ambiguity_has_no_single_pixel_target(self):
        view = synthetic_view('ambiguous')
        a = self.camera(scope_ui.render(self.frame, view, '', '', now=10))
        view['spot'] = (30, 30)
        b = self.camera(scope_ui.render(self.frame, view, '', '', now=10))
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.count_nonzero(a != self.frame), 0)

    def test_weak_and_rejected_estimates_show_heat_without_a_precise_marker(self):
        for view in (synthetic_view('weak'), dict(self.view, rejected=True)):
            view['support_mask'] = np.zeros_like(view['scores'], bool)
            out = scope_ui.render(self.frame, view, '', '', now=10)
            self.assertGreater(np.count_nonzero(self.camera(out) != self.frame), 0)
            view['spot'] = (10, 10)
            view['contour'] = np.array([[[10, 10]], [[200, 10]], [[200, 200]]])
            changed_spot = scope_ui.render(self.frame, view, '', '', now=10, diagnostics=True)
            np.testing.assert_array_equal(self.camera(out), self.camera(changed_spot))
            hidden = scope_ui.render(self.frame, view, '', '', now=10, heatmap=False, diagnostics=True)
            np.testing.assert_array_equal(self.camera(hidden), self.frame)

    def test_weak_map_is_graded_even_with_empty_support_mask(self):
        from unittest.mock import patch
        frame = np.full_like(self.frame, 80)
        scores = np.tile(np.linspace(0., .03, 96), (54, 1))
        view = synthetic_view('weak')
        view.update(scores=scores, support_mask=np.zeros(scores.shape, bool))
        with patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
            out = scope_ui.render(frame, view, '', '', now=10)
        camera = self.camera(out)
        low, high = camera[270, 240], camera[270, 840]
        self.assertGreater(np.linalg.norm(high.astype(float) - 80), np.linalg.norm(low.astype(float) - 80) + 1)
        # Weak evidence is deliberately subdued; require a visible graded tint,
        # not a particular palette's brightness or a minimum confidence signal.
        self.assertGreater(np.linalg.norm(low.astype(float) - 80), 1)
        labels = [str(call.args[1]) for call in text.call_args_list]
        self.assertIn('WEAK', labels)
        self.assertIn('HEATMAP / WEAK EVIDENCE', labels)
        self.assertIn('Direction unresolved', labels)

    def test_flat_or_invalid_maps_do_not_manufacture_heat(self):
        for scores in (np.zeros((6, 8)), np.full((6, 8), 0.3), np.full((6, 8), np.nan)):
            view = synthetic_view('weak')
            view.update(scores=scores, support_mask=np.zeros(scores.shape, bool))
            np.testing.assert_array_equal(self.camera(scope_ui.render(self.frame, view, '', '', now=10)), self.frame)

    def test_heatmap_switch_retains_uncertainty_boundary(self):
        on = self.camera(scope_ui.render(self.frame, self.view, '', '', now=10))
        off = self.camera(scope_ui.render(self.frame, self.view, '', '', now=10, heatmap=False))
        np.testing.assert_array_equal(off[340, 580], self.frame[340, 580])
        self.assertFalse(np.array_equal(on[340, 580], self.frame[340, 580]))
        self.assertGreater(np.count_nonzero(off != self.frame), 0)

    def test_palettes_change_only_display_and_keep_weak_uncertainty(self):
        view = synthetic_view('weak')
        before = copy.deepcopy(view)
        colors = []
        for palette in scope_ui.HEATMAP_PALETTES:
            with self.subTest(palette=palette):
                out = scope_ui.render(self.frame, view, '', '', now=10,
                                      heatmap_palette=palette, heatmap_opacity=.65)
                camera = self.camera(out)
                colors.append(camera[340, 580].tolist())
                np.testing.assert_array_equal(camera[:160, :400], self.frame[:160, :400])
                moved_spot = dict(view, spot=(10, 10))
                alternate = scope_ui.render(self.frame, moved_spot, '', '', now=10,
                                            heatmap_palette=palette, heatmap_opacity=.65)
                np.testing.assert_array_equal(self.camera(alternate), camera)
                zero = scope_ui.render(self.frame, view, '', '', now=10,
                                       heatmap_palette=palette, heatmap_opacity=0)
                np.testing.assert_array_equal(self.camera(zero), self.frame)
                stale = scope_ui.render(self.frame, view, '', '', now=20,
                                        heatmap_palette=palette, heatmap_opacity=.85)
                np.testing.assert_array_equal(self.camera(stale), self.frame)
        self.assertEqual(len({tuple(color) for color in colors}), len(colors))
        np.testing.assert_array_equal(view['scores'], before['scores'])
        self.assertEqual(view['quality'], before['quality'])

    def test_palette_controls_do_not_make_flat_evidence_visible(self):
        for palette in scope_ui.HEATMAP_PALETTES:
            view = synthetic_view('weak')
            view['scores'] = np.full_like(view['scores'], .2)
            out = scope_ui.render(self.frame, view, '', '', now=10,
                                  heatmap_palette=palette, heatmap_opacity=.85)
            np.testing.assert_array_equal(self.camera(out), self.frame)

    def test_palette_controls_reject_invalid_values(self):
        for palette in ('unknown', None, []):
            with self.assertRaises(ValueError):
                scope_ui.render(self.frame, None, '', '', heatmap_palette=palette)
        for opacity in (-.1, .86, float('nan'), float('inf'), 'bad'):
            with self.assertRaises(ValueError):
                scope_ui.render(self.frame, None, '', '', heatmap_opacity=opacity)

    def test_search_field_preserves_camera_and_shows_focused_controls(self):
        from unittest.mock import patch
        for query in ('', 'ringing near the door', 'long query ' * 70):
            with self.subTest(query=query[:30]):
                with patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
                    out = scope_ui.render(self.frame, None, '', 'Prior note', now=10,
                                          search_text=query, diagnostics=True)
                np.testing.assert_array_equal(self.camera(out), self.frame)
                labels = [str(call.args[1]) for call in text.call_args_list]
                for label in ('SEARCH SOUND HISTORY', 'ENTER', 'search', 'ESC', 'cancel', 'BACKSPACE'):
                    self.assertIn(label, labels)
                self.assertNotIn('Prior note', labels)
                self.assertNotIn('DIAGNOSTICS', labels)
                if query:
                    self.assertTrue(any(label.endswith(' |') for label in labels))
                else:
                    self.assertIn('Type a sound or description...', labels)

    def test_search_field_is_responsive_without_changing_camera_geometry(self):
        for viewport in ((1280, 720), (1366, 768), (1920, 1080)):
            ordinary = scope_ui.render(self.frame, None, '', '', now=10, viewport_size=viewport)
            searching = scope_ui.render(self.frame, None, '', '', now=10, viewport_size=viewport,
                                        search_text='phone ringing near the door')
            np.testing.assert_array_equal(scope_ui.camera_rgb_crop(ordinary, self.frame.shape, viewport),
                                          scope_ui.camera_rgb_crop(searching, self.frame.shape, viewport))

    def test_history_result_empty_long_and_input_priority(self):
        from unittest.mock import patch
        ordinary = scope_ui.render(self.frame, None, '', '', now=10)
        empty = scope_ui.render(self.frame, None, '', '', now=10, search_answer='')
        np.testing.assert_array_equal(ordinary, empty)
        for answer in ('A ring was recorded 12 seconds ago, near the door.', 'long history answer ' * 90):
            with self.subTest(answer=answer[:30]):
                with patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
                    out = scope_ui.render(self.frame, None, '', '', now=10, search_answer=answer)
                labels = [str(call.args[1]) for call in text.call_args_list]
                self.assertIn('HISTORY RESULT', labels)
                self.assertFalse(np.array_equal(self.camera(out), self.frame))
                # Results never cover the top of the image or move its geometry.
                np.testing.assert_array_equal(self.camera(out)[:200], self.frame[:200])
        with patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
            scope_ui.render(self.frame, None, '', '', now=10, search_text='', search_answer='Prior result')
        labels = [str(call.args[1]) for call in text.call_args_list]
        self.assertIn('SEARCH SOUND HISTORY', labels)
        self.assertNotIn('HISTORY RESULT', labels)

    def test_captions_never_overlap_after_screen_edge_clamping(self):
        from unittest.mock import patch
        image = np.zeros((360, 640, 3), np.uint8)
        captions = [{'text': f'Saved caption {i}', 'pixel': (320 + i * 4, 20), 'kind': 'speech'} for i in range(12)]
        with patch.object(scope_ui, '_round_rect', wraps=scope_ui._round_rect) as draw:
            scope_ui._captions(image, captions, 1.0)
        boxes = [call.args[1] for call in draw.call_args_list]
        self.assertGreater(len(boxes), 1)
        for i, (x, y, w, h) in enumerate(boxes):
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y + h, image.shape[0])
            for bx, by, bw, bh in boxes[i + 1:]:
                self.assertFalse(x < bx + bw and bx < x + w and y < by + bh and by < y + h)

    def test_recent_events_are_real_input_and_preserve_camera(self):
        from unittest.mock import patch
        events = [{'time': '2026-11-04T22:01:02-05:00', 'sound': {'label': 'Ringing'}}]
        with patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
            result = scope_ui.render(self.frame, None, '', '', now=10, recent_events=events)
        labels = [str(call.args[1]) for call in text.call_args_list]
        self.assertIn('RECENT EVENTS / B TO BROWSE', labels)
        self.assertIn('22:01:02', labels)
        self.assertIn('Ringing', labels)
        np.testing.assert_array_equal(self.camera(result), self.frame)

    def test_bearing_dial_withholds_arrow_for_weak_ambiguous_and_expired_views(self):
        from unittest.mock import patch
        for state, age in (('weak', .1), ('ambiguous', .1), ('candidate', 10)):
            with patch.object(scope_ui, '_bearing_dial', wraps=scope_ui._bearing_dial) as draw:
                scope_ui.render(self.frame, synthetic_view(state, age=age), '', '', now=10)
            self.assertIsNone(draw.call_args.args[3])
        with patch.object(scope_ui, '_bearing_dial', wraps=scope_ui._bearing_dial) as draw:
            scope_ui.render(self.frame, self.view, '', '', now=10)
        self.assertAlmostEqual(draw.call_args.args[3], np.degrees(np.arctan2(.12, 1.14)))

    def test_bearing_rejects_nonfinite_or_missing_coordinates(self):
        for view in ({}, {'point': [float('nan'), 0, 1]}, {'off_screen': ('left', float('inf'))},
                     {'off_screen': 'bad'}, {'point': [1, 0, -1]}):
            self.assertIsNone(scope_ui._view_bearing(view))

    def test_missing_and_degenerate_fields_are_safe(self):
        cases = [None, {}, {'fix': {}}, {'scores': np.zeros((5, 5))}, {'scores': np.full((5, 5), np.nan)},
                 {'scores': []}, {'off_screen': ('above', 20)}, {'scores': np.ones((3, 3)), 'support_mask': np.ones((4, 4))},
                 {'quality': {'state': 'candidate'}, 'measured': True, 'point': None},
                 {'time': float('nan'), 'expires_after': float('nan')}, {'quality': {'reason': 'x' * 900}}]
        for view in cases:
            with self.subTest(view_keys=list(view or {})):
                out = scope_ui.render(self.frame, view, 'keys jingling', 'x' * 400, now=10, diagnostics=True)
                self.assertEqual(out.ndim, 3)

    def test_no_input_mutation_and_deterministic_output(self):
        before_frame, before_view = self.frame.copy(), copy.deepcopy(self.view)
        kwargs = dict(now=10, fps=29.8, depth_coverage=0.88, diagnostics=True, calibration_active=True)
        a = scope_ui.render(self.frame, self.view, '', '', **kwargs)
        b = scope_ui.render(self.frame, self.view, '', '', **kwargs)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(self.frame, before_frame)
        np.testing.assert_array_equal(self.view['scores'], before_view['scores'])
        np.testing.assert_array_equal(self.view['support_mask'], before_view['support_mask'])
        self.assertEqual(self.view['quality'], before_view['quality'])

    def test_small_and_large_camera_frames_keep_native_dimensions(self):
        for shape in ((24, 32, 3), (1080, 1920, 3)):
            frame = np.full(shape, 127, np.uint8)
            out = scope_ui.render(frame, None, None, None, now=0, paused=True)
            np.testing.assert_array_equal(scope_ui.camera_rgb_crop(out, shape), frame)

    def test_viewport_fits_camera_without_stretching_or_cropping(self):
        for viewport in ((1280, 720), (1366, 768), (1920, 1080)):
            for source_shape in ((720, 1280, 3), (480, 640, 3), (960, 540, 3)):
                with self.subTest(viewport=viewport, source_shape=source_shape):
                    frame = synthetic_scene(source_shape[1], source_shape[0])
                    out = scope_ui.render(frame, None, '', '', now=10, viewport_size=viewport)
                    self.assertEqual(out.shape, (viewport[1], viewport[0], 3))
                    x, y, w, h = scope_ui.camera_rect(source_shape, viewport)
                    self.assertLessEqual(x + w, viewport[0])
                    self.assertLessEqual(y + h, viewport[1])
                    self.assertAlmostEqual(w / h, source_shape[1] / source_shape[0], delta=1 / h)
                    interpolation = cv2.INTER_AREA if w < source_shape[1] else cv2.INTER_LINEAR
                    expected = cv2.resize(frame, (w, h), interpolation=interpolation)[..., ::-1]
                    np.testing.assert_array_equal(scope_ui.camera_rgb_crop(out, source_shape, viewport), expected)

    def test_responsive_mouse_mapping_inverts_scaling_and_excludes_letterbox(self):
        for viewport in ((1280, 720), (1366, 768), (1920, 1080)):
            for shape in ((720, 1280, 3), (480, 640, 3), (960, 540, 3)):
                with self.subTest(viewport=viewport, shape=shape):
                    x, y, w, h = scope_ui.camera_rect(shape, viewport)
                    for u, v in ((0, 0), (shape[1] // 2, shape[0] // 2), (shape[1] - 1, shape[0] - 1)):
                        # A source pixel's displayed center maps back to that pixel.
                        shown_x = x + (u + .5) * w / shape[1]
                        shown_y = y + (v + .5) * h / shape[0]
                        self.assertEqual(scope_ui.canvas_to_frame(shown_x, shown_y, shape, viewport), (u, v))
                    for outside in ((x - 1, y), (x, y - 1), (x + w, y), (x, y + h)):
                        self.assertIsNone(scope_ui.canvas_to_frame(*outside, shape, viewport))

    def test_range_label_never_calls_depth_a_sound_distance(self):
        # Capture the visible text to protect this semantic distinction without OCR.
        from unittest.mock import patch
        original = scope_ui._text
        labels = []
        def record(img, text, *args, **kwargs):
            labels.append(str(text))
            return original(img, text, *args, **kwargs)
        with patch.object(scope_ui, '_text', side_effect=record):
            scope_ui.render(self.frame, self.view, '', '', now=10)
        self.assertIn('CANDIDATE SURFACE RANGE', labels)
        self.assertIn('Visible surface, not source distance.', labels)
        self.assertTrue(any('1.15 m' == label for label in labels))
        self.assertFalse(any('%' in label or 'confidence' in label.lower() for label in labels))

    def test_calibration_keeps_progress_and_error_notes_and_apply_controls(self):
        from unittest.mock import patch
        for note in ('1/3 observations; move the source to another position.',
                     'No valid depth at that pixel. Click inside the source.',
                     'Calibration ready: 3 positions. ENTER saves; ESC cancels.'):
            with self.subTest(note=note):
                with patch.object(scope_ui, '_wrap', wraps=scope_ui._wrap) as wrap, \
                     patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
                    scope_ui.render(self.frame, self.view, '', note, now=10, calibration_active=True)
                self.assertIn(note, [call.args[0] for call in wrap.call_args_list])
                labels = [str(call.args[1]) for call in text.call_args_list]
                for required in ('ENTER', 'apply', 'ESC', 'cancel'):
                    self.assertIn(required, labels)

    def test_visible_controls_map_to_actions_outside_camera(self):
        events = [{'id': 'saved-1'}, {'id': 'saved-2'}]
        for viewport in ((1280, 720), (1920, 1080)):
            for action, (x, y, w, h) in scope_ui.control_rects(self.frame.shape, viewport).items():
                self.assertGreaterEqual(x, 0)
                self.assertLessEqual(x + w, viewport[0])
                self.assertLessEqual(y + h, viewport[1])
                self.assertEqual(scope_ui.control_hit_test(x + w / 2, y + h / 2,
                                                          self.frame.shape, viewport), action)
                self.assertIsNone(scope_ui.canvas_to_frame(x + w / 2, y + h / 2, self.frame.shape, viewport))
            for record, (x, y, w, h) in zip(events, scope_ui.recent_event_rects(self.frame.shape, viewport)):
                self.assertEqual(scope_ui.control_hit_test(x + w / 2, y + h / 2,
                                                          self.frame.shape, viewport, events), ('event', record['id']))
            self.assertIsNone(scope_ui.control_hit_test(-10, -10, self.frame.shape, viewport, events))

    def test_recent_event_labels_keep_provenance_and_user_annotation(self):
        from unittest.mock import patch
        events = [
            {'id': 'one', 'sound': {'label': 'Predicted bell', 'source': 'muse-spark'},
             'annotation': {'label': 'My timer', 'source': 'user'}},
            {'id': 'two', 'sound': {'label': 'Telephone ringing', 'source': 'muse-spark'}},
            {'id': 'three', 'acoustic': {'character': 'Tonal event'}},
        ]
        with patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
            scope_ui.render(self.frame, None, '', '', now=10, recent_events=events)
        labels = [str(call.args[1]) for call in text.call_args_list]
        for expected in ('My timer', 'Your label', 'AI estimate', 'Acoustic pattern'):
            self.assertIn(expected, labels)
        self.assertNotIn('Predicted bell', labels)
        self.assertEqual(events[0]['sound']['label'], 'Predicted bell')

    def test_naming_editor_and_help_are_distinct_from_model_analysis(self):
        from unittest.mock import patch
        with patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
            out = scope_ui.render(self.frame, None, '', '', now=10, search_text='My desk timer', entry_kind='label')
        labels = [str(call.args[1]) for call in text.call_args_list]
        self.assertIn('NAME SAVED SOUND', labels)
        self.assertIn('ENTER saves your label / ESC cancels / does not train recognition', labels)
        self.assertNotIn('SEARCH SOUND HISTORY', labels)
        np.testing.assert_array_equal(self.camera(out), self.frame)
        with patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
            scope_ui.render(self.frame, None, '', '', now=10, show_help=True)
        labels = [str(call.args[1]) for call in text.call_args_list]
        self.assertIn('Make sense of a sound.', labels)
        self.assertIn('? to close help', labels)

    def test_style_controls_preserve_stale_and_weak_evidence_rules(self):
        for style in scope_ui.HEATMAP_STYLES:
            stale = scope_ui.render(self.frame, self.view, '', '', now=30, heatmap_style=style)
            np.testing.assert_array_equal(self.camera(stale), self.frame)
            weak = synthetic_view('weak')
            with_style = scope_ui.render(self.frame, weak, '', '', now=10, heatmap_style=style)
            smooth = scope_ui.render(self.frame, weak, '', '', now=10, heatmap_style='smooth')
            np.testing.assert_array_equal(self.camera(with_style), self.camera(smooth))
        for style in ('invented', None, []):
            with self.assertRaises(ValueError):
                scope_ui.render(self.frame, None, '', '', heatmap_style=style)


def write_preview(path):
    frame = synthetic_scene()
    view = synthetic_view()
    preview = scope_ui.render(frame, view, None, 'A short sound was heard. The highlighted region is a candidate.',
                              now=10, fps=29.8, depth_coverage=0.94)
    path = Path(path)
    if not cv2.imwrite(str(path), preview):
        raise OSError(f'Could not write preview to {path}')
    print(path.resolve())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--preview':
        write_preview(sys.argv[2])
    else:
        unittest.main()
