"""Offline history evidence rendering checks; no devices, model calls or files read."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import history_ui
from test_scope_ui import synthetic_scene


def event(index=0):
    return {'id': f'test-{index}', 'epoch': 1700000000 - index * 30,
            'time': '2023-11-14T17:13:20-05:00',
            'sound': {'label': 'Phone ringing', 'confidence': .8},
            'where': {'words': 'to the left', 'on_screen': True, 'pixel': [640, 360],
                      'distance_m': 1.2, 'distance_kind': 'candidate_surface',
                      'quality': 'candidate', 'source_verified': False},
            'loudness': {'snr_db': 12.4},
            'evidence': {'scene_valid': True, 'frame_offset_s': .07,
                         'original_image_size': [1280, 720], 'audio_normalized': True},
            'evidence_image': synthetic_scene(640, 360), 'evidence_audio': True}


class History(unittest.TestCase):
    def test_empty_and_corrupt_records_stay_renderable(self):
        for events in ([], [{}], [None], [{'epoch': 'bad', 'where': None, 'sound': 12}],
                       [{'epoch': float('nan'), 'evidence_image': 'not an image'}]):
            with self.subTest(events=str(events)[:50]):
                image = history_ui.render_history(events, now=1700000100)
                self.assertEqual(image.shape, (720, 1280, 3))
                self.assertEqual(image.dtype, np.uint8)

    def test_layout_click_mapping_and_paging(self):
        for viewport in ((1280, 720), (1920, 1080), (1366, 768), (800, 600)):
            for selected in (0, 6, 12):
                layout = history_ui.layout_positions(viewport, 14, selected)
                self.assertLessEqual(len(layout['rows']), 6)
                for index, rect in layout['rows']:
                    x, y, w, h = rect
                    self.assertGreaterEqual(x, 0)
                    self.assertLessEqual(x + w, viewport[0])
                    self.assertLessEqual(y + h, viewport[1])
                    self.assertEqual(history_ui.hit_test(x + w / 2, y + h / 2, viewport, 14, selected), ('select', index))
                for name in ('live', 'play', 'analyze', 'name', 'search'):
                    x, y, w, h = layout[name]
                    self.assertEqual(history_ui.hit_test(x + w / 2, y + h / 2, viewport, 14, selected), (name, None))
                if selected >= 6:
                    x, y, w, h = layout['previous']
                    self.assertEqual(history_ui.hit_test(x + 2, y + 2, viewport, 14, selected),
                                     ('select', layout['page_start'] - 6))
                if layout['page_start'] + 6 < 14:
                    x, y, w, h = layout['next']
                    self.assertEqual(history_ui.hit_test(x + 2, y + 2, viewport, 14, selected),
                                     ('select', layout['page_start'] + 6))
                self.assertIsNone(history_ui.hit_test(0, 0, viewport, 14, selected))

    def test_marker_transforms_original_to_saved_dimensions(self):
        item = event()
        self.assertEqual(history_ui.recorded_marker(item, (360, 640, 3)), (320, 180))
        item['evidence_image_size'] = [1920, 1080]
        item['where']['pixel'] = [960, 540]
        self.assertEqual(history_ui.recorded_marker(item, (360, 640, 3)), (320, 180))

    def test_no_marker_for_unassociated_weak_offscreen_or_invalid_data(self):
        cases = [({'scene_valid': False}, {}), ({'scene_valid': None}, {}),
                 ({}, {'quality': 'weak'}), ({}, {'quality': 'ambiguous'}),
                 ({}, {'on_screen': False}), ({}, {'pixel': [float('nan'), 30]}),
                 ({}, {'pixel': [-1, 30]}), ({}, {'pixel': [1280, 30]}),
                 ({}, {'pixel': 'invalid'}), ({'original_image_size': [0, 720]}, {})]
        for evidence, where in cases:
            item = event()
            item['evidence'].update(evidence)
            item['where'].update(where)
            self.assertIsNone(history_ui.recorded_marker(item, (360, 640, 3)))

    def test_historical_labels_never_claim_a_live_verified_source(self):
        item = event()
        with patch.object(history_ui, '_text', wraps=history_ui._text) as text, \
                patch.object(history_ui, '_button', wraps=history_ui._button) as buttons:
            history_ui.render_history([item], now=1700000100, search_query='phone ring', search_result_count=1)
        labels = [str(call.args[1]) for call in text.call_args_list]
        for required in ('RECORDED EVENT', 'HISTORICAL / NOT LIVE', '1 MATCHING EVENTS'):
            self.assertIn(required, labels)
        self.assertIn(('play', 'Play audio'), [(call.args[2], call.args[3]) for call in buttons.call_args_list])
        self.assertTrue(any('source identity unverified' in label for label in labels))
        self.assertTrue(any('candidate surface 1.20 m' in label for label in labels))
        self.assertTrue(any('relative to estimated room noise' in label for label in labels))
        self.assertNotIn('CAMERA LIVE', labels)

    def test_header_actions_are_disjoint_and_selection_actions_disable_without_events(self):
        for viewport in ((640, 360), (1280, 720), (800, 600), (1920, 1080)):
            layout = history_ui.layout_positions(viewport, 0)
            previous_right = 0
            for action in ('analyze', 'name', 'search', 'live'):
                x, y, w, h = layout[action]
                self.assertGreaterEqual(x, previous_right)
                self.assertLessEqual(x + w, viewport[0])
                self.assertLessEqual(y + h, viewport[1])
                previous_right = x + w
                actual = history_ui.hit_test(x + w / 2, y + h / 2, viewport, 0)
                self.assertEqual(actual, None if action in ('analyze', 'name') else (action, None))

    def test_actions_are_visible_icon_buttons_and_actual_transcript_is_preserved(self):
        item = event()
        item['speech'] = {'text': 'A transcript supplied by this recorded event.'}
        with patch.object(history_ui, '_button', wraps=history_ui._button) as buttons, \
                patch.object(history_ui, '_text', wraps=history_ui._text) as text:
            history_ui.render_history([item], now=1700000100)
        icon_labels = [(call.args[2], call.args[3]) for call in buttons.call_args_list]
        for control in (('analyze', 'Analyze'), ('name', 'Name'), ('search', 'Search'),
                        ('live', 'Live view'), ('play', 'Play audio'),
                        ('previous', 'Prev'), ('next', 'Next')):
            self.assertIn(control, icon_labels)
        labels = [str(call.args[1]) for call in text.call_args_list]
        self.assertIn('TRANSCRIPT  A transcript supplied by this recorded event.', labels)
        self.assertNotIn('B  RETURN TO LIVE', labels)
        self.assertNotIn('E  PLAY SAVED AUDIO', labels)
        with patch.object(history_ui, '_text', wraps=history_ui._text) as text:
            history_ui.render_history([event()], now=1700000100)
        self.assertFalse(any(str(call.args[1]).startswith('TRANSCRIPT') for call in text.call_args_list))

    def test_missing_and_late_frames_cannot_gain_a_marker(self):
        for image, valid in ((None, True), (synthetic_scene(640, 360), False)):
            item = event()
            item['evidence_image'] = image
            item['evidence']['scene_valid'] = valid
            with patch.object(history_ui, '_text', wraps=history_ui._text) as text:
                history_ui.render_history([item], now=1700000100)
            labels = [str(call.args[1]) for call in text.call_args_list]
            self.assertTrue(any('unavailable' in label.lower() or 'timing unverified' in label.lower() for label in labels))
            self.assertFalse(any('Circle: recorded acoustic candidate' in label for label in labels))

    def test_snapshot_is_not_stretched_and_inputs_unchanged(self):
        item = event()
        item['where']['quality'] = 'weak'  # compare saved pixels without a marker
        before = copy.deepcopy(item)
        for viewport in ((1280, 720), (1920, 1080), (800, 600)):
            result = history_ui.render_history([item], viewport_size=viewport, now=1700000100)
            x, y, w, h = history_ui.layout_positions(viewport, 1)['image_area']
            frame = item['evidence_image']
            ratio = min(w / frame.shape[1], h / frame.shape[0])
            iw, ih = round(frame.shape[1] * ratio), round(frame.shape[0] * ratio)
            ix, iy = x + (w - iw) // 2, y + (h - ih) // 2
            expected = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
            np.testing.assert_array_equal(result[iy:iy + ih, ix:ix + iw], expected)
        np.testing.assert_array_equal(item['evidence_image'], before['evidence_image'])
        self.assertEqual(item['where'], before['where'])

    def test_selection_clamps_and_long_labels_do_not_crash(self):
        records = [event(i) for i in range(15)]
        records[14]['sound']['label'] = 'Long label ' * 100
        records[14]['speech'] = {'text': 'A very long transcript. ' * 100}
        image = history_ui.render_history(records, 999, now=1700000100, status='Replay started',
                                          search_query='a long search query ' * 50)
        self.assertEqual(image.shape, (720, 1280, 3))
        self.assertEqual(history_ui.layout_positions(event_count=15, selected=999)['selected'], 14)

    def test_search_query_editor_has_focused_controls_and_keeps_record_metadata(self):
        for query in ('', 'phone ring', 'long query ' * 120):
            with patch.object(history_ui, '_text', wraps=history_ui._text) as text:
                image = history_ui.render_history([event()], now=1700000100, search_text=query)
            labels = [str(call.args[1]) for call in text.call_args_list]
            self.assertIn('SEARCH SOUND HISTORY', labels)
            self.assertIn('RECORDED EVENT', labels)
            self.assertIn('Search recorded sounds / Cancel to return', labels)
            self.assertEqual(image.shape, (720, 1280, 3))

    def test_editor_buttons_use_scaled_shared_geometry_and_block_background_actions(self):
        for viewport in ((640, 360), (1280, 720), (800, 600), (1920, 1080)):
            layout = history_ui.layout_positions(viewport, 10, 6)
            for query in ('', 'ring'):
                for action in ('submit', 'cancel', 'delete'):
                    x, y, w, h = layout['editor_' + action]
                    self.assertEqual(history_ui.hit_test(x + w / 2, y + h / 2, viewport,
                                                        10, 6, search_text=query), (action, None))
                for action in ('live', 'play', 'analyze', 'name', 'search'):
                    x, y, w, h = layout[action]
                    self.assertIsNone(history_ui.hit_test(x + w / 2, y + h / 2, viewport,
                                                         10, 6, search_text=query))
                _, (x, y, w, h) = layout['rows'][0]
                self.assertIsNone(history_ui.hit_test(x + w / 2, y + h / 2, viewport,
                                                     10, 6, search_text=query))

    def test_search_and_label_editors_have_visible_submit_cancel_delete_icons(self):
        for kind, label in (('search', 'Search'), ('label', 'Save label')):
            with patch.object(history_ui, '_button', wraps=history_ui._button) as buttons:
                image = history_ui.render_history([event()], now=1700000100,
                                                  search_text='doorbell', entry_kind=kind)
            controls = [(call.args[2], call.args[3]) for call in buttons.call_args_list]
            for action in (('submit', label), ('cancel', 'Cancel'), ('delete', 'Delete')):
                self.assertIn(action, controls)
            self.assertEqual(image.shape, (720, 1280, 3))


if __name__ == '__main__':
    unittest.main()
