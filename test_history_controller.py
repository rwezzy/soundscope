"""Controller tests use an in-memory store; no hardware, files, network or playback."""
import unittest

import numpy as np

from history_controller import HistoryController
from history_ui import layout_positions


class Store:
    def __init__(self):
        self.calls = []
        self.fail = False

    def load(self, artifact):
        self.calls.append(artifact['record_id'])
        if self.fail:
            raise ValueError('checksum mismatch')
        return np.full((4, 8, 3), 127, np.uint8), b'WAV fixture', {
            'original_image_size': [1280, 720], 'scene_valid': False, 'frame_offset_s': 2.0}


def record(index):
    return {'id': f'e{index}', 'epoch': 10 + index, 'sound': {'label': 'Sound'},
            'evidence': {'record_id': f'e{index}', 'scene_valid': True}}


class Controller(unittest.TestCase):
    def test_open_takes_independent_snapshot_and_hydrates_only_selection(self):
        store = Store()
        controller = HistoryController(store)
        source = [record(0), record(1)]
        result = controller.open(source, 'phone')
        source[0]['sound']['label'] = 'Changed externally'
        self.assertEqual(controller.events[0]['sound']['label'], 'Sound')
        self.assertEqual(controller.query, 'phone')
        self.assertTrue(controller.active)
        self.assertEqual(store.calls, ['e0'])
        self.assertTrue(result['evidence_audio'])
        self.assertFalse(result['evidence']['scene_valid'])
        self.assertEqual(result['evidence_image_size'], [1280, 720])
        self.assertNotIn('evidence_image', controller.events[0])
        self.assertEqual(controller.selected_audio(), b'WAV fixture')
        self.assertEqual(store.calls, ['e0'])

    def test_selection_clamps_and_cache_is_lru_bounded(self):
        store = Store()
        controller = HistoryController(store)
        controller.open([record(i) for i in range(10)])
        for i in range(1, 10):
            controller.select(i)
        self.assertEqual(len(controller._cache), 8)
        controller.select(9)
        self.assertEqual(len(store.calls), 10)
        controller.select(0)
        self.assertEqual(len(store.calls), 11)
        controller.next(-100)
        self.assertEqual(controller.selected, 0)
        controller.next(100)
        self.assertEqual(controller.selected, 9)

    def test_load_error_cached_and_open_retries_without_crashing(self):
        store = Store()
        store.fail = True
        controller = HistoryController(store)
        result = controller.open([record(0)])
        self.assertIsNone(result['evidence_image'])
        self.assertFalse(result['evidence_audio'])
        self.assertIn('checksum mismatch', controller.status)
        for _ in range(3):
            controller.current()
            self.assertIsNone(controller.selected_audio())
        self.assertEqual(store.calls, ['e0'])
        store.fail = False
        self.assertTrue(controller.open([record(0)])['evidence_audio'])
        self.assertEqual(len(store.calls), 2)

    def test_empty_and_no_artifact_do_not_call_store(self):
        store = Store()
        controller = HistoryController(store)
        self.assertEqual(controller.open([]), {})
        self.assertIsNone(controller.selected_audio())
        controller.next(1)
        result = controller.open([{'id': 'old-event'}])
        self.assertIsNone(result['evidence_image'])
        self.assertFalse(result['evidence_audio'])
        self.assertEqual(store.calls, [])

    def test_click_select_play_and_live(self):
        controller = HistoryController(Store())
        controller.open([record(0), record(1)])
        layout = layout_positions(event_count=2)
        x, y, _, _ = layout['rows'][1][1]
        self.assertEqual(controller.handle_click(x + 5, y + 5), ('select', 1))
        self.assertEqual(controller.selected, 1)
        x, y, _, _ = layout['play']
        self.assertEqual(controller.handle_click(x + 5, y + 5), ('play', None))
        x, y, _, _ = layout['live']
        self.assertEqual(controller.handle_click(x + 5, y + 5), ('live', None))
        self.assertFalse(controller.active)
        self.assertIsNone(controller.handle_click(x + 5, y + 5))

    def test_disabled_audio_does_not_request_playback(self):
        controller = HistoryController(Store())
        controller.open([{'id': 'no-audio'}])
        x, y, _, _ = layout_positions(event_count=1)['play']
        self.assertIsNone(controller.handle_click(x + 5, y + 5))
        self.assertIn('No saved audio', controller.status)


if __name__ == '__main__':
    unittest.main()
