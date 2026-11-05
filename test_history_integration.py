"""Actual local evidence round trip and replay exclusion; no devices or paid services."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import aim
import scope
import sound_memory
from meta_api import wav_bytes
from history_controller import HistoryController


class ReplayIsolation(unittest.TestCase):
    def test_final_event_that_overlaps_playback_is_excluded_even_after_it_ends(self):
        events = [
            {'kind': 'sound', 'onset_at': 9, 'observed_at': 14, 'clip_end_at': 15},
            {'kind': 'update', 'observed_at': 11},
            {'kind': 'sound', 'onset_at': 14, 'observed_at': 15, 'clip_end_at': 16},
            {'kind': 'voice', 'wav': b'command'},
            {'kind': 'warning', 'message': 'device'},
        ]
        self.assertEqual(scope.exclude_playback(events, [(10, 13)]), events[2:])

    def test_no_replay_preserves_reports(self):
        reports = [{'kind': 'sound', 'observed_at': 10}]
        self.assertEqual(scope.exclude_playback(reports, []), reports)


class EvidenceRoundTrip(unittest.TestCase):
    def test_session_records_searches_and_replays_actual_evidence(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(sound_memory, 'MEMORY_DIR', Path(folder)), \
                mock.patch.object(sound_memory, 'REFERENCES_DIR', Path(folder) / 'references'), \
                mock.patch('builtins.print'):
            cfg = aim.load_config(profile='carrier_v3')
            session = scope.Session(SimpleNamespace(memory=True), cfg, {})
            try:
                now = time.monotonic()
                wav = wav_bytes(np.sin(2 * np.pi * 800 * np.arange(8000) / 16000))
                frame = np.full((120, 160, 3), (50, 80, 110), np.uint8)
                fix = {'direction': aim.camera_to_array([0, 0, 1], cfg), 'z': 30.,
                       'snr_db': 15., 'seconds': .5, 'coherence': .3}
                view = {'spot': (80, 60), 'point': np.array([0., 0., 1.]), 'measured': True,
                        'quality': {'state': 'candidate'}}
                capture = {'method': 'single_raw_microphone', 'selected_device_channel': 2,
                           'channel_indexing': 'zero_based'}
                rec = session.record({'kind': 'sound', 'fix': fix, 'event_id': 1, 'wav': wav,
                                      'audio_capture': capture,
                                      'observed_at': now, 'onset_at': now - .5, 'clip_end_at': now},
                                     view, frame, frame_at=now)
                self.assertIsNotNone(rec)
                session.close()  # flush disk writer before reading
                saved = session.memory.get(rec['id'])
                self.assertEqual(saved['audio_capture'], capture)
                self.assertTrue(saved['evidence']['scene_valid'])
                self.assertIsNone(session.meta)
                session.memory.update(rec['id'], {'sound': {'label': 'phone ringing', 'confidence': .9}})
                answer = session.ask({'action': 'search', 'question': 'When did the phone ring?'}, '')
                self.assertIn('phone ringing', answer.lower())
                self.assertEqual(session.captions, [])
                history = HistoryController(session.evidence_store)
                history.open(session.search_results[1]['matches'], query='When did the phone ring?')
                hydrated = history.current()
                self.assertEqual(hydrated['evidence_image'].shape, frame.shape)
                self.assertEqual(history.selected_audio(), wav)
                self.assertIs(hydrated['evidence']['source_verified'], False)
                self.assertTrue((Path(folder) / 'evidence').is_dir())
            finally:
                session.close()


if __name__ == '__main__':
    unittest.main()
