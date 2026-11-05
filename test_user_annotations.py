"""Manual event names remain searchable user observations, never model recognition."""
import copy
import tempfile
import unittest
from unittest import mock

import sound_memory as sm


class UserAnnotations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = 1793857200.0
        self.memory = sm.Memory(self.temp.name, now=self.now)
        self.record = {
            'id': 'record-a', 'epoch': self.now - 30,
            'time': sm.timestamp(self.now - 30), 'duration_s': 2.0,
            'sound': {'label': 'door knock', 'confidence': 0.6, 'source': 'muse-spark'},
            'where': {'quality': 'ambiguous', 'words': 'to the left', 'pixel': [100, 200],
                      'point_m': [0.1, 0.2, 0.8], 'distance_m': 0.8, 'source_verified': False},
            'loudness': {'snr_db': 8, 'words': 'quiet', 'rms_dbfs': -42},
            'evidence': {'audio_path': 'record-a/audio.wav', 'image_path': 'record-a/frame.jpg'},
            'acoustic': {'character': 'mixed sound'},
        }
        self.memory.add(copy.deepcopy(self.record))

    def test_patch_preserves_model_measurements_evidence_and_event_time(self):
        result = sm.annotate_event(self.memory, 'record-a', 'my phone ringing')
        self.assertEqual(result['annotation'], {'label': 'my phone ringing', 'source': 'user'})
        without_annotation = {k: v for k, v in result.items() if k != 'annotation'}
        self.assertEqual(without_annotation, self.record)
        reloaded = sm.Memory(self.temp.name, now=self.now)
        self.assertEqual(reloaded.get('record-a'), result)

    def test_phone_query_retrieves_manual_name_with_explicit_provenance(self):
        sm.annotate_event(self.memory, 'record-a', 'phone ringing')
        result = sm.search('was there any point my phone rang?', self.memory, now=self.now)
        self.assertEqual([r['id'] for r in result['matches']], ['record-a'])
        self.assertEqual(result['source'], 'local')
        self.assertIn('you labelled "phone ringing"', result['answer'])
        self.assertIn('source_verified', result['highlights'][0])
        self.assertFalse(result['highlights'][0]['source_verified'])

    def test_table_and_index_mark_user_label_and_keep_model_data(self):
        rec = sm.annotate_event(self.memory, 'record-a', 'phone ringing')
        text = sm.table([rec], now=self.now)
        self.assertIn('user label: "phone ringing"', text)
        self.assertIn('door knock (muse-spark 0.60)', text)
        self.assertLess(text.index('user label:'), text.index('door knock'))
        self.assertTrue(sm.record_text(rec).startswith('user label: "phone ringing"'))

    def test_manual_exact_match_ranks_before_model_guess(self):
        sm.annotate_event(self.memory, 'record-a', 'phone ringing')
        guessed = copy.deepcopy(self.record)
        guessed.update(id='record-b', epoch=self.now - 1)
        guessed['sound']['label'] = 'phone ringing'
        self.memory.add(guessed)
        results = sm.local_search('phone ringing', self.memory.recent(), now=self.now)
        self.assertEqual([r['id'] for r in results], ['record-a', 'record-b'])

    def test_annotation_does_not_enroll_reference_or_name_future_events(self):
        with mock.patch.object(sm.ReferenceBank, 'add', side_effect=AssertionError('must not train')):
            sm.annotate_event(self.memory, 'record-a', 'phone ringing')
        later = copy.deepcopy(self.record)
        later.update(id='record-b', epoch=self.now)
        self.memory.add(later)
        self.assertNotIn('annotation', self.memory.get('record-b'))
        self.assertEqual(sm.what_it_was(later), 'probably door knock')

    def test_whitespace_control_cleanup_and_length_rejection_do_not_damage_existing_label(self):
        rec = sm.annotate_event(self.memory, 'record-a', '  my\r\nphone\t ringing\x00  ')
        self.assertEqual(rec['annotation']['label'], 'my phone ringing')
        for bad in ('', ' \r\n\t', '\x00', 'x' * 121, None, {'label': 'phone'}):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                sm.annotate_event(self.memory, 'record-a', bad)
        self.assertEqual(rec['annotation']['label'], 'my phone ringing')
        sm.annotate_event(self.memory, 'record-a', 'x' * 120)

    def test_unknown_event_is_not_created(self):
        with self.assertRaisesRegex(ValueError, 'no longer available'):
            sm.annotate_event(self.memory, 'missing', 'phone ringing')
        self.assertEqual(len(self.memory.recent()), 1)

    def test_annotation_cannot_claim_measured_hidden_location_or_model_origin(self):
        rec = sm.annotate_event(self.memory, 'record-a', 'phone under the pillow')
        self.assertIn('you labelled "phone under the pillow"', sm.phrase(rec, now=self.now))
        self.assertNotIn('spatial_evidence', rec)
        self.assertFalse(rec['where']['source_verified'])
        rec['annotation']['source'] = 'model'
        self.assertNotIn('you labelled', sm.what_it_was(rec))
        self.assertNotIn('pillow', sm.record_text(rec))


if __name__ == '__main__':
    unittest.main()
