"""Offline regression checks for sound history, provenance and local query filters."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from sound_events import EventStore, describe_event


NOW = datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc)


def event(id='ring', age=20, **overrides):
    values = dict(id=id, timestamp=(NOW - timedelta(seconds=age)).isoformat(),
                  label='phone ringing', label_source='spectrogram_model_hypothesis',
                  quality='candidate', bearing_deg=-30, elevation_deg=0,
                  range_m=1.2, level_dbfs=-20, snr_db=12, nearby=['keyboard'],
                  source_verified=False, acoustic_description='repeating tonal pulses')
    values.update(overrides)
    return values


class SoundEvents(unittest.TestCase):
    def setUp(self):
        self.store = EventStore(':memory:')

    def tearDown(self):
        self.store.close()

    def test_phrase_keeps_evidence_types_distinct(self):
        data = self.store.add(event())
        phrase = data['phrase']
        for part in ['possible phone ringing', 'candidate direction left',
                     'candidate visible surface at 1.20 m', '-20.0 dBFS (uncalibrated)',
                     'nearby visible objects: keyboard', 'emitting object unverified',
                     '2026-11-04']:
            self.assertIn(part, phrase)
        self.assertNotIn('dB SPL', phrase)
        self.assertEqual(data['label_source'], 'spectrogram_model_hypothesis')

    def test_synonym_direction_and_time_filter(self):
        self.store.add(event('left'))
        self.store.add(event('right', bearing_deg=30))
        self.store.add(event('old', age=600))
        self.store.add(event('weak', quality='weak'))
        self.store.add(event('future', age=-30))
        result = self.store.search('ringtone on the left last five minutes', now=NOW)
        self.assertEqual([e['id'] for e in result], ['left'])
        self.assertIn('no embeddings', result[0]['search_method'])
        self.assertIn('score', result[0])

    def test_visual_context_does_not_become_sound_identity(self):
        self.store.add(event('unknown', label='unidentified sound', nearby=['phone', 'keyboard']))
        self.assertEqual(self.store.search('phone ringing'), [])
        self.assertEqual(self.store.search('keyboard'), [])
        result = self.store.search('near the keyboard')
        self.assertEqual([e['id'] for e in result], ['unknown'])
        self.assertEqual(result[0]['label'], 'unidentified sound')
        self.assertFalse(result[0]['source_verified'])

    def test_recording_level_filters_exclude_unknown_measurements(self):
        self.store.add(event('loud'))
        self.store.add(event('quiet', level_dbfs=-55))
        self.store.add(event('unknown', level_dbfs=None))
        self.assertEqual([e['id'] for e in self.store.search('loud sounds last five minutes', now=NOW)], ['loud'])
        self.assertEqual([e['id'] for e in self.store.search('quiet sounds')], ['quiet'])
        self.assertEqual([e['id'] for e in self.store.search('sounds above -25 dBFS')], ['loud'])
        self.assertEqual([e['id'] for e in self.store.search('sounds below -45 dBFS')], ['quiet'])

    def test_unknown_values_are_not_fabricated(self):
        data = self.store.add({'timestamp':NOW.isoformat()})
        self.assertIn('unidentified sound', data['phrase'])
        self.assertIn('direction uncertain', data['phrase'])
        self.assertNotIn(' dBFS', data['phrase'])
        self.assertNotIn('surface at', data['phrase'])
        self.assertEqual(self.store.search('sounds left'), [])

    def test_update_regenerates_phrase_and_preserves_id(self):
        self.store.add(event())
        result = self.store.update('ring', {'id':'replacement', 'nearby':['cup']})
        self.assertEqual(result['id'], 'ring')
        self.assertIn('visible objects: cup', result['phrase'])
        self.assertIsNone(self.store.get('replacement'))
        self.assertEqual(len(self.store.list_events()), 1)
        with self.assertRaises(KeyError):
            self.store.update('missing', {'label':'clapping'})

    def test_disk_persistence_and_parameterized_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.sqlite'
            malicious="x'); DROP TABLE events; --"
            with EventStore(path) as store:
                store.add(event(malicious))
                self.assertEqual(store.get(malicious)['id'], malicious)
                self.assertEqual(store.search(malicious), [])
            with EventStore(path) as store:
                self.assertEqual(len(store.list_events()),1)

    def test_thread_safety_and_retention(self):
        with EventStore(':memory:',retention=10) as store:
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda i:store.add(event(str(i), age=i)), range(40)))
            ids=[e['id'] for e in store.list_events()]
            self.assertEqual(ids, [str(i) for i in range(10)])

    def test_returned_values_do_not_mutate_saved_records(self):
        initial=event()
        self.store.add(initial)
        initial['nearby'].append('plant')
        retrieved=self.store.get('ring')
        retrieved['nearby'].append('lamp')
        self.assertEqual(self.store.get('ring')['nearby'], ['keyboard'])

    def test_bad_timestamp_rejected_and_nonfinite_measurements_unknown(self):
        with self.assertRaises(ValueError):
            self.store.add(event(timestamp='not a time'))
        data=self.store.add(event(level_dbfs=float('nan'),range_m=float('inf')))
        self.assertIsNone(data['level_dbfs'])
        self.assertIsNone(data['range_m'])
        self.assertEqual(self.store.search('loud'), [])


if __name__ == '__main__':
    unittest.main()
