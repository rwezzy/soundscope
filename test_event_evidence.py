"""Local temporary-file/worker tests; no devices, playback, services or API calls."""
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import wave

import numpy as np

from event_evidence import EvidenceStore, EvidenceWriter


def wav_clip(seconds=0.25, rate=16000, channels=1, width=2):
    output = io.BytesIO()
    with wave.open(output, 'wb') as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(bytes(round(seconds * rate) * width * channels))
    return output.getvalue()


class EvidenceFiles(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / 'evidence'
        self.store = EvidenceStore(self.root)
        self.wav = wav_clip()
        self.frame = np.full((720, 1280, 3), (30, 80, 170), dtype=np.uint8)
        self.timing = dict(frame_at=100.1, event_observed_at=100.0,
                           event_onset_at=99.0, event_end_at=100.0)

    def save(self, **changes):
        return self.store.save('ev-20261104-235000-test', self.wav, self.frame, **{**self.timing, **changes})

    def test_actual_bytes_image_sizes_and_timing_roundtrip(self):
        artifact = self.save(geometry={'camera': {'position_mm': [0, 0, 0]}})
        image, audio, metadata = self.store.load(artifact)
        self.assertEqual(audio, self.wav)
        self.assertEqual(image.shape, (540, 960, 3))
        self.assertEqual(artifact['original_image_size'], [1280, 720])
        self.assertEqual(artifact['stored_image_size'], [960, 540])
        np.testing.assert_allclose(image[100, 100], self.frame[100, 100], atol=3)
        self.assertAlmostEqual(artifact['frame_offset_s'], 0.1)
        self.assertTrue(metadata['scene_valid'])
        self.assertFalse(metadata['frame_during_audio'])
        self.assertFalse(metadata['hardware_synchronized'])
        self.assertFalse(metadata['source_verified'])
        self.assertTrue(metadata['audio_normalized'])
        self.assertEqual(metadata['audio_duration_s'], 0.25)
        for key in ('image_path', 'audio_path', 'metadata_path'):
            self.assertFalse(Path(artifact[key]).is_absolute())
            self.assertTrue((self.root / artifact[key]).is_file())

    def test_late_or_early_scene_is_flagged_not_claimed_as_event_location(self):
        for frame_at in (102.0, 96.0):
            with self.subTest(frame_at=frame_at):
                artifact = self.save(frame_at=frame_at)
                _, _, metadata = self.store.load(artifact)
                self.assertFalse(artifact['scene_valid'])
                self.assertFalse(metadata['scene_valid'])
                self.assertIn('do not use it to locate', metadata['scene_reason'])

    def test_scene_inside_audio_window_is_explicitly_distinguished(self):
        artifact = self.save(frame_at=99.5)
        _, _, metadata = self.store.load(artifact)
        self.assertTrue(metadata['frame_during_audio'])
        self.assertEqual(metadata['distance_to_audio_s'], 0)

    def test_default_window_uses_observed_timestamp_without_inventing_onset(self):
        artifact = self.store.save('ev-no-onset', self.wav, self.frame, frame_at=99.8, event_observed_at=100)
        _, _, metadata = self.store.load(artifact)
        self.assertEqual(metadata['event_onset_at_monotonic'], 100)
        self.assertEqual(metadata['event_end_at_monotonic'], 100)

    def test_no_path_injection_through_ids_or_load_paths(self):
        for name in ('../other', 'folder/event', '..', 'C:\\outside', '/tmp/outside', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.save(name, self.wav, self.frame, **self.timing)
        artifact = self.save()
        for path in ('../outside.jpg', str(self.root.parent / 'outside.jpg'), 'C:\\outside.jpg'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.store.load({**artifact, 'image_path': path})

    def test_wrong_audio_format_long_empty_and_truncated_audio_fail(self):
        bad = [b'not a WAV', wav_clip(rate=8000), wav_clip(channels=2), wav_clip(width=1),
               wav_clip(seconds=10.1), wav_clip(seconds=0), self.wav[:-40]]
        for audio in bad:
            with self.subTest(size=len(audio)), self.assertRaises(ValueError):
                self.store.save('ev-audio', audio, self.frame, **self.timing)

    def test_ten_second_clip_is_accepted_without_truncation(self):
        audio = wav_clip(seconds=10)
        artifact = self.store.save('ev-ten', audio, self.frame, **self.timing)
        _, actual, metadata = self.store.load(artifact)
        self.assertEqual(actual, audio)
        self.assertEqual(metadata['audio_duration_s'], 10)

    def test_bad_image_geometry_and_timestamps_fail_before_publication(self):
        for frame in (np.zeros((10, 10)), np.zeros((10, 10, 3), dtype=float),
                      np.zeros((10, 10, 4), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                self.store.save('ev-image', self.wav, frame, **self.timing)
        for changes in ({'frame_at': float('nan')}, {'event_observed_at': float('inf')},
                        {'event_onset_at': 101}, {'geometry': {'position': float('nan')}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.save(**changes)

    def test_missing_or_changed_files_are_clear_failures(self):
        artifact = self.save()
        (self.root / artifact['audio_path']).unlink()
        with self.assertRaisesRegex(ValueError, 'missing or unreadable'):
            self.store.load(artifact)
        artifact = self.save()
        (self.root / artifact['image_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.store.load(artifact)

    def test_metadata_edit_cannot_turn_a_late_frame_into_valid_evidence(self):
        artifact = self.save(frame_at=105)
        path = self.root / artifact['metadata_path']
        metadata = json.loads(path.read_text())
        metadata['scene_valid'] = True
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'metadata has changed'):
            self.store.load(artifact)

    def test_failed_publication_leaves_no_partial_visible_bundle(self):
        with mock.patch('event_evidence.os.replace', side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):
                self.save()
        self.assertEqual(list(self.root.iterdir()), [])


class EvidenceQueue(unittest.TestCase):
    def test_bounded_submission_copies_frame_and_geometry_and_reports_results(self):
        entered, release = threading.Event(), threading.Event()
        saved = []

        def save(record_id, wav, frame, **kwargs):
            entered.set()
            release.wait(2)
            saved.append((record_id, frame.copy(), copy.deepcopy(kwargs)))
            return {'record_id': record_id}

        store = mock.Mock()
        store.save.side_effect = save
        writer = EvidenceWriter(store, max_pending=1)
        self.addCleanup(lambda: (release.set(), writer.close()))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        geometry = {'x': [1, 2, 3]}
        self.assertTrue(writer.submit('ev-first', wav_clip(), frame, frame_at=100,
                                      event_observed_at=100, geometry=geometry))
        self.assertTrue(entered.wait(1))
        frame[:] = 255
        geometry['x'][0] = 99
        self.assertTrue(writer.submit('ev-second', wav_clip(), frame, frame_at=100, event_observed_at=100))
        self.assertFalse(writer.submit('ev-full', wav_clip(), frame, frame_at=100, event_observed_at=100))
        release.set()
        results = writer.close(timeout=3)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result[2] is None for result in results))
        self.assertEqual(saved[0][1].max(), 0)
        self.assertEqual(saved[0][2]['geometry']['x'][0], 1)
        self.assertFalse(writer.submit('ev-closed', wav_clip(), frame, frame_at=100, event_observed_at=100))
        self.assertFalse(writer.thread.is_alive())

    def test_worker_error_is_reported_and_next_event_can_succeed(self):
        store = mock.Mock()
        store.save.side_effect = [ValueError('invalid clip'), {'record_id': 'ev-good'}]
        writer = EvidenceWriter(store)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        writer.submit('ev-bad', wav_clip(), frame, frame_at=100, event_observed_at=100)
        writer.submit('ev-good', wav_clip(), frame, frame_at=100, event_observed_at=100)
        results = writer.close(timeout=3)
        self.assertEqual(results[0], ('ev-bad', None, 'invalid clip'))
        self.assertEqual(results[1], ('ev-good', {'record_id': 'ev-good'}, None))


if __name__ == '__main__':
    unittest.main()
