"""Offline ring-buffer extraction tests: sample timing, pulse gaps and truncation."""
import unittest

import numpy as np

from arm_follow import Listener
from doa import FS, HOP


class EventClip(unittest.TestCase):
    def listener(self, samples):
        listener = Listener(lambda _: None)
        listener._keep(np.repeat(np.asarray(samples, np.float32)[:, None], 6, axis=1))
        return listener

    def test_pulsed_event_keeps_early_and_late_pulses_despite_quiet_gaps(self):
        audio = np.zeros(8 * FS, np.float32)
        # 100 ms pulses separated by 300 ms gaps: each gap is shorter than the
        # detector's 500 ms end threshold, yet active time totals only 1.6 s.
        for i in range(16):
            start = int(round((0.6 + i * 0.4) * FS))
            audio[start:start + int(0.1 * FS)] = 0.1 if i == 0 else 0.2 if i == 15 else 0.15
        listener = self.listener(audio)
        clip, timing = listener._event_clip(int(0.5 * FS), int(7.4 * FS), 108.0)
        self.assertAlmostEqual(timing['clip_start_at'], 100.1)
        self.assertAlmostEqual(timing['clip_end_at'], 107.4)
        self.assertAlmostEqual(len(clip) / FS, 7.3)
        self.assertTrue(np.any(clip[:, 0] == np.float32(0.1)))
        self.assertTrue(np.any(clip[:, 0] == np.float32(0.2)))
        self.assertFalse(timing['truncated'])
        self.assertTrue(timing['detected_onset_retained'])

    def test_long_ring_uses_full_available_buffer_and_reports_lost_start(self):
        audio = np.full(int(9.1 * FS), 0.2, np.float32)
        listener = self.listener(audio)
        clip, timing = listener._event_clip(FS, 9 * FS, 109.1)
        self.assertAlmostEqual(len(clip) / FS, 7.9)
        self.assertAlmostEqual(timing['requested_clip_start_at'], 100.6)
        self.assertAlmostEqual(timing['clip_start_at'], 101.1)
        self.assertAlmostEqual(timing['clip_end_at'], 109.0)
        self.assertTrue(timing['head_truncated'])
        self.assertTrue(timing['truncated'])
        self.assertFalse(timing['tail_truncated'])
        self.assertFalse(timing['detected_onset_retained'])
        self.assertTrue(timing['saved_audio_available'])

    def test_finalized_hop_excludes_later_samples_already_kept_from_callback(self):
        listener = self.listener(np.repeat([0.1, 0.2, 0.9], HOP))
        clip, timing = listener._event_clip(HOP, HOP * 2, 10.0 + 3 * HOP / FS)
        np.testing.assert_array_equal(clip[:, 0], np.repeat(np.array([0.1, 0.2], np.float32), HOP))
        self.assertAlmostEqual(timing['clip_start_at'], 10.0)
        self.assertAlmostEqual(timing['clip_end_at'], 10.0 + 2 * HOP / FS)
        self.assertAlmostEqual(timing['clip_end_at'] - timing['clip_start_at'], len(clip) / FS)
        self.assertTrue(timing['head_truncated'])  # startup pre-roll was unavailable
        self.assertFalse(np.any(clip == 0))  # no uninitialised startup padding was invented

    def test_history_reset_does_not_join_audio_across_missing_callbacks(self):
        listener = self.listener(np.full(FS, 0.7, np.float32))
        listener.pos = 0  # same reset performed after a missing/overflowed callback
        listener._keep(np.full((HOP * 2, 6), 0.2, np.float32))
        clip, timing = listener._event_clip(HOP, HOP * 2, 20.0)
        self.assertEqual(len(clip), HOP * 2)
        self.assertTrue(np.all(clip == np.float32(0.2)))
        self.assertTrue(timing['head_truncated'])

    def test_fully_overwritten_event_is_unavailable_not_replaced_with_new_audio(self):
        listener = self.listener(np.full(9 * FS, 0.4, np.float32))
        clip, timing = listener._event_clip(HOP, HOP * 2, 109.0)
        self.assertEqual(len(clip), 0)
        self.assertIsNone(timing['clip_start_at'])
        self.assertIsNone(timing['clip_end_at'])
        self.assertFalse(timing['saved_audio_available'])
        self.assertTrue(timing['truncated'])
        self.assertIn('no longer', timing['unavailable_reason'])


if __name__ == '__main__':
    unittest.main()
