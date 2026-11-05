"""Offline recordings: selecting a raw channel avoids unaligned array cancellation."""
import json
import math
import unittest

import numpy as np

from arm_follow import select_event_audio, sound_level
from meta_api import wav_bytes
from sound_memory import wav_samples


class EventAudio(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(16000) / 16000
        self.ring = (0.02 * np.sin(2 * np.pi * 1400 * self.t) +
                     0.008 * np.sin(2 * np.pi * 1800 * self.t)).astype(np.float32)

    def clip(self, channels):
        out = np.zeros((len(self.t), 6), dtype=np.float32)
        out[:, 2:6] = np.asarray(channels).T
        return out

    def test_antiphase_ring_survives_with_both_frequencies_and_no_input_change(self):
        clip = self.clip([self.ring, -self.ring, self.ring, -self.ring])
        original = clip.copy()
        averaged = clip[:, 2:6].mean(axis=1)
        self.assertEqual(float(np.max(np.abs(averaged))), 0.0)
        selected, metadata = select_event_audio(clip, [5, 3, 2, 4])
        np.testing.assert_array_equal(selected, self.ring)
        np.testing.assert_array_equal(clip, original)
        self.assertEqual(metadata['method'], 'single_raw_microphone')
        self.assertEqual(metadata['selected_device_channel'], 2)
        replay = wav_samples(wav_bytes(selected))
        spectrum = np.abs(np.fft.rfft(replay))
        self.assertGreater(spectrum[1400], 100)
        self.assertAlmostEqual(spectrum[1800] / spectrum[1400], 0.4, places=3)
        self.assertGreater(sound_level(selected)['peak_dbfs'], -40)
        self.assertEqual(sound_level(averaged)['peak_dbfs'], -120)

    def test_dead_dc_and_clipped_channels_are_not_selected(self):
        clipped = np.sign(self.ring)
        clip = self.clip([np.zeros_like(self.ring), clipped,
                          np.full_like(self.ring, 0.4), self.ring])
        selected, metadata = select_event_audio(clip, [2, 3, 4, 5])
        np.testing.assert_array_equal(selected, self.ring)
        self.assertEqual(metadata['selected_device_channel'], 5)
        self.assertGreater(metadata['per_channel_clipped_fraction'][1], 0.9)
        self.assertIsNone(metadata['fallback'])

    def test_single_loud_faulty_microphone_does_not_dominate(self):
        clip = self.clip([self.ring, self.ring * 1.1, self.ring * 0.9, self.ring * 20])
        selected, metadata = select_event_audio(clip, [2, 3, 4, 5])
        self.assertEqual(metadata['rms_outlier_channels'], [5])
        self.assertEqual(metadata['selected_device_channel'], 3)
        np.testing.assert_array_equal(selected, clip[:, 3])

    def test_nonfinite_channel_is_reported_and_excluded(self):
        clip = self.clip([self.ring] * 4)
        clip[20, 2] = math.nan
        clip[30, 3] = math.inf
        selected, metadata = select_event_audio(clip, [2, 3, 4, 5])
        self.assertTrue(np.isfinite(selected).all())
        self.assertEqual(metadata['nonfinite_channels'], [2, 3])
        self.assertEqual(metadata['selected_device_channel'], 4)
        self.assertIsNone(metadata['per_channel_peak'][0])
        json.dumps(metadata, allow_nan=False)
        clip[:, 2:6] = math.nan
        with self.assertRaisesRegex(ValueError, 'Every raw microphone'):
            select_event_audio(clip, [2, 3, 4, 5])

    def test_all_clipped_or_silent_audio_uses_an_explicit_fallback(self):
        clip = self.clip([np.sign(self.ring)] * 4)
        selected, metadata = select_event_audio(clip, [2, 3, 4, 5])
        self.assertEqual(metadata['fallback'], 'all_active_channels_clipped')
        np.testing.assert_array_equal(selected, clip[:, 2])
        clip.fill(0)
        selected, metadata = select_event_audio(clip, [2, 3, 4, 5])
        self.assertEqual(metadata['fallback'], 'no_ac_signal')
        self.assertFalse(np.any(selected))

    def test_raw_levels_are_not_normalised_until_wav_encoding(self):
        clip = self.clip([self.ring] * 4)
        selected, metadata = select_event_audio(clip, [2, 3, 4, 5])
        self.assertAlmostEqual(metadata['per_channel_peak'][0], float(np.max(np.abs(self.ring))))
        self.assertAlmostEqual(metadata['per_channel_rms'][0], float(np.sqrt(np.mean(self.ring.astype(float) ** 2))))
        self.assertLess(float(np.max(np.abs(selected))), 0.03)
        self.assertGreater(float(np.max(np.abs(wav_samples(wav_bytes(selected))))), 0.69)

    def test_invalid_shapes_and_channel_lists_fail_clearly(self):
        for clip, channels in ((np.zeros(10), [2]), (np.zeros((0, 6)), [2]),
                               (np.zeros((10, 6)), []), (np.zeros((10, 6)), [2, 2]),
                               (np.zeros((10, 6)), [6]), (np.zeros((10, 6)), [True])):
            with self.subTest(shape=clip.shape, channels=channels), self.assertRaises(ValueError):
                select_event_audio(clip, channels)


if __name__ == '__main__':
    unittest.main()
