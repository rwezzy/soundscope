"""Synthetic cancellation and export tests. No microphone/output devices opened."""
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

import audio_check


class AudioComparison(unittest.TestCase):
    def samples(self):
        t = np.arange(1600) / audio_check.FS
        tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * .1
        data = np.column_stack([tone * .5, tone * .5, tone, -tone, tone, -tone])
        return data

    def test_unaligned_mean_cancels_but_one_shared_gain_keeps_selected_intact(self):
        data = self.samples()
        selected, mean, metadata = audio_check.prepare_comparison(data, [2, 3, 4, 5])
        self.assertEqual(metadata['selection']['selected_device_channel'], 2)
        np.testing.assert_allclose(selected, data[:, 2] * metadata['shared_replay_gain'])
        np.testing.assert_allclose(mean, data[:, [2, 3, 4, 5]].mean(axis=1) * metadata['shared_replay_gain'])
        self.assertGreater(np.max(np.abs(selected)), .79)
        self.assertEqual(np.max(np.abs(mean)), 0)
        self.assertLess(metadata['old_mean_relative_rms_db'], -100)
        self.assertEqual(metadata['classification'], 'none')

    def test_saved_channels_rates_and_relative_level_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, metadata, path = audio_check.save_capture(self.samples(), [2, 3, 4, 5], ['test warning'], folder)
            for name, channels, width in [('all_six_channels_original_gain.wav', 6, 4),
                                          ('1_selected_raw_shared_gain.wav', 1, 2),
                                          ('2_old_mean_shared_gain.wav', 1, 2)]:
                with wave.open(str(path / name)) as audio:
                    self.assertEqual(audio.getnchannels(), channels)
                    self.assertEqual(audio.getsampwidth(), width)
                    self.assertEqual(audio.getframerate(), audio_check.FS)
                    self.assertEqual(audio.getnframes(), 1600)
            saved = json.loads((path / 'levels.json').read_text())
            self.assertEqual(saved['shared_replay_gain'], metadata['shared_replay_gain'])
            self.assertIn('test warning', saved['capture_warnings'])

    def test_silence_and_clipping_are_reported_and_nonfinite_withheld(self):
        silence = np.zeros((100, 6), np.float32)
        selected, mean, metadata = audio_check.prepare_comparison(silence, [2, 3, 4, 5])
        self.assertIsNone(metadata['old_mean_relative_rms_db'])
        self.assertTrue(np.isfinite(selected).all())
        self.assertEqual(metadata['shared_replay_gain'], 1)
        clipped = silence.copy()
        clipped[10, 2] = 1
        self.assertGreater(audio_check.channel_stats(clipped)[2]['clipped_fraction'], 0)
        clipped[0, 4] = np.nan
        with self.assertRaises(ValueError):
            audio_check.prepare_comparison(clipped, [2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
