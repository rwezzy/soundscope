"""Regression checks for full-hemisphere acoustic ambiguity (no hardware)."""
import unittest

import numpy as np

from doa import Localizer, N_FFT


class DirectionalAmbiguity(unittest.TestCase):
    def setUp(self):
        self.loc = Localizer()

    def evidence(self, bins):
        # Equal phase across the four microphones: true direction is normal to
        # their plane. A periodic narrow tone may also fit very different angles.
        C = np.zeros((len(self.loc.pairs), N_FFT // 2 + 1), complex)
        W = np.zeros(N_FFT // 2 + 1)
        W2 = np.zeros_like(W)
        C[:, bins] = 100
        W[bins] = W2[bins] = 100
        return C, W, W2

    def test_high_coherence_high_z_tone_can_still_be_ambiguous(self):
        fix = self.loc.peak(*self.evidence([192]))  # 6 kHz at 16 kHz / 512
        self.assertGreater(fix['z'], 12)
        self.assertGreater(fix['coherence'], .99)
        self.assertTrue(fix['directional_ambiguous'])
        self.assertGreater(fix['directional_support_radius_deg'], 50)

    def test_broadband_centered_evidence_stays_compact(self):
        fix = self.loc.peak(*self.evidence(np.arange(8, 220)))
        self.assertFalse(fix['directional_ambiguous'])
        self.assertLess(fix['directional_support_radius_deg'], 20)
        np.testing.assert_allclose(fix['direction'], [0., 0., 1.], atol=1e-12)

    def test_peak_and_input_are_unchanged_by_uncertainty_measurement(self):
        evidence = self.evidence([192])
        originals = tuple(x.copy() for x in evidence)
        gcc = self.loc.gcc(*evidence[:2])
        scores = np.take_along_axis(gcc, self.loc.lag_idx, axis=1).mean(axis=0)
        expected = int(np.argmax(scores))
        fix = self.loc.peak(*evidence)
        np.testing.assert_array_equal(fix['direction'], self.loc.dirs[expected])
        for value, original in zip(evidence, originals):
            np.testing.assert_array_equal(value, original)


if __name__ == '__main__':
    unittest.main()
