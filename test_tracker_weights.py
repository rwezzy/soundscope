"""Offline invariants for exponentially weighted acoustic evidence."""
import unittest

import numpy as np

from doa import Localizer, N_FFT, Tracker


class TrackerWeightMoments(unittest.TestCase):
    def test_running_evidence_matches_explicitly_aged_sample_weights(self):
        rng = np.random.default_rng(42)
        loc = Localizer()
        tracker = Tracker(loc, memory_s=.08)
        bins = N_FFT // 2 + 1
        spectra = np.exp(1j * rng.uniform(-np.pi, np.pi, (12, 4, bins)))
        weights = rng.uniform(.05, 1, (12, bins))
        for spectrum, weight in zip(spectra, weights):
            tracker._add(spectrum, weight[None])

        ages = np.arange(len(weights) - 1, -1, -1)
        effective = weights * tracker.decay ** ages[:, None]
        reference = loc.evidence(spectra, effective)
        for actual, expected in zip(tracker.run, reference):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

        # Kish effective sample count must agree with the explicitly aged
        # samples. This quantity determines the random-phase variance in peak().
        actual_neff = tracker.run[1].sum() ** 2 / tracker.run[2].sum()
        expected_neff = effective.sum() ** 2 / np.square(effective).sum()
        self.assertAlmostEqual(actual_neff, expected_neff, places=9)
        self.assertGreaterEqual(actual_neff, 1)
        self.assertLessEqual(actual_neff, effective.size)

        # Completed events intentionally keep their original, unaged evidence.
        for actual, expected in zip(tracker.event, loc.evidence(spectra, weights)):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_aging_without_new_evidence_preserves_effective_sample_count(self):
        loc = Localizer()
        tracker = Tracker(loc, memory_s=.08)
        bins = N_FFT // 2 + 1
        spectrum = np.ones((4, bins), complex)
        weight = np.ones((1, bins))
        for _ in range(4):
            tracker._add(spectrum, weight)
        before = tracker.run[1].sum() ** 2 / tracker.run[2].sum()
        for _ in range(7):
            tracker._add(spectrum, np.zeros_like(weight))
        after = tracker.run[1].sum() ** 2 / tracker.run[2].sum()
        self.assertAlmostEqual(after, before, places=9)


if __name__ == '__main__':
    unittest.main()
