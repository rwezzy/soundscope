"""Offline regression checks: loud transients must not relax spatial evidence gates."""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from doa import FS, Tracker
from sound_memory import worth_remembering
from test_doa_noise import events


class RememberingGate(unittest.TestCase):
    def test_preserves_existing_evidence_thresholds(self):
        self.assertTrue(worth_remembering({'z': 20, 'snr_db': -8}))
        self.assertTrue(worth_remembering({'z': 10, 'snr_db': 0}))
        self.assertFalse(worth_remembering({'z': 19, 'snr_db': -0.1}))
        self.assertFalse(worth_remembering({'z': 6, 'snr_db': 60}))
        self.assertFalse(worth_remembering({'z': 9.7, 'snr_db': 22}))

    def test_invalid_values_cannot_be_rescued_by_a_large_z_score(self):
        for field in ('z', 'snr_db', 'coherence'):
            for value in (math.nan, math.inf, -math.inf, 'invalid', []):
                with self.subTest(field=field, value=value):
                    fix = {'z': 60, 'snr_db': 20, 'coherence': 0.3}
                    fix[field] = value
                    self.assertFalse(worth_remembering(fix))
        for fix in (None, [], {}, {'z': 60, 'snr_db': 20, 'coherence': 0}):
            self.assertFalse(worth_remembering(fix))

    def test_ambiguous_or_invalid_direction_is_not_a_remembered_location(self):
        base = {'z': 60, 'snr_db': 20, 'coherence': 0.3}
        self.assertTrue(worth_remembering(dict(base, direction=[0, 0, 1])))
        self.assertFalse(worth_remembering(dict(base, directional_ambiguous=True)))
        for direction in ([0, 0, 0], [0, 1], [0, 0, math.nan], [0, math.inf, 1], None):
            with self.subTest(direction=direction):
                self.assertFalse(worth_remembering(dict(base, direction=direction)))

    def test_loud_independent_microphone_noise_is_not_spatial_evidence(self):
        rng = np.random.default_rng(82)
        audio = rng.standard_normal((8 * FS, 4)) * 1e-3
        audio[2 * FS:4 * FS] *= 30  # loud, but no coherent arrival direction
        found = events(audio)
        self.assertTrue(found, 'The detector must encounter the noise burst in this fixture.')
        self.assertFalse(any(worth_remembering(fix) for _, fix in found),
                         [(round(t, 2), fix['z'], fix['snr_db']) for t, fix in found])


class TrackerLoudnessMeaning(unittest.TestCase):
    def test_transient_snr_keeps_event_average_meaning(self):
        loc = SimpleNamespace(pairs=range(6), peak=lambda *args: {'z': 30})
        tracker = Tracker(loc)
        tracker.ratios = [101.0] + [1.0] * 31
        tracker.frames = 32
        result = tracker._result(tracker.event)
        self.assertAlmostEqual(result['snr_db'], 10 * math.log10(100 / 32))


if __name__ == '__main__':
    unittest.main()
