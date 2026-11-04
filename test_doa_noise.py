"""Steady noise that starts after calibration must not swallow real sounds: python -m unittest test_doa_noise

Reproduces what the live viewer did on Nov 4, 02:31: a heater or vent switched on above the
camera, the tracker treated it as one endless sound, and a ring, speech and claps were lost in it.
"""
import unittest

import numpy as np

from doa import FS, MIC_XYZ, SPEED_OF_SOUND, Localizer, Tracker
from mic_test import band_limit, unit


def plane_wave(x, direction, m):
    f = np.fft.rfftfreq(m, 1 / FS)
    S = np.fft.rfft(x, m)
    X = S[None] * np.exp(2j * np.pi * f[None] * (MIC_XYZ @ direction / SPEED_OF_SOUND)[:, None])
    return np.fft.irfft(X, m)[:, :len(x)].T


def clap(n, rng):
    x = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.015 * FS))
    return band_limit(x, 300, 7000)


def room(seconds=12.0, heater_db=10.0, clap_db=24.0, clap_times=(7.0, 9.5), heater_from=2.0, seed=4):
    """4-mic recording: self-noise, a heater from 50 deg up starting at heater_from, claps from 40 deg left."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    out = rng.standard_normal((n, 4)) * 1e-3                       # mic self-noise, power 1e-6
    heater = np.zeros(n)
    start = int(heater_from * FS)
    heater[start:] = band_limit(rng.standard_normal(n - start), 100, 3000)
    heater *= np.sqrt(1e-6 * 10 ** (heater_db / 10) / np.mean(heater[start:] ** 2))
    out += plane_wave(heater, unit(10, 50), n + FS)
    for t in clap_times:
        burst = np.zeros(n)
        k = int(0.25 * FS)
        burst[int(t * FS):int(t * FS) + k] = clap(k, rng)
        burst *= np.sqrt(1e-6 * 10 ** (clap_db / 10) / np.mean(burst[int(t * FS):int(t * FS) + k] ** 2))
        out += plane_wave(burst, unit(-40, 0), n + FS)
    return out


def events(x, calibrate_s=1.5, threshold_db=1.5):
    loc = Localizer()
    loc.calibrate(x[:int(calibrate_s * FS)])
    tracker = Tracker(loc, threshold_db=threshold_db)
    X = loc.stft(x)
    found = []
    for i in range(len(X)):
        out = tracker.push(X[i])
        if out and out[0] == 'end' and out[1]:
            found.append((i * 256 / FS, out[1]))
    return found


class SteadyNoiseAfterCalibration(unittest.TestCase):
    def test_claps_still_come_out_separately_and_on_target(self):
        found = events(room())
        claps = [(t, r) for t, r in found if 6.5 <= t <= 11.5]
        self.assertGreaterEqual(len(claps), 2, f'events: {[(round(t, 1), round(r["seconds"], 1)) for t, r in found]}')
        for t, r in claps[:2]:
            bearing = -(r['azimuth'] + 180) % 360 - 180   # doa azimuth is counterclockwise; bearing + = right
            self.assertAlmostEqual(bearing, -40, delta=20, msg=f'clap at {t:.1f} s heard at {bearing:.0f} deg')
            self.assertLess(r['seconds'], 2.0)
        self.assertFalse(any(r['seconds'] > 6.0 for _, r in found), 'the heater became one endless sound')

    def test_a_sound_that_never_stops_is_still_reported(self):
        rng = np.random.default_rng(8)
        n = int(20 * FS)
        x = rng.standard_normal((n, 4)) * 1e-3
        hum = band_limit(rng.standard_normal(n - FS), 300, 3000)
        hum *= np.sqrt(1e-6 * 10 ** (12 / 10) / np.mean(hum ** 2))
        x[FS:] += plane_wave(hum, unit(-30, 0), n - FS + FS)[:n - FS]
        loc = Localizer()
        loc.calibrate(x[:FS])
        tracker = Tracker(loc, threshold_db=1.5, min_window_s=60.0)  # floor tracking off: only the cap acts
        ends = [out[1] for out in (tracker.push(frame) for frame in loc.stft(x)) if out and out[0] == 'end']
        self.assertGreaterEqual(len(ends), 2)
        self.assertTrue(all(r['seconds'] <= 8.05 for r in ends))
        self.assertEqual(ends[0].get('final_reason'), 'max_length')

    def test_quiet_room_behaviour_is_unchanged(self):
        found = events(room(heater_db=-99.0))
        self.assertEqual(len([1 for t, _ in found if 6.5 <= t <= 11.5]), 2)


if __name__ == '__main__':
    unittest.main()
