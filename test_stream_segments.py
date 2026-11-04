"""Finite streaming regressions: no audio devices, cameras, robot or API calls."""
import contextlib
import io
import unittest
import wave
from unittest import mock

import numpy as np

import arm_follow
from doa import FS, HOP, MIC_XYZ, N_FFT, Localizer, Tracker


class CheapLocalizer:
    """Deterministic acoustic evidence fixture for long streaming lifecycle tests."""
    pairs = np.zeros((1, 2), int)
    band = np.ones(N_FFT // 2 + 1, bool)
    noise = np.ones(N_FFT // 2 + 1)

    def power(self, x): return np.full(N_FFT // 2 + 1, abs(float(x)))
    def weights(self, power, past): return np.ones_like(power), past
    def evidence(self, x, w):
        return np.full((1, N_FFT // 2 + 1), float(x[0]), complex), np.ones(N_FFT // 2 + 1), np.ones(N_FFT // 2 + 1)
    def peak(self, c, w, w2): return {'direction': np.array([1., 0., 0.]), 'coherence': 1., 'z': 20.}


class SegmentTrackerTests(unittest.TestCase):
    def test_continuous_sound_emits_bounded_segments_and_memory(self):
        tracker = Tracker(CheapLocalizer())
        ends = []
        for index in range(2000):
            out = tracker.push(np.array(1e12))
            self.assertLessEqual(len(tracker.window), tracker.final_window_frames)
            self.assertLessEqual(len(tracker.ratios), tracker.final_window_frames)
            self.assertLessEqual(len(tracker.recent), 4)
            if out and out[0] == 'end':
                ends.append((index, out[1]))
                self.assertEqual(tracker.quiet, 0, 'segment cap must not invent silence')
                self.assertEqual(out[1]['final_reason'], 'segment_limit')
                self.assertLessEqual(out[1]['segment_seconds'], 3.)
                self.assertLessEqual(out[1]['evidence_seconds'], 1.)
        self.assertGreaterEqual(len(ends), 10)
        self.assertTrue(all(b[0] - a[0] == tracker.max_segment_frames for a, b in zip(ends, ends[1:])))

    def test_final_direction_tracks_recent_source_not_whole_path(self):
        loc = Localizer()
        loc.noise = np.ones(N_FFT // 2 + 1)
        tracker = Tracker(loc)
        freqs = np.fft.rfftfreq(N_FFT, 1 / FS)
        final = None
        # Strong source spends 2s on +X, then almost 1s on +Y. Old whole-event
        # accumulation would select +X, whereas final evidence should select +Y.
        for index in range(tracker.max_segment_frames):
            direction = np.array([1., 0., 0.]) if index < 124 else np.array([0., 1., 0.])
            phase = 2j * np.pi * (MIC_XYZ @ direction)[:, None] * freqs[None] / 343.
            # Vary the envelope: perfectly static spectra are deliberately
            # suppressed by the existing echo/fan rejection weighting.
            out = tracker.push(100 * (1.2 + .6 * np.sin(index * .7)) * np.exp(phase))
            if out and out[0] == 'end': final = out[1]
        self.assertIsNotNone(final)
        self.assertGreater(float(np.dot(final['direction'], [0, 1, 0])), .99)
        self.assertGreater(final['evidence_start_sample'], HOP * 120)

    def test_short_transient_retains_onset_and_waits_for_quiet(self):
        tracker = Tracker(CheapLocalizer(), threshold_db=10, gap_s=.05)
        tracker.push(np.array(0.))
        tracker.push(np.array(0.))
        tracker.push(np.array(1e3))
        outputs = []
        for _ in range(100):
            out = tracker.push(np.array(0.))
            if out and out[0] == 'end': outputs.append(out[1])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]['final_reason'], 'quiet')
        self.assertEqual(outputs[0]['evidence_start_sample'], 0)
        self.assertLess(outputs[0]['evidence_seconds'], 1.)


class ClipRangeTests(unittest.TestCase):
    def test_ring_wrap_and_callback_pending_are_explicit_sample_ranges(self):
        listener = arm_follow.Listener(lambda _: None, keep_s=.032)
        values = np.arange(800, dtype=np.float32)
        listener._keep(np.repeat(values[:, None], 6, axis=1))
        clip, start, end = listener._sample_range(450, 650)
        self.assertEqual((start, end), (450, 650))
        np.testing.assert_array_equal(clip[:, 0], values[450:650])
        clip, start, end = listener._sample_range(0, 650)
        self.assertEqual((start, end), (288, 650))
        self.assertEqual(len(listener.buf), 512)

    def run_listener(self, reason):
        events = []
        listener = None

        def post(event):
            events.append(event)
            if event['kind'] in ('sound', 'error'):
                listener.stop_event.set()

        class Loc:
            def calibrate(self, _): pass
            def stft(self, frame): return [round(float(frame[-1, 0]))]

        class Script:
            def __init__(self, _): self.quiet, self.gap = 1, 1
            def push(self, value):
                self.quiet = 0 if value else 1
                ending = value == 3 if reason == 'segment_limit' else value == 0
                if ending:
                    end = 3 * HOP if reason == 'segment_limit' else 2 * HOP
                    return 'end', {'seconds': .1, 'direction': np.array([1., 0., 0.]),
                                   'evidence_start_sample': HOP, 'evidence_end_sample': end,
                                   'final_reason': reason, 'segment_seconds': .1}
                return None

        values = [1, 2, 3 if reason == 'segment_limit' else 0, 99]
        raw = np.repeat(np.repeat(np.array(values, np.float32), HOP)[:, None], 6, axis=1)

        class Stream:
            def __init__(self, callback, **_): self.callback = callback
            def __enter__(self):
                # Callback boundary is NOT aligned with the processing hop.
                self.callback(raw[:HOP + 13], HOP + 13, None, None)
                self.callback(raw[HOP + 13:], len(raw) - HOP - 13, None, None)
                return self
            def __exit__(self, *_): pass

        listener = arm_follow.Listener(post)
        with mock.patch.object(arm_follow, 'find_array', return_value=0), \
                mock.patch.object(arm_follow, 'raw_channels', return_value=[2, 3, 4, 5]), \
                mock.patch.object(arm_follow, 'Localizer', Loc), \
                mock.patch.object(arm_follow, 'Tracker', Script), \
                mock.patch.object(arm_follow, 'record', return_value=np.zeros((512, 6), np.int32)), \
                mock.patch.object(arm_follow.sd, 'InputStream', Stream), \
                mock.patch.object(listener, '_capture_time', side_effect=[100 + (HOP + 13) / FS, 100 + len(raw) / FS]), \
                contextlib.redirect_stdout(io.StringIO()):
            listener.run()
        self.assertIsNone(listener.error)
        return next(e for e in events if e['kind'] == 'sound')

    def test_quiet_tail_audio_and_evidence_end_at_last_active_sample(self):
        event = self.run_listener('quiet')
        self.assertAlmostEqual(event['observed_at'], 100 + 2 * HOP / FS)
        self.assertAlmostEqual(event['audio_ended_at'], event['observed_at'])
        self.assertAlmostEqual(event['evidence_ended_at'], event['observed_at'])
        self.assertAlmostEqual(event['audio_started_at'], 100 + HOP / FS)
        with wave.open(io.BytesIO(event['wav']), 'rb') as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        self.assertEqual(len(samples), HOP)
        self.assertTrue(np.all(samples == samples[0]))
        self.assertGreater(samples[0], 20000)  # No trailing zero/99 or pre-window 1.

    def test_continuous_segment_has_no_fictional_quiet_tail(self):
        event = self.run_listener('segment_limit')
        self.assertAlmostEqual(event['audio_ended_at'], 100 + 3 * HOP / FS)
        self.assertEqual(event['final_reason'], 'segment_limit')
        self.assertTrue(event['audio_complete'])
        with wave.open(io.BytesIO(event['wav']), 'rb') as wav:
            self.assertEqual(wav.getnframes(), 2 * HOP)


if __name__ == '__main__':
    unittest.main()
