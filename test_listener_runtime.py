"""Offline Listener lifecycle/timestamp tests. No device, camera, or network is opened."""
import contextlib
import io
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import arm_follow
from doa import FS, HOP


class SilentStream:
    """An open input stream that never calls back."""
    instances = []

    def __init__(self, callback, **_):
        self.callback, self.closed = callback, False
        self.entered = threading.Event()
        self.instances.append(self)

    def __enter__(self):
        self.entered.set()
        return self

    def __exit__(self, *_):
        self.closed = True


class SimpleLocalizer:
    def calibrate(self, _samples):
        pass

    def stft(self, frame):
        return [int(frame[-1, 0])]


class ScriptTracker:
    """1 emits an active update; 2 stays active; 0 ends after a quiet frame."""
    def __init__(self, _loc, **_settings):  # e.g. threshold_db
        self.gap, self.quiet = 1, 1
        self.active = False

    def push(self, value):
        fix = {'seconds': 0.1, 'direction': np.array([1., 0., 0.])}
        if value:
            self.quiet, self.active = 0, True
            return ('update', fix) if value == 1 else None
        self.quiet += 1
        if self.active:
            self.active = False
            return 'end', fix
        return None


class ListenerRuntime(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.find = self.stack.enter_context(mock.patch.object(arm_follow, 'find_array', return_value=0))
        self.stack.enter_context(mock.patch.object(arm_follow, 'raw_channels', return_value=[2, 3, 4, 5]))
        self.stack.enter_context(mock.patch.object(arm_follow, 'Localizer', SimpleLocalizer))
        self.stack.enter_context(mock.patch.object(arm_follow, 'Tracker', ScriptTracker))
        self.calibration = self.stack.enter_context(mock.patch.object(
            arm_follow, 'record', return_value=np.zeros((512, 6), np.int32)))
        self.stack.enter_context(mock.patch.object(arm_follow.sd, 'InputStream', SilentStream))
        SilentStream.instances = []
        self.events = []

    def listener(self, **kwargs):
        listener = arm_follow.Listener(self.events.append, **kwargs)
        self.addCleanup(listener.stop)
        return listener

    def start(self, **kwargs):
        listener = self.listener(**kwargs)
        listener.start()
        self.assertTrue(listener.wait_ready(1.0), 'mock input did not become ready')
        self.assertIsNone(listener.error)
        return listener

    def wait_until(self, predicate, timeout=1.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(predicate(), 'mock worker did not complete before deadline')

    @staticmethod
    def block(value=0):
        return np.full((HOP, 6), value, np.float32)

    def test_construction_does_not_query_devices(self):
        self.listener()
        self.find.assert_not_called()

    def test_startup_failure_signals_ready_and_posts_error(self):
        self.find.side_effect = SystemExit('No mock microphone')
        listener = self.listener()
        listener.start()
        self.assertTrue(listener.wait_ready(1))
        self.assertTrue(listener.stop())
        self.assertIsInstance(listener.error, SystemExit)
        self.assertEqual(self.events[0]['kind'], 'error')
        self.assertEqual(self.events[0]['message'], 'No mock microphone')

    def test_readiness_timeout_requests_stop_without_waiting_on_discovery(self):
        release = threading.Event()
        self.addCleanup(release.set)
        self.find.side_effect = lambda: release.wait(1)
        listener = self.listener()
        listener.start()
        started = time.monotonic()
        self.assertFalse(listener.wait_ready(0.04))
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertTrue(listener.stop_event.is_set())
        release.set()
        self.assertTrue(listener.stop())
        self.calibration.assert_not_called()

    def test_stop_exits_idle_queue_and_closes_stream(self):
        listener = self.start()
        started = time.monotonic()
        self.assertTrue(listener.stop(timeout=0.5))
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertTrue(SilentStream.instances[-1].closed)
        self.assertIsNone(listener.error)
        self.assertEqual(self.events, [])

    def test_missing_callbacks_fail_instead_of_hanging(self):
        listener = self.start(input_timeout=0.08)
        self.assertTrue(listener.stopped.wait(0.6))
        self.assertIsInstance(listener.error, TimeoutError)
        self.assertEqual(self.events[-1]['kind'], 'error')
        self.assertTrue(SilentStream.instances[-1].closed)

    def test_queue_is_bounded_and_retains_latest_audio(self):
        listener = self.listener(queue_blocks=2)
        for value in range(5):
            listener._capture(self.block(value), HOP, None, None)
        self.assertEqual(listener.audio_queue.qsize(), 2)
        self.assertEqual(listener.dropped_blocks, 3)
        kept = [listener.audio_queue.get_nowait()[0][0, 0] for _ in range(2)]
        self.assertEqual(kept, [3, 4])
        listener._report_status()
        self.assertEqual(self.events[0]['kind'], 'warning')
        self.assertEqual(self.events[0]['dropped_blocks'], 3)
        self.assertNotIn('fix', self.events[0])

    def test_overflow_is_reported_separately_from_sound(self):
        listener = self.start()
        listener._capture(self.block(), HOP, None, SimpleNamespace(input_overflow=True))
        self.wait_until(lambda: bool(self.events))
        self.assertEqual(self.events[0]['kind'], 'warning')
        self.assertEqual(self.events[0]['overflows'], 1)
        self.assertNotIn('fix', self.events[0])

    def test_capture_time_maps_portaudio_clock_and_supports_missing_time_info(self):
        timing = SimpleNamespace(inputBufferAdcTime=20.0, currentTime=20.2)
        listener = self.listener()
        with mock.patch.object(arm_follow.time, 'monotonic', return_value=100.0):
            self.assertAlmostEqual(listener._capture_time(timing, HOP), 99.816)
            self.assertEqual(listener._capture_time(None, HOP), 100.0)

    def test_wdmks_independent_adc_origin_produces_fresh_capture_times(self):
        listener = self.listener()
        # These are actual values observed from the reSpeaker's WDM-KS driver.
        timing = SimpleNamespace(inputBufferAdcTime=3.56, currentTime=251135.1419084)
        with mock.patch.object(arm_follow.time, 'monotonic', return_value=500.0):
            self.assertAlmostEqual(listener._capture_time(timing, 384), 500.0)
        timing.inputBufferAdcTime += 0.024
        timing.currentTime += 0.024
        with mock.patch.object(arm_follow.time, 'monotonic', return_value=500.024):
            self.assertAlmostEqual(listener._capture_time(timing, 384), 500.024)

    def test_independent_adc_clock_keeps_delayed_audio_old(self):
        listener = self.listener()
        timing = SimpleNamespace(inputBufferAdcTime=0.0, currentTime=251131.58)
        with mock.patch.object(arm_follow.time, 'monotonic', return_value=500.0):
            self.assertAlmostEqual(listener._capture_time(timing, HOP), 500.0)
        timing.inputBufferAdcTime += HOP / FS
        timing.currentTime += 3.0
        with mock.patch.object(arm_follow.time, 'monotonic', return_value=503.0):
            self.assertAlmostEqual(listener._capture_time(timing, HOP), 500.016)

    def test_independent_adc_clock_restart_reanchors(self):
        listener = self.listener()
        timing = SimpleNamespace(inputBufferAdcTime=5.0, currentTime=251131.58)
        with mock.patch.object(arm_follow.time, 'monotonic', return_value=500.0):
            listener._capture_time(timing, HOP)
        timing.inputBufferAdcTime = 0.0
        with mock.patch.object(arm_follow.time, 'monotonic', return_value=501.0):
            self.assertEqual(listener._capture_time(timing, HOP), 501.0)

    def test_final_time_is_last_active_capture_and_id_survives_quiet_tail(self):
        listener = self.start(updates=True)
        with mock.patch.object(listener, '_capture_time', side_effect=[10., 11., 12., 13., 14.]):
            for value in [1, 2, 0, 1, 0]:
                listener._capture(self.block(value), HOP, None, None)
        self.wait_until(lambda: len(self.events) == 4)
        first, end, second, second_end = self.events
        self.assertEqual([ev['kind'] for ev in self.events], ['update', 'sound', 'update', 'sound'])
        self.assertEqual(first['observed_at'], 10.)
        self.assertEqual(end['observed_at'], 11.)
        self.assertEqual(first['event_id'], end['event_id'])
        self.assertEqual(end['onset_at'], 10.)
        self.assertNotEqual(first['event_id'], second['event_id'])
        self.assertEqual(second['event_id'], second_end['event_id'])
        self.assertEqual(second_end['observed_at'], 13.)
        self.assertTrue(all(ev['emitted_at'] >= ev['observed_at'] for ev in self.events))

    def test_missing_audio_span_starts_new_event(self):
        listener = self.start(updates=True)
        listener._capture(self.block(1), HOP, None, None)
        self.wait_until(lambda: len(self.events) == 1)
        listener._sequence += 1  # a callback was lost between the observed blocks
        listener._capture(self.block(1), HOP, None, None)
        self.wait_until(lambda: len(self.events) == 2)
        self.assertNotEqual(self.events[0]['event_id'], self.events[1]['event_id'])

    def test_final_wav_ends_at_finalized_hop_not_callback_end(self):
        from sound_memory import wav_samples
        listener = self.start(updates=True)
        block = np.concatenate([self.block(1), self.block(0), self.block(2)])
        callback_end = 10.0 + len(block) / FS
        with mock.patch.object(listener, '_capture_time', return_value=callback_end):
            listener._capture(block, len(block), None, None)
        self.wait_until(lambda: any(ev['kind'] == 'sound' for ev in self.events))
        event = next(ev for ev in self.events if ev['kind'] == 'sound')
        self.assertEqual(len(wav_samples(event['wav'])), 2 * HOP)
        self.assertAlmostEqual(event['clip_end_at'], callback_end - HOP / FS)
        self.assertAlmostEqual(event['clip_start_at'], 10.0)
        self.assertAlmostEqual(event['clip_seconds'], 2 * HOP / FS)
        self.assertEqual(event['level']['peak_dbfs'], 0.0)  # the following value-2 audio was excluded
        self.assertTrue(event['audio_capture']['head_truncated'])
        self.assertEqual(event['audio_capture']['method'], 'single_raw_microphone')

    def test_stop_releases_voice_recording_wait(self):
        listener = self.start()
        result = []
        recording = threading.Thread(target=lambda: result.append(listener.record_command(4)), daemon=True)
        recording.start()
        self.wait_until(lambda: listener.voice is not None)
        listener.stop()
        recording.join(0.5)
        self.assertFalse(recording.is_alive())
        self.assertEqual(result, [None])


class CalibrationRuntime(unittest.TestCase):
    def test_no_input_calibration_times_out_and_closes(self):
        SilentStream.instances = []
        started = time.monotonic()
        with mock.patch.object(arm_follow.sd, 'InputStream', SilentStream):
            with self.assertRaisesRegex(TimeoutError, 'calibration timed out'):
                arm_follow.record(0, 0.01, timeout=0.05)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertTrue(SilentStream.instances[-1].closed)

    def test_stop_during_calibration_closes_listener(self):
        with mock.patch.object(arm_follow, 'find_array', return_value=0), \
                mock.patch.object(arm_follow, 'raw_channels', return_value=[2, 3, 4, 5]), \
                mock.patch.object(arm_follow, 'Localizer', SimpleLocalizer), \
                mock.patch.object(arm_follow.sd, 'InputStream', SilentStream), \
                contextlib.redirect_stdout(io.StringIO()):
            SilentStream.instances = []
            listener = arm_follow.Listener(lambda _: None)
            listener.start()
            deadline = time.monotonic() + 1
            while not SilentStream.instances and time.monotonic() < deadline:
                time.sleep(0.005)
            try:
                self.assertTrue(SilentStream.instances, 'mock calibration stream did not open')
                self.assertTrue(listener.stop(0.5))
                self.assertTrue(SilentStream.instances[-1].closed)
                self.assertIsNone(listener.error)
            finally:
                listener.stop()


class SimulatedAudioRuntime(unittest.TestCase):
    def test_finite_keys_recording_localizes_and_stops_cleanly(self):
        from mic_test import simulate
        raw = simulate('keys', -5, 40.0, np.random.default_rng(3), seconds=5.0, tail=1.5) * 1e-3
        six = np.zeros((len(raw), 6), np.float32)
        six[:, 2:6], six[:, :2] = raw, raw.mean(axis=1, keepdims=True)
        calibration = np.round(six[:16000].astype(np.float64) * 2 ** 31).astype(np.int32)
        received, events = threading.Event(), []

        def post(event):
            events.append(event)
            if event['kind'] in ('sound', 'error'):
                received.set()

        class FiniteStream(SilentStream):
            def __enter__(self):
                super().__enter__()

                def feed():
                    for i in range(16000, len(six), HOP):
                        if self.closed:
                            break
                        self.callback(six[i:i + HOP], HOP, None, None)
                        time.sleep(0.002)

                self.producer = threading.Thread(target=feed, daemon=True)
                self.producer.start()
                return self

            def __exit__(self, *_):
                self.closed = True
                self.producer.join(0.5)

        with mock.patch.object(arm_follow, 'find_array', return_value=0), \
                mock.patch.object(arm_follow, 'raw_channels', return_value=[2, 3, 4, 5]), \
                mock.patch.object(arm_follow, 'record', return_value=calibration), \
                mock.patch.object(arm_follow.sd, 'InputStream', FiniteStream), \
                contextlib.redirect_stdout(io.StringIO()):
            listener = arm_follow.Listener(post, updates=True)
            listener.start()
            try:
                self.assertTrue(listener.wait_ready(1))
                self.assertTrue(received.wait(3), 'finite recording never produced a result')
                self.assertIsNone(listener.error)
                sounds = [ev for ev in events if ev['kind'] == 'sound']
                self.assertTrue(sounds)
                final = sounds[-1]
                self.assertAlmostEqual(arm_follow.bearing(final['fix']['azimuth']), 40, delta=4)
                updates = [ev for ev in events if ev['kind'] == 'update'
                           and ev['event_id'] == final['event_id']]
                self.assertTrue(updates)
                self.assertLessEqual(updates[-1]['observed_at'], final['observed_at'])
                self.assertLess(final['observed_at'], final['emitted_at'])
                self.assertTrue(final['wav'].startswith(b'RIFF'))
                self.assertEqual(listener.dropped_blocks, 0)
            finally:
                self.assertTrue(listener.stop(1))


if __name__ == '__main__':
    unittest.main()
