"""Hardware-free regression tests for the live viewer's coordination helpers.

Run ``python -B -m unittest test_scope_runtime``. No sensor, config write, network,
or Meta client is started. Camera bearings and a recording sink are synthetic.
"""
import copy
from contextlib import ExitStack
import queue
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

import aim
from camera import backproject
import scope


class LiveCalibration(unittest.TestCase):
    def setUp(self):
        self.cfg = {'array': {'mount': 'flat', 'position_mm': [0, 0, 0], 'front_yaw_deg': 25.0},
                    'camera': {'mount': 'base', 'base': {'position_mm': [0, 0, 0],
                                                       'yaw_deg': 0.0, 'pitch_deg': 0.0}}}
        self.k = {'fx': 80.0, 'fy': 80.0, 'cx': 60.0, 'cy': 40.0}
        self.depth = np.full((80, 120), 1000.0, np.float32)
        self.controller = scope.CalibrationController()
        self.controller.start(self.cfg)
        self.save = patch.object(scope, 'save_front_yaw').start()
        self.addCleanup(patch.stopall)

    def event(self, pixel=(60, 40), event_id=1, timestamp=10.0):
        point = aim.camera_to_base(backproject(*pixel, 1.0, self.k) * 1000, self.cfg)
        # The actual flat array is aligned with the base. The configured yaw is
        # deliberately wrong by +25 degrees; these clicks should recover zero.
        direction = point / np.linalg.norm(point)
        return {'kind': 'update', 'event_id': event_id, 'observed_at': timestamp,
                'fix': {'direction': direction, 'z': 20.0, 'coherence': 0.3}}

    def click(self, pixel=(60, 40), event=None, now=10.1):
        return self.controller.click(pixel, self.event(pixel) if event is None else event,
                                     self.depth, self.k, self.cfg, now=now)

    def ready(self):
        for i, pixel in enumerate(((30, 40), (60, 40), (90, 40)), start=1):
            self.click(pixel, self.event(pixel, i))
        self.assertTrue(self.controller.result.ready)

    def test_three_fresh_distinct_events_do_not_save_until_apply(self):
        before = copy.deepcopy(self.cfg)
        self.ready()
        self.assertEqual(self.cfg, before)
        self.save.assert_not_called()
        self.assertAlmostEqual(self.controller.result.correction_deg, -25)
        message = self.controller.apply(self.cfg)
        self.save.assert_called_once()
        self.assertAlmostEqual(self.save.call_args.args[0], 0)
        self.assertAlmostEqual(self.cfg['array']['front_yaw_deg'], 0)
        self.assertFalse(self.controller.active)
        self.assertIn('NEW position', message)

    def test_missing_stale_and_future_audio_are_rejected(self):
        for event in (None, self.event(timestamp=9.0), self.event(timestamp=10.2),
                      {'kind': 'update', 'fix': self.event()['fix'], 'event_id': 9}):
            with self.subTest(event=event):
                message = self.controller.click((60, 40), event, self.depth, self.k, self.cfg, now=10.1)
                self.assertIn('No fresh sound', message)
                self.assertEqual(len(self.controller.session.samples), 0)
        self.save.assert_not_called()

    def test_exact_clicked_depth_is_required_despite_valid_neighbors(self):
        for value in (0, 100, float('nan'), float('inf')):
            with self.subTest(depth=value):
                self.depth[40, 60] = value
                self.assertIn('No valid depth', self.click())
                self.assertEqual(len(self.controller.session.samples), 0)
        self.depth[40, 60] = 1000
        self.click()
        self.assertEqual(len(self.controller.session.samples), 1)

    def test_event_update_and_end_cannot_count_twice(self):
        first = self.event()
        self.click(event=first)
        repeated = dict(first, kind='sound', observed_at=10.05)
        self.assertIn('new sound', self.click(event=repeated))
        self.assertEqual(len(self.controller.session.samples), 1)
        self.assertEqual(self.controller.used_events, {1})

    def test_weak_rejected_event_can_be_retried_with_good_evidence(self):
        event = self.event()
        event['fix']['z'] = 1
        self.assertIn('weak', self.click(event=event))
        self.assertEqual(len(self.controller.session.samples), 0)
        self.assertFalse(self.controller.used_events)
        event['fix']['z'] = 20
        self.click(event=event)
        self.assertEqual(len(self.controller.session.samples), 1)

    def test_incomplete_or_failed_application_keeps_live_yaw(self):
        self.controller.apply(self.cfg)
        self.save.assert_not_called()
        self.ready()
        self.save.side_effect = OSError('synthetic write failure')
        with self.assertRaises(OSError):
            self.controller.apply(self.cfg)
        self.assertEqual(self.cfg['array']['front_yaw_deg'], 25)
        self.assertTrue(self.controller.active)


class LatestReport(unittest.TestCase):
    def test_uses_capture_time_not_arrival_order_or_emission_time(self):
        latest = {'kind': 'update', 'observed_at': 9.9, 'emitted_at': 10.0}
        reports = [latest, {'kind': 'sound', 'observed_at': 9.4, 'emitted_at': 12.0},
                   {'kind': 'warning', 'message': 'late blocks'},
                   {'kind': 'sound', 'observed_at': 8.0, 'emitted_at': 10.0}]
        self.assertIs(scope.newest_sound(reports, 10.0), latest)

    def test_final_wins_tie_without_extending_the_capture_lifetime(self):
        update = {'kind': 'update', 'observed_at': 9.0}
        final = {'kind': 'sound', 'observed_at': 9.0}
        self.assertIs(scope.newest_sound([final, update], 10.0), final)
        self.assertIsNone(scope.newest_sound([update, final], 10.3))

    def test_skips_missing_nonfinite_future_and_stale_times(self):
        reports = [{'kind': 'sound'}, {'kind': 'ready'},
                   {'kind': 'update', 'observed_at': float('nan')},
                   {'kind': 'update', 'observed_at': float('inf')},
                   {'kind': 'sound', 'observed_at': 11.0},
                   {'kind': 'sound', 'observed_at': 2.0}]
        self.assertIsNone(scope.newest_sound(reports, 10))


class _Writer:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class ElapsedRecording(unittest.TestCase):
    def test_resizing_window_letterboxes_recording_instead_of_changing_writer_shape(self):
        writer = _Writer()
        recorder = scope.TimedRecorder('unused.mp4', fps=1, factory=Mock(return_value=writer))
        recorder.write(np.zeros((12, 16, 3), np.uint8), 0)
        square = np.full((20, 20, 3), 255, np.uint8)
        recorder.write(square, 1)
        recorder.write(square, 2)
        result = writer.frames[-1]
        self.assertEqual(result.shape, (12, 16, 3))
        self.assertTrue(np.all(result[:, :2] == 0))
        self.assertTrue(np.all(result[:, 2:14] == 255))
        self.assertTrue(np.all(result[:, 14:] == 0))

    def test_variable_render_cadence_produces_real_elapsed_duration(self):
        writer = _Writer()
        factory = Mock(return_value=writer)
        recorder = scope.TimedRecorder('unused.mp4', fps=4, factory=factory)
        frame = np.zeros((12, 16, 3), np.uint8)
        for index, now in enumerate((20.0, 20.125, 20.5, 21.25)):
            frame[:] = index
            recorder.write(frame, now)
        self.assertEqual(len(writer.frames), 6)  # samples at 0, .25, .5, .75, 1, 1.25 s
        self.assertEqual(factory.call_args.args[2:], (4, (16, 12)))
        self.assertEqual(writer.frames[0][0, 0, 0], 0)
        frame[:] = 99
        self.assertFalse(any(np.any(saved == 99) for saved in writer.frames))
        recorder.close()
        self.assertTrue(writer.released)

    def test_repeated_same_time_does_not_duplicate_frames(self):
        writer = _Writer()
        recorder = scope.TimedRecorder('unused.mp4', fps=15, factory=Mock(return_value=writer))
        frame = np.zeros((12, 16, 3), np.uint8)
        for _ in range(10):
            recorder.write(frame, 10.0)
        self.assertEqual(len(writer.frames), 1)

    def test_closed_or_unopened_writer_is_handled(self):
        recorder = scope.TimedRecorder('unused.mp4', factory=Mock(return_value=_Writer(False)))
        with self.assertRaisesRegex(RuntimeError, 'Could not open video writer'):
            recorder.write(np.zeros((12, 16, 3), np.uint8), 0)
        recorder.close()
        scope.TimedRecorder('never-started.mp4').close()


class LiveLoopSmoke(unittest.TestCase):
    def test_base_camera_one_sound_render_quit_and_sensor_cleanup(self):
        frame = np.full((24, 32, 3), 80, np.uint8)
        depth = np.full((24, 32), 1000, np.float32)
        k = {'fx': 30., 'fy': 30., 'cx': 16., 'cy': 12.}
        observed_at = time.monotonic()
        event = {'kind': 'sound', 'observed_at': observed_at, 'event_id': 1,
                 'fix': {'direction': np.array([1., 0., 0.]), 'z': 20., 'coherence': .3}}
        view = {'time': observed_at, 'final': True, 'fix': event['fix'],
                'point': np.array([0., 0., 1.]), 'measured': True,
                'quality': {'state': 'candidate', 'reason': 'Synthetic region.'}}
        reports = queue.Queue()
        reports.put({'kind': 'ready'})
        reports.put(event)
        reports.close = Mock()
        stop = Mock()
        process = Mock(pid=123)
        process.is_alive.side_effect = [True, False]
        camera = Mock()
        camera.read.return_value = (frame, depth, k)
        args = SimpleNamespace(target=None, no_heatmap=False, diagnostics=False, record=None,
                               seconds=None, reach=False)
        cfg = {'camera': {'mount': 'base'}}
        with ExitStack() as patches:
            patches.enter_context(patch('multiprocessing.Queue', return_value=reports))
            patches.enter_context(patch('multiprocessing.Event', return_value=stop))
            patches.enter_context(patch('multiprocessing.Process', return_value=process))
            sensor = patches.enter_context(patch('camera.Gemini', return_value=camera))
            patches.enter_context(patch.object(scope, 'Localizer', return_value=Mock()))
            make_view = patches.enter_context(patch.object(scope, 'make_view', return_value=view))
            meta = patches.enter_context(patch.object(scope, 'Meta'))
            draw = patches.enter_context(patch.object(scope, 'draw', wraps=scope.draw))
            for name in ('namedWindow', 'resizeWindow', 'setMouseCallback', 'imshow'):
                patches.enter_context(patch.object(scope.cv2, name))
            destroyed = patches.enter_context(patch.object(scope.cv2, 'destroyAllWindows'))
            patches.enter_context(patch.object(scope.cv2, 'waitKey', return_value=ord('q')))
            patches.enter_context(patch.object(scope.cv2, 'waitKeyEx', return_value=ord('q')))
            patches.enter_context(patch.object(scope.cv2, 'getWindowProperty', return_value=1))
            patches.enter_context(patch('builtins.print'))
            scope.run(args, cfg)
        process.start.assert_called_once()
        sensor.assert_called_once()
        make_view.assert_called_once()
        self.assertIs(make_view.call_args.args[0], event)
        draw.assert_called_once()
        self.assertGreaterEqual(draw.call_args.kwargs['now'], observed_at)
        self.assertEqual(draw.call_args.kwargs['depth_coverage'], 1.0)
        meta.assert_not_called()
        camera.close.assert_called_once()
        stop.set.assert_called_once()
        process.join.assert_called_once_with(timeout=3)
        process.terminate.assert_not_called()
        reports.close.assert_called_once()
        destroyed.assert_called_once()


if __name__ == '__main__':
    unittest.main()
