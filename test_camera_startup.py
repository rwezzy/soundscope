"""Mocked Gemini startup and timeout tests: never import/open the device SDK."""
import unittest
from types import SimpleNamespace
from unittest import mock

import camera


def profile(width, height, fmt, fps=30):
    return SimpleNamespace(get_width=lambda: width, get_height=lambda: height,
                           get_format=lambda: fmt, get_fps=lambda: fps)


class Config:
    def __init__(self):
        self.streams = []

    def enable_stream(self, stream):
        self.streams.append(stream)

    def set_frame_aggregate_output_mode(self, value):
        self.aggregate = value


class Startup(unittest.TestCase):
    def setUp(self):
        self.sdk = SimpleNamespace(Config=Config, OBFormat=SimpleNamespace(MJPG='MJPG', RGB='RGB', Y16='Y16'),
                                   OBAlignMode=SimpleNamespace(SW_MODE='SW'),
                                   OBSensorType=SimpleNamespace(DEPTH_SENSOR='depth'),
                                   OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE='full'))
        self.primary = profile(1280, 720, 'RGB')
        self.depth = profile(1280, 800, 'Y16')
        self.low_depth = profile(640, 480, 'Y16')
        self.colors = mock.Mock()
        self.colors.get_video_stream_profile.side_effect = lambda w, h, f, fps: profile(w, h, f, fps)
        self.depths = mock.Mock()
        self.depths.get_video_stream_profile.return_value = self.low_depth
        self.pipeline = mock.Mock()
        self.pipeline.get_d2c_depth_profile_list.return_value = self.depths

    def start(self):
        return camera._start_streams(self.sdk, self.pipeline, self.primary, self.depth, self.colors)

    def test_success_preserves_original_profile_without_retry(self):
        result = self.start()
        self.assertEqual(result[:4], (1280, 720, 'RGB', 30))
        config = self.pipeline.start.call_args.args[0]
        self.assertEqual(config.streams, [self.primary, self.depth])
        self.colors.get_video_stream_profile.assert_not_called()
        self.pipeline.stop.assert_not_called()

    def test_mft_failure_stops_then_uses_supported_lower_bandwidth_pair(self):
        self.pipeline.start.side_effect = [RuntimeError('HResult0xc00d3704 HardwareMFT'), None]
        result = self.start()
        self.assertEqual(result[:4], (640, 480, 'MJPG', 30))
        self.assertEqual(self.pipeline.start.call_count, 2)
        self.pipeline.stop.assert_called_once()
        self.colors.get_video_stream_profile.assert_called_once_with(640, 480, 'MJPG', 30)
        self.pipeline.get_d2c_depth_profile_list.assert_called_once()
        self.depths.get_video_stream_profile.assert_called_once_with(640, 480, 'Y16', 30)
        self.assertEqual(self.pipeline.start.call_args.args[0].streams[1], self.low_depth)
        calls = [call[0] for call in self.pipeline.method_calls]
        self.assertLess(calls.index('stop'), calls.index('get_d2c_depth_profile_list'))

    def test_all_failures_are_bounded_and_report_original_windows_error(self):
        self.pipeline.start.side_effect = RuntimeError('HardwareMFT lacks resources')
        with self.assertRaisesRegex(RuntimeError, 'HardwareMFT lacks resources'):
            self.start()
        self.assertEqual(self.pipeline.start.call_count, 3)
        self.assertEqual(self.pipeline.stop.call_count, 3)

    def test_unsupported_fallback_is_not_started(self):
        self.pipeline.start.side_effect = RuntimeError('initial failure')
        self.colors.get_video_stream_profile.side_effect = RuntimeError('profile unavailable')
        with self.assertRaisesRegex(RuntimeError, 'bounded retries'):
            self.start()
        self.pipeline.start.assert_called_once()

    def test_stop_failure_does_not_prevent_bounded_retry(self):
        self.pipeline.start.side_effect = [RuntimeError('first failure'), None]
        self.pipeline.stop.side_effect = RuntimeError('already stopped')
        self.assertEqual(self.start()[:3], (640, 480, 'MJPG'))

    def test_missing_frames_timeout_closes_and_never_loops_forever(self):
        cam = camera.Gemini.__new__(camera.Gemini)
        cam.pipeline = mock.Mock()
        cam.pipeline.wait_for_frames.return_value = None
        cam._started, cam._last_valid_frame = True, 100.0
        with mock.patch.object(camera.time, 'monotonic', return_value=105.0):
            self.assertIsNone(cam.read())
        with mock.patch.object(camera.time, 'monotonic', return_value=113.0):
            with self.assertRaisesRegex(RuntimeError, 'no valid aligned'):
                cam.read()
        cam.close()
        cam.pipeline.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
