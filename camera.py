"""Orbbec Gemini 2: color frames plus depth aligned to them, through pyorbbecsdk2.

Same SDK calls as Meta_Hack's tools/gemini_capture.py. Plug the camera straight into a USB 3
port and close Orbbec Viewer first.
"""
import os
import subprocess
import time

import cv2
import numpy as np

ORBBEC_USB_VENDOR = 'VID_2BC5'  # Gemini 2 datasheet: VID/PID 0x2BC5/0x0670
FRAME_TIMEOUT_S = 12.0


def _stop_quietly(pipeline):
    try:
        pipeline.stop()
    except Exception:
        pass  # partially started/failed pipelines may report that they are already stopped


def _start_streams(sdk, pipeline, color, depth, colors):
    """Keep the original first choice; at most two supported lower-bandwidth retries."""
    failures, tried = [], set()
    options = [(None, None), (sdk.OBFormat.MJPG, 30), (sdk.OBFormat.RGB, 30)]
    for fmt, fps in options:
        try:
            if fmt is not None:
                # Never invent profiles: the device must advertise the requested mode.
                color = colors.get_video_stream_profile(640, 480, fmt, fps)
                try:
                    depths = pipeline.get_d2c_depth_profile_list(color, sdk.OBAlignMode.SW_MODE)
                except (AttributeError, RuntimeError):
                    # Older firmware may not expose the compatibility query; software AlignFilter
                    # still uses the actual stream profiles/extrinsics and verifies output size.
                    depths = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
                try:
                    depth = depths.get_video_stream_profile(640, 480, sdk.OBFormat.Y16, fps)
                except Exception:
                    depth = depths.get_video_stream_profile(0, 0, sdk.OBFormat.Y16, fps)
            key = (color.get_width(), color.get_height(), str(color.get_format()), color.get_fps(),
                   depth.get_width(), depth.get_height(), str(depth.get_format()), depth.get_fps())
            if key in tried:
                continue
            tried.add(key)
            config = sdk.Config()
            config.enable_stream(color)
            config.enable_stream(depth)
            config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            pipeline.start(config)
            return key
        except Exception as exc:
            failures.append(str(exc))
            _stop_quietly(pipeline)
    detail = failures[-1] if failures else 'no supported stream pair'
    # Keep the original Windows/MFT error too; fallback may merely be unsupported.
    first = failures[0] if failures else detail
    raise RuntimeError('Gemini 2 could not start color + depth after bounded retries. Close Windows Camera, '
                       'Orbbec Viewer and other camera apps; reconnect its USB cable directly to a USB 3 port. '
                       f'First error: {first}. Last error: {detail}')


def camera_connected():
    """True/False on Windows (is any Orbbec USB device present?); None when it cannot be checked.

    The Orbbec SDK crashes the whole process (heap corruption, no Python error) when it starts with no
    camera attached (seen Nov 4), so the viewer asks Windows first. SOUNDSCOPE_SKIP_CAMERA_CHECK=1 skips it.
    """
    if os.name != 'nt' or os.environ.get('SOUNDSCOPE_SKIP_CAMERA_CHECK') == '1':
        return None
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              f"@(Get-PnpDevice -PresentOnly | Where-Object {{ $_.InstanceId -match '{ORBBEC_USB_VENDOR}' }}).Count"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        return int(out.splitlines()[-1]) > 0 if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class Gemini:
    def __init__(self, size=(1280, 720)):
        if camera_connected() is False:
            raise SystemExit('No Orbbec camera is plugged in (Windows sees no Orbbec USB device). Plug the Gemini 2 '
                             'into a USB 3 port, or use the mics alone: python soundscope.py listen --assist')
        import pyorbbecsdk as sdk
        self.sdk = sdk
        self._started = False
        sdk.Context.set_logger_level(sdk.OBLogLevel.NONE)
        try:
            self.pipeline = sdk.Pipeline()
        except Exception as exc:
            raise SystemExit(f'No Gemini 2 found ({exc}). Plug it into a USB 3 port and close Orbbec Viewer.')
        colors = self.pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        color = None
        for width, height in (size, (640, 480), (0, 0)):
            for fmt in (sdk.OBFormat.RGB, sdk.OBFormat.MJPG, sdk.OBFormat.BGR, sdk.OBFormat.YUYV):
                try:
                    color = colors.get_video_stream_profile(width, height, fmt, 0)
                    break
                except Exception:
                    continue
            if color:
                break
        if color is None:
            raise SystemExit('The Gemini 2 offers no usable color stream.')
        depth = self.pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR) \
            .get_video_stream_profile(0, 0, sdk.OBFormat.Y16, 0)
        self.align = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
        self.intrinsics = None
        self.stream_profile = _start_streams(sdk, self.pipeline, color, depth, colors)
        self._started = True
        self._last_valid_frame = time.monotonic()

    def _missing_frame(self):
        if time.monotonic() - self._last_valid_frame > FRAME_TIMEOUT_S:
            self.close()
            raise RuntimeError('Gemini 2 produced no valid aligned color/depth frames for 12 seconds. '
                               'Close other camera apps and reconnect its cable to a direct USB 3 port.')
        return None

    def read(self, timeout_ms=1000):
        """(BGR image, depth in mm aligned to it as float32 with 0 = unknown, intrinsics) or None."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None or frames.get_color_frame() is None or frames.get_depth_frame() is None:
            return self._missing_frame()
        aligned = self.align.process(frames)
        depth = aligned.as_frame_set().get_depth_frame() if aligned is not None else None
        if depth is None:
            return self._missing_frame()
        color = frames.get_color_frame()
        image = self._bgr(color)
        mm = np.frombuffer(depth.get_data(), np.uint16).reshape(depth.get_height(), depth.get_width())
        mm = mm.astype(np.float32) * depth.get_depth_scale()
        if image is None or mm.shape != image.shape[:2]:
            return self._missing_frame()
        if self.intrinsics is None:
            k = color.get_stream_profile().as_video_stream_profile().get_intrinsic()
            self.intrinsics = {'fx': k.fx, 'fy': k.fy, 'cx': k.cx, 'cy': k.cy}
        self._last_valid_frame = time.monotonic()
        return image, mm, self.intrinsics

    def _bgr(self, frame):
        sdk, fmt = self.sdk, frame.get_format()
        data = np.frombuffer(frame.get_data(), np.uint8)
        h, w = frame.get_height(), frame.get_width()
        if fmt == sdk.OBFormat.RGB:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == sdk.OBFormat.BGR:
            return data.reshape(h, w, 3).copy()
        if fmt == sdk.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == sdk.OBFormat.YUYV:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
        return None

    def close(self):
        if self._started:
            self._started = False
            _stop_quietly(self.pipeline)


def backproject(u, v, depth_m, k):
    """Pixel(s) + depth (m) -> camera-frame points (m): x right, y down, z out of the lens."""
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(depth_m, float))
    return np.stack([(u - k['cx']) / k['fx'] * z, (v - k['cy']) / k['fy'] * z, z], axis=-1)


def project(points, k):
    """Camera-frame points (m) -> pixel (u, v); None where behind the camera."""
    p = np.asarray(points, float)
    if p[..., 2] <= 0:
        return None
    return k['fx'] * p[..., 0] / p[..., 2] + k['cx'], k['fy'] * p[..., 1] / p[..., 2] + k['cy']
