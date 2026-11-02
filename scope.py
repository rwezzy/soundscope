"""SoundScope: acoustic source candidates over the Gemini 2 image.

  python scope.py                            live view; make a sound
  python scope.py --target "keys jingling"   also ask Muse Spark what the sound is
  python scope.py --record demo.mp4          also save the annotated video

Keys: C = three-position calibration, Enter = apply, Esc = cancel,
H = heat map, D = diagnostics, Space = pause, S = evidence snapshot, Q = quit.

Visible 3D points are scored against the four microphones. The support region is a
candidate, not a confirmed source. A depth contour is available in diagnostics.
Default mode makes no Meta requests and does not connect to the robot.
"""
import argparse
import math
import json
import os
import queue
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import aim
from camera import backproject, project
from doa import Localizer
from calibration import CalibrationSession, wrap_deg
from localization_quality import assess
import scope_ui
from meta_api import ENV_FILE, Meta, MetaError, load_env

WINDOW = 'SoundScope'
GRID_STEP = 8        # score every 8th pixel
NO_DEPTH_M = 2.5     # assumed distance where the depth camera sees nothing
MAX_AGE_S = 1.2     # from capture, not from when a queued report reaches the screen
SNAPSHOTS = Path(__file__).with_name('recordings') / 'snapshots'


def score_view(loc, gcc, depth_mm, k, cfg, step=GRID_STEP):
    """Heat map over the picture: (scores, pixel u, pixel v, camera points in m, has depth), on a grid."""
    h, w = depth_mm.shape
    v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = depth_mm[v, u] / 1000.0
    has_depth = np.isfinite(z) & (z > 0.1)
    points = backproject(u, v, np.where(has_depth, z, NO_DEPTH_M), k)
    in_array = aim.base_to_array(aim.camera_to_base(points.reshape(-1, 3) * 1000.0, cfg), cfg)
    return loc.srp_points(gcc, in_array).reshape(z.shape), u, v, points, has_depth


def best_spot(scores, has_depth, points):
    """Best acoustic score, with no arbitrary preference for a closer surface.

    The peak is only a candidate: depth at its pixel is not measured acoustic range.
    Missing-depth pixels use an assumed scoring distance and cannot supply a range.
    """
    pick = np.where(np.isfinite(scores), scores, -np.inf)
    return np.unravel_index(int(np.argmax(pick)), scores.shape)


def camera_direction(direction, cfg):
    """An array-frame direction in camera coordinates (x right, y down, z out of the lens)."""
    d = aim.array_axes(cfg) @ np.asarray(direction, float)
    c = cfg['camera']['base']
    return np.array([d @ axis for axis in aim.camera_axes(c['yaw_deg'], c['pitch_deg'])])


def off_screen(direction, cfg, k, shape):
    """None if the sound's line (from the board's real position) crosses the picture, else
    (which edge: 'left' | 'right' | 'above' | 'below', degrees off the camera's center)."""
    c = cfg['camera']['base']
    axes = aim.camera_axes(c['yaw_deg'], c['pitch_deg'])
    d = aim.array_axes(cfg) @ np.asarray(direction, float)
    start = np.asarray(cfg['array']['position_mm'], float) - np.asarray(c['position_mm'], float)
    h, w = shape
    for r in (300, 500, 800, 1200, 2000, 3500):  # mm along the line
        pixel = project(np.array([(start + r * d) @ a for a in axes]), k)
        if pixel is not None and 0 <= pixel[0] < w and 0 <= pixel[1] < h:
            return None
    x, y, z = camera_direction(direction, cfg)
    if abs(x) >= abs(y):
        return ('right' if x > 0 else 'left'), abs(math.degrees(math.atan2(x, z)))
    return ('below' if y > 0 else 'above'), abs(math.degrees(math.atan2(y, z)))


def nearest_valid(depth_mm, u, v, reach=20):
    """Closest pixel to (u, v) that has depth, or (None, None)."""
    h, w = depth_mm.shape
    y0, x0 = max(0, v - reach), max(0, u - reach)
    ys, xs = np.nonzero(depth_mm[y0:min(h, v + reach + 1), x0:min(w, u + reach + 1)] > 100)
    if len(ys) == 0:
        return None, None
    i = int(np.argmin((ys + y0 - v) ** 2 + (xs + x0 - u) ** 2))
    return int(xs[i] + x0), int(ys[i] + y0)


def dominant_plane(points, rng, tries=60, tol=0.008):
    """(unit normal, offset) of the biggest flat surface among points (N, 3), if it covers >= 30%."""
    if len(points) < 50:
        return None
    sample = points[rng.choice(len(points), min(len(points), 3000), replace=False)]
    best, best_count = None, 0
    for _ in range(tries):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        count = np.count_nonzero(np.abs(sample @ n - n @ a) < tol)
        if count > best_count:
            best, best_count = (n, -n @ a), count
    return best if best_count >= 0.3 * len(sample) else None


def outline(depth_mm, k, u, v, radius_m=0.12, rng=None):
    """Contour (pixels) of the object at (u, v) from depth alone: the connected surface within
    radius_m, minus the table or floor it rests on. None if there is no depth there."""
    u, v = nearest_valid(depth_mm, u, v)
    if u is None:
        return None
    h, w = depth_mm.shape
    z0 = depth_mm[v, u] / 1000.0
    r = int(k['fx'] * 2 * radius_m / z0) + 4
    x0, x1, y0, y1 = max(0, u - r), min(w, u + r + 1), max(0, v - r), min(h, v + r + 1)
    vv, uu = np.mgrid[y0:y1, x0:x1]
    z = depth_mm[y0:y1, x0:x1] / 1000.0
    pts, p0 = backproject(uu, vv, z, k), backproject(u, v, z0, k)
    valid = z > 0.1
    dist = np.linalg.norm(pts - p0, axis=-1)
    inside = valid & (dist < radius_m)
    plane = dominant_plane(pts[valid], rng or np.random.default_rng(0))
    if plane is not None:
        n, d = plane
        if abs(p0 @ n + d) > 0.015:          # the spot stands off the surface: cut the surface away
            inside &= np.abs(pts @ n + d) > 0.012
        else:                                # the spot is on it (a phone lying flat): small disk
            inside &= dist < radius_m / 2
    count, labels = cv2.connectedComponents(inside.astype(np.uint8))
    blob = labels == labels[v - y0, u - x0]
    if labels[v - y0, u - x0] == 0 or blob.sum() < 20:
        return None
    contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) + np.array([x0, y0]) if contours else None


def make_view(ev, final, loc, depth_mm, k, cfg):
    """Everything to draw for one sound estimate."""
    fix = ev['fix']
    view = {'time': ev.get('observed_at', time.monotonic()), 'final': final, 'fix': fix,
            'event_id': ev.get('event_id'), 'expires_after': MAX_AGE_S}
    away = off_screen(fix['direction'], cfg, k, depth_mm.shape)
    if away:
        view['off_screen'] = away
        strong = fix.get('z', 0) >= 12 and fix.get('coherence', 0) >= 0.10
        view['quality'] = {'state': 'candidate' if strong else 'weak',
                           'reason': 'Direction outside the camera view; object unverified.',
                           'peak_score': fix.get('coherence', 0), 'support_fraction': 0.0}
        return view
    scores, us, vs, points, has_depth = score_view(loc, loc.gcc(*fix['evidence'][:2]), depth_mm, k, cfg)
    i, j = best_spot(scores, has_depth, points)
    view.update(scores=scores, spot=(int(us[i, j]), int(vs[i, j])), point=points[i, j],
                measured=bool(has_depth[i, j]))
    view['quality'], view['support_mask'] = assess(scores, fix)
    if final:
        view['contour'] = outline(depth_mm, k, *view['spot'])
        view['reach'] = aim.reach_check(aim.camera_to_base(points[i, j] * 1000.0, cfg), cfg)
    return view


def calibrate_yaw(cfg, fix, click, depth_mm, k):
    """front_yaw_deg that makes the heard direction point at the clicked spot: (value, problem)."""
    u, v = nearest_valid(depth_mm, *click)
    if u is None:
        return None, 'no depth where you clicked; click on the object itself'
    patch = depth_mm[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
    p = aim.camera_to_base(backproject(u, v, np.median(patch[patch > 100]) / 1000.0, k) * 1000.0, cfg)
    to_click = p - np.asarray(cfg['array']['position_mm'], float)
    heard = aim.array_axes(cfg) @ fix['direction']
    delta = math.degrees(math.atan2(to_click[1], to_click[0]) - math.atan2(heard[1], heard[0]))
    return cfg['array']['front_yaw_deg'] + (delta + 180) % 360 - 180, None


def say(ui, text):
    """Show a status line in the window and print it in the terminal."""
    ui['note'] = text
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


def calibrate_step(cal, ui, cfg, depth_mm, k):
    """Finish calibrating once a sound was heard up to 3 s before the click, or any time after it."""
    clicked = cal['click'][2]
    if ui['last'] and ui['last'][0] > clicked - 3:
        value, problem = calibrate_yaw(cfg, ui['last'][1], cal['click'][:2], depth_mm, k)
        if problem:
            cal['click'] = None
            say(ui, f'Calibrate: {problem}')
        else:
            say(ui, f"Calibrated: front_yaw_deg {cfg['array']['front_yaw_deg']:.1f} -> {value:.1f} (saved)")
            cfg['array']['front_yaw_deg'] = value
            save_front_yaw(value)
            cal['active'] = False
    elif time.monotonic() - clicked > 10:
        cal['click'] = None
        say(ui, 'Calibrate: heard no sound. Play it louder or closer, then click on the phone again.')


def save_front_yaw(value):
    text = aim.CONFIG_FILE.read_text(encoding='utf-8')
    updated, count = re.subn(r'(?m)^(front_yaw_deg\s*=\s*)-?[\d.]+', rf'\g<1>{value:.3f}', text, count=1)
    if count != 1 or not math.isfinite(value):
        raise ValueError('Cannot save calibration: missing yaw setting or invalid value.')
    temporary = aim.CONFIG_FILE.with_suffix('.toml.tmp')
    temporary.write_text(updated, encoding='utf-8')
    os.replace(temporary, aim.CONFIG_FILE)


def check_sound(meta, wav, target, view):
    """Background thread: Muse Spark's verdict on the sound, written into the view."""
    try:
        verdict = meta.sound_matches(wav, target)
    except MetaError as exc:
        verdict = {'match': False, 'heard': f'check failed ({exc})', 'confidence': 0.0}
    view['rejected'] = not (verdict['match'] and verdict['confidence'] >= 0.5)
    view['verdict'] = verdict


def summary(view):
    quality = view['quality']['state']
    if 'off_screen' in view:
        edge, angle = view['off_screen']
        return f'{quality}: direction {angle:.0f} deg {edge}; source unverified'
    distance = f"{np.linalg.norm(view['point']):.2f} m" if view['measured'] else 'unknown'
    text = f"{quality}: candidate surface {distance}; {view['quality']['reason']}"
    if view.get('show_reach') and 'reach' in view:
        inside, radius = view['reach']
        text += f"; configured workspace: {'inside' if inside else 'outside'} ({radius:.0f} mm)"
    return text


def draw(frame, view, target, note, **kwargs):
    return scope_ui.render(frame, view, target, note, **kwargs)


def initial_viewport():
    """Start inside the screen; do not let HighGUI squash a tall native canvas."""
    width, height = 1280, 720
    if os.name == 'nt':
        try:
            import ctypes
            width = min(width, ctypes.windll.user32.GetSystemMetrics(0) - 100)
            height = min(height, ctypes.windll.user32.GetSystemMetrics(1) - 140)
        except (OSError, AttributeError):
            pass
    return max(640, width), max(400, height)


def window_viewport(previous):
    """Use the actual drawing area, not an assumed monitor resolution."""
    try:
        _, _, width, height = cv2.getWindowImageRect(WINDOW)
        if width > 0 and height > 0:
            size = (max(320, int(width)), max(240, int(height)))
            if size != (width, height):
                cv2.resizeWindow(WINDOW, *size)
            return size
    except cv2.error:
        pass
    return previous


class CalibrationController:
    """UI session: fresh audio + exact clicked depth, then explicit application.

    No automatic nearest-depth fallback: a hole on the phone must not calibrate
    using a wall behind it. Only one observation per sound event is admitted.
    """
    def __init__(self):
        self.active = False
        self.session = CalibrationSession()
        self.used_events = set()
        self.result = None
        self.initial_yaw = 0.0

    def start(self, cfg):
        self.__init__()
        self.active = True
        self.initial_yaw = cfg['array']['front_yaw_deg']
        return 'Calibration: sound at LEFT, CENTRE, RIGHT. Click each source while it plays; pause between positions.'

    def click(self, xy, ev, depth, k, cfg, now=None):
        now = time.monotonic() if now is None else now
        if ev is None or not 0 <= now - ev.get('observed_at', -float('inf')) <= 0.75:
            return 'No fresh sound: keep the source still and playing while you click it.'
        event_id = ev.get('event_id')
        if event_id is None or event_id in self.used_events:
            return 'Pause the sound for a second, move it, then play a new sound and click again.'
        u, v = xy
        h, w = depth.shape
        if not (0 <= u < w and 0 <= v < h) or not np.isfinite(depth[v, u]) or depth[v, u] <= 100:
            return 'No valid depth at that pixel. Click inside the source, not its edge.'
        p = aim.camera_to_base(backproject(u, v, float(depth[v, u]) / 1000, k) * 1000, cfg)
        relative = p - np.asarray(cfg['array']['position_mm'], float)
        heard = aim.array_axes(cfg) @ ev['fix']['direction']
        if np.linalg.norm(relative[:2]) < 1 or np.linalg.norm(heard[:2]) < 1e-6:
            return 'This position has no usable horizontal bearing; move the source.'
        expected_angle = math.degrees(math.atan2(relative[1], relative[0]))
        heard_angle = math.degrees(math.atan2(heard[1], heard[0]))
        self.result = self.session.add(expected_angle, heard_angle,
                                      timestamp=ev['observed_at'], z=ev['fix'].get('z', 0),
                                      coherence=ev['fix'].get('coherence', 0))
        if self.result.accepted:
            self.used_events.add(event_id)
        if self.result.ready:
            print(self.result.message, flush=True)
            return (f'Calibration ready: {self.result.count} positions, residual '
                    f'{self.result.residual_deg:.1f} deg. ENTER saves; ESC cancels.')
        return self.result.message

    def apply(self, cfg, save=None):
        if not self.active or self.result is None or not self.result.ready:
            return 'Calibration needs three separated, consistent source positions before saving.'
        value = wrap_deg(self.initial_yaw + self.result.correction_deg)
        (save or save_front_yaw)(value)  # save must succeed before changing live state
        cfg['array']['front_yaw_deg'] = value
        self.active = False
        return f'Yaw {value:+.1f} deg saved. Test a NEW position; pitch, roll and offsets are unchanged.'


def post_latest(events, event):
    """Never block capture on a slow renderer; bounded queue keeps newest reports."""
    try:
        events.put_nowait(event)
    except queue.Full:
        try:
            events.get_nowait()
        except queue.Empty:
            pass
        try:
            events.put_nowait(event)
        except queue.Full:
            pass


def listen(events, stop):
    """Keep audio in its own process; the camera SDK can hold Python's lock."""
    from arm_follow import Listener
    listener = Listener(lambda ev: post_latest(events, ev), updates=True)
    try:
        listener.start()
        if not listener.wait_ready(timeout=20):
            post_latest(events, {'kind': 'error', 'message': 'Microphone startup timed out.'})
            return
        if listener.error:
            post_latest(events, {'kind': 'error', 'message': str(listener.error)})
            return
        post_latest(events, {'kind': 'ready'})
        while listener.is_alive() and not stop.wait(0.1):
            pass
    except Exception as exc:
        post_latest(events, {'kind': 'error', 'message': str(exc)})
    finally:
        listener.stop()
        events.cancel_join_thread()


def pending(events):
    while True:
        try:
            yield events.get_nowait()
        except queue.Empty:
            return


def newest_sound(reports, now, max_age=MAX_AGE_S):
    """Skip diagnostics and stale audio; never redraw the entire queued backlog."""
    usable = [ev for ev in reports if ev.get('kind') in ('update', 'sound')
              and 0 <= now - ev.get('observed_at', -float('inf')) <= max_age]
    return max(usable, key=lambda ev: (ev['observed_at'], ev['kind'] == 'sound'), default=None)


class TimedRecorder:
    """Resample display frames to 15 fps using elapsed time, without speeding up video."""
    def __init__(self, path, fps=15, factory=cv2.VideoWriter):
        self.path, self.fps, self.factory = path, fps, factory
        self.writer, self.previous, self.started, self.written = None, None, None, 0
        self.size = None

    def write(self, frame, now):
        if self.writer is None:
            self.size = (frame.shape[1], frame.shape[0])
            self.writer = self.factory(self.path, cv2.VideoWriter_fourcc(*'mp4v'),
                                       self.fps, self.size)
            if not self.writer.isOpened():
                raise RuntimeError(f'Could not open video writer: {self.path}')
            self.started = now
            self.previous = frame.copy()
        due = int((now - self.started) * self.fps) + 1
        while self.written < due:
            self.writer.write(self.previous)
            self.written += 1
        if (frame.shape[1], frame.shape[0]) != self.size:
            width, height = self.size
            scale = min(width / frame.shape[1], height / frame.shape[0])
            fitted = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)),
                                        max(1, round(frame.shape[0] * scale))))
            canvas = np.zeros((height, width, 3), np.uint8)
            x, y = (width - fitted.shape[1]) // 2, (height - fitted.shape[0]) // 2
            canvas[y:y + fitted.shape[0], x:x + fitted.shape[1]] = fitted
            self.previous = canvas
        else:
            self.previous = frame.copy()

    def close(self):
        if self.writer is not None:
            self.writer.release()


def save_snapshot(out, frame, depth, k, cfg, ev, loc, *, frame_at=None):
    """Save screen plus aligned sensor evidence for offline diagnosis; no API call."""
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    stem = time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000:06d}'
    png = SNAPSHOTS / (stem + '.png')
    if not cv2.imwrite(str(png), out):
        raise OSError(f'Could not save {png}')
    if ev is not None:
        fix = ev['fix']
        np.savez_compressed(SNAPSHOTS / (stem + '.npz'), frame=frame, depth_mm=depth,
                            intrinsics=json.dumps(k), config=json.dumps(cfg),
                            gcc=loc.gcc(*fix['evidence'][:2]),
                            direction=fix['direction'], z=fix['z'], coherence=fix['coherence'],
                            observed_at=ev['observed_at'], snapshot_at=time.monotonic(),
                            frame_at=time.monotonic() if frame_at is None else frame_at)
    return png


def run(args, cfg):
    import multiprocessing
    from camera import Gemini

    if cfg['camera']['mount'] != 'base':
        raise SystemExit('This viewer needs a fixed camera/array. Moving-camera transforms are not implemented here.')
    meta = Meta() if args.target else None
    events, stop = multiprocessing.Queue(maxsize=32), multiprocessing.Event()
    process = multiprocessing.Process(target=listen, args=(events, stop), daemon=True)
    cam, recorder = None, None
    ui = {'note': 'Listening. Fixed camera + microphone. Robot control is off.', 'last': None}
    cal = CalibrationController()
    shape, click = None, None
    view, latest, meta_worker, evidence_frame = None, None, None, None
    heatmap, diagnostics, paused = not args.no_heatmap, args.diagnostics, False
    pause_time, held = None, None
    viewport = initial_viewport()
    displayed_viewport = None
    displayed = None

    def on_mouse(event, x, y, flags, param):
        nonlocal click
        if cal.active and not paused and shape is not None and event == cv2.EVENT_LBUTTONDOWN:
            xy = scope_ui.canvas_to_frame(x, y, shape, viewport_size=displayed_viewport)
            if xy is not None and displayed is not None:
                # Bind the click to the image/audio the person actually saw,
                # rather than replacing them with next iteration's sensor data.
                click = (xy, *displayed, time.monotonic())

    try:
        process.start()
        deadline = time.monotonic() + 25
        while True:
            try:
                first = events.get(timeout=max(.01, deadline - time.monotonic()))
            except queue.Empty:
                raise RuntimeError('Microphone did not start within 25 seconds.')
            if first['kind'] == 'error':
                raise RuntimeError('Microphone: ' + first['message'])
            if first['kind'] == 'ready':
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Microphone did not start within 25 seconds.')
        cam = Gemini()
        loc = Localizer()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(WINDOW, *viewport)
        cv2.setMouseCallback(WINDOW, on_mouse)
        say(ui, ui['note'])
        started = last_frame_at = time.monotonic()
        fps, last_timestamp = None, -float('inf')
        recorder = TimedRecorder(args.record) if args.record else None
        while not (args.seconds and time.monotonic() - started > args.seconds):
            got = cam.read()
            reports = list(pending(events))
            now = time.monotonic()  # includes callbacks posted while draining the queue
            for report in reports:
                if report['kind'] == 'error':
                    raise RuntimeError('Microphone: ' + report['message'])
                if report['kind'] == 'warning':
                    say(ui, report['message'])
            if not process.is_alive():
                raise RuntimeError('Microphone worker stopped; restart after checking its USB connection.')
            if got is None:
                if now - last_frame_at > 5:
                    raise RuntimeError('No camera frames for 5 seconds; check its USB connection.')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            interval = now - last_frame_at
            if interval > 0:
                instantaneous = 1 / interval
                fps = instantaneous if fps is None else .9 * fps + .1 * instantaneous
            last_frame_at = now
            if not paused:
                frame, depth, k = got
                held = got
                shape = frame.shape
                ev = newest_sound(reports, now)
                if ev is not None and ev['observed_at'] >= last_timestamp:
                    latest = ev
                    last_timestamp = ev['observed_at']
                    final = ev['kind'] == 'sound'
                    view = make_view(ev, final, loc, depth, k, cfg)
                    evidence_frame = (frame.copy(), depth.copy(), dict(k), now)
                    view['show_reach'] = args.reach
                    if final or now - ui.get('printed', 0) > 1:
                        print(summary(view), flush=True)
                        ui['printed'] = now
                    if (final and meta and ev.get('wav') and view['quality']['state'] != 'weak'
                            and time.monotonic() - ev['observed_at'] <= MAX_AGE_S
                            and (meta_worker is None or not meta_worker.is_alive())):
                        meta_worker = threading.Thread(target=check_sound,
                                                       args=(meta, ev['wav'], args.target, view), daemon=True)
                        meta_worker.start()
            else:
                frame, depth, k = held
            now = time.monotonic()  # include scoring work in freshness checks
            if cal.active and click is not None:
                xy, clicked_event, clicked_depth, clicked_k, clicked_at = click
                say(ui, cal.click(xy, clicked_event, clicked_depth, clicked_k, cfg, clicked_at))
                click = None
            if displayed_viewport is not None:
                viewport = window_viewport(viewport)
            out = draw(frame, view, args.target, ui['note'], now=pause_time if paused else now,
                       fps=fps, depth_coverage=float(np.mean(np.isfinite(depth) & (depth > 100))),
                       heatmap=heatmap, diagnostics=diagnostics, paused=paused, calibration_active=cal.active,
                       viewport_size=viewport)
            cv2.imshow(WINDOW, out)
            displayed_viewport = viewport
            displayed = (latest, depth, dict(k))
            if recorder:
                recorder.write(out, now)
            key = cv2.waitKey(1) & 0xFF
            if ord('A') <= key <= ord('Z'):
                key += ord('a') - ord('A')
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('c') and not paused:
                say(ui, cal.start(cfg))
            elif key == 27:
                cal.active, click = False, None
                say(ui, 'Calibration cancelled. Existing settings kept.')
            elif key in (10, 13) and cal.active and not paused:
                try:
                    say(ui, cal.apply(cfg))
                    view, latest, evidence_frame = None, None, None
                except (OSError, ValueError) as exc:
                    say(ui, f'Could not save calibration: {exc}')
            elif key == ord('h'):
                heatmap = not heatmap
            elif key == ord('d'):
                diagnostics = not diagnostics
            elif key == ord(' '):
                paused = not paused
                pause_time = now if paused else None
                if not paused:
                    view, latest, evidence_frame = None, None, None
            elif key == ord('s'):
                source, source_depth, source_k, frame_at = evidence_frame or (frame, depth, k, now)
                path = save_snapshot(out, source, source_depth, source_k, cfg, latest, loc, frame_at=frame_at)
                say(ui, f'Saved {path.name} and available sensor evidence.')
    finally:
        stop.set()
        if process.pid is not None:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for cleanup in (events.close, cam.close if cam is not None else None,
                        recorder.close if recorder is not None else None, cv2.destroyAllWindows):
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as exc:
                    print(f'Cleanup warning: {exc}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', help='also ask Muse Spark whether the sound is this')
    p.add_argument('--record', help='save the annotated video to this .mp4 file')
    p.add_argument('--seconds', type=float, help='stop after this long')
    p.add_argument('--diagnostics', action='store_true', help='show raw evidence metrics')
    p.add_argument('--no-heatmap', action='store_true', help='hide the acoustic overlay initially')
    p.add_argument('--reach', action='store_true', help="print a configured workspace check; requires camera-to-base calibration")
    p.add_argument('--env-file', help='also load keys from this .env file')
    args = p.parse_args()
    load_env(ENV_FILE)
    if args.env_file:
        load_env(args.env_file)
    try:
        run(args, aim.load_config())
    except KeyboardInterrupt:
        pass
    except (RuntimeError, MetaError, OSError) as exc:
        raise SystemExit(str(exc))


if __name__ == '__main__':
    main()
