"""SoundScope: acoustic source candidates over the Gemini 2 image, with a searchable memory of sounds.

  python scope.py                            live view; make a sound (tripod / desk mount)
  python scope.py --mount wrist              camera and mics clamped on the arm's gripper
  python scope.py --meta                     also let Meta models label sounds, caption speech,
                                             name what is near the spot and outline found targets
  python scope.py --assist --name Keith      sound awareness: alerts for alarms, doorbell, knocks,
                                             phone, baby, kettle, and when someone says your name
  python scope.py --target "keys jingling"   only highlight sounds Muse Spark hears as this
  python scope.py --record demo.mp4          also save the annotated video

Keys: V = voice command, L = last sounds, C = three-position calibration, Enter = apply,
Esc = cancel, H = heat map, M = map colours, D = diagnostics, Space = pause,
T = type a history question, S = evidence snapshot, Q = quit.
Voice: press V and say "find my keys", "follow the phone ringing", "stop", "show everything",
"when did the phone ring?", "who is talking?" or "what was that?".

Visible 3D points are scored against the four microphones. The support region is a
candidate, not a confirmed source. A depth contour is available in diagnostics.
Default mode makes no Meta requests (V asks only when pressed) and never moves the robot.
Every heard sound is saved to the local sound memory (recordings/memory, git-ignored);
`python soundscope.py search "..."` answers questions about it later. --no-memory turns that off.
"""
import argparse
import collections
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
import sound_memory
from insight import ASSIST_WATCH, FEATURES, Enricher
from meta_api import COMMAND_WORDS, ENV_FILE, Meta, MetaError, load_env

WINDOW = 'SoundScope'
GRID_STEP = 8        # score every 8th pixel
NO_DEPTH_M = 2.5     # assumed distance where the depth camera sees nothing
MAX_AGE_S = 1.2     # from capture, not from when a queued report reaches the screen
SNAPSHOTS = Path(__file__).with_name('recordings') / 'snapshots'
CHECK_S = 8.0        # how long a sound stays on screen while Muse Spark is still judging it
AFTER_VERDICT_S = 3.0
CAPTION_S = 6.0
YAW_FILE = None      # set from the mount profile, so calibration saves where it was loaded from


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


def export_target(writer, view, *, paused=False, calibrating=False, hunt=None, now=None):
    """Export only current acoustic/depth evidence, never a frozen display or a remembered sound.

    The optional consumer is a separate, explicitly armed program. This viewer never controls
    the arm. A selected surface is an acoustic candidate, not a verified sounding object.
    """
    if writer is None:
        return
    hunt = hunt or {}
    verdict = (view or {}).get('verdict') or {}
    confirmed = verdict.get('match') is True and isinstance(verdict.get('confidence'), (int, float)) \
        and math.isfinite(verdict['confidence']) and verdict['confidence'] >= 0.5
    reason = ('Viewer paused' if paused or hunt.get('mode') == 'paused' else
              'Calibration active; restart the consumer after saving' if calibrating else
              'Waiting for a new sound' if view is None else
              'Semantic target is not freshly confirmed' if
              hunt.get('target') and not confirmed else None)
    if reason:
        return writer.invalidate(reason)
    return writer.publish(view, now=now)


def edit_search_input(text, key):
    """Small nonblocking ASCII input editor for the OpenCV viewer."""
    if key == 27:
        return None, 'cancel'
    if key in (10, 13):
        return text.strip(), 'submit' if text.strip() else 'edit'
    if key in (8, 127):
        return text[:-1], 'edit'
    if 32 <= key <= 126 and len(text) < 240:
        return text + chr(key), 'edit'
    return text, 'edit'


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
    target = Path(YAW_FILE) if YAW_FILE else aim.CONFIG_FILE
    text = target.read_text(encoding='utf-8')
    updated, count = re.subn(r'(?m)^(front_yaw_deg\s*=\s*)-?[\d.]+', rf'\g<1>{value:.3f}', text, count=1)
    if count != 1 or not math.isfinite(value):
        raise ValueError('Cannot save calibration: missing yaw setting or invalid value.')
    temporary = target.with_suffix('.toml.tmp')
    temporary.write_text(updated, encoding='utf-8')
    os.replace(temporary, target)


def check_sound(meta, wav, target, view, hunt=None, ui=None, sheet=None):
    """Background thread: Muse Spark's verdict on the sound, written into the view. In find mode the
    first match is held on screen until the next command."""
    try:
        verdict = meta.sound_matches(wav, target, sheet) if sheet else meta.sound_matches(wav, target)
    except MetaError as exc:
        verdict = {'match': False, 'heard': f'check failed ({exc})', 'confidence': 0.0}
    match = bool(verdict['match']) and verdict['confidence'] >= 0.5
    age = max(0.0, time.monotonic() - view.get('time', time.monotonic()))
    view.update(rejected=not match, verdict=verdict, expires_after=max(MAX_AGE_S, age + AFTER_VERDICT_S))
    if match and hunt is not None and hunt['mode'] == 'find' and hunt['target'] == target and hunt['found'] is None:
        view['hold'] = True
        hunt['found'] = view
        if ui is not None:
            say(ui, f"Found it ({verdict['heard']}). Press V for a new command.")


def handle_voice(meta, wav, ui, hunt, ask=None):
    """Background thread: spoken command -> Muse Voice Transcribe -> Muse Spark -> what to do.
    ask(cmd, text) answers questions about past sounds (search / who / describe)."""
    if not wav:
        say(ui, "Didn't hear a command. Press V, then speak.")
        return
    try:
        try:
            text = meta.transcribe(wav, keywords=COMMAND_WORDS).strip()
        except TypeError:  # stand-ins without keyword support
            text = meta.transcribe(wav).strip()
        if not text:
            say(ui, "Didn't catch that. Press V and try again.")
            return
        cmd = meta.parse_command(text)
    except MetaError as exc:
        say(ui, f'Voice command failed: {exc}')
        return
    sound = (cmd.get('sound') or '').strip() or None
    action = cmd.get('action')
    if action in ('find', 'follow'):
        hunt.update(target=sound, mode=action, found=None)
        what = f'looking for "{sound}"' if sound else 'showing every sound'
    elif action == 'home':
        hunt.update(target=None, mode='follow', found=None)
        what = 'showing every sound'
    elif action == 'stop':
        hunt.update(target=None, mode='paused', found=None)
        what = 'paused'
    elif action in ('search', 'who', 'describe', 'watch'):
        if ask is None:
            what = 'no sound memory in this session'
        else:
            say(ui, f'You asked "{text}". Looking it up...')
            try:
                say(ui, ask(cmd, text))
            except Exception as exc:
                say(ui, f'Could not answer: {exc}')
            return
    else:
        what = 'not a SoundScope command'
    say(ui, f'You said "{text}": {what}')


def beep():
    """Short alert tone on Windows; silent elsewhere."""
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(1320, 180), daemon=True).start()
    except (ImportError, RuntimeError):
        pass


def hunt_status(hunt):
    if hunt['found'] is not None:
        return f"FOUND: {hunt['target']}"
    if hunt['mode'] == 'paused':
        return 'Paused (press V)'
    if hunt['target']:
        return f"{'Finding' if hunt['mode'] == 'find' else 'Following'}: {hunt['target']}"
    return 'Showing every sound'


def camera_angles(direction, cfg):
    """(azimuth, elevation) in degrees of an array-frame direction, seen from the camera:
    azimuth + = right of the lens axis, elevation + = up."""
    x, y, z = camera_direction(direction, cfg)
    return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(-y, math.hypot(x, z)))


def crop_jpeg(frame, pixel, size_frac=0.36, max_side=448):
    """Square crop of the camera picture centered on the picked pixel, with a thin crosshair."""
    h, w = frame.shape[:2]
    half = int(min(h, w) * size_frac / 2)
    u, v = int(pixel[0]), int(pixel[1])
    x0, y0 = max(0, min(w - 2 * half, u - half)), max(0, min(h - 2 * half, v - half))
    crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
    cu, cv_ = u - x0, v - y0
    cv2.line(crop, (cu - 12, cv_), (cu + 12, cv_), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(crop, (cu, cv_ - 12), (cu, cv_ + 12), (0, 255, 255), 1, cv2.LINE_AA)
    if crop.shape[0] > max_side:
        crop = cv2.resize(crop, (max_side, max_side), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', blur_faces(crop), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else None


_FACES = None


def blur_faces(image):
    """Best-effort: blur the faces OpenCV's bundled face finders see before a picture leaves this computer.
    They miss angled or dim faces (tested Nov 4), so --private is the real switch: it sends no pictures.
    This only hides people; it never decides where a sound came from."""
    global _FACES
    if os.environ.get('SOUNDSCOPE_NO_BLUR') == '1':
        return image
    if _FACES is None:
        _FACES = [cv2.CascadeClassifier(cv2.data.haarcascades + name)
                  for name in ('haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml')]
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    w_img = image.shape[1]
    side = max(24, min(image.shape[:2]) // 12)
    boxes = []
    for finder in _FACES:
        for flipped in (False, True):  # the profile finder only knows one side; a mirror covers the other
            found = finder.detectMultiScale(cv2.flip(gray, 1) if flipped else gray, 1.1, 3, minSize=(side, side))
            boxes += [(w_img - x - w, y, w, h) if flipped else (x, y, w, h) for x, y, w, h in found]
    out = image.copy()
    for x, y, w, h in boxes:
        pad = int(0.3 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(out.shape[1], x + w + pad), min(out.shape[0], y + h + pad)
        k = max(15, (max(w, h) // 3) | 1)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (k, k), 0)
    return out


def crop_box(shape, pixel, size_frac=0.36):
    h, w = shape[:2]
    half = int(min(h, w) * size_frac / 2)
    u, v = int(pixel[0]), int(pixel[1])
    return max(0, min(w - 2 * half, u - half)), max(0, min(h - 2 * half, v - half)), half


def is_dark(frame, pixel, limit=28.0):
    """True when the camera crop around the pixel is too dark to show anything (night)."""
    x0, y0, half = crop_box(frame.shape, pixel)
    crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()) < limit if crop.size else True


def depth_relief_jpeg(depth_mm, pixel, max_side=448):
    """Night mode: the depth camera's 3D surface around the pixel, shaded like a plaster relief, so a
    model can still see shapes (a mug, a phone, a hand) when the color image is black."""
    x0, y0, half = crop_box(depth_mm.shape, pixel)
    d = depth_mm[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(np.float32)
    valid = np.isfinite(d) & (d > 100)
    if valid.mean() < 0.2:
        return None
    d = np.where(valid, d, np.median(d[valid]))
    d = cv2.GaussianBlur(d, (5, 5), 0)
    gy, gx = np.gradient(d)
    scale = np.median(d) / 600.0                      # gradient per pixel shrinks with distance
    n = np.dstack([-gx, -gy, np.full_like(d, scale)])
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    light = np.array([-0.45, -0.55, 0.7])
    shade = np.clip(n @ (light / np.linalg.norm(light)), 0, 1)
    near = np.clip(1.4 - (d - d[valid].min()) / max(1.0, np.ptp(d[valid])), 0.4, 1.0)  # nearer = brighter
    img = (255 * shade * near).astype(np.uint8)
    img[~valid] = 0
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cu, cv_ = int(pixel[0]) - x0, int(pixel[1]) - y0
    cv2.line(out, (cu - 12, cv_), (cu + 12, cv_), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(out, (cu, cv_ - 12), (cu, cv_ + 12), (0, 255, 255), 1, cv2.LINE_AA)
    if out.shape[0] > max_side:
        out = cv2.resize(out, (max_side, max_side), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def frame_jpeg(frame, pixel, max_width=1280):
    """Whole picture for SAM, shrunk to max_width: (jpeg, (w, h), pixel in that picture, scale)."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
    ok, buf = cv2.imencode('.jpg', blur_faces(small), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return ((buf.tobytes() if ok else None), (small.shape[1], small.shape[0]),
            (int(pixel[0] * scale), int(pixel[1] * scale)), scale)


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


def draw(frame, view, target, note, status=None, **kwargs):
    return scope_ui.render(frame, view, target, note, status=status, **kwargs)


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


def listen(events, stop, requests=None, threshold_db=0.5):
    """Keep audio in its own process; the camera SDK can hold Python's lock. Also records a spoken
    command (from the board's speech channel) when the window asks through `requests`."""
    from arm_follow import Listener
    listener = Listener(lambda ev: post_latest(events, ev), updates=True, threshold_db=threshold_db)
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
            if requests is None:
                continue
            try:
                seconds = requests.get_nowait()
            except queue.Empty:
                continue
            wav = listener.record_command(seconds)
            try:
                events.put({'kind': 'voice', 'wav': wav}, timeout=1.0)
            except queue.Full:
                post_latest(events, {'kind': 'voice', 'wav': wav})
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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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


class Session:
    """Sound memory, Meta features and on-screen results for one viewer run.

    Everything here is optional: with memory off and no Meta features it does nothing, so the
    plain viewer keeps working exactly as before.
    """

    def __init__(self, args, cfg, ui):
        self.cfg, self.ui = cfg, ui
        self.mode = 'wrist' if (cfg.get('array') or {}).get('on_arm') else 'tripod'
        assist = getattr(args, 'assist', False)
        self.features = (set(FEATURES) if getattr(args, 'meta', False) or assist
                         else {f for f in FEATURES if getattr(args, f, False)})
        if getattr(args, 'private', False):  # no camera pictures leave this computer
            self.features -= {'context', 'outline'}
        self.names = [n.strip() for n in getattr(args, 'name', None) or [] if n.strip()]  # alert when said
        self.memory = sound_memory.Memory() if getattr(args, 'memory', False) else None
        self.references = sound_memory.ReferenceBank() if self.memory is not None or self.features else None
        self.speakers = sound_memory.SpeakerMap()
        self.results = collections.deque()  # (record id, kind, payload) from the worker thread
        self.captions = []                  # {'text', 'pixel', 'until', 'kind'}
        self.outline = None                 # (record id, polygons, until)
        self.outlines = []                  # (polygons, until): SAM outlines drawn on their own
        self.tracks = {}                    # event id -> [(capture time, azimuth, elevation)]
        self.event_records = {}             # event id -> record id
        self.frames = collections.deque(maxlen=24)  # (time, small frame, scale): ~5 s for "before" crops
        self.watch = []                     # sounds to alert on ("tell me when ...")
        self.alert = None                   # (text, until) banner
        self.meta, self.enricher = None, None
        if self.features:
            self.meta = Meta()
        if self.features or (self.references is not None and self.references.items):
            # Your recorded examples label sounds (and match watches) for free, with or without Meta.
            self.enricher = Enricher(self.memory, self.meta, self.features,
                                     on_result=lambda *r: self.results.append(r),
                                     references=self.references, speakers=self.speakers)
            self.enricher.listen_for_names = bool(self.names)
            self.enricher.start()
        for sound in (list(ASSIST_WATCH) if assist else []) + list(getattr(args, 'watch', None) or []):
            self.add_watch(sound)

    @property
    def active(self):
        return self.memory is not None or self.enricher is not None

    def get_meta(self):
        if self.meta is None:
            self.meta = Meta()
        return self.meta

    def status(self, hunt):
        text = hunt_status(hunt)
        if self.watch:
            text += '  |  Watching: ' + ', '.join(self.watch)
        if self.meta is not None:
            text += '  |  ' + self.meta.budget.short()
        return text

    def remember_frame(self, now, frame):
        """Keep a small copy every ~0.2 s, so a sound's spot can also be seen from just before it."""
        if self.enricher is None or 'context' not in self.features:
            return
        if self.frames and now - self.frames[-1][0] < 0.2:
            return
        scale = min(1.0, 640.0 / frame.shape[1])
        small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
        self.frames.append((now, small, scale))

    def frame_before(self, when):
        """(small frame, scale) captured closest to, but before, `when`; or (None, None)."""
        older = [f for f in self.frames if f[0] <= when]
        if not older:
            return None, None
        _, small, scale = older[-1]
        return small, scale

    def add_watch(self, sound):
        sound = sound.strip()
        if sound and sound.lower() not in (w.lower() for w in self.watch):
            self.watch = (self.watch + [sound])[-10:]
        if self.enricher is not None:
            self.enricher.watch = list(self.watch)
        return self.watch

    def track(self, report):
        if not self.active or 'fix' not in report:
            return
        try:
            az, el = camera_angles(report['fix']['direction'], self.cfg)
        except (KeyError, TypeError, ValueError):
            return
        events = self.tracks.setdefault(report.get('event_id'), [])
        events.append((report.get('observed_at', time.monotonic()), az, el))
        if len(self.tracks) > 64:  # forget the oldest sounds' tracks
            self.tracks.pop(next(iter(self.tracks)))

    def record(self, report, view, frame, outline=False, depth=None):
        """Save one finished sound and hand it to the Meta worker. Never raises into the viewer."""
        if not self.active or not sound_memory.worth_remembering(report.get('fix') or {}):
            return None
        try:
            fix = report['fix']
            direction = camera_direction(fix['direction'], self.cfg)
            wav = report.get('wav')
            profile = sound_memory.acoustic_profile(sound_memory.wav_samples(wav)) if wav else None
            if profile and report.get('level'):
                profile.pop('rms_dbfs', None)   # the WAV is normalized; the listener measured the raw level
                profile.pop('peak_dbfs', None)
            onset = report.get('onset_at') or report.get('observed_at')
            started = time.time() - (time.monotonic() - onset) if onset else None  # wall clock of the onset
            rec = sound_memory.make_record(fix, direction, profile=profile, view=view, level=report.get('level'),
                                           mode=self.mode, now=started)
            if self.memory is not None:
                self.memory.add(rec)
            self.event_records[report.get('event_id')] = rec['id']
            print(sound_memory.phrase(rec), flush=True)
            if self.enricher is not None:
                task = {'record': rec, 'wav': wav, 'speech_wav': report.get('speech_wav'),
                        'track': self.tracks.pop(report.get('event_id'), []),
                        'clip_end_at': report.get('clip_end_at', report.get('observed_at', 0.0)),
                        'clip_seconds': report.get('clip_seconds', 0.0)}
                pixel = rec['where']['pixel']
                if pixel is not None and frame is not None and 'context' in self.features:
                    relief = depth_relief_jpeg(depth, pixel) if depth is not None and is_dark(frame, pixel) else None
                    if relief is not None:  # night: shapes from the depth camera instead of a black crop
                        task.update(crop_jpeg=relief, crop_kind='depth')
                    else:
                        task['crop_jpeg'] = crop_jpeg(frame, pixel)
                        before, scale = self.frame_before(report.get('onset_at', report.get('observed_at', 0.0)) - 0.25)
                        if before is not None:  # same spot, just before the sound began: what changed there?
                            task['crop_before'] = crop_jpeg(before, (pixel[0] * scale, pixel[1] * scale))
                if pixel is not None and frame is not None and 'outline' in self.features and \
                        (outline or ('context' in self.features and 'crop_kind' not in task)):
                    # The whole picture rides along so SAM can outline the object when sound and sight
                    # agree (or right away for a found target). Not in night mode: nothing to outline.
                    jpeg, size, small_pixel, scale = frame_jpeg(frame, pixel)
                    task.update(outline=outline, frame_jpeg=jpeg, frame_size=size, pixel=small_pixel, scale=scale)
                self.enricher.submit(task)
            return rec
        except Exception as exc:
            print(f'Sound memory warning: {exc}', flush=True)
            return None

    def outline_found(self, view, frame):
        """Ask SAM 3.1 to outline a held (found) target at the pixel the sound picked."""
        rid = view.get('record_id')
        if self.enricher is None or 'outline' not in self.features or rid is None or frame is None:
            return
        if 'spot' not in view or 'off_screen' in view:
            return
        rec = self.memory.get(rid) if self.memory is not None else None
        jpeg, size, small_pixel, scale = frame_jpeg(frame, view['spot'])
        self.enricher.submit({'record': rec or {'id': rid, 'where': {}, 'acoustic': {}},
                              'outline': True, 'frame_jpeg': jpeg, 'frame_size': size,
                              'pixel': small_pixel, 'scale': scale})
        view['outline_requested'] = True

    def drain(self, now):
        """Apply the worker's results to the screen; returns notices to show."""
        notes = []
        while self.results:
            rid, kind, payload = self.results.popleft()
            rec = self.memory.get(rid) if self.memory is not None else None
            pixel = (rec or {}).get('where', {}).get('pixel')
            age = time.time() - rec['epoch'] if rec is not None else 0.0
            if age > (25.0 if kind == 'captions' else 15.0) and kind not in ('notice', 'alert'):
                pixel = 'late'  # still saved and printed, but no longer drawn on the live picture
            if kind == 'notice':
                notes.append(str(payload))
            elif kind == 'captions':
                for turn in payload:
                    text = f"{turn['name']} ({turn['where']}): {turn['text']}"
                    self.show(text, pixel, now + CAPTION_S, 'speech')
                    print('  ' + text, flush=True)
                    called = sound_memory.name_called(turn['text'], self.names)
                    if called:  # someone is calling you: say who and from where
                        self.raise_alert(rid, rec, f"{turn['name']} said \"{called}\" ({turn['where']})",
                                         'name called', pixel, now)
                        notes.append(f"ALERT: {turn['name']} said \"{called}\" ({turn['where']})")
            elif kind in ('identify', 'label') and rec is not None:
                self.show(sound_memory.what_it_was(rec), pixel, now + 4.0, 'label')
                print('  ' + sound_memory.phrase(rec), flush=True)
            elif kind == 'context' and rec is not None:
                print(f"  near the spot: {rec['near']['text']}", flush=True)
            elif kind == 'source' and rec is not None:
                src = rec.get('source') or {}
                mark = {'yes': 'sound + sight agree', 'no': 'sound + sight disagree', 'unsure': 'unconfirmed'}
                text = f"{sound_memory.what_it_was(rec)}: {src.get('object', '?')} ({mark.get(src.get('consistent'), '?')})"
                self.show(text, pixel, now + 6.0, 'label')
                print('  ' + sound_memory.phrase(rec), flush=True)
            elif kind == 'alert':
                where = (rec or {}).get('where', {}).get('words', '')
                text = f"ALERT: {payload['watch']}" + (f' ({where})' if where else '')
                self.raise_alert(rid, rec, text[len('ALERT: '):], payload['watch'], pixel, now)
                notes.append(f"{text}, matched by {payload['how']}")
            elif kind == 'outline':
                scale = payload.get('scale') or 1.0
                polygons = [(np.asarray(poly, float) / scale).astype(np.int32) for poly in payload['polygons']]
                self.outline = (rid, polygons, now + 8.0)
                if pixel != 'late' and self.mode == 'tripod':  # a still camera: the outline stays valid
                    self.outlines = (self.outlines + [(polygons, now + 8.0)])[-3:]
                print(f'  SAM 3.1 outlined the object at the spot ({len(polygons)} part(s))', flush=True)
        self.captions = [c for c in self.captions if c['until'] > now][-5:]
        self.outlines = [o for o in self.outlines if o[1] > now]
        if self.alert and self.alert[1] <= now:
            self.alert = None
        if self.outline and self.outline[2] <= now:
            self.outline = None
        return notes

    def raise_alert(self, rid, rec, text, tag, pixel, now):
        """Banner + caption at the sound + beep, and the alert saved on the sound's record."""
        self.alert = ('ALERT: ' + text, now + 8.0)
        self.captions.append({'text': 'ALERT: ' + text, 'pixel': pixel if pixel != 'late' else None,
                              'until': now + 8.0, 'kind': 'alert'})
        if self.memory is not None and rec is not None:
            self.memory.update(rid, {'alerts': sorted(set((rec.get('alerts') or []) + [tag]))})
        beep()

    def show(self, text, pixel, until, kind):
        """A caption at the sound's pixel, unless the result came in too late to be useful on screen."""
        if pixel != 'late':
            self.captions.append({'text': text, 'pixel': pixel, 'until': until, 'kind': kind})

    def ask(self, cmd, text):
        """Answer a spoken question about past sounds (search / who / describe), or set a watch."""
        action = cmd.get('action')
        if action == 'watch':
            sound = (cmd.get('sound') or '').strip()
            if not sound:
                return 'Watch for what? Say for example "tell me when the kettle whistles".'
            watching = self.add_watch(sound)
            how = ('Muse Spark checks each clear sound' if self.enricher is not None and
                   {'identify', 'context'} & self.features else 'only your recorded examples can match (start with --meta for Muse Spark)')
            return f'Watching for: {", ".join(watching)}. {how}.'
        if action == 'who':
            people = self.speakers.summary()
            if not people:
                return 'Nobody has talked since this session started (captions need --meta or --captions).'
            return 'Talkers: ' + '; '.join(f'{name} {where} (last {when})' for name, where, when in people)
        if self.memory is None:
            return 'The sound memory is off in this session.'
        if action == 'describe':
            last = self.memory.recent(1)
            return sound_memory.phrase(last[0]) if last else 'No sound heard yet.'
        question = (cmd.get('question') or '').strip() or text
        meta = self.meta if self.meta is not None else None
        result = sound_memory.search(question, self.memory, meta)
        now = time.monotonic()
        for rec in result['matches']:
            print('  - ' + sound_memory.phrase(rec), flush=True)
        if self.mode == 'tripod':  # the camera has not moved: show where those sounds happened
            for rec in result['matches'][:3]:
                pixel = (rec.get('where') or {}).get('pixel')
                if pixel:
                    when = sound_memory.ago(rec['epoch'], time.time())
                    self.captions.append({'text': f'{sound_memory.what_it_was(rec)} here, {when}', 'pixel': pixel,
                                          'until': now + 10.0, 'kind': 'recall'})
        return result['answer']

    def close(self):
        if self.enricher is not None:
            self.enricher.stop()


def run(args, cfg):
    import multiprocessing
    from camera import Gemini
    global YAW_FILE

    if cfg['camera']['mount'] != 'base' and not (cfg.get('array') or {}).get('on_arm'):
        raise SystemExit('This viewer needs the camera and mics fixed together: the tripod mount, or the '
                         'wrist mount with both on the gripper (--mount wrist).')
    profile = cfg.get('profile') or {}
    YAW_FILE = profile.get('path') if profile.get('has_yaw') else None
    ui = {'note': 'Listening. Fixed camera + microphone. Robot control is off.', 'last': None}
    target_writer = None
    if getattr(args, 'target_file', None):
        if cfg['camera']['mount'] != 'base' or cfg.get('array', {}).get('on_arm'):
            raise SystemExit('Depth target export currently requires a fixed camera and microphone.')
        from target_link import TargetWriter
        target_writer = TargetWriter(args.target_file, cfg)
        target_writer.invalidate('Viewer starting; no measured target yet')
        ui['note'] = 'Listening. Local depth-target export enabled; this viewer does not move the arm.'
    try:
        session = Session(args, cfg, ui)
    except MetaError as exc:
        raise SystemExit(f'Meta: {exc}')
    hunt = {'target': getattr(args, 'target', None), 'mode': 'follow', 'found': None}
    events, stop = multiprocessing.Queue(maxsize=32), multiprocessing.Event()
    requests = multiprocessing.Queue(maxsize=4)
    threshold = (cfg.get('listen') or {}).get('threshold_db', 0.5)
    process = multiprocessing.Process(target=listen, args=(events, stop, requests, threshold), daemon=True)
    cam, recorder = None, None
    cal = CalibrationController()
    shape, click = None, None
    view, latest, meta_worker, evidence_frame = None, None, None, None
    heatmap, diagnostics, paused = not args.no_heatmap, args.diagnostics, False
    heatmap_palette = getattr(args, 'palette', 'turbo')
    heatmap_opacity = getattr(args, 'heatmap_opacity', 0.65)
    pause_time, held = None, None
    viewport = initial_viewport()
    displayed_viewport = None
    displayed = None
    search_text, search_worker = None, None

    def ask_history(question):
        try:
            # Explicit user query: use the existing Meta session if enabled; otherwise free local search.
            answer = session.ask({'action': 'search', 'question': question}, question)
            say(ui, answer)
            ui['search_answer'] = (answer, time.monotonic() + 20)
        except Exception as exc:
            say(ui, f'History search unavailable: {exc}')

    def on_mouse(event, x, y, flags, param):
        nonlocal click
        if cal.active and not paused and shape is not None and event == cv2.EVENT_LBUTTONDOWN:
            xy = scope_ui.canvas_to_frame(x, y, shape, viewport_size=displayed_viewport)
            if xy is not None and displayed is not None:
                # Bind the click to the image/audio the person actually saw,
                # rather than replacing them with next iteration's sensor data.
                click = (xy, *displayed, time.monotonic())

    def start_check(report, target_view):
        nonlocal meta_worker
        try:
            meta = session.get_meta()
        except MetaError as exc:
            say(ui, f'Sound check unavailable: {exc}')
            return
        target_view['verdict'] = None
        target_view['expires_after'] = CHECK_S
        refs = session.references
        sheet = refs.sheet(report['wav']) if refs is not None and refs.items else None
        meta_worker = threading.Thread(target=check_sound, args=(meta, report['wav'], hunt['target'], target_view,
                                                                 hunt, ui, sheet), daemon=True)
        meta_worker.start()

    try:
        # The camera starts first: its projector and USB traffic can add noise the mics hear, and the
        # listener must learn the room's noise floor with that noise already present.
        cam = Gemini()
        warm = time.monotonic()
        while cam.read() is None and time.monotonic() - warm < 5:
            pass
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
        loc = Localizer()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(WINDOW, *viewport)
        cv2.setMouseCallback(WINDOW, on_mouse)
        say(ui, ui['note'])
        started = last_frame_at = time.monotonic()
        fps, last_timestamp = None, -float('inf')
        debug = os.environ.get('SOUNDSCOPE_DEBUG') == '1'
        stats = {'at': started, 'reports': 0, 'finals': 0, 'loops': 0}
        recorder = TimedRecorder(args.record) if args.record else None
        while not (args.seconds and time.monotonic() - started > args.seconds):
            got = cam.read()
            reports = list(pending(events))
            now = time.monotonic()  # includes callbacks posted while draining the queue
            if debug:
                stats['loops'] += 1
                stats['reports'] += len(reports)
                stats['finals'] += sum(1 for r in reports if r.get('kind') == 'sound')
                for r in reports:
                    if r.get('kind') == 'sound' and 'fix' in r:
                        f = r['fix']
                        print(f"[debug] finished sound: z {f.get('z', 0):.1f}, SNR {f.get('snr_db', 0):+.1f} dB, "
                              f"{f.get('seconds', 0):.2f} s, level {r.get('level')}", flush=True)
                if now - stats['at'] >= 5:
                    queued = session.enricher.tasks.qsize() if session.enricher is not None else '-'
                    print(f"[debug] {stats['loops'] / (now - stats['at']):.1f} loops/s, {stats['reports']} reports, "
                          f"{stats['finals']} finished sounds, Meta queue {queued}", flush=True)
                    stats.update(at=now, reports=0, finals=0, loops=0)
            for report in reports:
                if report['kind'] == 'error':
                    raise RuntimeError('Microphone: ' + report['message'])
                if report['kind'] == 'warning':
                    say(ui, report['message'])
                if report['kind'] == 'voice':
                    try:
                        meta = session.get_meta()
                    except MetaError as exc:
                        say(ui, f'Voice needs Meta: {exc}')
                        continue
                    threading.Thread(target=handle_voice, args=(meta, report['wav'], ui, hunt, session.ask),
                                     daemon=True).start()
                if report['kind'] in ('update', 'sound'):
                    session.track(report)
            for note in session.drain(now):
                say(ui, note)
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
                session.remember_frame(now, frame)
                ev = newest_sound(reports, now)
                live = hunt['mode'] != 'paused' and hunt['found'] is None
                if ev is not None and ev['observed_at'] >= last_timestamp and live:
                    latest = ev
                    last_timestamp = ev['observed_at']
                    final = ev['kind'] == 'sound'
                    view = make_view(ev, final, loc, depth, k, cfg)
                    evidence_frame = (frame.copy(), depth.copy(), dict(k), now)
                    view['show_reach'] = args.reach
                    if hunt['target']:
                        view['rejected'] = True  # gray until Muse Spark says it is the right sound
                    if final or now - ui.get('printed', 0) > 1:
                        print(summary(view), flush=True)
                        ui['printed'] = now
                    if (final and hunt['target'] and ev.get('wav') and view['quality']['state'] != 'weak'
                            and time.monotonic() - ev['observed_at'] <= MAX_AGE_S
                            and (meta_worker is None or not meta_worker.is_alive())):
                        start_check(ev, view)
                for report in reports:  # every finished sound goes to memory, shown or not
                    if report.get('kind') != 'sound' or 'fix' not in report or not session.active:
                        continue
                    shown = report is ev and view is not None and live
                    report_view = view if shown else make_view(report, True, loc, depth, k, cfg)
                    rec = session.record(report, report_view, frame, depth=depth)
                    if rec is not None and shown:
                        view['record_id'] = rec['id']
                if hunt['found'] is not None and not hunt['found'].get('outline_requested'):
                    session.outline_found(hunt['found'], evidence_frame[0] if evidence_frame else frame)
            else:
                frame, depth, k = held
            now = time.monotonic()  # include scoring work in freshness checks
            export_target(target_writer, view, paused=paused, calibrating=cal.active, hunt=hunt, now=now)
            if cal.active and click is not None:
                xy, clicked_event, clicked_depth, clicked_k, clicked_at = click
                say(ui, cal.click(xy, clicked_event, clicked_depth, clicked_k, cfg, clicked_at))
                click = None
            if displayed_viewport is not None:
                viewport = window_viewport(viewport)
            shown_view = hunt['found'] or view
            if session.outline is not None and shown_view is not None and \
                    session.outline[0] == shown_view.get('record_id'):
                shown_view = {**shown_view, 'sam_outline': session.outline[1]}
            out = draw(frame, shown_view, hunt['target'], ui['note'], session.status(hunt),
                       now=pause_time if paused else now,
                       fps=fps, depth_coverage=float(np.mean(np.isfinite(depth) & (depth > 100))),
                       heatmap=heatmap, diagnostics=diagnostics, paused=paused, calibration_active=cal.active,
                       heatmap_palette=heatmap_palette, heatmap_opacity=heatmap_opacity,
                       search_text=search_text,
                       search_answer=(ui['search_answer'][0] if ui.get('search_answer') and
                                      now < ui['search_answer'][1] else None),
                       viewport_size=viewport, captions=session.captions,
                       outlines=[poly for polys, _ in session.outlines for poly in polys],
                       alert=session.alert[0] if session.alert else None)
            cv2.imshow(WINDOW, out)
            displayed_viewport = viewport
            displayed = (latest, depth, dict(k))
            if recorder:
                recorder.write(out, now)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if search_text is not None:
                search_text, action = edit_search_input(search_text, key)
                if action == 'submit':
                    question, search_text = search_text, None
                    say(ui, f'Searching sound history: {question}')
                    search_worker = threading.Thread(target=ask_history, args=(question,), daemon=True)
                    search_worker.start()
                elif action == 'cancel':
                    say(ui, 'Search cancelled.')
                continue
            if ord('A') <= key <= ord('Z'):
                key += ord('a') - ord('A')
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('v'):
                try:
                    session.get_meta()
                    requests.put_nowait(4.0)
                    say(ui, 'Listening for your command (4 s), e.g. "find my keys" or "when did the phone ring?"')
                except MetaError as exc:
                    say(ui, f'Voice needs MODEL_API_KEY in meta-soundscope/.env: {exc}')
                except queue.Full:
                    say(ui, 'Still recording the last command.')
            elif key == ord('t'):
                if search_worker is not None and search_worker.is_alive():
                    say(ui, 'Finishing the previous history search.')
                else:
                    search_text = ''
                    say(ui, 'Type a question about recorded sounds; ENTER searches, ESC cancels.')
            elif key == ord('l'):
                last = session.memory.recent(5) if session.memory is not None else []
                for rec in last:
                    print('  ' + sound_memory.phrase(rec), flush=True)
                say(ui, sound_memory.phrase(last[-1]) if last else 'No sounds in memory yet.')
            elif key == ord('c') and not paused:
                say(ui, cal.start(cfg))
                if target_writer:
                    target_writer.invalidate('Calibration active')
            elif key == 27:
                cal.active, click = False, None
                say(ui, 'Calibration cancelled. Existing settings kept.')
            elif key in (10, 13) and cal.active and not paused:
                try:
                    say(ui, cal.apply(cfg))
                    view, latest, evidence_frame = None, None, None
                    if target_writer:
                        # Calibration changed the geometry: a running consumer must reload its profile.
                        from target_link import TargetWriter
                        target_writer = TargetWriter(args.target_file, cfg)
                        target_writer.invalidate('Calibration changed; restart arm_target.py')
                except (OSError, ValueError) as exc:
                    say(ui, f'Could not save calibration: {exc}')
            elif key == ord('h'):
                heatmap = not heatmap
            elif key == ord('m'):
                choices = tuple(scope_ui.HEATMAP_PALETTES)
                heatmap_palette = choices[(choices.index(heatmap_palette) + 1) % len(choices)]
                say(ui, f'Heatmap colour: {heatmap_palette}. M changes colour; H hides/shows the overlay.')
            elif key == ord('d'):
                diagnostics = not diagnostics
            elif key == ord(' '):
                paused = not paused
                pause_time = now if paused else None
                if paused and target_writer:
                    target_writer.invalidate('Viewer paused')
                if not paused:
                    view, latest, evidence_frame = None, None, None
            elif key == ord('s'):
                source, source_depth, source_k, frame_at = evidence_frame or (frame, depth, k, now)
                path = save_snapshot(out, source, source_depth, source_k, cfg, latest, loc, frame_at=frame_at)
                say(ui, f'Saved {path.name} and available sensor evidence.')
    finally:
        if target_writer:
            try:
                target_writer.invalidate('Viewer closed')
            except OSError as exc:
                print(f'Target cleanup warning: {exc}', flush=True)
        session.close()
        stop.set()
        if process.pid is not None:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        closers = [events.close] + ([requests.close] if requests is not events else [])
        for cleanup in (*closers, cam.close if cam is not None else None,
                        recorder.close if recorder is not None else None, cv2.destroyAllWindows):
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as exc:
                    print(f'Cleanup warning: {exc}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mount', choices=sorted(aim.profiles()), help='mount profile (default: SOUNDSCOPE_MOUNT or none)')
    p.add_argument('--target', help='only highlight sounds Muse Spark hears as this')
    p.add_argument('--assist', action='store_true',
                   help='sound-awareness mode: all Meta features + alerts for ' + ', '.join(ASSIST_WATCH))
    p.add_argument('--name', action='append', help='alert when someone says this name (captions), e.g. --name Keith')
    p.add_argument('--private', action='store_true',
                   help='send no camera pictures to Meta (labels from sound only, captions, alerts still work)')
    p.add_argument('--meta', action='store_true', help='all Meta features: ' + ', '.join(FEATURES))
    for feature, text in (('identify', 'Muse Spark labels each clear sound'),
                          ('captions', 'Muse Voice Transcribe captions speech, talkers named by direction'),
                          ('context', 'Muse Spark names what is near the spot the sound picked'),
                          ('outline', 'SAM 3.1 outlines a found target at the spot the sound picked')):
        p.add_argument('--' + feature, action='store_true', help=text)
    p.add_argument('--watch', action='append', help='alert when this sound happens, e.g. --watch "phone ringing"')
    p.add_argument('--no-memory', dest='memory', action='store_false', help='do not save heard sounds')
    p.add_argument('--record', help='save the annotated video to this .mp4 file')
    p.add_argument('--seconds', type=float, help='stop after this long')
    p.add_argument('--diagnostics', action='store_true', help='show raw evidence metrics')
    p.add_argument('--no-heatmap', action='store_true', help='hide the acoustic overlay initially')
    p.add_argument('--palette', choices=tuple(scope_ui.HEATMAP_PALETTES), default='turbo',
                   help='heatmap colours (M cycles while running)')
    p.add_argument('--heatmap-opacity', type=float, default=0.65, help='heatmap strength, 0 to 0.85')
    p.add_argument('--reach', action='store_true', help="print a configured workspace check; requires camera-to-base calibration")
    p.add_argument('--env-file', help='also load keys from this .env file')
    p.add_argument('--target-file', help='export fresh depth candidates locally for arm_target.py; no motion here')
    args = p.parse_args()
    if not math.isfinite(args.heatmap_opacity) or not 0 <= args.heatmap_opacity <= .85:
        p.error('--heatmap-opacity must be between 0 and 0.85')
    load_env(ENV_FILE)
    if args.env_file:
        load_env(args.env_file)
    try:
        run(args, aim.load_config(profile=args.mount))
    except KeyboardInterrupt:
        pass
    except (RuntimeError, MetaError, OSError) as exc:
        raise SystemExit(str(exc))


if __name__ == '__main__':
    main()
