"""SoundScope live view: the Gemini 2 picture with the sound's location drawn on it.

  python scope.py                            live view; make a sound
  python scope.py --target "keys jingling"   also ask Muse Spark what the sound is
  python scope.py --record demo.mp4          also save the annotated video

Keys in the window: c = calibrate, s = snapshot, q = quit.
Calibrate: press c, play a sound, and click its source in the picture. The board's direction
(front_yaw_deg in arm_config.toml) is corrected from that.

The sound alone picks the spot: every 3D point the depth camera sees is scored by how well a
sound from there fits what the four mics heard (the heat map). The outline comes from depth only.
No image detection.
"""
import argparse
import math
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
from meta_api import ENV_FILE, Meta, MetaError, load_env
from mic_test import side, sureness

WINDOW = 'SoundScope'
GRID_STEP = 8        # score every 8th pixel
NO_DEPTH_M = 2.5     # assumed distance where the depth camera sees nothing
HOLD_S, FADE_S = 3.0, 1.0
SNAPSHOTS = Path(__file__).with_name('recordings') / 'snapshots'


def score_view(loc, gcc, depth_mm, k, cfg, step=GRID_STEP):
    """Heat map over the picture: (scores, pixel u, pixel v, camera points in m, has depth), on a grid."""
    h, w = depth_mm.shape
    v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = depth_mm[v, u] / 1000.0
    has_depth = z > 0.1
    points = backproject(u, v, np.where(has_depth, z, NO_DEPTH_M), k)
    in_array = aim.base_to_array(aim.camera_to_base(points.reshape(-1, 3) * 1000.0, cfg), cfg)
    return loc.srp_points(gcc, in_array).reshape(z.shape), u, v, points, has_depth


def best_spot(scores, has_depth, points):
    """Grid index of the sound's spot: the best-scoring measured point, with near-ties going to the
    nearer surface (anything farther along the same line is hidden behind it)."""
    norm = (scores - np.median(scores)) / max(float(scores.max() - np.median(scores)), 1e-9)
    pick = np.where(has_depth, norm - 0.05 * points[..., 2], -np.inf)
    if not np.isfinite(pick).any() or norm[has_depth].max() < 0.8:
        pick = norm  # the sound is somewhere the depth camera can't measure
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
    view = {'time': time.monotonic(), 'final': final, 'fix': fix}
    away = off_screen(fix['direction'], cfg, k, depth_mm.shape)
    if away:
        view['off_screen'] = away
        return view
    scores, us, vs, points, has_depth = score_view(loc, loc.gcc(*fix['evidence'][:2]), depth_mm, k, cfg)
    i, j = best_spot(scores, has_depth, points)
    view.update(scores=scores, spot=(int(us[i, j]), int(vs[i, j])), point=points[i, j],
                measured=bool(has_depth[i, j]))
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
    aim.CONFIG_FILE.write_text(re.sub(r'(?m)^(front_yaw_deg\s*=\s*)-?[\d.]+', rf'\g<1>{value:.1f}', text, count=1),
                               encoding='utf-8')


def check_sound(meta, wav, target, view):
    """Background thread: Muse Spark's verdict on the sound, written into the view."""
    try:
        verdict = meta.sound_matches(wav, target)
    except MetaError as exc:
        verdict = {'match': False, 'heard': f'check failed ({exc})', 'confidence': 0.0}
    view['rejected'] = not (verdict['match'] and verdict['confidence'] >= 0.5)
    view['verdict'] = verdict


def label(img, text, org, color=(255, 255, 255), scale=0.7):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 5, y - th - 7), (x + tw + 5, y + base + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def heat_overlay(img, scores, strength, gray=False):
    s = scores - np.median(scores)
    s = np.clip(s / max(float(s.max()), 1e-9), 0, 1) ** 2
    heat = np.clip(cv2.resize(s.astype(np.float32), (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_CUBIC), 0, 1)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if gray:
        colored = cv2.cvtColor(cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    alpha = (heat * 0.55 * strength)[..., None]
    img[:] = (img * (1 - alpha) + colored * alpha).astype(np.uint8)


def summary(view):
    """One line: where the sound is and how sure."""
    fix = view['fix']
    sure = f"{sureness(fix['z'])} (z {fix['z']:.0f})"
    if 'off_screen' in view:
        where, deg = view['off_screen']
        return f'sound off camera, {deg:.0f} deg {where}   {sure}'
    x, y, z = view['point']
    dist = f'{math.sqrt(x * x + y * y + z * z):.2f} m away' if view['measured'] else 'distance unknown'
    return f"{dist}, {side(math.degrees(math.atan2(x, z))).replace('°', ' deg')}   {sure}"


def draw(frame, view, target, note):
    out = frame.copy()
    h, w = out.shape[:2]
    label(out, note, (12, h - 16), (200, 255, 200) if note.startswith('Calibrat') else (255, 255, 255), 0.6)
    if view is None:
        return out
    age = time.monotonic() - view['time']
    if view['final'] and age > HOLD_S + FADE_S:
        return out
    fade = 1.0 if not view['final'] or age <= HOLD_S else 1 - (age - HOLD_S) / FADE_S
    if 'off_screen' in view:
        where, _ = view['off_screen']
        tip = {'right': (w - 30, h // 2), 'left': (30, h // 2), 'above': (w // 2, 30), 'below': (w // 2, h - 50)}[where]
        tail = (tip[0] + (w // 2 - tip[0]) // 3, tip[1] + (h // 2 - tip[1]) // 3)
        cv2.arrowedLine(out, tail, tip, (0, 200, 255), 8, tipLength=0.4)
        label(out, summary(view), (12, 32), (0, 200, 255))
        return out
    rejected = view.get('rejected', False)
    heat_overlay(out, view['scores'], fade * (0.4 if rejected else 1.0), gray=rejected)
    color = (130, 130, 130) if rejected else (0, 255, 0)
    if view.get('contour') is not None:
        cv2.drawContours(out, [view['contour']], -1, color, 3)
    u, v = view['spot']
    cv2.circle(out, (u, v), 18, color, 3)
    cv2.line(out, (u - 30, v), (u + 30, v), color, 2)
    cv2.line(out, (u, v - 30), (u, v + 30), color, 2)
    label(out, summary(view), (12, 32))
    if target:
        verdict = view.get('verdict')
        if verdict is None:
            label(out, f'Muse Spark: checking for "{target}"...', (12, 66), (0, 220, 255))
        else:
            yes = not rejected
            label(out, f"Muse Spark: {verdict['heard']} - {'YES' if yes else 'not'} {target}", (12, 66),
                  (0, 255, 0) if yes else (160, 160, 160))
    if 'reach' in view:
        ok, radius = view['reach']
        label(out, f"{'within' if ok else 'out of'} the arm's reach ({radius / 1000:.2f} m from its base)",
              (12, 100 if target else 66), (0, 255, 0) if ok else (200, 200, 200), 0.6)
    return out


def listen(events):
    """Mic listener, run in its own process: the camera SDK holds Python's lock while it waits for
    frames, which froze a listener thread in the same process completely (measured: 0 frames/s)."""
    from arm_follow import Listener
    listener = Listener(events.put, updates=True)
    listener.start()
    listener.ready.wait()
    events.put({'kind': 'error', 'message': str(listener.error)} if listener.error else {'kind': 'ready'})
    listener.join()


def pending(events):
    """Sound reports waiting in the queue from the listener process."""
    while True:
        try:
            yield events.get_nowait()
        except queue.Empty:
            return


def run(args, cfg):
    import multiprocessing
    from camera import Gemini

    meta = None
    if args.target:
        try:
            meta = Meta()
        except MetaError as exc:
            raise SystemExit(f'Meta: {exc}')
    events = multiprocessing.Queue()
    multiprocessing.Process(target=listen, args=(events,), daemon=True).start()
    try:
        first = events.get(timeout=30)
    except queue.Empty:
        raise SystemExit('The microphone listener did not start within 30 s.')
    if first['kind'] == 'error':
        raise SystemExit(f"Microphone problem: {first['message']}")
    cam, loc = Gemini(), Localizer()
    ui = {'note': 'Listening. c = calibrate, s = snapshot, q = quit', 'last': None}  # last = (time, fix)
    cal = {'active': False, 'click': None}

    def on_mouse(event, x, y, flags, param):
        if cal['active'] and event == cv2.EVENT_LBUTTONDOWN:
            cal['click'] = (x, y, time.monotonic())
            say(ui, f'Calibrating: got your click at ({x}, {y}), listening for the sound...')

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    say(ui, ui['note'])
    writer, view, started = None, None, time.monotonic()
    try:
        while not (args.seconds and time.monotonic() - started > args.seconds):
            got = cam.read()
            if got is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            frame, depth, k = got
            for ev in pending(events):
                final = ev['kind'] == 'sound'
                view = make_view(ev, final, loc, depth, k, cfg)
                ui['last'] = (time.monotonic(), ev['fix'])  # a sound still playing counts too
                if final or time.monotonic() - ui.get('printed', 0) > 1.0:
                    print(f"{time.strftime('%H:%M:%S')} {'heard' if final else 'hearing'}: {summary(view)}", flush=True)
                    ui['printed'] = time.monotonic()
                if final and meta and ev.get('wav'):
                    view['verdict'] = None
                    threading.Thread(target=check_sound, args=(meta, ev['wav'], args.target, view), daemon=True).start()
            if cal['active'] and cal['click']:
                calibrate_step(cal, ui, cfg, depth, k)
            out = draw(frame, view, args.target, ui['note'])
            if cal['active'] and cal['click']:
                cv2.drawMarker(out, cal['click'][:2], (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
            cv2.imshow(WINDOW, out)
            if args.record:
                writer = writer or cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                                   (out.shape[1], out.shape[0]))
                writer.write(out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('c'):
                cal.update(active=True, click=None)
                say(ui, 'Calibrating: play the sound from your phone in view, then click on the phone')
            if key == ord('s'):
                SNAPSHOTS.mkdir(parents=True, exist_ok=True)
                path = SNAPSHOTS / time.strftime('%Y%m%d-%H%M%S.png')
                cv2.imwrite(str(path), out)
                say(ui, f'Saved {path.name}')
    finally:
        cam.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', help='also ask Muse Spark whether the sound is this')
    p.add_argument('--record', help='save the annotated video to this .mp4 file')
    p.add_argument('--seconds', type=float, help='stop after this long')
    p.add_argument('--env-file', help='also load keys from this .env file')
    args = p.parse_args()
    load_env(ENV_FILE)
    if args.env_file:
        load_env(args.env_file)
    try:
        run(args, aim.load_config())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
