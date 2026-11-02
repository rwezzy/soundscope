"""Hardware-free OpenCV dashboard for SoundScope.

The camera is native resolution by default, or uniformly fitted when a viewport
is supplied. Mouse coordinates must pass through ``canvas_to_frame`` with the
same viewport; ``camera_rect`` describes the image bounds excluding letterbox.
Acoustic support is a region of agreement, never an object identification or a
confidence percentage.  Depth describes a candidate visible surface only.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import time

import cv2
import numpy as np


# All colors are BGR. A restrained palette keeps camera detail readable.
BG = (18, 15, 12)
PANEL = (29, 25, 21)
LINE = (57, 49, 41)
TEXT = (239, 233, 223)
MUTED = (154, 143, 127)
CYAN = (220, 207, 75)
AMBER = (91, 187, 244)
GRAY = (143, 137, 126)
DEFAULT_EXPIRY = 1.2


@dataclass(frozen=True)
class _Layout:
    scale: float
    main_width: int
    width: int
    height: int
    camera: tuple
    sidebar_x: int
    sidebar_width: int
    camera_area_height: int
    note_y: int
    note_height: int
    compact: bool


def _layout(frame_shape, viewport_size=None):
    h, w = (int(n) for n in frame_shape[:2])
    if h <= 0 or w <= 0:
        raise ValueError('frame must have positive width and height')
    if viewport_size is not None:
        width, height = (int(n) for n in viewport_size)
        if width < 320 or height < 240:
            raise ValueError('viewport_size must be at least 320 x 240 pixels')
        s = min(1.25, width / 1000.0, height / 754.0)
        px = lambda value: int(round(value * s))
        main_width = width - px(384)
        note_y = height - px(158)
        area_height = note_y - px(16) - px(128)
        factor = min(main_width / w, area_height / h)
        fitted_w, fitted_h = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
        camera = (px(24) + (main_width - fitted_w) // 2,
                  px(128) + (area_height - fitted_h) // 2, fitted_w, fitted_h)
        return _Layout(s, main_width, width, height, camera,
                       main_width + px(48), px(312), area_height, note_y, px(72), True)
    s = max(1.0, min(1.5, w / 1280.0))
    px = lambda value: int(round(value * s))
    main_width = max(w, px(520))
    return _Layout(s, main_width, main_width + px(384), max(px(754), h + px(348)),
                   (px(24) + (main_width - w) // 2, px(128), w, h),
                   main_width + px(48), px(312), h, px(128) + h + px(24), px(94), False)


def camera_rect(frame_shape, viewport_size=None):
    """Return displayed camera (x, y, width, height), excluding all letterboxing.

    Pass the same ``viewport_size=(width, height)`` used by ``render``. Without a
    viewport the camera is native resolution; with one it is uniformly fitted.
    """
    return _layout(frame_shape, viewport_size).camera


def canvas_to_frame(x, y, frame_shape, viewport_size=None):
    """Map a canvas point to an integer camera pixel, or None outside the image."""
    left, top, width, height = camera_rect(frame_shape, viewport_size)
    if left <= x < left + width and top <= y < top + height:
        return (min(frame_shape[1] - 1, int((x - left) * frame_shape[1] / width)),
                min(frame_shape[0] - 1, int((y - top) * frame_shape[0] / height)))
    return None


def camera_rgb_crop(canvas, frame_shape, viewport_size=None):
    """Copy the displayed camera viewport as RGB, useful for previews and QA."""
    x, y, w, h = camera_rect(frame_shape, viewport_size)
    return cv2.cvtColor(canvas[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)


def _number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError, OverflowError):
        return default


def _clean(value):
    # Hershey fonts do not support Unicode. Never render replacement glyphs.
    return str(value).replace('\n', ' ').encode('ascii', 'replace').decode('ascii')


def _text(img, text, x, y, scale=0.5, color=TEXT, thickness=1):
    cv2.putText(img, _clean(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _fit(text, width, scale):
    text = _clean(text)
    while text and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
        text = text[:-1]
    return text


def _wrap(text, width, scale, limit=3):
    words = _clean(text).split()
    lines, line = [], ''
    for word in words:
        proposed = f'{line} {word}'.strip()
        if line and cv2.getTextSize(proposed, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
            lines.append(line)
            line = word
        else:
            line = proposed
    if line:
        lines.append(line)
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = _fit(lines[-1], width - 18, scale) + '...'
    return [_fit(line, width, scale) for line in lines]


def _round_rect(img, rect, color, radius=10, thickness=-1):
    x, y, w, h = map(int, rect)
    r = max(1, min(int(radius), w // 2, h // 2))
    if thickness != -1:
        cv2.rectangle(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.rectangle(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
        for center, angle in (((x + r, y + r), 180), ((x + w - r, y + r), 270),
                              ((x + w - r, y + h - r), 0), ((x + r, y + h - r), 90)):
            cv2.ellipse(img, center, (r, r), 0, angle, angle + 90, color, thickness, cv2.LINE_AA)
        return
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _state(view, now):
    if not view:
        return 'LISTENING', 'Make a sound in the camera field.', None, DEFAULT_EXPIRY
    timestamp = _number(view.get('time'), now)
    age = max(0.0, now - timestamp)
    expiry = max(0.0, _number(view.get('expires_after'), DEFAULT_EXPIRY))
    if age > expiry:
        return 'STALE', 'Last estimate expired. Waiting for a fresh sound.', age, expiry
    quality = view.get('quality') or {}
    state = str(quality.get('state', '')).upper()
    if state not in ('CANDIDATE', 'AMBIGUOUS', 'WEAK'):
        state = 'AMBIGUOUS' if any(k in view for k in ('scores', 'spot', 'off_screen')) else 'WEAK'
    defaults = {
        'CANDIDATE': 'A region agrees with the acoustic evidence.',
        'AMBIGUOUS': 'Several positions fit. The source is unresolved.',
        'WEAK': 'Too little acoustic evidence to place the sound.',
    }
    if view.get('rejected'):
        return 'WEAK', 'The sound did not match the requested target.', age, expiry
    return state, quality.get('reason') or defaults[state], age, expiry


def _support(view, shape):
    """Relative score strength and support, without assuming calibrated confidence."""
    scores = view.get('scores')
    try:
        scores = np.asarray(scores, dtype=np.float32)
    except (ValueError, TypeError):
        return None, None
    if scores.ndim != 2 or not scores.size or not np.isfinite(scores).any():
        return None, None
    valid = np.isfinite(scores)
    floor, peak = float(np.min(scores[valid])), float(np.max(scores[valid]))
    spread = peak - floor
    relative = np.zeros(scores.shape, np.float32)
    nonflat = spread > max(1e-9, 1e-6 * max(abs(floor), abs(peak)))
    if nonflat:
        relative[valid] = np.clip((scores[valid] - floor) / spread, 0, 1)
    mask = view.get('support_mask')
    if mask is not None:
        mask = np.asarray(mask)
        mask = mask.astype(bool) & valid if mask.shape == scores.shape else None
    if mask is None:
        mask = (relative >= 0.8) & valid if nonflat else np.zeros(scores.shape, bool)
    h, w = shape[:2]
    support = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    strength = cv2.resize(relative, (w, h), interpolation=cv2.INTER_LINEAR)
    if not valid.all():
        strength *= cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return strength, support


@lru_cache(maxsize=2)
def _heat_tables(weak):
    """Premultiplied BGR heat and camera retention, computed at 256 levels.

    OpenCV applies these tiny tables with native byte operations. Computing the
    same color/alpha expressions on full-frame NumPy RGB arrays dominated the
    live loop; byte blending preserves the gradient within rounding precision.
    """
    level = np.linspace(0, 1, 256, dtype=np.float32)
    alpha = (0.40 if weak else 0.44) * level ** 1.2
    blend = np.clip((level - 0.25) / 0.75, 0, 1)[:, None]
    color = np.asarray(CYAN, np.float32) * (1 - blend) + np.asarray(AMBER, np.float32) * blend
    retention = np.rint((1 - alpha) * 255).astype(np.uint8)
    contribution = np.rint(color * alpha[:, None]).astype(np.uint8).reshape(256, 1, 3)
    return retention, contribution


def _brackets(img, bounds, color, scale):
    x, y, w, h = bounds
    pad = int(round(7 * scale))
    x0, x1 = max(2, x - pad), min(img.shape[1] - 3, x + w + pad)
    y0, y1 = max(2, y - pad), min(img.shape[0] - 3, y + h + pad)
    length = max(4, min(int(19 * scale), (x1 - x0) // 3, (y1 - y0) // 3))
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, 2, cv2.LINE_AA)


def _camera_overlay(image, view, state, heatmap, diagnostics, scale):
    if state in ('STALE', 'LISTENING'):
        return
    color = CYAN if state == 'CANDIDATE' else AMBER
    strength, support = _support(view, image.shape)
    # Display the actual graded score field even if quality gates reject a lock.
    # The support mask controls candidate contours, never visibility of evidence.
    # A zero/flat/invalid field has no heat; nonflat values are relative SRP, not
    # probability. Strongest heat retains at least 56% of the camera image.
    if heatmap and strength is not None and cv2.minMaxLoc(strength)[1] > 0:
        retention_table, heat_table = _heat_tables(state == 'WEAK')
        level = cv2.convertScaleAbs(strength, alpha=255)
        retention = cv2.cvtColor(cv2.LUT(level, retention_table), cv2.COLOR_GRAY2BGR)
        heat = cv2.applyColorMap(level, heat_table)
        cv2.multiply(image, retention, scale=1 / 255.0, dst=image)
        cv2.add(image, heat, dst=image)
    # Weak or semantically rejected estimates can show evidence, never a precise
    # point, direction arrow, support boundary, or depth-object outline.
    if state not in ('CANDIDATE', 'AMBIGUOUS'):
        return
    offscreen = view.get('off_screen')
    if offscreen:
        try:
            edge, _ = offscreen
            h, w = image.shape[:2]
            inset = min(int(28 * scale), w // 4, h // 4)
            tips = {'left': (inset, h // 2), 'right': (w - inset - 1, h // 2),
                    'above': (w // 2, inset), 'below': (w // 2, h - inset - 1)}
            tip = tips.get(edge)
            if tip:
                tail = (tip[0] + (w // 2 - tip[0]) // 4, tip[1] + (h // 2 - tip[1]) // 4)
                cv2.arrowedLine(image, tail, tip, color, max(1, int(2 * scale)), cv2.LINE_AA, tipLength=0.3)
        except (TypeError, ValueError):
            pass
        return
    if support is None or not np.any(support):
        return
    contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 4]
    lines = image.copy()
    cv2.drawContours(lines, contours, -1, color, max(1, int(scale)), cv2.LINE_AA)
    cv2.addWeighted(lines, 0.7, image, 0.3, 0, dst=image)
    # Brackets enclose the entire acoustic support, not a supposedly exact spot.
    # Broad/multiple regions retain their actual uncertainty contours.
    if state == 'CANDIDATE' and np.mean(support) < 0.35 and len(contours) == 1:
        _brackets(image, cv2.boundingRect(contours[0]), color, scale)
    if diagnostics and view.get('contour') is not None:
        try:
            contour = np.asarray(view['contour'], np.int32).reshape(-1, 1, 2)
            cv2.drawContours(image, [contour], -1, GRAY, 1, cv2.LINE_AA)
        except (ValueError, TypeError):
            pass


def _sparkline(img, view, rect, color):
    """A real horizontal profile of the relative score map, not decorative audio."""
    strength, _ = _support(view, (24, max(2, rect[2])))
    if strength is None:
        return
    x, y, w, h = rect
    profile = strength.max(axis=0)
    pts = np.column_stack((np.arange(len(profile)) + x, y + h - profile * h)).astype(np.int32)
    cv2.line(img, (x, y + h), (x + w, y + h), LINE, 1)
    cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)


def render(frame, view, target, note, *, now=None, fps=None, depth_coverage=None,
           heatmap=True, diagnostics=False, paused=False, calibration_active=False,
           viewport_size=None):
    """Return a BGR dashboard without modifying ``frame`` or ``view``.

    ``now`` and ``view['time']`` use the same monotonic clock. Both final and live
    estimates expire after ``expires_after`` seconds (default 1.2). A caller that
    pauses a snapshot may pass its frozen monotonic time. ``depth_coverage`` is a
    measured fraction in [0, 1]. No fields are interpreted as confidence.
    ``viewport_size=(width, height)`` produces exactly that canvas size and fits
    the camera without stretching. Use this same size for mouse mapping helpers.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('frame must be a uint8 BGR image with three channels')
    view = view or {}
    now = _number(now, time.monotonic())
    state, reason, age, expiry = _state(view, now)
    layout = _layout(frame.shape, viewport_size)
    s = layout.scale
    p = lambda value: int(round(value * s))
    out = np.empty((layout.height, layout.width, 3), np.uint8)
    cv2.rectangle(out, (0, 0), (layout.width - 1, layout.height - 1), BG, -1)
    color = CYAN if state == 'CANDIDATE' else AMBER if state in ('AMBIGUOUS', 'WEAK') else GRAY
    x, y, w, h = layout.camera
    sx, sw = layout.sidebar_x, layout.sidebar_width

    # Brand and live camera status.
    for i, size in enumerate((10, 22, 32, 19, 9)):
        xx = p(26 + i * 6)
        cv2.line(out, (xx, p(38 - size / 2)), (xx, p(38 + size / 2)), CYAN, p(2), cv2.LINE_AA)
    _text(out, 'SoundScope', p(70), p(43), 0.84 * s, TEXT, max(1, p(2)))
    _text(out, 'SPATIAL AUDIO / VISION', p(71), p(62), 0.32 * s, MUTED)
    badge = 'PAUSED' if paused else 'CAMERA LIVE'
    badge_color = AMBER if paused else CYAN
    badge_x = layout.width - p(184)
    _round_rect(out, (badge_x, p(24), p(160), p(32)), PANEL, p(9))
    cv2.circle(out, (badge_x + p(14), p(40)), p(3), badge_color, -1, cv2.LINE_AA)
    _text(out, badge, badge_x + p(26), p(45), 0.42 * s, badge_color)
    cv2.line(out, (p(24), p(78)), (layout.width - p(24), p(78)), LINE, 1)

    # The card includes letterboxing, while camera_rect names only actual pixels.
    _round_rect(out, (p(16), p(92), layout.main_width + p(16), layout.camera_area_height + p(44)), PANEL, p(12))
    _text(out, 'CAMERA FIELD', p(28), p(114), 0.38 * s, MUTED)
    mode = ('CALIBRATION' if calibration_active else 'HEATMAP OFF' if not heatmap
            else 'HEATMAP / WEAK EVIDENCE' if state == 'WEAK'
            else 'HEATMAP / RELATIVE EVIDENCE')
    mode_width = cv2.getTextSize(mode, cv2.FONT_HERSHEY_SIMPLEX, 0.34 * s, 1)[0][0]
    _text(out, mode, p(24) + layout.main_width - mode_width - p(6), p(114), 0.34 * s,
          AMBER if calibration_active else MUTED)
    camera = frame.copy()
    _camera_overlay(camera, view, state, heatmap, diagnostics, s)
    if (w, h) != (frame.shape[1], frame.shape[0]):
        interpolation = cv2.INTER_AREA if w < frame.shape[1] else cv2.INTER_LINEAR
        camera = cv2.resize(camera, (w, h), interpolation=interpolation)
    out[y:y + h, x:x + w] = camera
    cv2.rectangle(out, (x - 1, y - 1), (x + w, y + h), LINE, 1)

    # Notes are outside the picture. Long device messages cannot cover the scene.
    note_y = layout.note_y
    _round_rect(out, (p(16), note_y, layout.main_width + p(16), layout.note_height), PANEL, p(12))
    notes_heading = 'CALIBRATION' if calibration_active else 'FIELD NOTES'
    _text(out, notes_heading, p(30), note_y + p(22),
          0.34 * s, AMBER if calibration_active else MUTED)
    message = note or ('Play a sound, then click its source in the camera field.' if calibration_active
                       else 'Listening for spatial audio. Make a short, clear sound.')
    for i, line in enumerate(_wrap(message, layout.main_width - p(28), 0.43 * s, 2)):
        _text(out, line, p(30), note_y + p(45 + i * 20), 0.43 * s, TEXT)

    def card(top, height, title):
        _round_rect(out, (sx, p(top), sw, p(height)), PANEL, p(12))
        _text(out, title, sx + p(18), p(top + 24), 0.34 * s, MUTED)

    card(92, 140, 'LOCALIZATION')
    cv2.circle(out, (sx + p(21), p(144)), p(4), color, -1, cv2.LINE_AA)
    _text(out, state, sx + p(34), p(152), 0.68 * s, color, max(1, p(2)))
    for i, line in enumerate(_wrap(reason, sw - p(36), 0.4 * s, 3)):
        _text(out, line, sx + p(18), p(179 + i * 18), 0.4 * s, MUTED)

    card(246, 174, 'ACOUSTIC EVIDENCE')
    fix = view.get('fix') or {}
    fresh = state not in ('STALE', 'LISTENING')
    coherence = _number(fix.get('coherence')) if fresh else None
    snr = _number(fix.get('snr_db')) if fresh else None
    fraction = _number((view.get('quality') or {}).get('support_fraction')) if fresh else None
    coverage = ('broad' if fraction >= 0.35 else 'distributed' if fraction >= 0.12 else 'localized') if fraction is not None else '--'
    if state == 'WEAK':
        coverage = 'unresolved'
    rows = [('Pair coherence', f'{coherence:.2f}' if coherence is not None else '--'),
            ('Signal / noise', f'{snr:+.1f} dB' if snr is not None else '--'), ('Field support', coverage)]
    for i, (label, value) in enumerate(rows):
        yy = p(295 + i * 25)
        _text(out, label, sx + p(18), yy, 0.4 * s, MUTED)
        text_width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.44 * s, 1)[0][0]
        _text(out, value, sx + sw - p(18) - text_width, yy, 0.44 * s, TEXT)
    if fresh:
        _sparkline(out, view, (sx + p(18), p(369), sw - p(36), p(23)), color)
    _text(out, 'Relative agreement across camera field', sx + p(18), p(407), 0.29 * s, MUTED)

    card(434, 126, 'CANDIDATE SURFACE RANGE')
    distance = None
    if fresh and state in ('CANDIDATE', 'AMBIGUOUS') and view.get('measured') and not view.get('off_screen'):
        try:
            point = np.asarray(view.get('point'), dtype=float)
            if point.shape == (3,) and np.isfinite(point).all() and point[2] > 0:
                distance = float(np.linalg.norm(point))
        except (ValueError, TypeError):
            pass
    _text(out, f'{distance:.2f} m' if distance is not None else '--', sx + p(18), p(501), 0.92 * s,
          TEXT if distance is not None else MUTED, max(1, p(2)))
    _text(out, 'Visible surface, not source distance.', sx + p(18), p(528), 0.37 * s, MUTED)
    _text(out, 'Depth unavailable' if fresh and distance is None else 'Sound-source depth is unconfirmed.',
          sx + p(18), p(546), 0.31 * s, MUTED)

    card(574, 88, 'EVENT AGE')
    _text(out, f'{age:.1f} s' if age is not None else '--', sx + p(18), p(628), 0.64 * s,
          AMBER if state == 'STALE' else TEXT)
    phase = 'expired' if state == 'STALE' else 'completed sound' if view.get('final') else 'live estimate' if view else 'awaiting audio'
    _text(out, phase, sx + p(120), p(626), 0.35 * s, MUTED)
    bar_x, bar_w = sx + p(18), sw - p(36)
    cv2.line(out, (bar_x, p(646)), (bar_x + bar_w, p(646)), LINE, max(1, p(2)))
    if age is not None and age <= expiry and expiry > 0:
        cv2.line(out, (bar_x, p(646)), (bar_x + int(bar_w * (1 - age / expiry)), p(646)), color, max(1, p(2)))

    # Target verification is descriptive; it never becomes a localization score.
    lower_y = layout.height - p(67) if layout.compact else max(note_y + p(94), p(662)) + p(22)
    available_width = layout.width - p(48)
    if target:
        verdict = view.get('verdict') or {}
        heard = verdict.get('heard')
        verification = f'TARGET  {target}  |  ' + (str(heard) if heard else 'Waiting for sound check')
        if lower_y < layout.height - p(61):
            _text(out, _fit(verification, available_width, 0.36 * s), p(24), lower_y, 0.36 * s, MUTED)

    footer_y = layout.height - p(56)
    cv2.line(out, (p(24), footer_y), (layout.width - p(24), footer_y), LINE, 1)
    hints = ([('ENTER', 'apply'), ('ESC', 'cancel'), ('C', 'restart'), ('H', 'heatmap'), ('S', 'snapshot'), ('Q', 'quit')]
             if calibration_active else
             [('C', 'calibrate'), ('H', 'heatmap'), ('D', 'diagnostics'), ('SPACE', 'resume' if paused else 'pause'),
              ('S', 'snapshot'), ('Q', 'quit')])
    cursor = p(24)
    for key, action in hints:
        key_width = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.32 * s, 1)[0][0] + p(12)
        _round_rect(out, (cursor, footer_y + p(17), key_width, p(22)), PANEL, p(4))
        _text(out, key, cursor + p(6), footer_y + p(32), 0.32 * s, TEXT)
        cursor += key_width + p(7)
        _text(out, action, cursor, footer_y + p(32), 0.34 * s, MUTED)
        cursor += cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.34 * s, 1)[0][0] + p(19)
    fps_value = _number(fps)
    if fps_value is not None and layout.width - cursor > p(90):
        _text(out, f'{max(0, fps_value):.1f} FPS', layout.width - p(98), footer_y + p(32), 0.36 * s, MUTED)

    if diagnostics:
        z = _number(fix.get('z')) if fresh else None
        peak = _number((view.get('quality') or {}).get('peak_score')) if fresh else None
        depth = _number(depth_coverage)
        info = ['DIAGNOSTICS', f'peak z {z:.1f}' if z is not None else 'peak z --',
                f'SRP {peak:.3f}' if peak is not None else 'SRP --',
                f'depth coverage {np.clip(depth, 0, 1) * 100:.0f}%' if depth is not None else 'depth coverage --',
                'gray outline = depth surface']
        heading_width = cv2.getTextSize(notes_heading, cv2.FONT_HERSHEY_SIMPLEX, 0.34 * s, 1)[0][0]
        diag_x = p(30) + heading_width + p(24) if layout.compact else p(30)
        diag_y = note_y + p(22 if layout.compact else 81)
        _text(out, _fit('  |  '.join(info), layout.main_width - diag_x, 0.29 * s), diag_x, diag_y, 0.29 * s, CYAN)
    return out
