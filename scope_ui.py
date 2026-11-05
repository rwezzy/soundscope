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
import os
import time

import cv2
import numpy as np
from heatmap_render import blend_heatmap, HEATMAP_PALETTES, HEATMAP_STYLES, palette_lut
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # retain the OpenCV fallback on a minimal installation
    Image = ImageDraw = ImageFont = None


# All colors are BGR. Color is interface hierarchy, never inferred confidence.
BG = (0, 0, 0)
PANEL = (17, 17, 17)
PANEL2 = (28, 28, 28)
LINE = (55, 55, 55)
TEXT = (244, 244, 244)
MUTED = (163, 163, 163)
PINK = (178, 82, 255)
YELLOW = (94, 229, 245)
CYAN = PINK
AMBER = YELLOW
GRAY = (143, 137, 126)
ALERT = (80, 80, 240)
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
        main_width = width - px(360)
        note_y = height - px(138)
        area_height = note_y - px(18) - px(108)
        factor = min(main_width / w, area_height / h)
        fitted_w, fitted_h = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
        camera = (px(24) + (main_width - fitted_w) // 2,
                  px(108) + (area_height - fitted_h) // 2, fitted_w, fitted_h)
        return _Layout(s, main_width, width, height, camera,
                       main_width + px(48), px(288), area_height, note_y, px(64), True)
    s = max(1.0, min(1.5, w / 1280.0))
    px = lambda value: int(round(value * s))
    main_width = max(w, px(520))
    return _Layout(s, main_width, main_width + px(360), max(px(754), h + px(304)),
                   (px(24) + (main_width - w) // 2, px(108), w, h),
                   main_width + px(48), px(288), h, px(108) + h + px(20), px(74), False)


# The action buttons sit in a toolbar under the header (they were hard to see along the bottom
# edge). render() draws the body as before and moves it down by BAR; the public geometry
# helpers below apply the same shift, so mouse mapping stays exact.
BAR = 46
BODY_TOP = 78


def _shift(layout):
    return int(round(BAR * layout.scale))


def camera_rect(frame_shape, viewport_size=None):
    """Return displayed camera (x, y, width, height), excluding all letterboxing.

    Pass the same ``viewport_size=(width, height)`` used by ``render``. Without a
    viewport the camera is native resolution; with one it is uniformly fitted.
    """
    layout = _layout(frame_shape, viewport_size)
    x, y, w, h = layout.camera
    return x, y + _shift(layout), w, h


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


def control_rects(frame_shape, viewport_size=None, *, search_text=None,
                  calibration_active=False, show_help=False, shifted=True):
    """Named clickable controls; shared by drawing and mouse hit testing."""
    layout = _layout(frame_shape, viewport_size)
    p = lambda value: int(round(value * layout.scale))
    sx, sw = layout.sidebar_x, layout.sidebar_width
    if show_help:
        hw = min(layout.width - p(48), p(670))
        hh = min(layout.height - p(40), p(550))
        hx, hy = (layout.width - hw) // 2, (layout.height - hh) // 2
        return {'help': (hx + hw - p(116), hy + hh - p(46), p(92), p(30))}
    d = _shift(layout) if shifted else 0  # render() draws the body unshifted, then moves it
    controls = {
        'search': (layout.width - p(386), p(24), p(126), p(32)),
        'history': (layout.width - p(248), p(24), p(126), p(32)),
        'help': (layout.width - p(110), p(24), p(86), p(32)),
        'palette': (sx + p(16), p(540) + d, sw - p(32), p(30)),
        'style': (sx + p(16), p(580) + d, sw - p(32), p(30)),
        'opacity_down': (sx + sw - p(82), p(624) + d, p(28), p(27)),
        'opacity_up': (sx + sw - p(44), p(624) + d, p(28), p(27)),
    }
    actions = (('submit', 'cancel', 'delete') if search_text is not None else
               ('submit', 'cancel', 'calibrate', 'heatmap', 'snapshot') if calibration_active else
               ('voice', 'analyze', 'calibrate', 'heatmap', 'pause', 'fullscreen', 'snapshot', 'quit'))
    available = layout.width - p(48)
    gap = p(7)
    button_width = (available - gap * (len(actions) - 1)) // len(actions)
    for i, action in enumerate(actions):
        controls[action] = (p(24) + i * (button_width + gap), p(BODY_TOP) + p(6), button_width, p(34))
    return controls


def recent_event_rects(frame_shape, viewport_size=None, shifted=True):
    layout = _layout(frame_shape, viewport_size)
    p = lambda value: int(round(value * layout.scale))
    return [(layout.sidebar_x + p(10), p(354 + i * 43) + (_shift(layout) if shifted else 0),
             layout.sidebar_width - p(20), p(39)) for i in range(3)]


def control_hit_test(x, y, frame_shape, viewport_size=None, recent_events=(), *,
                     search_text=None, calibration_active=False, show_help=False):
    """Return a control name, ('event', id), or None; never perform an action."""
    for name, (rx, ry, rw, rh) in control_rects(frame_shape, viewport_size,
            search_text=search_text, calibration_active=calibration_active, show_help=show_help).items():
        if rx <= x < rx + rw and ry <= y < ry + rh:
            return name
    if show_help:
        return None
    for record, (rx, ry, rw, rh) in zip(recent_events, recent_event_rects(frame_shape, viewport_size)):
        if rx <= x < rx + rw and ry <= y < ry + rh and record.get('id') is not None:
            return ('event', record['id'])
    return None


def _number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError, OverflowError):
        return default


def _clean(value):
    return str(value).replace('\r', ' ').replace('\n', ' ')


# Text is drawn with a real typeface (Segoe UI on Windows, DejaVu elsewhere) through PIL, cached
# per string; OpenCV's Hershey font is only a fallback when no font file is found.
_FONT_DIRS = (os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'), '/usr/share/fonts/truetype/dejavu',
              '/Library/Fonts', '/System/Library/Fonts/Supplemental')
_FACES = {False: ('segoeui.ttf', 'DejaVuSans.ttf', 'Arial.ttf', 'arial.ttf'),
          True: ('seguisb.ttf', 'DejaVuSans-Bold.ttf', 'Arial Bold.ttf', 'arialbd.ttf')}


@lru_cache(maxsize=64)
def _font(px, bold):
    if ImageFont is None:
        return None
    for name in _FACES[bool(bold)]:
        for folder in _FONT_DIRS:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, px)
                except OSError:
                    continue
    return None


def _px(scale):
    return max(8, int(round(scale * 32)))


@lru_cache(maxsize=2048)
def _glyphs(text, px, bold):
    """An anti-aliased coverage mask of the text and the row of its baseline."""
    font = _font(px, bold)
    ascent, descent = font.getmetrics()
    width = int(math.ceil(font.getlength(text))) + 4
    image = Image.new('L', (max(1, width), ascent + descent + 4), 0)
    ImageDraw.Draw(image).text((2, 2), text, font=font, fill=255)
    return np.asarray(image), ascent + 2


def _width(text, scale, bold=False):
    text = _clean(text)
    font = _font(_px(scale), bold)
    if font is None:
        return cv2.getTextSize(text.encode('ascii', 'replace').decode('ascii'), cv2.FONT_HERSHEY_SIMPLEX,
                               scale, 2 if bold else 1)[0][0]
    return int(math.ceil(font.getlength(text)))


def _text(img, text, x, y, scale=0.5, color=TEXT, thickness=1):
    """Draw text with its baseline at y (like cv2.putText). thickness >= 2 means semibold."""
    text = _clean(text)
    if not text:
        return
    bold = thickness >= 2
    font = _font(_px(scale), bold)
    if font is None:
        cv2.putText(img, text.encode('ascii', 'replace').decode('ascii'), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return
    mask, base = _glyphs(text, _px(scale), bold)
    x0, y0 = int(x) - 2, int(y) - base
    h, w = mask.shape
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    a = mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].astype(np.float32)[..., None] / 255.0
    roi = img[iy0:iy1, ix0:ix1]
    roi[:] = (roi * (1 - a) + np.asarray(color, np.float32) * a + 0.5).astype(np.uint8)


def _fit(text, width, scale, bold=False):
    text = _clean(text)
    if _width(text, scale, bold) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # longest prefix that fits with an ellipsis
        mid = (lo + hi + 1) // 2
        if _width(text[:mid].rstrip() + '\u2026', scale, bold) <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + '\u2026' if lo else ''


def _wrap(text, width, scale, limit=3):
    words = _clean(text).split()
    lines, line = [], ''
    for word in words:
        proposed = f'{line} {word}'.strip()
        if line and _width(proposed, scale) > width:
            lines.append(line)
            line = word
        else:
            line = proposed
    if line:
        lines.append(line)
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = _fit(lines[-1] + ' \u2026\u2026', width, scale)
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


def _icon(img, name, center, size, color=TEXT):
    """Small native vector icons: no external fonts or bitmap assets required."""
    cx, cy = center
    unit = max(.25, size / 24.)
    thick = max(1, round(unit * 1.7))
    point = lambda x, y: (round(cx + (x - 12) * unit), round(cy + (y - 12) * unit))
    def line(a, b):
        cv2.line(img, point(*a), point(*b), color, thick, cv2.LINE_AA)
    def circle(x, y, r, fill=False):
        cv2.circle(img, point(x, y), max(1, round(r * unit)), color, -1 if fill else thick, cv2.LINE_AA)
    def path(points):
        cv2.polylines(img, [np.array([point(*v) for v in points], np.int32)], False,
                      color, thick, cv2.LINE_AA)
    if name == 'voice':
        _round_rect(img, (*point(8, 3), max(2, round(8 * unit)), max(2, round(12 * unit))), color,
                    round(4 * unit), thick)
        cv2.ellipse(img, point(12, 12), (round(7 * unit), round(7 * unit)), 0, 0, 180, color, thick, cv2.LINE_AA)
        line((12, 19), (12, 22)); line((8, 22), (16, 22))
    elif name == 'search':
        circle(10, 10, 6); line((15, 15), (21, 21))
    elif name == 'history':
        circle(12, 12, 9); path([(12, 6), (12, 12), (17, 15)])
    elif name == 'calibrate':
        circle(12, 12, 6); circle(12, 12, 1, True)
        for a, b in (((12, 1), (12, 6)), ((12, 18), (12, 23)), ((1, 12), (6, 12)), ((18, 12), (23, 12))):
            line(a, b)
    elif name in ('heatmap', 'analyze'):
        if name == 'heatmap':
            for r in (3, 7, 11):
                cv2.ellipse(img, point(12, 12), (round(r * unit), max(1, round(r * .72 * unit))),
                            -25, 0, 360, color, thick, cv2.LINE_AA)
        else:
            for xx, top in ((3, 9), (7, 4), (11, 8), (15, 2), (19, 7)):
                line((xx, top), (xx, 22 - top))
            circle(18, 17, 4); line((21, 20), (23, 23))
    elif name == 'pause':
        line((8, 4), (8, 20)); line((16, 4), (16, 20))
    elif name == 'play':
        cv2.fillConvexPoly(img, np.array([point(7, 3), point(21, 12), point(7, 21)], np.int32), color, cv2.LINE_AA)
    elif name == 'fullscreen':
        for pts in ([(9, 3), (3, 3), (3, 9)], [(15, 3), (21, 3), (21, 9)],
                    [(3, 15), (3, 21), (9, 21)], [(15, 21), (21, 21), (21, 15)]):
            path(pts)
    elif name == 'snapshot':
        path([(2, 7), (7, 7), (9, 4), (15, 4), (17, 7), (22, 7), (22, 20), (2, 20), (2, 7)])
        circle(12, 13, 4)
    elif name == 'help':
        circle(12, 12, 10)
        cv2.ellipse(img, point(12, 8), (round(3 * unit), round(3 * unit)), 0, 180, 480, color, thick, cv2.LINE_AA)
        line((12, 12), (12, 14)); circle(12, 18, .8, True)
    elif name == 'palette':
        for xx, yy, rgb in ((6, 6, PINK), (17, 6, YELLOW), (6, 17, TEXT), (17, 17, MUTED)):
            cv2.circle(img, point(xx, yy), max(1, round(4 * unit)), rgb, -1, cv2.LINE_AA)
    elif name == 'style':
        for yy, end in ((5, 20), (12, 17), (19, 22)):
            line((3, yy), (end, yy))
    elif name == 'submit':
        path([(4, 12), (10, 18), (21, 6)])
    elif name in ('cancel', 'quit'):
        line((5, 5), (19, 19)); line((5, 19), (19, 5))
    elif name == 'delete':
        path([(8, 5), (22, 5), (22, 19), (8, 19), (2, 12), (8, 5)])
        line((12, 9), (18, 15)); line((12, 15), (18, 9))
    elif name == 'name':
        path([(4, 16), (16, 4), (21, 9), (9, 21), (3, 22), (4, 16)])
        line((13, 7), (18, 12)); line((4, 16), (9, 21))
    elif name == 'live':
        circle(12, 12, 3, True)
        for radius in (7, 11):
            cv2.ellipse(img, point(12, 12), (round(radius * unit), round(radius * unit)),
                        0, -55, 55, color, thick, cv2.LINE_AA)
            cv2.ellipse(img, point(12, 12), (round(radius * unit), round(radius * unit)),
                        0, 125, 235, color, thick, cv2.LINE_AA)
    elif name in ('previous', 'next'):
        path([(15, 4), (7, 12), (15, 20)] if name == 'previous' else [(9, 4), (17, 12), (9, 20)])


def _button(img, rect, icon, label, scale, *, active=False, accent=None):
    x, y, w, h = rect
    p = lambda value: int(round(value * scale))
    tint = accent or (YELLOW if active else TEXT)
    _round_rect(img, rect, PANEL2 if active else PANEL, p(8))
    _round_rect(img, rect, tint if active else LINE, p(8), 1)
    icon_size = max(8, p(18))
    text_scale = .40 * scale
    shown = _fit(label, w - p(42), text_scale, True)
    group_width = icon_size + p(8) + _width(shown, text_scale, True)
    left = x + max(p(7), (w - group_width) // 2)
    _icon(img, icon, (left + icon_size // 2, y + h // 2), icon_size, tint)
    _text(img, shown, left + icon_size + p(8), y + (h + p(10)) // 2, text_scale, tint, 2)


def _state(view, now):
    if not view:
        return 'LISTENING', 'Make a sound in the camera field.', None, DEFAULT_EXPIRY
    timestamp = _number(view.get('time'), now)
    age = max(0.0, now - timestamp)
    expiry = max(0.0, _number(view.get('expires_after'), DEFAULT_EXPIRY))
    if view.get('hold'):  # a found target stays until the next command
        expiry = max(expiry, age + 1.0)
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


def _brackets(img, bounds, color, scale):
    x, y, w, h = bounds
    pad = int(round(7 * scale))
    x0, x1 = max(2, x - pad), min(img.shape[1] - 3, x + w + pad)
    y0, y1 = max(2, y - pad), min(img.shape[0] - 3, y + h + pad)
    length = max(4, min(int(19 * scale), (x1 - x0) // 3, (y1 - y0) // 3))
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, 2, cv2.LINE_AA)


def _camera_overlay(image, view, state, heatmap, diagnostics, scale,
                    heatmap_palette='amber', heatmap_opacity=0.44, heatmap_style='classic'):
    if state in ('STALE', 'LISTENING'):
        return
    color = CYAN if state == 'CANDIDATE' else AMBER
    strength, support = _support(view, image.shape)
    # Display the actual graded score field even if quality gates reject a lock.
    # The support mask controls candidate contours, never visibility of evidence.
    # A zero/flat/invalid field has no heat; nonflat values are relative SRP, not
    # probability. User-selected opacity keeps at least 15% of the camera even
    # at a peak; lower scores preserve more of the scene. Palette is display only.
    if heatmap and strength is not None and cv2.minMaxLoc(strength)[1] > 0:
        blend_heatmap(image, strength, palette=heatmap_palette, style=heatmap_style,
                      opacity=heatmap_opacity, weak=state == 'WEAK',
                      support_mask=support,
                      outline=state in ('CANDIDATE', 'AMBIGUOUS') and not view.get('off_screen'))
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
    if diagnostics and not (heatmap and heatmap_style == 'classic'):
        lines = image.copy()
        cv2.drawContours(lines, contours, -1, color, max(1, int(scale)), cv2.LINE_AA)
        cv2.addWeighted(lines, 0.7, image, 0.3, 0, dst=image)
    # Brackets enclose the entire acoustic support, not a supposedly exact spot.
    # Broad/multiple regions retain their actual uncertainty contours.
    if ((not heatmap or heatmap_style != 'classic') and state == 'CANDIDATE'
            and np.mean(support) < 0.35 and len(contours) == 1):
        _brackets(image, cv2.boundingRect(contours[0]), color, scale)
    if diagnostics and view.get('contour') is not None:
        try:
            contour = np.asarray(view['contour'], np.int32).reshape(-1, 1, 2)
            cv2.drawContours(image, [contour], -1, GRAY, 1, cv2.LINE_AA)
        except (ValueError, TypeError):
            pass


def _sam_outline(image, view, scale):
    """SAM 3.1's outline of the object at the acoustic spot (a point prompt, never a search)."""
    polygons = view.get('sam_outline') or ()
    for poly in polygons:
        try:
            pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
        except (ValueError, TypeError):
            continue
        glow = image.copy()
        cv2.polylines(glow, [pts], True, CYAN, max(2, int(4 * scale)), cv2.LINE_AA)
        cv2.addWeighted(glow, 0.45, image, 0.55, 0, dst=image)
        cv2.polylines(image, [pts], True, TEXT, max(1, int(scale)), cv2.LINE_AA)


def _alert_banner(image, text):
    """A wide red bar across the top of the picture, readable from across a room."""
    h, w = image.shape[:2]
    bar = max(40, h // 9)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, bar), ALERT, -1)
    cv2.addWeighted(overlay, 0.88, image, 0.12, 0, dst=image)
    size = 0.9 * bar / 48
    text = _fit(text, w - 24, size)
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2)[0]
    cv2.putText(image, text, ((w - tw) // 2, (bar + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255),
                max(2, int(size * 2.2)), cv2.LINE_AA)


# One color per talker, so a subtitle and its marker on the picture's edge go together.
TALKER_COLORS = ((94, 229, 245), (178, 82, 255), (120, 220, 120), (255, 196, 92), (90, 160, 255))


def _talker_color(name):
    words = str(name or '').split()
    if len(words) == 2 and words[0].lower() == 'speaker' and words[1].isdigit():
        return TALKER_COLORS[(int(words[1]) - 1) % len(TALKER_COLORS)]
    return TALKER_COLORS[sum(map(ord, str(name or ''))) % len(TALKER_COLORS)]


def _side(cap, w):
    """-1 left, +1 right, 0 ahead/unknown, from the talker's pixel or the direction words."""
    pixel = cap.get('pixel')
    try:
        if pixel is not None:
            off = (float(pixel[0]) - w / 2) / (w / 2)
            return -1 if off < -0.18 else 1 if off > 0.18 else 0
    except (TypeError, ValueError, IndexError):
        pass
    words = str(cap.get('where') or '').lower()
    return -1 if 'left' in words else 1 if 'right' in words else 0


def _subtitles(image, subs, scale):
    """Speech along the bottom of the picture, like film subtitles, never over anyone's face.

    Each line starts with the talker's name in their color and an arrow toward their side; a small
    marker in the same color sits on the picture's bottom edge right under where they are."""
    h, w = image.shape[:2]
    sc = max(0.6, scale)
    size = 0.5 * sc
    line_h = int(26 * sc)
    blocks = []
    for cap in list(subs)[-2:]:  # the two newest, newest at the bottom
        speaker = str(cap.get('speaker') or '').strip()
        spoken = _clean(cap.get('spoken') or cap.get('text') or '')
        if not spoken:
            continue
        color = _talker_color(speaker)
        lead = int(30 * sc) + (_width(speaker + ':', size, True) + int(8 * sc) if speaker else 0)
        lines = _wrap(spoken, int(w * 0.86) - lead, size, 2)
        blocks.append((cap, speaker, color, lead, lines))
    if not blocks:
        return
    total = sum(len(b[4]) for b in blocks) * line_h + int(14 * sc) * len(blocks) + int(10 * sc)
    top = h - total - int(16 * sc)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, top - int(6 * sc)), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, dst=image)
    y = top
    for cap, speaker, color, lead, lines in blocks:
        width = lead + max(_width(line, size) for line in lines)
        x0 = max(int(12 * sc), (w - width) // 2)
        mid = y + line_h // 2 + int(2 * sc)
        side, a = _side(cap, w), int(7 * sc)
        cx = x0 + int(10 * sc)
        if side == 0:
            cv2.circle(image, (cx, mid), max(3, a // 2), color, -1, cv2.LINE_AA)
        else:
            tip = cx + side * a
            cv2.fillConvexPoly(image, np.array([[tip, mid], [tip - side * 2 * a, mid - a], [tip - side * 2 * a, mid + a]],
                                               np.int32), color, cv2.LINE_AA)
        if speaker:
            _text(image, speaker + ':', x0 + int(30 * sc), y + line_h - int(6 * sc), size, color, 2)
        for i, line in enumerate(lines):
            _text(image, line, x0 + lead, y + line_h * (i + 1) - int(6 * sc), size, TEXT)
        pixel = cap.get('pixel')
        if pixel is not None:  # marker on the bottom edge, straight below the talker
            try:
                u = int(np.clip(float(pixel[0]), a, w - a - 1))
                cv2.fillConvexPoly(image, np.array([[u, h - int(13 * sc)], [u - a, h - 1], [u + a, h - 1]], np.int32),
                                   color, cv2.LINE_AA)
            except (TypeError, ValueError, IndexError):
                pass
        y += line_h * len(lines) + int(14 * sc)


def _captions(image, captions, scale):
    """Speech becomes subtitles at the bottom; other notes are bubbles near the spot their sound came from."""
    captions = list(captions or ())
    subs = [c for c in captions if c.get('kind') == 'speech']
    captions = [c for c in captions if c.get('kind') != 'speech']
    if subs:
        _subtitles(image, subs, scale)
    h, w = image.shape[:2]
    placed = []
    for cap in captions:
        text = _clean(cap.get('text', ''))[:140]
        if not text:
            continue
        size = 0.5 * max(0.6, scale)
        lines = _wrap(text, int(w * 0.5), size, 2)
        line_h = int(22 * max(0.6, scale))
        box_w = max(_width(line, size) for line in lines) + 16
        box_h = line_h * len(lines) + 10
        pixel = cap.get('pixel')
        if pixel is not None:
            u, v = int(pixel[0]), int(pixel[1])
            x, y = u - box_w // 2, v - box_h - int(30 * scale)
        else:
            x, y = (w - box_w) // 2, h - box_h - 12
        x, y = int(np.clip(x, 4, max(4, w - box_w - 4))), int(np.clip(y, 4, max(4, h - box_h - 4)))
        # Search above/below existing bubbles, rather than grouping only nearby
        # anchor pixels. When the picture is full, omit a bubble instead of
        # drawing overlapping, unreadable text.
        candidates = [y, 4, h - box_h - 4]
        for bx, by, bw, bh in placed:
            candidates.extend((by - box_h - 6, by + bh + 6))
        found = None
        for candidate in sorted(set(candidates), key=lambda yy: abs(yy - y)):
            if candidate < 4 or candidate + box_h > h - 4:
                continue
            if not any(x < bx + bw + 6 and bx < x + box_w + 6
                       and candidate < by + bh + 6 and by < candidate + box_h + 6
                       for bx, by, bw, bh in placed):
                found = candidate
                break
        if found is None:
            continue
        y = found
        placed.append((x, y, box_w, box_h))
        overlay = image.copy()
        _round_rect(overlay, (x, y, box_w, box_h), PANEL, 8)
        cv2.addWeighted(overlay, 0.82, image, 0.18, 0, dst=image)
        accent = {'speech': CYAN, 'alert': ALERT, 'recall': TEXT}.get(cap.get('kind'), AMBER)
        cv2.line(image, (x + 3, y + 5), (x + 3, y + box_h - 5), accent, 2, cv2.LINE_AA)
        for i, line in enumerate(lines):
            _text(image, line, x + 10, y + 6 + line_h * (i + 1) - 6, size, TEXT)
        if pixel is not None:
            cv2.circle(image, (int(np.clip(pixel[0], 0, w - 1)), int(np.clip(pixel[1], 0, h - 1))), 3, accent, -1,
                       cv2.LINE_AA)


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


def _view_bearing(view):
    """Candidate camera-relative bearing, never a compass/north measurement."""
    off = view.get('off_screen')
    if off:
        try:
            edge, value = off
            angle = _number(value)
            return (-abs(angle) if edge == 'left' else abs(angle) if edge == 'right' else None) if angle is not None else None
        except (ValueError, TypeError):
            return None
    try:
        point = np.asarray(view.get('point'), dtype=float)
        if point.shape == (3,) and np.isfinite(point).all() and point[2] > 0:
            return math.degrees(math.atan2(point[0], point[2]))
    except (ValueError, TypeError):
        pass
    return None


def _bearing_dial(image, center, radius, angle, color=CYAN):
    cx, cy = center
    cv2.circle(image, center, radius, LINE, 1, cv2.LINE_AA)
    cv2.circle(image, center, max(2, radius // 2), LINE, 1, cv2.LINE_AA)
    k = max(2, radius // 8)
    cv2.fillConvexPoly(image, np.array([[cx, cy - k - 2], [cx - k, cy + k], [cx + k, cy + k]], np.int32), MUTED)
    if angle is not None:
        radians = math.radians(angle)
        tip = (round(cx + radius * .88 * math.sin(radians)), round(cy - radius * .88 * math.cos(radians)))
        cv2.arrowedLine(image, center, tip, color, max(1, radius // 15), cv2.LINE_AA, tipLength=.25)


def render(frame, view, target, note, *, now=None, fps=None, depth_coverage=None,
           heatmap=True, diagnostics=False, paused=False, calibration_active=False,
           viewport_size=None, status=None, captions=(), outlines=(), alert=None,
           heatmap_palette='amber', heatmap_opacity=0.44, search_text=None,
           search_answer=None, recent_events=(), heatmap_style='classic', show_help=False,
           entry_kind='search', ai_status=None):
    """Return a BGR dashboard without modifying ``frame`` or ``view``.

    ``now`` and ``view['time']`` use the same monotonic clock. Both final and live
    estimates expire after ``expires_after`` seconds (default 1.2). A caller that
    pauses a snapshot may pass its frozen monotonic time. ``depth_coverage`` is a
    measured fraction in [0, 1]. No fields are interpreted as confidence.
    ``viewport_size=(width, height)`` produces exactly that canvas size and fits
    the camera without stretching. Use this same size for mouse mapping helpers.
    ``status`` is a one-line mode text (voice command, budget); ``captions`` are bubbles
    ({'text', 'pixel' or None, 'kind'}) placed at the pixel each sound came from.
    ``heatmap_palette`` and ``heatmap_opacity`` change color/alpha only, never
    evidence, quality, geometry or persistence. Zero opacity removes the tint.
    ``search_text`` shows a focused local-history query field; None closes it.
    ``search_answer`` shows the caller's answer; the caller controls its expiry.
    This renderer never submits the query or calls a model.
    ``ai_status`` is optional actual controller state: phase, text, detail and
    transcript. No model activity or transcript is inferred by this renderer.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('frame must be a uint8 BGR image with three channels')
    if not isinstance(heatmap_palette, str) or heatmap_palette not in HEATMAP_PALETTES:
        raise ValueError('Unknown heatmap palette')
    if not isinstance(heatmap_style, str) or heatmap_style not in HEATMAP_STYLES:
        raise ValueError('Unknown heatmap style')
    heatmap_opacity = _number(heatmap_opacity)
    if heatmap_opacity is None or not 0 <= heatmap_opacity <= .85:
        raise ValueError('Heatmap opacity must be a finite number from 0 to 0.85')
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

    # Brand, navigation, and a calm canvas around the measured camera scene.
    for i, size in enumerate((10, 22, 32, 19, 9)):
        xx = p(26 + i * 6)
        cv2.line(out, (xx, p(37 - size / 2)), (xx, p(37 + size / 2)),
                 YELLOW if i == 2 else PINK, max(1, p(3)), cv2.LINE_AA)
    _text(out, 'Sound', p(70), p(43), 0.85 * s, PINK, max(2, p(2)))
    _text(out, 'Scope', p(70) + _width('Sound', .85 * s, True), p(43), .85 * s,
          YELLOW, max(2, p(2)))
    _text(out, 'SEE SOUND. REMEMBER WHAT HAPPENED.', p(71), p(62), 0.28 * s, MUTED)
    controls = control_rects(frame.shape, viewport_size, shifted=False, search_text=search_text,
                             calibration_active=calibration_active)
    for name, label in (('search', 'Search'), ('history', 'History'), ('help', 'Help')):
        _button(out, controls[name], name, label, s, accent=PINK if name == 'search' else None)
    cv2.line(out, (p(24), p(73)), (layout.width - p(24), p(73)), LINE, 1)

    # The card includes letterboxing, while camera_rect names only actual pixels.
    _round_rect(out, (p(16), p(82), layout.main_width + p(16), layout.camera_area_height + p(34)), PANEL, p(14))
    cv2.circle(out, (p(31), p(97)), max(1, p(3)), YELLOW if not paused else MUTED, -1, cv2.LINE_AA)
    _text(out, 'PAUSED FRAME' if paused else 'LIVE VIEW', p(41), p(102), .32 * s, TEXT, 2)
    mode = ('CALIBRATION' if calibration_active else 'HEATMAP OFF' if not heatmap
            else 'HEATMAP / WEAK EVIDENCE' if state == 'WEAK'
            else 'HEATMAP / RELATIVE EVIDENCE')
    mode_width = _width(mode, 0.34 * s)
    _text(out, mode, p(24) + layout.main_width - mode_width - p(6), p(102), 0.34 * s,
          AMBER if calibration_active else MUTED)
    camera = frame.copy()
    _camera_overlay(camera, view, state, heatmap, diagnostics, s,
                    heatmap_palette, heatmap_opacity, heatmap_style)
    if state == 'CANDIDATE':
        _sam_outline(camera, view, s)
    # Retained polygons belong to past events. Without object tracking, painting
    # them on this frame would imply a current source lock after evidence expires.
    _captions(camera, captions, frame.shape[1] / 1280.0 * 1.4)
    if alert:
        _alert_banner(camera, alert)
    if (w, h) != (frame.shape[1], frame.shape[0]):
        interpolation = cv2.INTER_AREA if w < frame.shape[1] else cv2.INTER_LINEAR
        camera = cv2.resize(camera, (w, h), interpolation=interpolation)
    out[y:y + h, x:x + w] = camera
    cv2.rectangle(out, (x - 1, y - 1), (x + w, y + h), LINE, 1)

    # A search result is a historical answer, separate from live localization.
    if search_answer and search_text is None and w > p(100) and h > p(90):
        result_lines = _wrap(search_answer, w - p(52), 0.39 * s, 5)
        max_lines = max(1, min(5, int((h - p(68)) / max(1, p(18)))))
        result_lines = result_lines[:max_lines]
        if result_lines:
            result_h = p(42 + 18 * len(result_lines))
            result_y = y + h - result_h - p(12)
            result_x, result_w = x + p(12), w - p(24)
            overlay = out.copy()
            _round_rect(overlay, (result_x, result_y, result_w, result_h), PANEL, p(8))
            cv2.addWeighted(overlay, .94, out, .06, 0, dst=out)
            cv2.line(out, (result_x + p(2), result_y + p(9)),
                     (result_x + p(2), result_y + result_h - p(9)), CYAN, max(1, p(2)))
            _text(out, 'HISTORY RESULT', result_x + p(14), result_y + p(22), 0.3 * s, CYAN)
            for i, line in enumerate(result_lines):
                _text(out, line, result_x + p(14), result_y + p(43 + i * 18), .39 * s, TEXT)

    # Notes are outside the picture. Long device messages cannot cover the scene.
    note_y = layout.note_y
    has_ai_status = isinstance(ai_status, dict) or bool(status)
    ai_width = min(p(320), max(p(215), layout.main_width // 3)) if has_ai_status else 0
    notes_width = layout.main_width + p(16) - (ai_width + p(10) if has_ai_status else 0)
    _round_rect(out, (p(16), note_y, notes_width, layout.note_height), PANEL, p(12))
    searching = search_text is not None
    naming = searching and entry_kind == 'label'
    latest_transcript = (str(ai_status.get('transcript') or '') if isinstance(ai_status, dict) else '')
    ai_features = ai_status.get('features') if isinstance(ai_status, dict) else None
    captions_enabled = isinstance(ai_features, (list, tuple, set)) and 'Transcripts' in ai_features
    show_transcript = (bool(latest_transcript) or captions_enabled) and not searching and not calibration_active
    transcript_message = latest_transcript
    if show_transcript and not transcript_message:
        phase = str(ai_status.get('phase') or '').lower()
        transcript_message = {
            'off': 'No transcript yet. Automatic transcription is off.',
            'budget': 'No transcript yet. Transcription is paused.',
            'error': 'No transcript yet. Check the Meta AI status.',
        }.get(phase, 'Listening for speech. Captions appear after a short pause.')
    notes_heading = ('NAME SAVED SOUND' if naming else 'SEARCH SOUND HISTORY' if searching
                     else 'CALIBRATION' if calibration_active else 'LATEST TRANSCRIPT' if show_transcript else 'SOUND NOTES')
    _text(out, notes_heading, p(30), note_y + p(19),
          0.34 * s, CYAN if searching else AMBER if calibration_active else MUTED)
    message = transcript_message if show_transcript else note or ('Play a sound, then click its source in the camera field.' if calibration_active
                       else 'Listening for spatial audio. Make a short, clear sound.')
    if searching:
        _round_rect(out, (p(16), note_y, notes_width, layout.note_height), CYAN, p(12), 1)
        query = _clean(search_text)
        available = notes_width - p(32)
        clipped = False
        # Keep the insertion point visible when a query exceeds the field width.
        while query and _width(('... ' if clipped else '') + query + ' |', 0.43 * s) > available:
            query, clipped = query[1:], True
        shown = ('... ' if clipped else '') + query + ' |' if query else ('Name this saved event...' if naming else 'Type a sound or description...')
        _text(out, shown, p(30), note_y + p(40), 0.43 * s, TEXT if query else MUTED)
        hint = 'Saved label only; does not train recognition.' if naming else 'Searches recorded events. Use Search below to submit.'
        _text(out, _fit(hint, notes_width - p(28), .28 * s), p(30), note_y + p(57), .28 * s, MUTED)
    else:
        for i, line in enumerate(_wrap(message, notes_width - p(28), 0.43 * s, 2)):
            _text(out, line, p(30), note_y + p(40 + i * 18), 0.40 * s, TEXT)
    if has_ai_status:
        ax = p(26) + notes_width
        _round_rect(out, (ax, note_y, ai_width, layout.note_height), PANEL, p(12))
        ai = ai_status if isinstance(ai_status, dict) else {}
        phase = str(ai.get('phase') or 'status').lower()
        phase_label = {'off': 'OFF', 'idle': 'READY', 'processing': 'PROCESSING',
                       'budget': 'PAUSED', 'error': 'ERROR'}.get(phase, 'STATUS')
        ai_color = PINK if phase == 'processing' else YELLOW if phase in ('budget', 'error') else MUTED
        _text(out, 'META AI', ax + p(13), note_y + p(19), .30 * s, TEXT, 2)
        _text(out, phase_label, ax + ai_width - _width(phase_label, .27 * s, True) - p(13),
              note_y + p(19), .27 * s, ai_color, 2)
        cv2.line(out, (ax + p(1), note_y + p(12)), (ax + p(1), note_y + layout.note_height - p(12)),
                 ai_color, max(1, p(2)), cv2.LINE_AA)
        ai_text = str(ai.get('text') or status or 'No activity reported.')
        second_line = str(ai.get('detail') or ai.get('budget') or '')
        _text(out, _fit(ai_text, ai_width - p(26), .34 * s, True), ax + p(13), note_y + p(39), .34 * s, TEXT, 2)
        if second_line:
            _text(out, _fit(second_line, ai_width - p(26), .28 * s), ax + p(13), note_y + p(56), .28 * s, MUTED)

    def card(top, height, title):
        _round_rect(out, (sx, p(top), sw, p(height)), PANEL, p(14))
        _text(out, title, sx + p(16), p(top + 24), 0.32 * s, MUTED, 2)

    # One spatial card, not a wall of scores. The candidate qualifier stays explicit.
    card(82, 224, 'LOCALIZATION')
    cv2.circle(out, (sx + p(20), p(131)), max(1, p(3)), color, -1, cv2.LINE_AA)
    _text(out, state, sx + p(31), p(136), .48 * s, color, 2)
    bearing = _view_bearing(view) if state == 'CANDIDATE' else None
    direction = ('Straight ahead' if abs(bearing) < 10 else
                 'To the right' if bearing > 0 else 'To the left') if bearing is not None else 'Listening...' if state == 'LISTENING' else 'Direction unresolved'
    _text(out, _fit(direction, sw - p(99), .54 * s, True), sx + p(16), p(164), .54 * s, TEXT, 2)
    for i, line in enumerate(_wrap(reason, sw - p(105), .32 * s, 2)):
        _text(out, line, sx + p(16), p(185 + i * 16), .32 * s, MUTED)
    _bearing_dial(out, (sx + sw - p(49), p(169)), p(31), bearing, color)
    _text(out, 'camera bearing', sx + sw - p(84), p(214), .22 * s, MUTED)
    cv2.line(out, (sx + p(16), p(224)), (sx + sw - p(16), p(224)), LINE, 1)
    _text(out, 'CANDIDATE SURFACE RANGE', sx + p(16), p(244), .27 * s, MUTED, 2)
    fix = view.get('fix') or {}
    fresh = state not in ('STALE', 'LISTENING')
    distance = None
    if fresh and state in ('CANDIDATE', 'AMBIGUOUS') and view.get('measured') and not view.get('off_screen'):
        try:
            point = np.asarray(view.get('point'), dtype=float)
            if point.shape == (3,) and np.isfinite(point).all() and point[2] > 0:
                distance = float(np.linalg.norm(point))
        except (ValueError, TypeError):
            pass
    _text(out, f'{distance:.2f} m' if distance is not None else '--', sx + p(16), p(276), .70 * s,
          YELLOW if distance is not None else MUTED, 2)
    _text(out, 'Visible surface, not source distance.', sx + p(16), p(295), .28 * s, MUTED)

    card(320, 170, 'RECENT EVENTS / CLICK TO OPEN')
    if recent_events:
        for event, rect in zip(tuple(recent_events)[:3], recent_event_rects(frame.shape, viewport_size, shifted=False)):
            ex, ey, ew, eh = rect
            _round_rect(out, rect, PANEL2, p(8))
            sound = event.get('sound') or {}
            source = str(sound.get('source') or '').lower()
            annotation = event.get('annotation') or {}
            speech = event.get('speech') or {}
            speech_text = str(speech.get('text') or '') if isinstance(speech, dict) else str(speech)
            is_transcript = False
            if annotation.get('source') == 'user' and annotation.get('label'):
                title, origin = annotation['label'], 'Your label'
            elif speech_text:
                speaker = str(speech.get('speaker') or '') if isinstance(speech, dict) else ''
                title = (speaker + ': ' if speaker else '') + '\u201c' + speech_text + '\u201d'
                origin, is_transcript = 'Transcript', True
            elif sound.get('label'):
                title = sound['label']
                origin = 'User label' if source in ('user', 'manual', 'user-provided') else 'AI estimate'
            elif sound.get('local_label'):
                title, origin = sound['local_label'], 'Your example'
            else:
                title = (event.get('acoustic') or {}).get('character') or 'Unidentified sound'
                origin = 'Acoustic pattern'
            origin_w = 0 if is_transcript else _width(origin, .23 * s)
            _text(out, _fit(title, ew - origin_w - p(26), .38 * s, True), ex + p(9), ey + p(17), .38 * s, TEXT, 2)
            if not is_transcript:
                _text(out, origin, ex + ew - origin_w - p(9), ey + p(17), .23 * s, PINK)
            stamp = str(event.get('time') or '')
            stamp = stamp[11:19] if len(stamp) >= 19 else '--:--:--'
            _text(out, stamp, ex + p(9), ey + p(32), .27 * s, MUTED)
            where = ('Transcript / ' if is_transcript else '') + str((event.get('where') or {}).get('words') or 'saved event')
            _text(out, _fit(where, ew - p(88), .26 * s), ex + p(80), ey + p(32), .26 * s, MUTED)
    else:
        _text(out, 'Your sound history starts here.', sx + p(16), p(374), .37 * s, TEXT)
        _text(out, 'Make a sound, then inspect its evidence.', sx + p(16), p(398), .29 * s, MUTED)
        if age is not None:
            _text(out, f'Last estimate: {age:.1f} s ago', sx + p(16), p(448), .30 * s, MUTED)
        else:
            _text(out, 'No recorded events supplied.', sx + p(16), p(448), .30 * s, MUTED)

    card(504, 158, 'HEATMAP APPEARANCE')
    button_labels = {'palette': ('Palette', HEATMAP_PALETTES[heatmap_palette]),
                     'style': ('Style', HEATMAP_STYLES[heatmap_style])}
    for name, (label, value) in button_labels.items():
        bx, by, bw, bh = controls[name]
        _round_rect(out, (bx, by, bw, bh), PANEL2, p(7))
        _icon(out, name, (bx + p(17), by + bh // 2), p(16), MUTED)
        _text(out, label, bx + p(33), by + p(20), .30 * s, MUTED)
        _text(out, _fit(value, bw - p(146 if name == 'palette' else 103), .32 * s, True),
              bx + p(92), by + p(20), .32 * s, TEXT, 2)
        if name == 'palette':
            swatch = cv2.resize(palette_lut(heatmap_palette).reshape(1, 256, 3),
                                (max(1, p(31)), max(1, p(8))))
            gx, gy = bx + bw - p(41), by + p(12)
            out[gy:gy + swatch.shape[0], gx:gx + swatch.shape[1]] = swatch
    _text(out, 'Opacity', sx + p(16), p(642), .31 * s, MUTED)
    _text(out, f'{heatmap_opacity:.2f}', sx + p(87), p(643), .36 * s, YELLOW, 2)
    for name, label in (('opacity_down', '-'), ('opacity_up', '+')):
        bx, by, bw, bh = controls[name]
        _round_rect(out, (bx, by, bw, bh), PANEL2, p(7))
        _text(out, label, bx + (bw - _width(label, .45 * s)) // 2, by + p(21), .45 * s, TEXT)

    # Target verification is descriptive; it never becomes a localization score.
    lower_y = layout.height - p(67)
    available_width = layout.width - p(150 if _number(fps) is not None else 48)
    line = ''
    if target:
        verdict = view.get('verdict') or {}
        heard = verdict.get('heard')
        line = f'TARGET  {target}  |  ' + (str(heard) if heard else 'Waiting for sound check')
    if show_transcript and note:
        line = (line + '   |   ' if line else '') + str(note)
    if line and lower_y < layout.height - p(61):
        _text(out, _fit(line, available_width, 0.36 * s), p(24), lower_y, 0.36 * s,
              CYAN if view.get('hold') else MUTED)

    footer_y = layout.height - p(56)
    buttons = ([('submit', 'Save label' if naming else 'Search'), ('cancel', 'Cancel'), ('delete', 'Delete')]
               if searching else
               [('submit', 'Apply calibration'), ('cancel', 'Cancel'), ('calibrate', 'Restart'),
                ('heatmap', 'Heatmap'), ('snapshot', 'Snapshot')] if calibration_active else
               [('voice', 'Voice'), ('analyze', 'Analyze sound'), ('calibrate', 'Calibrate'),
                ('heatmap', 'Heatmap'), ('pause', 'Resume' if paused else 'Pause'),
                ('fullscreen', 'Fullscreen'), ('snapshot', 'Snapshot'), ('quit', 'Finish')])
    fps_value = _number(fps)

    if diagnostics and not searching:
        coherence = _number(fix.get('coherence')) if fresh else None
        snr = _number(fix.get('snr_db')) if fresh else None
        z = _number(fix.get('z')) if fresh else None
        peak = _number((view.get('quality') or {}).get('peak_score')) if fresh else None
        depth = _number(depth_coverage)
        info = ['DIAGNOSTICS', f'SNR {snr:+.1f} dB' if snr is not None else 'SNR --',
                f'coherence {coherence:.2f}' if coherence is not None else 'coherence --',
                f'peak z {z:.1f}' if z is not None else 'peak z --',
                f'SRP {peak:.3f}' if peak is not None else 'SRP --',
                f'depth coverage {np.clip(depth, 0, 1) * 100:.0f}%' if depth is not None else 'depth coverage --',
                'gray outline = depth surface']
        heading_width = _width(notes_heading, 0.34 * s)
        diag_x = p(30) + heading_width + p(24) if layout.compact else p(30)
        diag_y = note_y + p(19) if layout.compact else note_y - p(5)
        _text(out, _fit('  |  '.join(info), layout.main_width - diag_x, 0.29 * s), diag_x, diag_y, 0.29 * s, CYAN)
    # Toolbar: move the body (everything between the header and the old footer) down by one bar,
    # then draw the action buttons in the gap, right under the header where they are easy to see.
    top, bottom, d = p(BODY_TOP), footer_y, _shift(layout)
    out[top + d:bottom + d] = out[top:bottom].copy()
    out[top:top + d] = BG
    for action, label in buttons:
        active = (action == 'heatmap' and heatmap) or (action == 'pause' and paused)
        icon = 'play' if action == 'pause' and paused else action
        _button(out, controls[action], icon, label, s, active=active,
                accent=PINK if action in ('voice', 'analyze', 'submit') else None)
    if fps_value is not None:
        _text(out, f'{max(0, fps_value):.1f} FPS', layout.width - p(98), layout.height - p(14), 0.28 * s, MUTED)
    if show_help:
        shade = np.full_like(out, BG)
        cv2.addWeighted(out, .32, shade, .68, 0, dst=out)
        help_w = min(layout.width - p(48), p(670))
        help_h = min(layout.height - p(40), p(550))
        hx, hy = (layout.width - help_w) // 2, (layout.height - help_h) // 2
        _round_rect(out, (hx, hy, help_w, help_h), PANEL, p(18))
        _round_rect(out, (hx, hy, help_w, help_h), (85, 51, 99), p(18), 1)
        _text(out, 'Make sense of a sound.', hx + p(28), hy + p(46), .72 * s, TEXT, 2)
        _text(out, 'Live evidence, a searchable history, and the original recording.',
              hx + p(28), hy + p(72), .34 * s, MUTED)
        help_rows = [
            ('T  /  Search', 'Ask about sounds already recorded.'),
            ('B  /  R', 'Browse history / open the latest saved event.'),
            ('V  /  Voice', 'Speak a command; enabled AI features use credits.'),
            ('M  /  X  /  [ ]', 'Palette, heatmap style, and opacity.'),
            ('C', 'Collect calibration points; Enter applies the fit.'),
            ('H  /  D', 'Show heatmap / measured diagnostics.'),
            ('N  /  E', 'Name a saved event / replay its available audio.'),
            ('SPACE  /  S', 'Pause display / save a diagnostic snapshot.'),
            ('F  /  Q', 'Fullscreen / finish and save cleanly.'),
        ]
        for i, (keys, description) in enumerate(help_rows):
            ry = hy + p(110 + i * 40)
            _text(out, keys, hx + p(28), ry, .36 * s, YELLOW, 2)
            _text(out, _fit(description, help_w - p(215), .34 * s), hx + p(190), ry, .34 * s, TEXT)
        _text(out, 'The map shows acoustic agreement, not a verified object identity.',
              hx + p(28), hy + help_h - p(58), .31 * s, MUTED)
        _text(out, 'Buttons work by click. Keyboard shortcuts are optional.', hx + p(28),
              hy + help_h - p(28), .29 * s, MUTED)
        _button(out, control_rects(frame.shape, viewport_size, show_help=True)['help'],
                'cancel', 'Close', s, accent=PINK)
    return out
