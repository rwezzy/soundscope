"""Local, hardware-free evidence browser. Pixels and records are supplied by the caller.

No file, audio-device, network or model access occurs here. Events are newest
first. ``evidence_image`` is a BGR ndarray; ``evidence_audio`` is a truthy flag
enabling the replay button. Saved candidate pixels are never called a live fix.
"""
import math
import time
from datetime import datetime

import cv2
import numpy as np

from scope_ui import (BG, PANEL, LINE, TEXT, MUTED, CYAN, AMBER, _text, _fit,
                      _wrap, _round_rect, _clean, _width, _button)

PAGE_SIZE = 6


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _dict(value):
    return value if isinstance(value, dict) else {}


def layout_positions(viewport_size=(1280, 720), event_count=0, selected=0):
    """Shared geometry for drawing and hit-testing, including automatic paging."""
    width, height = (int(n) for n in viewport_size)
    if width < 640 or height < 360:
        raise ValueError('History viewport must be at least 640 x 360')
    s = min(width / 1280.0, height / 720.0)
    ox, oy = (width - round(1280 * s)) // 2, (height - round(720 * s)) // 2
    rect = lambda x, y, w, h: (ox + round(x * s), oy + round(y * s), round(w * s), round(h * s))
    count = max(0, int(event_count))
    selection = min(max(0, int(selected)), max(0, count - 1))
    start = selection // PAGE_SIZE * PAGE_SIZE
    rows = [(index, rect(24, 160 + row * 81, 336, 73))
            for row, index in enumerate(range(start, min(start + PAGE_SIZE, count)))]
    return {'width': width, 'height': height, 'scale': s, 'offset': (ox, oy),
            'selected': selection, 'page_start': start, 'rows': rows,
            'analyze': rect(648, 26, 124, 42), 'name': rect(788, 26, 120, 42),
            'search': rect(924, 26, 132, 42),
            'live': rect(1076, 26, 180, 42), 'play': rect(1062, 632, 194, 40),
            'editor_submit': rect(780, 410, 144, 34),
            'editor_cancel': rect(936, 410, 120, 34),
            'editor_delete': rect(1072, 410, 136, 34),
            'image_area': rect(400, 164, 840, 300),
            'previous': rect(24, 640, 76, 32), 'next': rect(284, 640, 76, 32)}


def hit_test(x, y, viewport_size=(1280, 720), event_count=0, selected=0, *, search_text=None):
    """Return ('select', index), (action, None), or None.

    Play availability must also be checked by the caller. Page arrows select the
    first record of the adjacent page; no record is modified. Header actions are
    live, analyze, name and search; analyze/name require an existing event.
    With an open editor, only submit/cancel/delete are actionable; clicking a
    record behind the editor cannot silently change the sound being named.
    """
    layout = layout_positions(viewport_size, event_count, selected)
    inside = lambda r: r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]
    if search_text is not None:
        for action in ('submit', 'cancel', 'delete'):
            if inside(layout['editor_' + action]):
                return action, None
        return None
    for action in ('live', 'play', 'search'):
        if inside(layout[action]):
            return action, None
    if event_count > 0:
        for action in ('analyze', 'name'):
            if inside(layout[action]):
                return action, None
    for index, rect in layout['rows']:
        if inside(rect):
            return 'select', index
    start = layout['page_start']
    if start > 0 and inside(layout['previous']):
        return 'select', max(0, start - PAGE_SIZE)
    if start + PAGE_SIZE < event_count and inside(layout['next']):
        return 'select', start + PAGE_SIZE
    return None


def _label(record):
    annotation = _dict(record.get('annotation'))
    if annotation.get('label'):
        return str(annotation['label']), 'Your label / not an AI identification'
    sound, speech, acoustic = (_dict(record.get(k)) for k in ('sound', 'speech', 'acoustic'))
    if sound.get('label'):
        return str(sound['label']), 'Model interpretation / unverified'
    if speech.get('text'):
        return 'Speech', 'Transcribed speech'
    if sound.get('local_label'):
        return str(sound['local_label']), 'Local example match / unverified'
    return str(acoustic.get('character') or 'Unidentified sound'), 'Acoustic description'


def _stamp(record, now):
    epoch = _number(record.get('epoch'))
    if epoch is None:
        return '--:--:--', 'time unavailable'
    try:
        stamp = datetime.fromtimestamp(epoch).strftime('%H:%M:%S')
    except (ValueError, OSError, OverflowError):
        return '--:--:--', 'time unavailable'
    age = now - epoch
    if age < 0:
        return stamp, 'clock mismatch'
    if age < 60:
        return stamp, f'{int(age)} s ago'
    if age < 3600:
        return stamp, f'{int(age / 60)} min ago'
    if age < 86400:
        return stamp, f'{age / 3600:.1f} h ago'
    return stamp, f'{age / 86400:.0f} d ago'


def _image(value):
    return (value if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3
            and value.dtype == np.uint8 and value.shape[0] > 0 and value.shape[1] > 0 else None)


def recorded_marker(record, image_shape):
    """Map the saved original-camera pixel into stored pixels, or withhold it.

    Only an explicitly time-associated scene and a candidate-quality, on-screen
    estimate gets a marker. This does not verify the acoustic source's identity.
    """
    evidence, where = _dict(record.get('evidence')), _dict(record.get('where'))
    if evidence.get('scene_valid') is not True or where.get('on_screen') is not True:
        return None
    if str(where.get('quality', '')).lower() != 'candidate':
        return None
    size = record.get('evidence_image_size')
    if size is None:
        size = evidence.get('original_image_size')
    try:
        original = np.asarray(size, dtype=float)
        point = np.asarray(where.get('pixel'), dtype=float)
        if original.shape != (2,) or point.shape != (2,) or not np.isfinite(original).all() or not np.isfinite(point).all():
            return None
        if np.any(original <= 0) or np.any(point < 0) or np.any(point >= original):
            return None
        return tuple((point * np.array([image_shape[1], image_shape[0]]) / original).astype(int))
    except (ValueError, TypeError, OverflowError):
        return None


def render_history(events, selected=0, *, viewport_size=(1280, 720), now=None,
                   search_query='', search_result_count=None, status='', search_text=None,
                   entry_kind='search'):
    """Render a recorded-events browser; ``now`` is UNIX wall time, not monotonic.

    Supply newest-first record dictionaries. Optional ``evidence_image`` is the
    selected saved image, ``evidence_image_size`` its ORIGINAL [width, height],
    and ``evidence_audio`` indicates replay availability. Metadata in evidence
    includes scene_valid, scene_reason, frame_offset_s and audio_normalized.
    """
    events = tuple(_dict(record) for record in (events or ()))
    layout = layout_positions(viewport_size, len(events), selected)
    now = _number(now)
    now = time.time() if now is None else now
    s, (ox, oy) = layout['scale'], layout['offset']
    p = lambda n: round(n * s)
    pt = lambda x, y: (ox + p(x), oy + p(y))
    box = lambda x, y, w, h: (ox + p(x), oy + p(y), p(w), p(h))
    out = np.empty((layout['height'], layout['width'], 3), np.uint8)
    out[:] = BG
    text = lambda message, x, y, size=.42, color=TEXT, thick=1: _text(out, message, *pt(x, y), size * s, color, thick)
    text('SoundScope', 24, 47, .86, TEXT, max(1, p(2)))
    text('SOUND HISTORY / RECORDED EVIDENCE', 26, 70, .32, MUTED)
    for action, label in (('analyze', 'Analyze'), ('name', 'Name'), ('search', 'Search')):
        enabled = bool(events) or action == 'search'
        _button(out, layout[action], action, label, s, accent=TEXT if enabled else LINE)
    _button(out, layout['live'], 'live', 'Live view', s, accent=CYAN)
    cv2.line(out, pt(24, 90), pt(1256, 90), LINE, 1)
    query = str(search_query or '')
    result_count = _number(search_result_count)
    count = max(0, int(result_count)) if result_count is not None else len(events)
    text(f'{count} MATCHING EVENTS' if query else f'{len(events)} RECORDED EVENTS', 24, 124, .39, CYAN)
    text(_fit(query if query else 'Newest first / choose an event', p(332), .34 * s), 24, 145, .34, MUTED)

    for index, rect in layout['rows']:
        record = events[index]
        active = index == layout['selected']
        _round_rect(out, rect, (32, 32, 32) if active else PANEL, p(9))
        if active:
            _round_rect(out, rect, CYAN, p(9), 1)
        rx, ry, rw, rh = rect
        label, origin = _label(record)
        stamp, age = _stamp(record, now)
        _text(out, stamp, rx + p(13), ry + p(20), .32 * s, CYAN if active else MUTED)
        _text(out, age, rx + rw - p(89), ry + p(20), .3 * s, MUTED)
        _text(out, _fit(label, rw - p(26), .47 * s), rx + p(13), ry + p(43), .47 * s, TEXT)
        location = str(_dict(record.get('where')).get('words') or 'direction unresolved')
        _text(out, _fit(location, rw - p(26), .32 * s), rx + p(13), ry + p(62), .32 * s, MUTED)
    if not events:
        _round_rect(out, box(24, 160, 336, 150), PANEL, p(9))
        text('No events here yet', 42, 198, .55)
        for i, line in enumerate(_wrap('Return to live and make a short sound. Only recorded events appear here.', p(298), .4 * s, 4)):
            text(line, 42, 228 + i * 20, .4, MUTED)
    start = layout['page_start']
    for name, enabled in [('previous', start > 0), ('next', start + PAGE_SIZE < len(events))]:
        rect = layout[name]
        _button(out, rect, name, 'Prev' if name == 'previous' else 'Next', s,
                accent=CYAN if enabled else LINE)
    text(f'{start + 1 if events else 0}-{min(start + PAGE_SIZE, len(events))} of {len(events)}', 128, 661, .34, MUTED)

    _round_rect(out, box(384, 106, 872, 510), PANEL, p(12))
    text('RECORDED EVENT', 400, 130, .5, AMBER, max(1, p(2)))
    text('HISTORICAL / NOT LIVE', 1051, 130, .32, AMBER)
    record = events[layout['selected']] if events else {}
    evidence, where, loud = (_dict(record.get(k)) for k in ('evidence', 'where', 'loudness'))
    image = _image(record.get('evidence_image'))
    area = layout['image_area']
    cv2.rectangle(out, (area[0], area[1]), (area[0] + area[2], area[1] + area[3]), BG, -1)
    marker = None
    if image is not None:
        ratio = min(area[2] / image.shape[1], area[3] / image.shape[0])
        iw, ih = max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))
        ix, iy = area[0] + (area[2] - iw) // 2, area[1] + (area[3] - ih) // 2
        display = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
        out[iy:iy + ih, ix:ix + iw] = display
        marker = recorded_marker(record, image.shape)
        if marker is not None:
            center = (ix + round(marker[0] * ratio), iy + round(marker[1] * ratio))
            cv2.circle(out, center, max(6, p(14)), AMBER, max(1, p(2)), cv2.LINE_AA)
            cv2.line(out, (center[0] - p(5), center[1]), (center[0] + p(5), center[1]), AMBER, 1)
            cv2.line(out, (center[0], center[1] - p(5)), (center[0], center[1] + p(5)), AMBER, 1)
    else:
        text('No saved camera frame', 650, 299, .66, MUTED)
        text('This event may still have sound and direction metadata.' if record else 'Recorded evidence will appear here.',
             537 if record else 635, 331, .38, MUTED)
    offset = _number(evidence.get('frame_offset_s'))
    if image is None:
        association = 'Visual evidence unavailable'
    elif evidence.get('scene_valid') is not True:
        association = 'Frame timing unverified - no spatial marker shown'
    elif marker is None:
        association = 'Saved frame / sound position unresolved'
    else:
        association = 'Circle: recorded acoustic candidate / source identity unverified'
    if offset is not None:
        association += f' / frame offset {offset:+.2f} s'
    text(_fit(association, p(836), .33 * s), 400, 484, .33, AMBER if marker is None else MUTED)

    if record:
        label, origin = _label(record)
        stamp, age = _stamp(record, now)
        text(_fit(label, p(810), .65 * s), 400, 515, .65, TEXT, max(1, p(2)))
        text(_fit(f'{stamp} / {age} / {origin}', p(810), .33 * s), 400, 537, .33, MUTED)
        direction = str(where.get('words') or 'unresolved direction')
        snr, distance = _number(loud.get('snr_db')), _number(where.get('distance_m'))
        spatial = 'DIRECTION  ' + direction
        if distance is not None and distance > 0:
            spatial += f' / candidate surface {distance:.2f} m'
        text(_fit(spatial, p(810), .36 * s), 400, 561, .36, CYAN)
        level = f'{snr:+.1f} dB relative to estimated room noise' if snr is not None else 'level unavailable'
        text('LEVEL  ' + level, 400, 584, .36, MUTED)
        speech = _dict(record.get('speech')).get('text')
        if speech:
            text(_fit('TRANSCRIPT  ' + str(speech), p(810), .33 * s), 400, 605, .33, TEXT)
        else:
            # Match memory's handling of unsupported hidden-object relations.
            from sound_memory import witnessed_context
            nearby = witnessed_context(record, _dict(record.get('near')).get('text'))
            if nearby:
                text(_fit('VISUAL CONTEXT  ' + str(nearby), p(810), .33 * s), 400, 605, .33, TEXT)
    else:
        text('Select a saved event to inspect its evidence.', 400, 527, .54, MUTED)

    audio = bool(record.get('evidence_audio'))
    _button(out, layout['play'], 'play', 'Play audio' if audio else 'No saved audio', s,
            accent=CYAN if audio else MUTED)
    warning = ('Saved replay may be normalized; playback volume is not the original sound level.' if audio
               else 'Saved locations describe the scene at event time, not its current state.')
    text(_fit(warning, p(642), .31 * s), 400, 646, .31, MUTED)
    if status:
        text(_fit(status, p(642), .31 * s), 400, 665, .31, CYAN)
    if search_text is not None:
        # A transient query editor uses the saved-image panel, never the live
        # scene or record metadata. Caller owns input, submission and API use.
        _round_rect(out, box(416, 318, 808, 140), PANEL, p(9))
        _round_rect(out, box(416, 318, 808, 140), CYAN, p(9), 1)
        text('NAME SAVED SOUND' if entry_kind == 'label' else 'SEARCH SOUND HISTORY', 432, 343, .33, CYAN)
        query_text = _clean(search_text)
        clipped = False
        while query_text and _width(('... ' if clipped else '') + query_text + ' |', .52 * s) > p(768):
            query_text, clipped = query_text[1:], True
        shown = ('... ' if clipped else '') + query_text + ' |' if query_text else ('Your name for this event...' if entry_kind == 'label' else 'Type a sound or description...')
        text(shown, 432, 378, .52, TEXT if query_text else MUTED)
        text('Your label describes this event; it does not train recognition.' if entry_kind == 'label'
             else 'Searches sounds already recorded in this history.', 432, 398, .32, MUTED)
        _button(out, layout['editor_submit'], 'submit', 'Save label' if entry_kind == 'label' else 'Search',
                s, accent=CYAN)
        _button(out, layout['editor_cancel'], 'cancel', 'Cancel', s)
        _button(out, layout['editor_delete'], 'delete', 'Delete', s)
    cv2.line(out, pt(24, 685), pt(1256, 685), LINE, 1)
    text(('Save label to apply it / Cancel to return' if entry_kind == 'label' else
          'Search recorded sounds / Cancel to return') if search_text is not None else
         'UP / DOWN select    B live    E replay    A analyze    N name    T search', 24, 707, .33, MUTED)
    text('LOCAL EVENT EVIDENCE', 1088, 707, .28, MUTED)
    return out
