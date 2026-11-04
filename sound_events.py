"""Local sound-event memory and concept search. No network, embeddings or robot calls.

Visual context is stored separately from acoustic identity. Levels are uncalibrated
dBFS, not physical sound-pressure levels. A depth range belongs to a candidate
visible surface and does not establish that the surface emitted the sound.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import uuid


SEARCH_METHOD = 'local concept search (synonyms and structured filters; no embeddings)'
_NUMBERS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'ten': 10, 'fifteen': 15, 'thirty': 30, 'sixty': 60}
_GROUPS = (
    {'ring', 'rings', 'ringing', 'ringtone', 'ringtones'},
    {'beep', 'beeps', 'beeping', 'bleep', 'bleeps'},
    {'clap', 'claps', 'clapping', 'handclap', 'applause'},
    {'snap', 'snaps', 'snapping'},
    {'speak', 'speaking', 'speech', 'talk', 'talking', 'voice', 'voices', 'spoken'},
    {'jingle', 'jingles', 'jingling', 'rattle', 'rattling'},
    {'keyboard', 'keyboards', 'typing', 'keystrokes'},
    {'knock', 'knocks', 'knocking', 'tap', 'taps', 'tapping'},
    {'hum', 'humming', 'buzz', 'buzzing'},
    {'unknown', 'unidentified', 'unrecognized', 'unrecognised'},
)
_CANONICAL = {word: sorted(group)[0] for group in _GROUPS for word in group}
_STOP = set('a an the sound sounds event events find show me all any my that those of was were is are '
            'has have with and or please occurred happened heard from at on in to it i can you '
            'what which when where there just most recent recently'.split())


def _utc(value=None):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, timezone.utc)
    else:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _tokens(text):
    return {_CANONICAL.get(w, w) for w in re.findall(r"[a-z]+", str(text).lower()) if w not in _STOP}


def _nearby(event):
    result = []
    for obj in event.get('nearby') or []:
        label = obj.get('label', obj.get('name', '')) if isinstance(obj, dict) else obj
        if isinstance(label, str) and label.strip():
            result.append(label.strip()[:120])
    return result


def describe_event(event):
    """Describe observations without promoting a model guess into a known source."""
    at = _utc(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')
    label = str(event.get('label') or 'unidentified sound')
    identity = label if label.lower() == 'unidentified sound' or event.get('source_verified') else 'possible ' + label
    quality = event.get('quality', 'weak')
    bearing = _number(event.get('bearing_deg'))
    elevation = _number(event.get('elevation_deg'))
    if quality == 'weak':
        location = 'direction uncertain (weak acoustic evidence)'
    elif bearing is None:
        location = 'direction unavailable'
    else:
        side = 'left' if bearing < -8 else 'right' if bearing > 8 else 'ahead'
        location = f'{quality} direction {side} ({abs(bearing):.0f} degrees)'
        if elevation is not None and abs(elevation) > 12:
            location += ', ' + ('above' if elevation > 0 else 'below')
    fragments = [f'At {at}: {identity}', location]
    distance = _number(event.get('range_m'))
    if distance is not None and distance > 0:
        fragments.append(f'candidate visible surface at {distance:.2f} m')
    level = _number(event.get('level_dbfs'))
    if level is not None:
        fragments.append(f'recorded level {level:.1f} dBFS (uncalibrated)')
    snr = _number(event.get('snr_db'))
    if snr is not None:
        fragments.append(f'{snr:+.1f} dB above estimated background')
    nearby = _nearby(event)
    if nearby:
        fragments.append('nearby visible objects: ' + ', '.join(nearby[:5]))
    if not event.get('source_verified', False):
        fragments.append('emitting object unverified')
    return '; '.join(fragments) + '.'


def _normalize(event):
    data = dict(event)
    data['id'] = str(data.get('id') or uuid.uuid4().hex)
    data['timestamp'] = _utc(data.get('timestamp')).isoformat().replace('+00:00', 'Z')
    data.setdefault('mode', 'standalone')
    data.setdefault('label', 'unidentified sound')
    data.setdefault('label_source', 'unknown')
    data.setdefault('acoustic_description', '')
    data.setdefault('quality', 'weak')
    data.setdefault('source_verified', False)
    data.setdefault('nearby', [])
    for key in ('bearing_deg', 'elevation_deg', 'range_m', 'snr_db', 'level_dbfs', 'duration_s'):
        data[key] = _number(data.get(key))
    data['phrase'] = describe_event(data)
    # Round-trip strips shared mutable references; reject unsupported/NaN extras.
    return json.loads(json.dumps(data, allow_nan=False))


def _query_filters(query, now):
    q = str(query).lower().strip()
    filters = {}
    pattern = r'\b(?:last|past|within(?: the last)?)\s+(\d+(?:\.\d+)?|a|an|one|two|three|four|five|ten|fifteen|thirty|sixty)\s+(second|minute|hour|day)s?\b'
    match = re.search(pattern, q)
    if match:
        n = _NUMBERS.get(match[1])
        n = n if n is not None else float(match[1])
        seconds = n * {'second': 1, 'minute': 60, 'hour': 3600, 'day': 86400}[match[2]]
        filters['after'] = now - timedelta(seconds=seconds)
        q = q[:match.start()] + ' ' + q[match.end():]
    elif re.search(r'\btoday\b', q):
        filters['after'] = now.replace(hour=0, minute=0, second=0, microsecond=0)
        q = re.sub(r'\btoday\b', ' ', q)
    if re.search(r'\b(loud|louder|loudest)\b', q):
        filters['level'] = 'loud'
        q = re.sub(r'\b(loud|louder|loudest)\b', ' ', q)
    elif re.search(r'\b(quiet|quieter|quietest|faint|soft)\b', q):
        filters['level'] = 'quiet'
        q = re.sub(r'\b(quiet|quieter|quietest|faint|soft)\b', ' ', q)
    # Parse numeric level before 'above/below' can be read as an image direction.
    db = re.search(r'\b(?:over|above|under|below)\s*(-?\d+(?:\.\d+)?)\s*dbfs\b', q)
    if db:
        filters['dbfs'] = ('min' if db[0].startswith(('over', 'above')) else 'max', float(db[1]))
        q = q[:db.start()] + ' ' + q[db.end():]
    for direction in ('left', 'right', 'ahead', 'above', 'below'):
        if re.search(r'\b' + direction + r'\b', q):
            filters['direction'] = direction
            q = re.sub(r'\b' + direction + r'\b', ' ', q)
            break
    nearby = re.search(r'\b(?:near|beside|next to)\s+(.+)', q)
    if nearby:
        filters['nearby'] = _tokens(nearby[1])
        q = q[:nearby.start()]
    return _tokens(q), filters


class EventStore:
    """Thread-safe bounded SQLite history. Caller chooses where local data is kept."""
    def __init__(self, path=':memory:', *, retention=500):
        self.retention = max(1, int(retention))
        self._lock = threading.RLock()
        if str(path) != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, at REAL NOT NULL, payload TEXT NOT NULL)')
        self._db.execute('CREATE INDEX IF NOT EXISTS events_at ON events(at DESC)')
        self._db.commit()

    def add(self, event):
        data = _normalize(event)
        with self._lock, self._db:
            self._db.execute('INSERT OR REPLACE INTO events (id, at, payload) VALUES (?, ?, ?)',
                             (data['id'], _utc(data['timestamp']).timestamp(), json.dumps(data, allow_nan=False)))
            self._db.execute('DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY at DESC, rowid DESC LIMIT -1 OFFSET ?)',
                             (self.retention,))
        return data

    def update(self, id, patch):
        with self._lock:
            previous = self.get(id)
            if previous is None:
                raise KeyError(id)
            data = dict(previous)
            data.update(patch)
            data['id'] = str(id)
            return self.add(data)

    def get(self, id):
        with self._lock:
            row = self._db.execute('SELECT payload FROM events WHERE id=?', (str(id),)).fetchone()
        return json.loads(row[0]) if row else None

    def list_events(self, limit=100):
        with self._lock:
            rows = self._db.execute('SELECT payload FROM events ORDER BY at DESC, rowid DESC LIMIT ?',
                                    (max(0, min(int(limit), self.retention)),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def search(self, query, limit=30, *, now=None):
        """Local synonym ranking and explicit filters, not a trained semantic model.

        'Loud' means >= -25 dBFS and 'quiet' <= -45 dBFS, using the recorded
        uncalibrated signal. These filters do not measure perceived loudness/SPL.
        Nearby visual labels match only explicit 'near/beside/next to' clauses.
        """
        clock = _utc(now)
        terms, filters = _query_filters(query, clock)
        found = []
        for event in self.list_events(self.retention):
            at = _utc(event['timestamp'])
            if 'after' in filters and not filters['after'] <= at <= clock:
                continue
            bearing, elevation = event.get('bearing_deg'), event.get('elevation_deg')
            direction = filters.get('direction')
            if direction:
                if event.get('quality') == 'weak':
                    continue
                matches = {'left': bearing is not None and bearing < -8,
                           'right': bearing is not None and bearing > 8,
                           'ahead': bearing is not None and abs(bearing) <= 8,
                           'above': elevation is not None and elevation > 12,
                           'below': elevation is not None and elevation < -12}
                if not matches[direction]:
                    continue
            level = event.get('level_dbfs')
            if filters.get('level') == 'loud' and (level is None or level < -25):
                continue
            if filters.get('level') == 'quiet' and (level is None or level > -45):
                continue
            if 'dbfs' in filters:
                comparison, threshold = filters['dbfs']
                if level is None or (comparison == 'min' and level < threshold) or (comparison == 'max' and level > threshold):
                    continue
            near_terms = filters.get('nearby', set())
            near_words = _tokens(' '.join(_nearby(event)))
            if near_terms and not near_terms <= near_words:
                continue
            auditory = _tokens(str(event.get('label', '')) + ' ' + str(event.get('acoustic_description', '')))
            overlap = terms & auditory
            if terms and not terms <= auditory:
                continue
            score = len(overlap) / max(1, len(terms)) + (0.25 if near_terms else 0) + len(filters) * 0.1
            result = dict(event, score=round(score, 3), search_method=SEARCH_METHOD)
            if 'level' in filters or 'dbfs' in filters:
                result['search_notes'] = 'Recording level only, not SPL: loud >= -25 dBFS; quiet <= -45 dBFS.'
            found.append(result)
        found.sort(key=lambda e: (e['score'], _utc(e['timestamp'])), reverse=True)
        return found[:max(0, int(limit))]

    def close(self):
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
