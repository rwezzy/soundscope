"""Sound memory: every sound SoundScope hears becomes a searchable record.

A record says when the sound happened, where it came from (physics: the four mics, plus depth),
how loud it was, what it sounded like (local acoustic measurements, free), and, when the Meta
features are on, what it most likely was (Muse Spark), what was said and by whom (Muse Voice
Transcribe), and what was near the spot (Muse Spark looking at the camera crop the sound picked).

  memory = Memory()
  memory.add(make_record(...))
  search('when did the phone ring?', memory, meta)   -> {'answer': '...', 'matches': [...]}

Records live in recordings/memory/YYYY-MM-DD.jsonl (git-ignored, local only). The file is
append-only: later results are written as {"id": ..., "patch": {...}} lines and merged on load.
Nothing here decides WHERE a sound is; it only stores what the localizer measured.
"""
import io
import json
import math
import os
import re
import threading
import time
import uuid
import wave
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FS = 16000
SPEECH_SCORE = 0.5   # at or above: treated as speech (captions instead of a sound label)
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('SOUNDSCOPE_DATA') or HERE / 'recordings')
MEMORY_DIR = Path(os.environ.get('SOUNDSCOPE_MEMORY') or DATA_DIR / 'memory')
REFERENCES_DIR = Path(os.environ.get('SOUNDSCOPE_REFERENCES') or HERE / 'references')


# ---------------------------------------------------------------------------------------------
# Local acoustic measurements (no model, no network)

def wav_samples(wav):
    """16-bit mono WAV bytes -> float samples in [-1, 1]."""
    with wave.open(io.BytesIO(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float) / 32768.0


def db(x, floor=-120.0):
    return float(max(floor, 20 * math.log10(max(float(x), 1e-12))))


def _envelope(x, fs, hop_s=0.01):
    hop = max(1, int(hop_s * fs))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1))


def _pitch(x, fs, lo=85.0, hi=320.0):
    """Strength (0..1) of a voice-range pitch in the loudest stretch, by autocorrelation."""
    if len(x) < int(0.06 * fs):
        return 0.0
    n = int(0.04 * fs)
    env = _envelope(x, fs, 0.02)
    start = int(np.argmax(env) * 0.02 * fs)
    seg = x[start:start + n]
    if len(seg) < n:
        seg = x[-n:]
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, 'full')[n - 1:]
    if ac[0] <= 0:
        return 0.0
    a, b = int(fs / hi), min(int(fs / lo), n - 1)
    return float(np.clip(ac[a:b].max() / ac[0], 0, 1)) if b > a else 0.0


def acoustic_profile(x, fs=FS):
    """Measurements of one sound from raw (not normalized) mono samples.

    Loudness is dBFS of the raw mic signal, so it compares sounds with each other; the localizer's
    SNR (dB over the room's noise floor) is the better "how loud" for people and is added later.
    """
    x = np.asarray(x, float)
    if len(x) < 64:
        return {'duration_s': len(x) / fs, 'speech_score': 0.0, 'character': 'too short to describe'}
    x = highpass(x - x.mean(), fs)  # mains hum (50/60 Hz and harmonics) is not part of any sound
    env = _envelope(x, fs)
    loudest = env.max() if len(env) else 0.0
    quiet = float(np.percentile(env, 20)) if len(env) else 0.0  # the clip's quiet tail = room level
    floor = min(quiet, loudest * 0.3)                              # a clip loud all through has no tail
    active = env > max(loudest * 0.1, floor * 2.0) if loudest > 0 else np.zeros(len(env), bool)
    duration = float(active.sum() * 0.01) if len(env) else len(x) / fs
    window = int(0.25 / 0.01)
    rms_loud = float(np.sqrt(np.convolve(env ** 2, np.ones(window) / window, 'valid').max())) if len(env) >= window else float(np.sqrt((x ** 2).mean()))
    peak = float(np.abs(x).max())

    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    band = (freqs >= 150) & (freqs <= 7800)
    p = spec[band] + 1e-20
    f = freqs[band]
    centroid = float((p * f).sum() / p.sum())
    dominant = float(f[int(np.argmax(p))])
    flatness = float(np.exp(np.log(p).mean()) / p.mean())
    voice_band = float(p[(f >= 300) & (f <= 3400)].sum() / p.sum())
    low_band = float(p[f <= 1000].sum() / p.sum())
    peak_share = float(p.max() / p.sum())  # one steady tone owns its bin; voices and noise spread out

    # Onsets: rises of the 10 ms envelope above half the loudest level (and well above the room),
    # at least 80 ms apart.
    onsets, last = 0, -1e9
    above = env > max(loudest * 0.5, min(floor * 4.0, loudest * 0.8)) if loudest > 0 else np.zeros(len(env), bool)
    for i in range(1, len(env)):
        if above[i] and not above[i - 1] and i - last >= 8:
            onsets, last = onsets + 1, i
    onsets = max(onsets, 1 if loudest > 0 else 0)
    peak_i = int(np.argmax(env)) if len(env) else 0
    rise = np.nonzero(env[:peak_i + 1] < loudest * 0.1)[0] if len(env) else []
    attack_ms = float((peak_i - (rise[-1] if len(rise) else 0)) * 10.0)

    # Speech: energy in the voice band and below 1 kHz, a strongly varying envelope at syllable rate
    # (2-8 Hz), and a voice-range pitch; never a single steady tone or broadband noise.
    mod, wobble = 0.0, 0.0
    if len(env) >= 50:
        e = env - env.mean()
        m = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
        mf = np.fft.rfftfreq(len(e), 0.01)
        tot = m[(mf > 0.5) & (mf < 25)].sum()
        mod = float(m[(mf >= 2) & (mf <= 8)].sum() / tot) if tot > 0 else 0.0
        on = env[active] if active.any() else env
        wobble = float(on.std() / on.mean()) if on.mean() > 0 else 0.0
    pitch = _pitch(x, fs)
    speech = (0.3 * np.clip((voice_band - 0.45) / 0.4, 0, 1)
              + 0.3 * np.clip(mod / 0.4, 0, 1) * np.clip(wobble / 0.5, 0, 1)
              + 0.2 * np.clip((pitch - 0.3) / 0.5, 0, 1) * np.clip(low_band / 0.3, 0, 1)
              + 0.2 * np.clip(low_band / 0.35, 0, 1))
    speech *= float(flatness < 0.45) * float(duration >= 0.35) * float(peak_share < 0.25)

    profile = {'duration_s': round(duration, 2), 'rms_dbfs': round(db(rms_loud), 1), 'peak_dbfs': round(db(peak), 1),
               'crest_db': round(db(peak) - db(rms_loud), 1), 'centroid_hz': round(centroid), 'dominant_hz': round(dominant),
               'flatness': round(flatness, 3), 'onsets': int(onsets), 'attack_ms': round(attack_ms),
               'voice_band': round(voice_band, 2), 'speech_score': round(float(speech), 2)}
    profile['contrast_db'] = round(db(loudest) - db(max(quiet, 1e-12)), 1) if 0 < quiet < loudest * 0.3 else None
    profile['character'] = character(profile)
    return profile


def highpass(x, fs=FS, cutoff=150.0):
    """Zero-phase spectral high-pass with a short ramp: removes hum and rumble, keeps the sound."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    spec *= np.clip((f - cutoff * 0.7) / (cutoff * 0.3), 0, 1)
    return np.fft.irfft(spec, n=len(x))


def _khz(hz):
    return f'{hz / 1000:.1f} kHz' if hz >= 1000 else f'{hz:.0f} Hz'


def character(p):
    """A few plain words for how the sound looked to the mics."""
    if p.get('speech_score', 0) >= SPEECH_SCORE:
        return 'speech-like'
    words = []
    if p['onsets'] >= 3:
        words.append(f"{p['onsets']} quick bursts")
    elif p['duration_s'] < 0.3 and p['attack_ms'] <= 30:
        words.append('short sharp hit')
    elif p['duration_s'] >= 1.0:
        words.append('sustained')
    if p['flatness'] < 0.12:
        words.append(f"tonal around {_khz(p['dominant_hz'])}")
    elif p['flatness'] > 0.45:
        words.append('noisy, broadband')
    elif p['centroid_hz'] > 3500:
        words.append('bright, high-pitched')
    elif p['centroid_hz'] < 700:
        words.append('low, dull')
    return ', '.join(words) or 'mixed sound'


def loudness_words(snr_db):
    if snr_db is None:
        return 'unknown loudness'
    for limit, word in ((6, 'faint'), (15, 'quiet'), (25, 'moderate'), (35, 'loud')):
        if snr_db < limit:
            return word
    return 'very loud'


def direction_words(azimuth_deg, elevation_deg=None):
    """Camera-relative words; azimuth + = right of the lens axis, elevation + = up."""
    if azimuth_deg is None:
        return 'unknown direction'
    a = abs(azimuth_deg)
    side = 'right' if azimuth_deg > 0 else 'left'
    words = ('straight ahead' if a < 8 else f'slightly {side}' if a < 25 else f'to the {side}' if a < 60
             else f'far {side}' if a < 100 else f'behind, to the {side}')
    if elevation_deg is not None and a < 100:
        if elevation_deg > 15:
            words += ', above'
        elif elevation_deg < -15:
            words += ', below'
    return words


# ---------------------------------------------------------------------------------------------
# Records

MIN_Z, MIN_SNR_DB = 10.0, 0.0  # a memory record needs a well-located sound above the room's level


def worth_remembering(fix):
    """Well located (z) AND above the room's average level (SNR): keep. Measured Nov 4 on the real
    array: laptop-speaker bursts, a ring, speech and claps had z 55-89 at +2.6..+14 dB, while the
    quiet room's own blips had z under 7 or SNR below 0 dB."""
    z = float(fix.get('z') or 0.0)
    snr = fix.get('snr_db')
    snr = float(snr) if snr is not None and math.isfinite(float(snr)) else -99.0
    return z >= MIN_Z and snr >= MIN_SNR_DB


def new_id(now=None):
    stamp = datetime.fromtimestamp(now or time.time()).strftime('%Y%m%d-%H%M%S')
    return f'ev-{stamp}-{uuid.uuid4().hex[:4]}'


def make_record(fix, direction_cam, *, profile=None, view=None, level=None, mode='tripod', now=None, extra=None):
    """One sound as a JSON-ready dict.

    fix: the localizer's result (z, snr_db, seconds, coherence). direction_cam: unit vector in the
    camera frame (x right, y down, z out of the lens). view: scope's view for the same sound, for
    the picked pixel, the depth-camera distance and the quality label.
    """
    now = time.time() if now is None else now
    x, y, z = (float(v) for v in direction_cam)
    azimuth = math.degrees(math.atan2(x, z))
    elevation = math.degrees(math.atan2(-y, math.hypot(x, z)))
    view = view or {}
    point = view.get('point')
    distance = float(np.linalg.norm(point)) if point is not None and view.get('measured') else None
    snr = fix.get('snr_db')
    snr = float(snr) if snr is not None and math.isfinite(float(snr)) else None
    record = {
        'id': new_id(now), 'time': datetime.fromtimestamp(now).isoformat(timespec='seconds'), 'epoch': round(now, 3),
        'mode': mode,
        'duration_s': round(float(fix.get('seconds') or (profile or {}).get('duration_s') or 0.0), 2),
        'where': {'azimuth_deg': round(azimuth, 1), 'elevation_deg': round(elevation, 1),
                  'words': direction_words(azimuth, elevation),
                  'distance_m': round(distance, 2) if distance is not None else None,
                  'on_screen': 'spot' in view and 'off_screen' not in view,
                  'pixel': [int(v) for v in view['spot']] if 'spot' in view and 'off_screen' not in view else None,
                  'point_m': [round(float(c), 3) for c in point] if distance is not None else None,
                  'quality': (view.get('quality') or {}).get('state'),
                  'z': round(float(fix.get('z', 0.0)), 1)},
        'loudness': {'snr_db': round(snr, 1) if snr is not None else None, 'words': loudness_words(snr),
                     **({k: level[k] for k in ('rms_dbfs', 'peak_dbfs') if k in level} if level else {})},
        'acoustic': profile or {},
        'sound': {'label': None, 'description': None, 'confidence': None, 'source': None,
                  'local_label': None, 'local_similarity': None, 'verified': False},
        'speech': None, 'near': None, 'source': None, 'alerts': [],
    }
    if extra:
        record.update(extra)
    return record


def merge(base, patch):
    """Deep-merge patch into base (dicts merge, everything else replaces)."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


class Memory:
    """Thread-safe, append-only store of sound records, one JSONL file per day."""

    def __init__(self, folder=None, days=1, now=None):
        self.folder = Path(folder or MEMORY_DIR)
        self.lock = threading.Lock()
        self.records, self.index = [], {}
        self.load(days, now)

    def _file(self, epoch):
        return self.folder / f"{datetime.fromtimestamp(epoch).strftime('%Y-%m-%d')}.jsonl"

    def load(self, days=1, now=None):
        now = time.time() if now is None else now
        with self.lock:
            self.records, self.index = [], {}
            for back in range(days - 1, -1, -1):
                path = self._file(now - back * 86400)
                if not path.exists():
                    continue
                for line in path.read_text(encoding='utf-8').splitlines():
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if 'patch' in item and item.get('id') in self.index:
                        merge(self.index[item['id']], item['patch'])
                    elif 'epoch' in item and 'id' in item:
                        self.records.append(item)
                        self.index[item['id']] = item
        return list(self.records)

    def _write(self, epoch, item):
        self.folder.mkdir(parents=True, exist_ok=True)
        with open(self._file(epoch), 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False, default=_json_default) + '\n')

    def add(self, record):
        with self.lock:
            self.records.append(record)
            self.index[record['id']] = record
            self._write(record['epoch'], record)
        return record

    def update(self, record_id, patch):
        with self.lock:
            record = self.index.get(record_id)
            if record is None:
                return None
            merge(record, patch)
            self._write(record['epoch'], {'id': record_id, 'patch': patch})
            return record

    def get(self, record_id):
        with self.lock:
            return self.index.get(record_id)

    def recent(self, n=50, since=None):
        with self.lock:
            items = [r for r in self.records if since is None or r['epoch'] >= since]
        return items[-n:]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


# ---------------------------------------------------------------------------------------------
# Semantic phrases

def clock(epoch):
    return datetime.fromtimestamp(epoch).strftime('%I:%M:%S %p').lstrip('0')


def ago(epoch, now):
    s = max(0, now - epoch)
    if s < 60:
        return f'{s:.0f} s ago'
    if s < 3600:
        return f'{s / 60:.0f} min ago'
    if s < 86400:
        return f'{s / 3600:.1f} h ago'
    return f'{s / 86400:.0f} days ago'


def what_it_was(record):
    s = record.get('sound') or {}
    if s.get('label'):
        sure = s.get('confidence') or 0
        return s['label'] if sure >= 0.75 else f"probably {s['label']}"
    if s.get('local_label'):
        return f"sounds like your \"{s['local_label']}\" example"
    speech = record.get('speech') or {}
    if speech.get('text'):
        return 'speech'
    snr = (record.get('loudness') or {}).get('snr_db')
    if snr is not None and snr < 0:  # barely above the room: its shape is mostly noise
        return 'a faint sound'
    return (record.get('acoustic') or {}).get('character') or 'a sound'


def place_words(where):
    """'about 1.2 m away, to the right of the camera, above it' from a record's where-dict."""
    words = where.get('words') or 'unknown direction'
    side, _, vertical = words.partition(', ')
    if side in ('straight ahead', 'unknown direction'):
        place = side + (f', {vertical} the camera' if vertical else '')
    elif side.startswith('behind'):
        place = side + ' of the camera'
    else:
        place = f'{side} of the camera' + (f', {vertical} it' if vertical else '')
    if where.get('distance_m'):
        place = f"about {where['distance_m']:.1f} m away, {place}"
    return place


def phrase(record, now=None):
    """One sentence answering where, when, how loud, what, and what was near."""
    now = time.time() if now is None else now
    w, loud = record.get('where') or {}, record.get('loudness') or {}
    parts = [f"{what_it_was(record)[:1].upper()}{what_it_was(record)[1:]} at {clock(record['epoch'])} "
             f"({ago(record['epoch'], now)})"]
    parts.append(place_words(w))
    if loud.get('snr_db') is not None:
        parts.append(f"{loud['words']} ({loud['snr_db']:+.0f} dB over the room)")
    if record.get('duration_s'):
        parts.append(f"{record['duration_s']:.1f} s long")
    src = record.get('source') or {}
    obj = src.get('object', '')
    if obj and obj.lower() not in ('nothing distinct', 'nothing', 'unknown', ''):
        agree = {'yes': 'sound and sight agree', 'no': 'sound and sight disagree; the source may be hidden',
                 'unsure': 'unconfirmed'}.get(src.get('consistent'), 'unconfirmed')
        action = src.get('action', '')
        quiet = ('nothing', 'not checked', 'unknown', 'no before', 'none')
        seen = obj + (f' ({action})' if action and not any(w in action.lower() for w in quiet) else '')
        parts.append(f'at the spot: {seen}, {agree}')
    near = (record.get('near') or {}).get('text')
    if near and near.lower() not in obj.lower():
        parts.append(f'near {near}')
    text = ', '.join(parts) + '.'
    speech = record.get('speech') or {}
    if speech.get('text'):
        who = speech.get('speaker') or 'Someone'
        text += f' {who} said: "{speech["text"]}"'
    return text


# ---------------------------------------------------------------------------------------------
# Search

STOP_WORDS = set('a an the is was were did do does when where what who which how my me i it its at in on of to '
                 'for and or that this there any some sound sounds noise noises heard hear last time today ago '
                 'happened happen happening minute minutes min mins hour hours hr hrs second seconds sec secs day '
                 'days week yesterday past previous recent recently anything everything all show list tell give '
                 'event events morning afternoon tonight just'.split())
NUMBER_WORDS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'five': 5, 'ten': 10, 'few': 3, 'couple': 2}


def time_window(question, now=None):
    """(start, end) epoch from phrases like 'last 5 minutes', 'past hour', 'today', 'yesterday', or None."""
    now = time.time() if now is None else now
    q = question.lower()
    m = re.search(r'(?:last|past|previous)\s+(\d+|a|an|one|two|three|five|ten|few|couple(?: of)?)?\s*'
                  r'(second|sec|minute|min|hour|hr|day)s?', q)
    if m:
        count = m.group(1) or '1'
        count = NUMBER_WORDS.get(count.replace(' of', ''), None) if not count.isdigit() else int(count)
        unit = {'second': 1, 'sec': 1, 'minute': 60, 'min': 60, 'hour': 3600, 'hr': 3600, 'day': 86400}[m.group(2)]
        return now - (count or 1) * unit, now
    m = re.search(r'(\d+|a|an|one|two|three|five|ten|few|couple)\s+(second|minute|hour)s?\s+ago', q)
    if m:
        count = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        unit = {'second': 1, 'minute': 60, 'hour': 3600}[m.group(2)]
        return now - count * unit * 2 - 60, now
    midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
    if 'yesterday' in q:
        return (midnight - timedelta(days=1)).timestamp(), midnight.timestamp()
    if 'today' in q or 'this morning' in q or 'tonight' in q or 'this afternoon' in q:
        return midnight.timestamp(), now
    return None


def record_text(record):
    """Everything searchable about a record, lowercase."""
    s, sp, near = record.get('sound') or {}, record.get('speech') or {}, record.get('near') or {}
    src = record.get('source') or {}
    bits = [s.get('label'), s.get('description'), s.get('local_label'), (record.get('acoustic') or {}).get('character'),
            sp.get('text'), sp.get('speaker'), near.get('text'), (record.get('where') or {}).get('words'),
            (record.get('loudness') or {}).get('words'), src.get('object'), src.get('action'), src.get('summary'),
            ' '.join(record.get('alerts') or [])]
    return ' '.join(str(b) for b in bits if b).lower()


def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOP_WORDS and len(t) > 1]


def _stem(t):
    for suffix in ('ing', 'ed', 'es', 's'):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            return t[:-len(suffix)]
    return t


SYNONYMS = {'ring': {'ringtone', 'phone', 'bell', 'alarm', 'beep', 'tonal'}, 'phone': {'ring', 'ringtone', 'cell'},
            'talk': {'speech', 'said', 'speak', 'voice', 'speaker'}, 'speak': {'speech', 'talk', 'voice'},
            'clap': {'bursts', 'hit', 'sharp', 'clapping'}, 'knock': {'hit', 'sharp', 'bursts', 'tap'},
            'key': {'jingle', 'jingling', 'metal', 'bright'}, 'loud': {'very'}, 'alarm': {'beep', 'tonal', 'ring'},
            'beep': {'tonal', 'alarm'}, 'door': {'knock', 'slam'}, 'music': {'tonal', 'song'}}


def local_search(question, records, now=None, limit=5):
    """Keyword + synonym + recency ranking over records; works with no network at all."""
    now = time.time() if now is None else now
    window = time_window(question, now)
    pool = [r for r in records if window is None or window[0] <= r['epoch'] <= window[1]]
    want = {_stem(t) for t in _tokens(question)}
    for t in list(want):
        want |= {_stem(s) for s in SYNONYMS.get(t, ())}
    scored = []
    for r in pool:
        have = {_stem(t) for t in _tokens(record_text(r))}
        overlap = len(want & have)
        if want and overlap == 0:
            continue
        recency = math.exp(-(now - r['epoch']) / 3600)
        loud = 'loud' in want and (r.get('loudness') or {}).get('snr_db') or 0
        scored.append((overlap + 0.3 * recency + 0.01 * loud, r))
    if not want:  # "what happened in the last 5 minutes": everything in the window, newest first
        scored = [(r['epoch'], r) for r in pool]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [r for _, r in scored[:limit]]


def table(records, now=None):
    """Compact one-line-per-record text for Muse Spark (facts only, newest last)."""
    now = time.time() if now is None else now
    lines = []
    for r in records:
        w, loud, s = r.get('where') or {}, r.get('loudness') or {}, r.get('sound') or {}
        what = s.get('label') or (f"like example '{s['local_label']}'" if s.get('local_label') else '') or \
            (r.get('acoustic') or {}).get('character', '')
        conf = f" ({s['source']} {s['confidence']:.2f})" if s.get('label') and s.get('confidence') is not None else ''
        bits = [r['id'], datetime.fromtimestamp(r['epoch']).strftime('%a %H:%M:%S'), f'{what}{conf}',
                w.get('words', '?') + (f", {w['distance_m']:.1f} m" if w.get('distance_m') else ''),
                f"{loud.get('words', '?')} {loud['snr_db']:+.0f} dB" if loud.get('snr_db') is not None else loud.get('words', '?'),
                f"{r.get('duration_s', 0):.1f} s"]
        src = r.get('source') or {}
        if src.get('object'):
            bits.append(f"at spot: {src['object']}" + (f" ({src['action']})" if src.get('action') else '')
                        + f" [sound/sight agree: {src.get('consistent', 'unsure')}]")
        if (r.get('near') or {}).get('text'):
            bits.append('near: ' + r['near']['text'])
        if r.get('alerts'):
            bits.append('alert: ' + ', '.join(r['alerts']))
        sp = r.get('speech') or {}
        if sp.get('text'):
            bits.append(f"{sp.get('speaker') or 'someone'} said: \"{sp['text'][:160]}\"")
        lines.append(' | '.join(bits))
    return '\n'.join(lines)


def search(question, memory_or_records, meta=None, now=None, limit=5, max_rows=150):
    """Answer a question about past sounds with a grounded sentence.

    With meta (Muse Spark), the model reads a compact table of the records and picks the matching
    ids plus a short answer; ids it invents are dropped. Without meta (or if the call fails or the
    budget is used up), a local keyword search answers. Either way the matches are rendered from
    the stored facts by phrase(), so where/when/how loud never come from the model's imagination.
    """
    now = time.time() if now is None else now
    records = memory_or_records.recent(5000) if hasattr(memory_or_records, 'recent') else list(memory_or_records)
    if not records:
        return {'answer': 'No sounds recorded yet.', 'matches': [], 'source': 'local'}
    window = time_window(question, now)
    pool = [r for r in records if window is None or window[0] <= r['epoch'] <= window[1]] or records
    if meta is not None:
        try:
            rows = pool[-max_rows:]
            reply = meta.answer_query(question, table(rows, now),
                                      datetime.fromtimestamp(now).strftime('%A %Y-%m-%d %H:%M:%S'))
            by_id = {r['id']: r for r in rows}
            matches = [by_id[i] for i in reply.get('matches', []) if i in by_id][:limit]
            answer = (reply.get('answer') or '').strip()
            if matches or answer:
                if not answer:
                    answer = phrase(matches[0], now)
                return {'answer': answer, 'matches': matches, 'source': 'muse-spark'}
        except Exception as exc:  # any model/budget problem falls back to the free local search
            fallback = f' (Muse Spark unavailable: {exc})'
        else:
            fallback = ''
    else:
        fallback = ''
    matches = local_search(question, pool, now, limit)
    if not matches:
        return {'answer': 'No matching sound in memory' + (' for that time.' if window else '.') + fallback,
                'matches': [], 'source': 'local'}
    answer = phrase(matches[0], now)
    if len(matches) > 1:
        answer += f' ({len(matches) - 1} more match{"es" if len(matches) > 2 else ""}.)'
    return {'answer': answer + fallback, 'matches': matches, 'source': 'local'}


# ---------------------------------------------------------------------------------------------
# Your own example sounds: a free nearest-neighbour label, and a few-shot sheet for Muse Spark

def _mel_filters(n_fft=1024, n_mels=40, fs=FS, lo=80.0, hi=7600.0):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    points = inv(np.linspace(mel(lo), mel(hi), n_mels + 2))
    bins = np.floor((n_fft + 1) * points / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        fb[i, a:b] = (np.arange(a, b) - a) / max(1, b - a)
        fb[i, b:c] = (c - np.arange(b, c)) / max(1, c - b)
    return fb


_FB = _mel_filters()


def features(x, fs=FS):
    """Loudness-independent fingerprint: mean and spread of the log-mel spectrum over the active part."""
    x = np.asarray(x, float)
    n, hop = 1024, 256
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = x[np.arange(n) + hop * np.arange((len(x) - n) // hop + 1)[:, None]] * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    energy = power.sum(axis=1)
    keep = energy >= energy.max() * 0.05
    logmel = np.log(power[keep] @ _FB.T + 1e-12)
    logmel -= logmel.mean()  # removes overall level
    v = np.concatenate([logmel.mean(axis=0), logmel.std(axis=0)])
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def slug(label):
    return re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')[:40] or 'sound'


class ReferenceBank:
    """references/<label>/<n>.wav recorded with `python soundscope.py enroll "<label>"`."""

    def __init__(self, folder=None):
        self.folder = Path(folder or REFERENCES_DIR)
        self.items = []  # (label, path, feature vector)
        self.reload()

    def reload(self):
        self.items = []
        if not self.folder.exists():
            return self
        for sub in sorted(p for p in self.folder.iterdir() if p.is_dir()):
            label = (sub / 'label.txt').read_text(encoding='utf-8').strip() if (sub / 'label.txt').exists() else sub.name
            for wav in sorted(sub.glob('*.wav')):
                try:
                    self.items.append((label, wav, features(wav_samples(wav.read_bytes()))))
                except (OSError, wave.Error, EOFError, ValueError):
                    continue
        return self

    def labels(self):
        return sorted({label for label, _, _ in self.items})

    def add(self, label, wav):
        sub = self.folder / slug(label)
        sub.mkdir(parents=True, exist_ok=True)
        (sub / 'label.txt').write_text(label, encoding='utf-8')
        path = sub / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}.wav"
        path.write_bytes(wav)
        self.items.append((label, path, features(wav_samples(wav))))
        return path

    def nearest(self, x):
        """(label, cosine similarity in 0..1) of the closest example, or (None, 0.0)."""
        if not self.items:
            return None, 0.0
        f = features(x)
        best = max(self.items, key=lambda item: float(item[2] @ f))
        return best[0], round(max(0.0, float(best[2] @ f)), 3)

    def sheet(self, query_wav, max_labels=6, per_label=2):
        """One JPEG (data URL): a row of spectrograms per example label, and the unknown clip last.
        One picture keeps the few-shot comparison to a single image's worth of tokens."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        rows = []
        for label in self.labels()[:max_labels]:
            clips = [wav_samples(p.read_bytes()) for lab, p, _ in self.items if lab == label][:per_label]
            rows.append((label, clips))
        rows.append(('UNKNOWN (judge this one)', [wav_samples(query_wav)]))
        cols = max(len(c) for _, c in rows)
        fig, axes = plt.subplots(len(rows), cols, figsize=(2.6 * cols, 1.25 * len(rows)), dpi=80, squeeze=False)
        for r, (label, clips) in enumerate(rows):
            for c in range(cols):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])
                if c < len(clips):
                    ax.imshow(_spectrogram_db(clips[c]), origin='lower', aspect='auto', cmap='magma')
                if c == 0:
                    ax.set_ylabel(label[:22], rotation=0, ha='right', va='center', fontsize=8)
        fig.suptitle('Spectrograms (time ->, 0-8 kHz up). Rows above are labeled examples.', fontsize=8)
        buf = io.BytesIO()
        fig.savefig(buf, format='jpeg', bbox_inches='tight')
        plt.close(fig)
        import base64
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def _spectrogram_db(x, n=512, hop=128):
    x = np.asarray(x, float)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = x[np.arange(n) + hop * np.arange((len(x) - n) // hop + 1)[:, None]] * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    d = 10 * np.log10(power + power.max() * 1e-8)
    return np.maximum(d, d.max() - 80).T


# ---------------------------------------------------------------------------------------------
# Who is talking: speaker names that stay put by direction

class SpeakerMap:
    """Gives a persistent name to each talker by where they are: physics, not voice identity.

    Muse Voice Transcribe's speaker letters (A, B, ...) only hold within one request, so names
    across requests come from direction clusters (within tolerance_deg).
    """

    def __init__(self, tolerance_deg=20.0, forget_s=900.0):
        self.tolerance, self.forget = tolerance_deg, forget_s
        self.clusters = []  # [name, azimuth, last_seen]
        self.count = 0

    def name_for(self, azimuth_deg, now=None):
        now = time.time() if now is None else now
        self.clusters = [c for c in self.clusters if now - c[2] <= self.forget]
        if azimuth_deg is None:
            return 'Someone'
        near = min(self.clusters, key=lambda c: abs(c[1] - azimuth_deg), default=None)
        if near is not None and abs(near[1] - azimuth_deg) <= self.tolerance:
            near[1] = 0.7 * near[1] + 0.3 * azimuth_deg
            near[2] = now
            return near[0]
        self.count += 1
        name = f'Speaker {self.count}'
        self.clusters.append([name, azimuth_deg, now])
        return name

    def summary(self, now=None):
        now = time.time() if now is None else now
        live = [c for c in self.clusters if now - c[2] <= self.forget]
        return [(c[0], direction_words(c[1]), ago(c[2], now)) for c in sorted(live, key=lambda c: c[1])]


def turn_directions(turns, track, clip_end_at, clip_seconds):
    """Median camera azimuth heard during each diarized turn.

    turns: [{'start_s', 'end_s', ...}] relative to the clip start. track: [(capture time, azimuth,
    elevation)] from the localizer's running updates. Returns one azimuth (or None) per turn.
    """
    start = clip_end_at - clip_seconds
    out = []
    for t in turns:
        a, b = start + t['start_s'] - 0.15, start + t['end_s'] + 0.15
        hits = [az for when, az, _ in track if a <= when <= b and az is not None]
        out.append(float(np.median(hits)) if hits else None)
    return out
