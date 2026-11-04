"""Background enrichment: adds Meta model results to sound records without slowing the viewer.

One worker thread sends one request at a time, under rate limits and the Meta budget. The
localizer has already decided WHERE each sound is; these steps only add what, who and what's near:

  label     free: nearest of your own recorded examples (references/, `soundscope.py enroll`)
  source    identify + context together, for a sound the camera can see: ONE Muse Spark call gets
            the spectrogram and camera crops of the spot the mics picked (before and after the
            sound) and names the sound, the object at the spot, what visibly happened there, and
            whether sound and sight agree ("sound object detection" with physics choosing the spot)
  identify  Muse Spark labels the sound from its spectrogram alone (off screen, or no context)
  captions  Muse Voice Transcribe (DIARIZATION) writes speech down; each turn's direction names
            the talker ("Speaker 1, to the left"), because the model's speaker letters reset per clip
  context   Muse Spark names what is near the spot the sound picked, from a camera crop
  outline   SAM 3.1 outlines the object at that spot (point prompt), for a found target
  watch     sounds you asked to be told about ("tell me when the kettle whistles") are matched in the
            same identify/source call, or for free against your own examples, and raise an alert

Results go into the sound memory and to on_result(record_id, kind, payload) for the screen.
"""
import queue
import threading
import time

from meta_api import BudgetError, MetaError
from sound_memory import SPEECH_SCORE, SpeakerMap, direction_words, turn_directions, wav_samples

FEATURES = ('identify', 'captions', 'context', 'outline')
# Sound-awareness mode (--assist): sounds people most need to know about when they cannot hear them,
# or cannot tell where they came from. Matched by Muse Spark (and for free by your own examples).
ASSIST_WATCH = ('smoke alarm', 'doorbell', 'knock on the door', 'phone ringing', 'baby crying',
                'kettle whistling')


class RateLimit:
    """At most `per_minute` events per minute and one every `gap_s` seconds."""

    def __init__(self, gap_s, per_minute, clock=time.monotonic):
        self.gap, self.per_minute, self.clock, self.times = gap_s, per_minute, clock, []

    def allow(self):
        now = self.clock()
        self.times = [t for t in self.times if now - t < 60]
        if (self.times and now - self.times[-1] < self.gap) or len(self.times) >= self.per_minute:
            return False
        self.times.append(now)
        return True


class Enricher(threading.Thread):
    def __init__(self, memory, meta, features, on_result=None, references=None, speakers=None,
                 identify_limit=(3.0, 10), caption_limit=(1.0, 20), context_limit=(5.0, 6), outline_limit=(10.0, 3),
                 source_limit=(4.0, 8), context_cell_m=0.25, context_ttl_s=600.0, queue_size=6,
                 clock=time.monotonic):
        super().__init__(daemon=True)
        self.memory, self.meta = memory, meta
        self.features = set(features) & set(FEATURES)
        self.on_result = on_result or (lambda *_: None)
        self.references, self.speakers = references, speakers or SpeakerMap()
        self.limits = {'identify': RateLimit(*identify_limit, clock=clock),
                       'captions': RateLimit(*caption_limit, clock=clock),
                       'context': RateLimit(*context_limit, clock=clock),
                       'outline': RateLimit(*outline_limit, clock=clock)}
        self.cell, self.ttl, self.clock = context_cell_m, context_ttl_s, clock
        self.context_cache = {}  # cell -> (time, patch)
        self.tasks = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.exhausted = None    # budget message once the budget is used up
        self.last_error = 0.0
        self.watch = []          # sounds to alert on; replaced (not mutated) by the viewer
        self.max_age_s = 45.0    # older sounds get no Meta request
        self.limits['source'] = RateLimit(*source_limit, clock=clock)

    # -- queueing --------------------------------------------------------------------------------

    def submit(self, task):
        """Never blocks the caller; when full, the oldest waiting task is dropped."""
        while True:
            try:
                self.tasks.put_nowait(task)
                return
            except queue.Full:
                try:
                    self.tasks.get_nowait()
                except queue.Empty:
                    pass

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process(task)
            except Exception as exc:  # one bad sound must never kill the worker
                self.on_result(task['record']['id'], 'notice', f'Enrichment failed: {exc}')

    # -- steps -----------------------------------------------------------------------------------

    def _meta_ok(self, step, needs=None):
        needs = needs or (step,)
        return (self.meta is not None and all(n in self.features for n in needs) and self.exhausted is None
                and self.limits[step].allow())

    def _alerts(self, rid, items, how):
        for item in items:
            self.on_result(rid, 'alert', {'watch': item, 'how': how})

    def _call(self, record_id, fn, *args):
        try:
            return fn(*args)
        except BudgetError as exc:
            self.exhausted = str(exc)
            self.on_result(record_id, 'notice', f'Meta features paused: {exc}')
        except MetaError as exc:
            if self.clock() - self.last_error > 10:
                self.last_error = self.clock()
                self.on_result(record_id, 'notice', f'Meta call failed: {exc}')
        return None

    def _update(self, record_id, patch, kind, payload=None):
        record = self.memory.update(record_id, patch) if self.memory is not None else None
        self.on_result(record_id, kind, payload if payload is not None else (record or patch))

    def process(self, task):
        record = task['record']
        rid = record['id']
        # A backlog formed (many sounds at once): old sounds keep their free label only.
        stale = time.time() - record.get('epoch', time.time()) > self.max_age_s
        acoustic = record.get('acoustic') or {}
        wav = task.get('wav')
        speechy = acoustic.get('speech_score', 0) >= SPEECH_SCORE and record.get('duration_s', 0) >= 0.5

        watch = list(self.watch)
        if wav and self.references is not None and self.references.items:
            label, similarity = self.references.nearest(wav_samples(wav))
            if label is not None and similarity >= 0.8:
                record.setdefault('sound', {}).update(local_label=label, local_similarity=similarity)
                self._update(rid, {'sound': {'local_label': label, 'local_similarity': similarity}}, 'label')
                if similarity >= 0.9:
                    self._alerts(rid, [w for w in watch if _same_sound(w, label)], 'your example')
        if stale:
            return

        strong = (record.get('where') or {}).get('quality') in ('candidate', 'ambiguous') or \
            ((record.get('loudness') or {}).get('snr_db') or 0) >= 6
        trusted_local = ((record.get('sound') or {}).get('local_similarity') or 0) >= 0.92
        facts = {k: acoustic.get(k) for k in ('duration_s', 'dominant_hz', 'centroid_hz', 'onsets', 'character')}
        sheet = (self.references.sheet(wav) if wav and self.references is not None and self.references.items
                 else None)
        crop = task.get('crop_jpeg')
        checked = False
        if (wav and crop is not None and strong and not speechy and (not trusted_local or watch)
                and self._meta_ok('source', ('identify', 'context'))):
            reply = self._call(rid, self.meta.source_check, wav, crop, task.get('crop_before'), facts, watch, sheet,
                               task.get('crop_kind', 'color'))
            if reply:
                checked = True
                surface = _bare(reply['surface'])
                near = reply['near'].strip()
                patch = {'sound': {'label': reply['sound'], 'confidence': round(reply['sound_confidence'], 2),
                                   'source': 'muse-spark audio+visual'},
                         'source': {'object': reply['object'], 'action': reply['action'],
                                    'consistent': reply['consistent'], 'summary': reply['summary'],
                                    'seen_with': task.get('crop_kind', 'color')},
                         'near': {'text': near + (f' on the {surface}' if surface and surface.lower() != 'unknown'
                                                  and surface.lower() not in near.lower() else ''),
                                  'surface': surface, 'source': 'muse-spark'}}
                self.context_cache[self._cell(record)] = (self.clock(), {'near': patch['near']})
                self._update(rid, patch, 'source')
                self._alerts(rid, reply.get('watch') or [], 'Muse Spark')
                if reply['consistent'] == 'yes' and task.get('frame_jpeg') is not None:
                    task['outline'] = True  # sound and sight agree: outline that object (SAM 3.1)
        if (wav and strong and not speechy and not checked and (not trusted_local or watch)
                and self._meta_ok('identify')):
            reply = self._call(rid, self.meta.classify_sound, wav, facts, sheet, watch)
            if reply:
                self._update(rid, {'sound': {'label': reply['label'], 'category': reply['category'],
                                             'description': reply['description'],
                                             'confidence': round(reply['confidence'], 2), 'source': 'muse-spark'}},
                             'identify')
                self._alerts(rid, reply.get('watch') or [], 'Muse Spark')

        if speechy and self._meta_ok('captions'):
            speech_wav = task.get('speech_wav') or wav
            reply = self._call(rid, self.meta.transcribe_turns, speech_wav) if speech_wav else None
            if reply and reply['turns']:
                turns = reply['turns']
                azimuths = turn_directions(turns, task.get('track') or [], task.get('clip_end_at', 0.0),
                                           task.get('clip_seconds', 0.0))
                fallback = (record.get('where') or {}).get('azimuth_deg')
                out = []
                for turn, az in zip(turns, azimuths):
                    az = fallback if az is None else az
                    out.append({**turn, 'azimuth_deg': None if az is None else round(az, 1),
                                'name': self.speakers.name_for(az), 'where': direction_words(az)})
                main = max(out, key=lambda t: t['end_s'] - t['start_s'])
                patch = {'speech': {'text': reply['text'], 'speaker': main['name'], 'turns': out},
                         'sound': {'label': 'speech', 'category': 'speech', 'confidence': 0.9,
                                   'source': 'muse-voice-transcribe'}}
                self._update(rid, patch, 'captions', out)

        # Context alone only when the combined source check is not in use (no double spending), and
        # never for speech: the caption says enough, and the crop would show the person talking.
        if (crop is not None and 'context' in self.features and 'identify' not in self.features
                and not checked and not speechy):
            key = self._cell(record)
            cached = self.context_cache.get(key)
            if cached and self.clock() - cached[0] <= self.ttl:
                self._update(rid, cached[1], 'context')
            elif self._meta_ok('context'):
                reply = self._call(rid, self.meta.describe_near, crop, task.get('crop_kind', 'color'))
                if reply and reply.get('near'):
                    surface = _bare(reply.get('surface', ''))
                    text = reply['near'].strip() + (f' on the {surface}' if surface and surface.lower() != 'unknown'
                                                    and surface.lower() not in reply['near'].lower() else '')
                    patch = {'near': {'text': text, 'surface': surface, 'source': 'muse-spark'}}
                    self.context_cache[key] = (self.clock(), patch)
                    self._update(rid, patch, 'context')

        if task.get('outline') and task.get('frame_jpeg') is not None and self._meta_ok('outline'):
            w, h = task['frame_size']
            u, v = task['pixel']
            polygons = self._call(rid, self.meta.segment_at, task['frame_jpeg'], w, h, u, v)
            if polygons:
                self._update(rid, {'outline': {'source': 'sam-3.1', 'polygons': len(polygons)}}, 'outline',
                             {'polygons': polygons, 'scale': task.get('scale', 1.0)})

    def _cell(self, record):
        """Where a context answer can be reused: a 25 cm cell, else a pixel or direction bin."""
        w = record.get('where') or {}
        point = w.get('point_m')
        if point:
            return tuple(round(c / self.cell) for c in point)
        if w.get('pixel'):
            return ('px', w['pixel'][0] // 64, w['pixel'][1] // 64)
        return ('dir', round((w.get('azimuth_deg') or 0) / 10), round((w.get('elevation_deg') or 0) / 10))


def _bare(noun):
    """'a wall' -> 'wall', so it reads 'on the wall'."""
    words = noun.strip().split(' ', 1)
    return words[1] if len(words) == 2 and words[0].lower() in ('a', 'an', 'the') else noun.strip()


def _same_sound(watch_item, label):
    """Loose match between a watched sound and a label: shared content words ("kettle whistling" ~ "kettle")."""
    stop = {'a', 'an', 'the', 'my', 'is', 'sound', 'of', 'when', 'if'}
    a = {w.rstrip('s') for w in watch_item.lower().split() if w not in stop}
    b = {w.rstrip('s') for w in label.lower().split() if w not in stop}
    return bool(a & b)
