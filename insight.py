"""Background enrichment: adds Meta model results to sound records without slowing the viewer.

One worker thread sends one request at a time, under rate limits and the Meta budget. The
localizer supplies a candidate direction or surface; these steps add what, who and what's near.
An object remains an unverified candidate even when a model agrees with it.

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
import io
import math
import os
import re
import threading
import time
import wave

from meta_api import BudgetError, MetaError
from sound_memory import SPEECH_SCORE, SpeakerMap, direction_words, turn_directions, wav_samples

FEATURES = ('identify', 'captions', 'context', 'outline')
# Sound-awareness mode (--assist): sounds people most need to know about when they cannot hear them,
# or cannot tell where they came from. Matched by Muse Spark (and for free by your own examples).
ASSIST_WATCH = ('smoke alarm', 'doorbell', 'knock on the door', 'phone ringing', 'baby crying',
                'kettle whistling')
# Alerts must be worth interrupting someone for. Live test, Nov 4: every false alert (typing, a clap
# read as a knock, a hum read as a phone) was under +4 dB over the room; the real ring and knock were
# +14 and +19 dB. So: clearly audible, a reasonably sure label, and the same alert at most every 20 s.
# Live, Nov 4 evening: loud claps, snaps and whistles (+20 to +30 dB) came back as "knock" or "smoke alarm"
# at 0.4-0.6 and raised false alerts, so an alert now needs a label at least 75% sure (65% when very loud).
ALERT_MIN_SNR_DB, ALERT_MIN_CONFIDENCE, ALERT_REPEAT_S = 6.0, 0.75, 20.0
# A very clear sound (20 dB or more over the room) alerts with a less sure label: missing an alarm is worse
# than a false alarm. Live, Nov 4: a +33 dB ring was labeled "phone ringing" at only 0.4.
ALERT_LOUD_SNR_DB, ALERT_LOUD_CONFIDENCE = 20.0, 0.65
META_MIN_SNR_DB = 3.0  # quieter sounds keep their free measurements but get no Meta label
# A spend-control heuristic for this recording gain, NOT calibrated dB SPL.
# Quiet recordings remain available locally; explicit analysis can override this
# digital floor, but never spatial-evidence, rate, or budget checks.
# The XVF3800's raw mics record quietly: at 1 m a whistle peaked at -60 dBFS, a hum at -66, snaps and
# claps at -65 to -78 (Nov 4). A -50 dBFS floor blocked nearly every real sound, speech included,
# from automatic labels and captions. This floor only rejects near-silent glitches now; the
# over-the-room SNR gates (META_MIN_SNR_DB) decide what is worth a request.
AUTO_META_MIN_PEAK_DBFS = -80.0
_ACTIVITY_LABELS = {
    'source_check': 'Muse Spark: sound and scene',
    'classify_sound': 'Muse Spark: sound label',
    'transcribe_turns': 'Muse Voice: transcript',
    'describe_near': 'Muse Spark: scene context',
    'segment_at': 'SAM 3.1: outline',
}
_STEP_OPERATIONS = {'source': 'source_check', 'identify': 'classify_sound',
                    'captions': 'transcribe_turns', 'context': 'describe_near', 'outline': 'segment_at'}
PEOPLE = re.compile(r'\b(person|people|face|mouth|lips|head|man|woman|child|kid|speech|speaking|talking|voice)\b',
                    re.IGNORECASE)


def likely_speech(acoustic):
    """Route ambiguous narrow-band beeps to sound identification, not ASR alone.

    A modulated tone can have a syllable-rate envelope and score as speech.
    Dominant tonal energy plus low spectral flatness is contrary evidence;
    this heuristic chooses an analysis path, never a verified sound label.
    Missing/invalid tonal measurements do not override a speech indication.
    """
    try:
        score = float(acoustic.get('speech_score', 0))
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(score) or score < SPEECH_SCORE:
        return False
    try:
        share, flatness = float(acoustic.get('tone_share')), float(acoustic.get('flatness'))
        concentrated_tone = (math.isfinite(share) and math.isfinite(flatness)
                             and 0.60 <= share <= 1.0 and 0.0 <= flatness <= 0.05)
    except (TypeError, ValueError, OverflowError):
        concentrated_tone = False
    return not concentrated_tone


class RateLimit:
    """At most `per_minute` events per minute and one every `gap_s` seconds."""

    def __init__(self, gap_s, per_minute, clock=time.monotonic):
        self.gap, self.per_minute, self.clock, self.times = gap_s, per_minute, clock, []
        self.lock = threading.Lock()

    def allow(self):
        with self.lock:  # shared by the worker threads
            now = self.clock()
            self.times = [t for t in self.times if now - t < 60]
            if (self.times and now - self.times[-1] < self.gap) or len(self.times) >= self.per_minute:
                return False
            self.times.append(now)
            return True


class NewestFirst:
    """Bounded task stack: the newest sound is handled first; when full, the oldest is dropped."""

    def __init__(self, size):
        self.items, self.size, self.ready = [], size, threading.Condition()

    def put(self, item):
        with self.ready:
            self.items.append(item)
            del self.items[:-self.size]
            self.ready.notify()

    def get(self, timeout):
        with self.ready:
            if not self.items and not self.ready.wait(timeout):
                return None
            return self.items.pop() if self.items else None

    def qsize(self):
        with self.ready:
            return len(self.items)


class Enricher(threading.Thread):
    def __init__(self, memory, meta, features, on_result=None, references=None, speakers=None,
                 identify_limit=(3.0, 10), caption_limit=(1.0, 20), context_limit=(5.0, 6), outline_limit=(10.0, 3),
                 source_limit=(2.0, 12), context_cell_m=0.25, context_ttl_s=600.0, queue_size=6,
                 workers=3, clock=time.monotonic):
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
        self.tasks = NewestFirst(queue_size)
        self.workers = max(1, workers)  # Muse Spark takes 3-10 s; one at a time fell 40 s behind live
        self.lock = threading.Lock()
        self.listen_for_names = False   # set when someone's name should raise an alert
        self.stop_event = threading.Event()
        self.exhausted = None    # budget message once the budget is used up
        self.last_error = 0.0
        self.watch = []          # sounds to alert on; replaced (not mutated) by the viewer
        self.max_age_s = 45.0    # older sounds get no Meta request
        self.alerted = {}        # watched sound -> when it last raised an alert
        self._activity_lock = threading.Lock()
        self._activity_active = {}
        self._activity_counts = {key: 0 for key in ('started', 'completed', 'empty', 'errors', 'blocked', 'skipped')}
        self._activity_last = {'state': 'idle', 'operation': None, 'label': 'Meta analysis',
                               'message': 'Waiting for an eligible sound.', 'at': self.clock()}
        self.limits['source'] = RateLimit(*source_limit, clock=clock)
        try:
            self.auto_meta_min_peak_dbfs = float(os.environ.get(
                'SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS', AUTO_META_MIN_PEAK_DBFS))
        except (TypeError, ValueError) as exc:
            raise ValueError('SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS must be a finite number between -160 and 0.') from exc
        if not math.isfinite(self.auto_meta_min_peak_dbfs) or not -160 <= self.auto_meta_min_peak_dbfs <= 0:
            raise ValueError('SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS must be a finite number between -160 and 0.')

    # -- queueing --------------------------------------------------------------------------------

    def submit(self, task):
        """Never blocks the caller; when full, the oldest waiting task is dropped."""
        self.tasks.put(task)

    def activity_status(self):
        """Cheap, thread-safe UI snapshot. No request, ledger read or callback.

        `started` counts attempted model-method calls, including attempts the
        budget refuses before HTTP; it is not a billing/request-success count.
        Concurrent operations appear separately even when they use one model.
        """
        queued = self.tasks.qsize()
        with self._activity_lock:
            operations = list(self._activity_active.values())
            return {'queued': queued, 'active_operations': operations,
                    'active': [_ACTIVITY_LABELS.get(op, 'Meta analysis') for op in operations],
                    **self._activity_counts, 'last': dict(self._activity_last),
                    'budget_paused': self.exhausted is not None}

    def _activity_note(self, state, operation, message, token=None):
        operation = _STEP_OPERATIONS.get(operation, operation)
        counter = {'complete': 'completed', 'empty': 'empty', 'error': 'errors',
                   'blocked': 'blocked', 'skipped': 'skipped'}.get(state)
        with self._activity_lock:
            if token is not None:
                self._activity_active.pop(token, None)
            if counter:
                self._activity_counts[counter] += 1
            self._activity_last = {'state': state, 'operation': operation,
                                   'label': _ACTIVITY_LABELS.get(operation, 'Meta analysis'),
                                   'message': str(message)[:320], 'at': self.clock()}

    def _activity_begin(self, fn):
        operation = getattr(fn, '__name__', None)
        if not isinstance(operation, str) or operation not in _ACTIVITY_LABELS:
            operation = next((name for name in _ACTIVITY_LABELS
                              if fn is getattr(self.meta, name, None)), 'analysis')
        with self._activity_lock:
            self._activity_counts['started'] += 1
            token = self._activity_counts['started']
            self._activity_active[token] = operation
            label = _ACTIVITY_LABELS.get(operation, 'Meta analysis')
            self._activity_last = {'state': 'running', 'operation': operation, 'label': label,
                                   'message': label + ' in progress.', 'at': self.clock()}
        return token, operation

    @staticmethod
    def _activity_has_result(operation, reply):
        if operation == 'transcribe_turns':
            return isinstance(reply, dict) and any(
                isinstance(turn, dict) and isinstance(turn.get('text'), str) and turn['text'].strip()
                for turn in reply.get('turns') or ())
        key = {'source_check': 'sound', 'classify_sound': 'label', 'describe_near': 'near'}.get(operation)
        if key:
            return isinstance(reply, dict) and isinstance(reply.get(key), str) and bool(reply[key].strip())
        if operation == 'segment_at':
            return reply is not None and len(reply) > 0
        return reply is not None

    def automatic_audio_eligible(self, record):
        """Pure spend-policy check using the raw input peak, never normalised WAV samples.

        This recording-level heuristic is gain dependent and is not a dB SPL
        threshold. No request flag can override this automatic-only predicate.
        """
        try:
            raw_peak = (record.get('loudness') or {}).get('peak_dbfs')
            peak = float(raw_peak)
            return (not isinstance(raw_peak, bool) and math.isfinite(peak)
                    and peak >= self.auto_meta_min_peak_dbfs)
        except (TypeError, ValueError, OverflowError, AttributeError):
            return False

    def automatic_media_eligible(self, record):
        """Pure preflight for preparing automatic crops, face blur and JPEGs.

        Use before expensive image preparation; the worker repeats these checks
        before spending. Explicit historical requests use their separate path.
        """
        return self.automatic_audio_eligible(record) and usable_visual_evidence(record)

    def automatic_caption_eligible(self, record, speech_wav):
        """Allow only processed-speech ASR when a low-gain raw input fails the peak floor.

        The XVF3800 speech output has different gain/processing from its raw
        microphone channels. A raw dBFS floor must not veto otherwise audible
        speech on that separate channel. Require the existing speech/tone test,
        a finished >=0.5 s event, finite raw level above digital silence, and at
        least 3 dB event SNR over the learned room. Rates and budget still apply.
        This never enables sound classification, camera context, or segmentation.
        """
        if not isinstance(speech_wav, (bytes, bytearray)) or not speech_wav:
            return False
        try:
            loud = record.get('loudness') or {}
            values = (record.get('duration_s'), loud.get('snr_db'), loud.get('peak_dbfs'))
            if any(isinstance(value, bool) for value in values):
                return False
            duration, snr, peak = (float(value) for value in values)
            valid = (all(math.isfinite(value) for value in (duration, snr, peak))
                     and duration >= 0.5 and snr >= META_MIN_SNR_DB and -120.0 < peak <= 0.0
                     and likely_speech(record.get('acoustic') or {}))
            if not valid:
                return False
            # Inspect the real processed clip, not its normalized peak as a
            # loudness estimate. Reject missing/truncated/digital-silence audio.
            with wave.open(io.BytesIO(speech_wav)) as stream:
                frames = stream.getnframes()
                if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000) \
                        or not 8000 <= frames <= 160000:
                    return False
                if len(stream.readframes(frames)) != frames * 2:
                    return False
            samples = wav_samples(speech_wav)
            return bool(samples.size and samples.max() > samples.min())
        except (TypeError, ValueError, OverflowError, AttributeError, wave.Error, EOFError):
            return False

    def start(self):
        super().start()
        for _ in range(self.workers - 1):
            threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            task = self.tasks.get(timeout=0.2)
            if task is None:
                continue
            try:
                self.process(task)
            except Exception as exc:  # one bad sound must never kill the worker
                self.on_result(task['record']['id'], 'notice', f'Enrichment failed: {exc}')
            finally:
                if task.get('explicit') is True:
                    self.on_result(task['record']['id'], 'analysis_status',
                                   task.get('_analysis_status') or
                                   'Saved-event analysis finished. Review the available label or transcript; '
                                   'unclear evidence may remain unlabelled.')

    # -- steps -----------------------------------------------------------------------------------

    def _meta_ok(self, step, needs=None, on_block=None):
        needs = needs or (step,)
        if self.meta is None:
            reason = 'Meta is not configured.'
        elif not all(n in self.features for n in needs):
            reason = f'The {" + ".join(needs)} feature is not enabled.'
        elif self.exhausted is not None:
            reason = f'The estimated budget is paused: {self.exhausted}'
        elif not self.limits[step].allow():
            reason = f'The {step} rate limit is active; wait before requesting this event again.'
        else:
            return True
        self._activity_note('blocked', step, reason)
        if on_block is not None:
            on_block(reason)
        return False

    def _alerts(self, rid, record, items, how, confidence=None, label=None):
        """Raise alerts for watched sounds, only when they are clearly audible and reasonably sure.
        label: what the model called the sound; a label naming a watched sound counts even when the
        model did not list it as a watch hit ("doorbell chime" -> doorbell)."""
        snr = (record.get('loudness') or {}).get('snr_db') or 0.0
        if snr < (ALERT_MIN_SNR_DB if how == 'Muse Spark' else ALERT_MIN_SNR_DB / 2):
            return
        needed = ALERT_LOUD_CONFIDENCE if snr >= ALERT_LOUD_SNR_DB else ALERT_MIN_CONFIDENCE
        if confidence is not None and confidence < needed:
            return
        items = list(items)
        if label:
            items += [w for w in self.watch if _same_sound(w, label) and w not in items]
        with self.lock:
            now = self.clock()
            fresh = [item for item in items if now - self.alerted.get(item.lower(), -1e9) >= ALERT_REPEAT_S]
            for item in fresh:
                self.alerted[item.lower()] = now
        for item in fresh:
            self.on_result(rid, 'alert', {'watch': item, 'how': how})

    def _call(self, record_id, fn, *args):
        token, operation = self._activity_begin(fn)
        try:
            reply = fn(*args)
            usable = self._activity_has_result(operation, reply)
            message = ('Result received.' if usable else
                       'No speech was transcribed.' if operation == 'transcribe_turns' else
                       'No usable result was returned.')
            self._activity_note('complete' if usable else 'empty', operation, message, token)
            return reply
        except BudgetError as exc:
            self.exhausted = str(exc)
            self._activity_note('blocked', operation, f'Estimated budget paused: {exc}', token)
            self.on_result(record_id, 'notice', f'Meta features paused: {exc}')
        except MetaError as exc:
            self._activity_note('error', operation, str(exc), token)
            if self.clock() - self.last_error > 10:
                self.last_error = self.clock()
                self.on_result(record_id, 'notice', f'Meta call failed: {exc}')
        except Exception as exc:
            self._activity_note('error', operation, str(exc), token)
            raise
        return None

    def _update(self, record_id, patch, kind, payload=None):
        record = self.memory.update(record_id, patch) if self.memory is not None else None
        self.on_result(record_id, kind, payload if payload is not None else (record or patch))

    def process(self, task):
        record = task['record']
        rid = record['id']
        # A backlog formed (many sounds at once): old sounds keep their free label only.
        now = time.time()
        explicit = task.get('explicit') is True

        def manual_notice(message):
            message = 'Saved-event analysis: ' + message
            task['_analysis_status'] = message
            self.on_result(rid, 'notice', message)

        # User-requested historical analysis has a fresh request time, while the
        # original event timestamp remains untouched. Queued requests still expire.
        request_at = task.get('requested_at') if explicit else record.get('epoch', now)
        try:
            age = now - float(request_at)
            stale = not math.isfinite(age) or age < -1.0 or age > self.max_age_s
        except (TypeError, ValueError, OverflowError):
            stale = True
        acoustic = record.get('acoustic') or {}
        wav = task.get('wav')
        # Preserve the broader possible-speech flag for camera/privacy checks.
        # Tonal ambiguity only changes audio analysis routing.
        speech_like = acoustic.get('speech_score', 0) >= SPEECH_SCORE
        speech_likely = likely_speech(acoustic)
        speechy = speech_likely and record.get('duration_s', 0) >= 0.5  # long enough to caption

        watch = list(self.watch)
        if wav and self.references is not None and self.references.items:
            label, similarity = self.references.nearest(wav_samples(wav))
            if label is not None and similarity >= 0.8:
                record.setdefault('sound', {}).update(local_label=label, local_similarity=similarity)
                self._update(rid, {'sound': {'local_label': label, 'local_similarity': similarity}}, 'label')
                if similarity >= 0.9:
                    self._alerts(rid, record, [w for w in watch if _same_sound(w, label)], 'your example')
        if stale:
            self._activity_note('skipped', 'analysis', 'Analysis skipped: request is stale or its time is invalid.')
            if explicit:
                manual_notice('request expired or its request time is invalid; press A again.')
            return

        audio_allowed = explicit or self.automatic_audio_eligible(record)
        caption_only = (not audio_allowed and
                        self.automatic_caption_eligible(record, task.get('speech_wav')))
        if not audio_allowed and not caption_only:
            self._activity_note('skipped', 'analysis',
                                'Automatic analysis skipped: raw recording peak is below the configured floor '
                                'or its level is unavailable. Saved-event analysis is still available.')
            return  # free matching and local memory remain available
        if caption_only:
            self._activity_note('eligible', 'transcribe_turns',
                                'Processed speech is above the room noise: captions only. '
                                'Other automatic models remain blocked by the raw recording-level floor.')

        visual = usable_visual_evidence(record)

        # Worth a Meta label: at least a little above the room, and clearly located or clearly loud.
        # Live, Nov 4: blips at -2 dB came back as "knock on the door" and only cost credits.
        snr = (record.get('loudness') or {}).get('snr_db') or 0
        strong = snr >= META_MIN_SNR_DB and (
            (record.get('where') or {}).get('quality') in ('candidate', 'ambiguous') or snr >= 6)
        trusted_local = ((record.get('sound') or {}).get('local_similarity') or 0) >= 0.92
        # Send measurements rather than a heuristic category such as "speech-like",
        # which otherwise primes the model to repeat the routing mistake.
        facts = {k: acoustic.get(k) for k in ('duration_s', 'dominant_hz', 'second_hz', 'centroid_hz',
                                            'flatness', 'tone_share', 'voice_band', 'onsets',
                                            'attack_ms', 'crest_db')}
        sheet = (self.references.sheet(wav) if wav and self.references is not None and self.references.items
                 else None)
        crop = task.get('crop_jpeg')
        checked = False
        if (audio_allowed and wav and crop is not None and visual and strong and not speech_like and (not trusted_local or watch)
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
                self._alerts(rid, record, reply.get('watch') or [], 'Muse Spark', reply['sound_confidence'],
                             reply['sound'])
                person = PEOPLE.search(f"{reply['object']} {reply['sound']}")
                if reply['consistent'] == 'yes' and task.get('frame_jpeg') is not None and not person:
                    task['outline'] = True  # sound and sight agree: outline that object (SAM 3.1), never a person
        if (audio_allowed and not explicit and wav and strong and not speech_likely and not checked and (not trusted_local or watch)
                and self._meta_ok('identify')):
            reply = self._call(rid, self.meta.classify_sound, wav, facts, sheet, watch)
            if reply:
                self._update(rid, {'sound': {'label': reply['label'], 'category': reply['category'],
                                             'description': reply['description'],
                                             'confidence': round(reply['confidence'], 2), 'source': 'muse-spark'}},
                             'identify')
                self._alerts(rid, record, reply.get('watch') or [], 'Muse Spark', reply['confidence'],
                             reply['label'])

        # Listening for a name: also transcribe longer sounds that did not look like speech (a snap or
        # a door at the start can hide a sentence; live, Nov 4). A caption costs about $0.0003.
        long_enough = self.listen_for_names and record.get('duration_s', 0) >= 1.0
        captioned = False
        if (speechy or long_enough or (explicit and speech_likely and wav)) and self._meta_ok(
                'captions', on_block=manual_notice if explicit else None):
            speech_wav = task.get('speech_wav') or wav
            reply = self._call(rid, self.meta.transcribe_turns, speech_wav) if speech_wav else None
            turns = (reply.get('turns') or []) if reply else []
            if explicit:
                # An empty ASR response is not evidence that a speech-like ring
                # was speech. Keep the fallback bounded to one audio label call.
                turns = [turn for turn in turns if isinstance(turn, dict)
                         and isinstance(turn.get('text'), str) and turn['text'].strip()]
            if turns:
                captioned = True
                azimuths = turn_directions(turns, task.get('track') or [], task.get('clip_end_at', 0.0),
                                           task.get('clip_seconds', 0.0))
                fallback = (record.get('where') or {}).get('azimuth_deg')
                out = []
                for turn, az in zip(turns, azimuths):
                    az = fallback if az is None else az
                    out.append({**turn, 'azimuth_deg': None if az is None else round(az, 1),
                                'name': self._speaker(az), 'where': direction_words(az)})
                main = max(out, key=lambda t: t['end_s'] - t['start_s'])
                patch = {'speech': {'text': reply['text'], 'speaker': main['name'], 'turns': out},
                         'sound': {'label': 'speech', 'category': 'speech', 'confidence': 0.9,
                                   'source': 'muse-voice-transcribe'}}
                self._update(rid, patch, 'captions', out)

        if explicit:
            if checked:
                task['_analysis_status'] = 'Saved-event analysis finished: sound label and scene context updated.'
            elif captioned:
                task['_analysis_status'] = 'Saved-event analysis finished: transcript updated.'
            elif not wav:
                manual_notice('no saved audio is available to analyze.')
            elif self._meta_ok('identify', on_block=manual_notice):
                # A deliberate request can inspect weak/local-matched evidence.
                # This path is audio-only: it never relaxes the visual/source gates.
                reply = self._call(rid, self.meta.classify_sound, wav, facts, sheet, watch)
                if reply and isinstance(reply.get('label'), str) and reply['label'].strip():
                    self._update(rid, {'sound': {'label': reply['label'], 'category': reply['category'],
                                               'description': reply['description'],
                                               'confidence': round(reply['confidence'], 2),
                                               'source': 'muse-spark'}}, 'identify')
                    self._alerts(rid, record, reply.get('watch') or [], 'Muse Spark', reply['confidence'],
                                 reply['label'])
                    task['_analysis_status'] = 'Saved-event analysis finished: audio-only sound label updated.'
                elif self.exhausted is not None:
                    manual_notice(f'the estimated budget is paused: {self.exhausted}')
                else:
                    manual_notice('the audio-label request returned no usable label or failed; '
                                  'the saved recording is still available.')

        # Context alone only when the combined source check is not in use (no double spending), and
        # never for speech: the caption says enough, and the crop would show the person talking.
        if (audio_allowed and crop is not None and visual and 'context' in self.features and 'identify' not in self.features
                and not checked and not speech_like):
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

        if audio_allowed and visual and task.get('outline') and task.get('frame_jpeg') is not None and not speech_like \
                and self._meta_ok('outline'):
            w, h = task['frame_size']
            u, v = task['pixel']
            polygons = self._call(rid, self.meta.segment_at, task['frame_jpeg'], w, h, u, v)
            if polygons:
                self._update(rid, {'outline': {'source': 'sam-3.1', 'polygons': len(polygons)}}, 'outline',
                             {'polygons': polygons, 'scale': task.get('scale', 1.0)})

    def _speaker(self, azimuth):
        with self.lock:
            return self.speakers.name_for(azimuth)

    def _cell(self, record):
        """Where a context answer can be reused: a 25 cm cell, else a pixel or direction bin."""
        w = record.get('where') or {}
        point = w.get('point_m')
        if point:
            return tuple(round(c / self.cell) for c in point)
        if w.get('pixel'):
            return ('px', w['pixel'][0] // 64, w['pixel'][1] // 64)
        return ('dir', round((w.get('azimuth_deg') or 0) / 10), round((w.get('elevation_deg') or 0) / 10))


def usable_visual_evidence(record):
    """An isolated candidate with measured camera depth, never a guessed range.

    make_record only writes point_m/distance_m/candidate_surface for measured
    valid depth. A model's label or confidence cannot satisfy this gate.
    """
    if not isinstance(record, dict):
        return False
    where = record.get('where') or {}
    if not isinstance(where, dict):
        return False
    if (where.get('quality') != 'candidate' or where.get('on_screen') is not True or
            where.get('distance_kind') != 'candidate_surface'):
        return False
    point = where.get('point_m')
    if not isinstance(point, (list, tuple)) or len(point) != 3:
        return False
    try:
        values = [float(x) for x in point]
        distance = float(where.get('distance_m'))
        return (all(math.isfinite(x) for x in values) and values[2] > 0 and
                math.isfinite(distance) and distance > 0)
    except (TypeError, ValueError, OverflowError):
        return False


def _bare(noun):
    """'a wall' -> 'wall', so it reads 'on the wall'."""
    words = noun.strip().split(' ', 1)
    return words[1] if len(words) == 2 and words[0].lower() in ('a', 'an', 'the') else noun.strip()


def _stem(word):
    for suffix in ('ing', 'ed', 'es', 's'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _same_sound(watch_item, label):
    """Loose match between a watched sound and a label: a shared content word ("doorbell chime" ~ "doorbell",
    "knocking" ~ "knock on the door")."""
    stop = {'a', 'an', 'the', 'my', 'is', 'sound', 'of', 'on', 'when', 'if', 'probably', 'some', 'noise'}
    words = lambda text: {_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in stop}
    return bool(words(watch_item) & words(label))
