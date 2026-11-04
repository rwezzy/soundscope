"""Meta Model API calls for SoundScope. Direction always comes from the mics, never from a model.

  Muse Voice Transcribe   spoken commands -> text; captions with speaker turns (DIARIZATION mode)
  Muse Spark 1.3          what a sound most likely is (from a spectrogram picture, optionally next
                          to your own labeled examples), what a command means, answers about past
                          sounds, and the audio-visual source check: the spectrogram plus camera
                          crops of the spot the MICS picked, before and after the sound, to name the
                          object there, what visibly happened, and whether sound and sight agree
  SAM 3.1                 outline of the object at the pixel the sound picked (point prompt only;
                          never a text search for objects)

  python meta_api.py transcribe command.wav
  python meta_api.py match clip.wav "phone ringing"
  python meta_api.py command "turn toward my phone when it rings"
  python meta_api.py budget

Muse Spark is shown a spectrogram picture of the sound, not the audio itself: tested Nov 1, 2026,
the API accepts `input_audio` parts on muse-spark-1.2/1.3 but drops them (prompt token count
unchanged; see github.com/meta-models/meta-model-cookbook/issues/59). Image input does work.

Every call goes through Budget: an estimated-dollar limit per day (saved in
recordings/meta_usage.json) and per session. A call whose worst case would cross a limit is refused
before anything is sent. Limits: META_DAILY_LIMIT_USD (default 1.00) and META_SESSION_LIMIT_USD
(default 0.50) in .env. Contributor model tiers are never used (they train on your data).

Audio must be 16 kHz mono 16-bit WAV. Needs MODEL_API_KEY in meta-soundscope/.env (or pass
--env-file).
"""
import argparse
import base64
import io
import json
import math
import os
import threading
import time
import uuid
import wave
from datetime import date
from pathlib import Path
from urllib import error, request

import numpy as np

API_BASE = 'https://api.meta.ai/v1'
ENV_FILE = Path(__file__).with_name('.env')
USAGE_FILE = Path(os.environ.get('SOUNDSCOPE_DATA') or Path(__file__).with_name('recordings')) / 'meta_usage.json'

# USD, from the Meta Model API models page (Nov 2026).
PRICES = {
    'muse-spark-1.3': {'input': 1.25e-6, 'output': 4.25e-6},
    'muse-spark-1.2': {'input': 1.25e-6, 'output': 4.25e-6},
    'muse-spark-1.1': {'input': 1.25e-6, 'output': 4.25e-6},
    'muse-voice-transcribe-1.0': {'audio_hour': 0.18},
    'sam-3.1': {'image': 0.0025},
}
IMAGE_TOKENS = 1600  # conservative estimate for one picture, used only before the reply says

MATCH_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['match', 'heard', 'confidence'],
    'properties': {'match': {'type': 'boolean'}, 'heard': {'type': 'string'},
                   'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}},
}
COMMAND_WORDS = ['SoundScope', 'find', 'follow', 'watch', 'stop', 'show everything', 'who is talking',
                 'what was that', 'when did', 'where was', 'phone ringing', 'keys', 'clap', 'knock', 'alarm']
COMMAND_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['action', 'sound', 'question', 'reply'],
    'properties': {'action': {'type': 'string',
                              'enum': ['follow', 'find', 'watch', 'stop', 'home', 'search', 'who', 'describe',
                                       'ignore']},
                   'sound': {'type': 'string'}, 'question': {'type': 'string'}, 'reply': {'type': 'string'}},
}
COMMAND_PROMPT = (
    "You run SoundScope: microphones find where sounds come from, a camera shows the spot, and an "
    "optional robot arm turns toward sounds. Turn the user's words into one action. "
    "follow: keep showing (or turning toward) sounds, only the named sound if they name one. "
    "find: catch the next occurrence of the named sound once, then hold it. "
    "watch: keep listening in the background and alert when the named sound happens ('tell me when the "
    "kettle whistles', 'let me know if the baby cries'). "
    "stop: stop moving and stop following. home: go back to showing every sound / the starting direction. "
    "search: a question about sounds that already happened (when, where, how loud, how often). "
    "who: who is talking or who talked, and where they are. describe: what was that last sound. "
    "ignore: anything else. "
    "sound: the sound to find or follow in a few words (for example 'phone ringing'), or ''. "
    "question: for search, the user's question restated clearly; otherwise ''. "
    "reply: a short confirmation for the user."
)
MATCH_PROMPT = ('You are the ears of SoundScope. You get a spectrogram picture of a short sound clip: '
                'time runs left to right, frequency goes up to 8 kHz, brighter means louder. If labeled '
                'example rows are shown, compare the UNKNOWN row with them. Judge only the sound shown, '
                'and say briefly what it most likely is.')
CLASSIFY_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['label', 'category', 'description', 'confidence', 'watch'],
    'properties': {'label': {'type': 'string'},
                   'category': {'type': 'string', 'enum': ['speech', 'music', 'alarm or ringtone', 'beep or tone',
                                                           'clap or knock', 'impact', 'metal or keys', 'footsteps',
                                                           'water', 'animal', 'machine', 'other']},
                   'description': {'type': 'string'},
                   'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                   'watch': {'type': 'array', 'items': {'type': 'string'}}},
}
CLASSIFY_PROMPT = ('You are the ears of SoundScope, which already knows WHERE each sound came from. You get a '
                   'spectrogram picture of one sound (time left to right, 0-8 kHz bottom to top, brighter = '
                   'louder) and a few measurements. If labeled example rows are shown, the UNKNOWN row is the '
                   'sound to judge; prefer an example label when it clearly matches. label: 1-4 words for what '
                   'the sound most likely is (for example "phone ringing", "hand clap", "keys jingling"). '
                   'description: one short clause on what in the picture supports that. Be honest about doubt '
                   'in confidence. watch: the items of the given watch list this sound clearly is (copy them '
                   'exactly), or [] if none or no list.')
SOURCE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['sound', 'sound_confidence', 'object', 'action', 'consistent', 'near', 'surface', 'watch', 'summary'],
    'properties': {'sound': {'type': 'string'},
                   'sound_confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                   'object': {'type': 'string'}, 'action': {'type': 'string'},
                   'consistent': {'type': 'string', 'enum': ['yes', 'no', 'unsure']},
                   'near': {'type': 'string'}, 'surface': {'type': 'string'},
                   'watch': {'type': 'array', 'items': {'type': 'string'}},
                   'summary': {'type': 'string'}},
}
SOURCE_PROMPT = ('You are the eyes and ears of SoundScope. Its microphone array has ALREADY located a sound: the '
                 'thin crosshair in the camera crops marks where the sound came from. Never search the rest of '
                 'the picture for a source; describe only what is at or touching the crosshair. You get: (1) a '
                 'spectrogram of the sound (time left to right, 0-8 kHz bottom to top, brighter = louder), (2) '
                 'the camera crop when the sound ended, and possibly (3) the same crop just before it began. '
                 'sound: 1-4 words for what the sound most likely is, from the spectrogram. sound_confidence: '
                 'honest. object: the object at the crosshair in 1-5 words ("a black phone", "a hand", '
                 '"nothing distinct", "a wall"). action: what visibly changed at the crosshair between before and '
                 'after ("a mug was set down", "nothing visible"), or "not checked" when there is no before '
                 'image. consistent: yes if that object or action could '
                 'plausibly make that sound, no if clearly not (then the source may be hidden behind it or the '
                 'sound reflected), unsure otherwise. near: 1-3 objects right around the crosshair. surface: what '
                 'they rest on or are part of. watch: items of the given watch list this sound clearly is (copy '
                 'exactly), or []. summary: one short plain sentence for the user.')
NEAR_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['near', 'surface'],
    'properties': {'near': {'type': 'string'}, 'surface': {'type': 'string'}},
}
NEAR_PROMPT = ('A microphone array already located a sound at the center of this camera crop (marked with a '
               'thin crosshair). Do not guess what made the sound. near: list the 1-3 most noticeable objects '
               'at or right around the crosshair, as a short phrase ("a laptop and a mug"). surface: what '
               'they rest on or are part of ("desk", "shelf", "floor", "wall", "unknown"). Use few words.')
SEARCH_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['matches', 'answer'],
    'properties': {'matches': {'type': 'array', 'items': {'type': 'string'}}, 'answer': {'type': 'string'}},
}
SEARCH_PROMPT = ('You answer questions about sounds SoundScope recorded. Each line is one sound: id | when | '
                 'what it most likely was (with who judged it and their confidence; labels without a judge are '
                 'measurements only) | where relative to the camera | loudness over the room noise | duration '
                 '| optionally: the object seen at the spot the mics picked and whether sound and sight agree, '
                 'what was near, alerts raised, what was said. Use ONLY these facts. For summaries, group and '
                 'count similar sounds. matches: ids of the matching '
                 'sounds, most relevant first (at most 5; empty if none). answer: one or two plain sentences '
                 'saying when, where, how loud and what was near; say "probably" when a label is uncertain; if '
                 'nothing matches, say so.')


def load_env(path=ENV_FILE):
    """Read KEY=value lines into the environment; variables already set win."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def wav_bytes(samples, rate=16000):
    """Mono float audio -> 16-bit WAV bytes, peak-normalized (raw mic levels are very low)."""
    x = np.asarray(samples, float)
    peak = np.abs(x).max() if len(x) else 0.0
    if peak > 0:
        x = x * (0.7 / peak)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype('<i2').tobytes())
    return buf.getvalue()


def wav_seconds(wav):
    try:
        with wave.open(io.BytesIO(wav)) as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError):
        return 0.0


def spectrogram_jpeg(wav):
    """16 kHz WAV bytes -> spectrogram picture as a JPEG data URL, with labeled axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with wave.open(io.BytesIO(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float)
    n, hop = 512, 128
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = x[np.arange(n) + hop * np.arange(max(1, (len(x) - n) // hop + 1))[:, None]] * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    db = 10 * np.log10(power + power.max() * 1e-9 + 1e-20)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=90)
    ax.imshow(db.T, origin='lower', aspect='auto', cmap='magma', vmin=db.max() - 80,
              extent=(0, len(x) / 16000, 0, 8000))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close(fig)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def jpeg_data_url(jpeg):
    return 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()


class MetaError(Exception):
    pass


class BudgetError(MetaError):
    pass


class Budget:
    """Estimated spend per day (saved to disk, shared by every SoundScope program) and per session.

    check() refuses a call whose worst-case cost would cross either limit; spend() records the
    cost the reply reports (token usage), or the worst case when it reports none.
    """

    def __init__(self, daily_usd=None, session_usd=None, path=None, today=None):
        self.daily = float(daily_usd if daily_usd is not None else os.environ.get('META_DAILY_LIMIT_USD', 1.0))
        self.session = float(session_usd if session_usd is not None else os.environ.get('META_SESSION_LIMIT_USD', 0.5))
        self.path = Path(path or USAGE_FILE)
        self.today = today or (lambda: date.today().isoformat())
        self.session_spent, self.session_calls = 0.0, {}
        self.lock = threading.Lock()

    def _load(self):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            data = {}
        if data.get('day') != self.today():
            data = {'day': self.today(), 'spent': 0.0, 'calls': {}}
        return data

    def _save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=1), encoding='utf-8')
        os.replace(tmp, self.path)

    def check(self, worst_usd, kind):
        with self.lock:
            spent = self._load()['spent']
            if self.session_spent + worst_usd > self.session:
                raise BudgetError(f'session Meta budget reached (${self.session_spent:.3f} of ${self.session:.2f}); '
                                  'raise META_SESSION_LIMIT_USD to allow more')
            if spent + worst_usd > self.daily:
                raise BudgetError(f'daily Meta budget reached (${spent:.3f} of ${self.daily:.2f} today); '
                                  'raise META_DAILY_LIMIT_USD to allow more')

    def spend(self, usd, kind):
        with self.lock:
            data = self._load()
            data['spent'] = round(data['spent'] + usd, 6)
            data['calls'][kind] = data['calls'].get(kind, 0) + 1
            self._save(data)
            self.session_spent += usd
            self.session_calls[kind] = self.session_calls.get(kind, 0) + 1

    def summary(self):
        data = self._load()
        return {'today_usd': round(data['spent'], 4), 'daily_limit_usd': self.daily,
                'session_usd': round(self.session_spent, 4), 'session_limit_usd': self.session,
                'calls_today': data['calls'], 'calls_session': dict(self.session_calls)}

    def short(self):
        """One-line spend for the screen; re-read at most every 2 s (the viewer asks every frame)."""
        now = time.monotonic()
        if not hasattr(self, '_short') or now - self._short[0] > 2.0:
            s = self.summary()
            self._short = (now, f"Meta ${s['today_usd']:.2f}/{s['daily_limit_usd']:.2f} today")
        return self._short[1]


def _text_tokens(content):
    """Rough input-token estimate for a chat content list: ~3.5 characters per token, plus pictures."""
    if isinstance(content, str):
        return len(content) / 3.5 + 10
    total = 10.0
    for part in content:
        if part.get('type') == 'text':
            total += len(part.get('text', '')) / 3.5
        elif part.get('type') in ('image_url', 'input_image'):
            total += IMAGE_TOKENS
    return total


class Meta:
    def __init__(self, budget=None):
        self.key = os.environ.get('MODEL_API_KEY', '').strip()
        if not self.key:
            raise MetaError('Set MODEL_API_KEY in meta-soundscope/.env (same key as Meta_Hack).')
        self.image_model = os.environ.get('META_IMAGE_MODEL') or 'muse-spark-1.3'
        self.text_model = os.environ.get('META_TEXT_MODEL') or 'muse-spark-1.3'
        self.asr_model = os.environ.get('META_TRANSCRIPTION_MODEL') or 'muse-voice-transcribe-1.0'
        self.sam_model = os.environ.get('META_SEGMENT_MODEL') or 'sam-3.1'
        for model in (self.image_model, self.text_model):
            if 'contributor' in model:
                raise MetaError('Contributor model tiers train on your data; SoundScope does not use them.')
        self.budget = budget or Budget()

    def _open(self, endpoint, body, content_type, accept, timeout=60):
        req = request.Request(API_BASE + endpoint, data=body, method='POST',
                              headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': content_type,
                                       'Accept': accept})
        try:
            return request.urlopen(req, timeout=timeout)
        except error.HTTPError as exc:
            detail = ''
            try:
                detail = json.loads(exc.read()).get('error', {}).get('message', '')
            except Exception:
                pass
            raise MetaError(f'Meta API answered HTTP {exc.code} on {endpoint}' + (f': {detail}' if detail else '')) from None
        except (error.URLError, TimeoutError) as exc:
            raise MetaError(f'could not reach the Meta API ({exc})') from None

    def _post(self, endpoint, body, content_type='application/json'):
        with self._open(endpoint, body, content_type, 'application/json') as response:
            try:
                return json.loads(response.read())
            except ValueError:
                raise MetaError(f'unreadable reply from {endpoint}') from None

    # -- Muse Voice Transcribe -----------------------------------------------------------------

    def _asr(self, wav, mode, keywords=None):
        seconds = wav_seconds(wav)
        cost = PRICES.get(self.asr_model, {}).get('audio_hour', 0.18) * max(seconds, 1.0) / 3600
        self.budget.check(cost, 'transcribe')
        settings = {'model': self.asr_model, 'mode': mode, 'audioEncoding': 'WAV'}
        if keywords:
            settings['keywords'] = list(keywords)[:20]
        boundary = 'SoundScope' + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
                f'Content-Type: application/json\r\n\r\n{json.dumps(settings)}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="clip.wav"\r\n'
                'Content-Type: audio/wav\r\n\r\n').encode() + wav + f'\r\n--{boundary}--\r\n'.encode()
        result = self._post('/asr/transcribe', body, 'multipart/form-data; boundary=' + boundary)
        self.budget.spend(cost, 'transcribe')  # failed requests are not billed
        if not isinstance(result, dict) or not isinstance(result.get('transcript'), str):
            raise MetaError('no transcript in the reply')
        return result

    def transcribe(self, wav, keywords=None):
        """Muse Voice Transcribe: spoken audio -> text (push-to-talk). keywords bias recognition."""
        return self._asr(wav, 'PUSH_TO_TALK', keywords)['transcript']

    def transcribe_turns(self, wav, keywords=None):
        """Muse Voice Transcribe in DIARIZATION mode: {'text', 'turns': [{'start_s', 'end_s', 'text',
        'speaker'}]}. Speaker letters only mean something within this one clip."""
        result = self._asr(wav, 'DIARIZATION', keywords)
        turns = []
        for t in result.get('turns') or []:
            try:
                turns.append({'start_s': float(t['startMs']) / 1000, 'end_s': float(t['endMs']) / 1000,
                              'text': str(t.get('transcript', '')).strip(), 'speaker': t.get('speaker')})
            except (KeyError, TypeError, ValueError):
                continue
        turns = [t for t in turns if t['text']]
        if not turns and result['transcript'].strip():
            turns = [{'start_s': 0.0, 'end_s': wav_seconds(wav), 'text': result['transcript'].strip(), 'speaker': None}]
        return {'text': result['transcript'].strip(), 'turns': turns}

    # -- Muse Spark --------------------------------------------------------------------------------

    def chat_json(self, model, content, schema, name, system=None, max_tokens=2000, kind='chat', effort='low'):
        """Muse Spark with a strict JSON reply. It reasons before answering, so max_tokens must leave
        room for that (about 500-1000 tokens at low effort). effort 'minimal' (used for the automatic
        background checks) answered in about half the time of 'low' on Nov 4, with the same labels."""
        price = PRICES.get(model, PRICES['muse-spark-1.3'])
        tokens_in = _text_tokens(content) + (_text_tokens(system) if system else 0)
        worst = tokens_in * price['input'] + max_tokens * price['output']
        self.budget.check(worst, kind)
        messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': content}]
        payload = {'model': model, 'messages': messages, 'max_tokens': max_tokens,
                   'reasoning_effort': os.environ.get('META_REASONING_EFFORT') or effort,
                   'response_format': {'type': 'json_schema',
                                       'json_schema': {'name': name, 'strict': True, 'schema': schema}}}
        result = self._post('/chat/completions', json.dumps(payload).encode())
        cost = worst  # failed requests are not billed; answered ones cost what the reply reports
        usage = result.get('usage') if isinstance(result, dict) else None
        if isinstance(usage, dict) and 'prompt_tokens' in usage:
            cost = (usage.get('prompt_tokens', 0) * price['input']
                    + usage.get('completion_tokens', 0) * price['output'])
        self.budget.spend(cost, kind)
        try:
            if result['choices'][0].get('finish_reason') == 'length':
                raise MetaError('Muse Spark ran out of tokens before answering; raise max_tokens')
            reply = json.loads(result['choices'][0]['message']['content'])
        except (KeyError, IndexError, TypeError, ValueError):
            raise MetaError('Muse Spark did not return the expected JSON') from None
        if not isinstance(reply, dict) or set(reply) != set(schema['required']):
            raise MetaError('Muse Spark returned JSON with the wrong fields')
        return reply

    def sound_matches(self, wav, target, sheet=None):
        """Muse Spark, reading a spectrogram of the clip: is this the sound we're looking for?
        sheet: optional ReferenceBank.sheet() picture (labeled examples + this clip) instead."""
        content = [{'type': 'text', 'text': f'Is the main sound in this clip "{target}"?'},
                   {'type': 'image_url', 'image_url': {'url': sheet or spectrogram_jpeg(wav)}}]
        return self.chat_json(self.image_model, content, MATCH_SCHEMA, 'sound_match', MATCH_PROMPT, kind='match')

    def classify_sound(self, wav, measurements=None, sheet=None, watch=()):
        """Muse Spark: what the sound most likely is -> {label, category, description, confidence, watch}."""
        facts = f'Measurements: {json.dumps(measurements)}' if measurements else 'No measurements.'
        facts += f' Watch list: {json.dumps(list(watch))}.' if watch else ' No watch list.'
        content = [{'type': 'text', 'text': 'What is this sound? ' + facts},
                   {'type': 'image_url', 'image_url': {'url': sheet or spectrogram_jpeg(wav)}}]
        reply = self.chat_json(self.image_model, content, CLASSIFY_SCHEMA, 'sound_label', CLASSIFY_PROMPT,
                               kind='identify', effort='minimal')
        reply['confidence'] = float(min(1.0, max(0.0, reply['confidence'])))
        reply['watch'] = [w for w in reply.get('watch') or [] if w in set(watch)]
        return reply

    def source_check(self, wav, crop_after, crop_before=None, measurements=None, watch=(), sheet=None,
                     crop_kind='color'):
        """Muse Spark, audio-visual: the spectrogram plus camera crops of the spot the mics picked
        (after, and optionally before, the sound) -> {sound, sound_confidence, object, action,
        consistent, near, surface, watch, summary}. The crops only describe that spot; the location
        itself always comes from the microphones."""
        facts = f'Measurements: {json.dumps(measurements)}.' if measurements else ''
        facts += f' Watch list: {json.dumps(list(watch))}.' if watch else ' No watch list.'
        content = [{'type': 'text', 'text': 'Image 1: the sound. ' + facts},
                   {'type': 'image_url', 'image_url': {'url': sheet or spectrogram_jpeg(wav)}},
                   {'type': 'text', 'text': ('Image 2: camera crop at the sound\'s location, as the sound ended.'
                                             if crop_kind == 'color' else
                                             'Image 2: the room is dark, so this is the DEPTH camera\'s 3D relief '
                                             'at the sound\'s location: shapes only, no colors or text; nearer '
                                             'surfaces are brighter.')},
                   {'type': 'image_url', 'image_url': {'url': jpeg_data_url(crop_after)}}]
        if crop_before is not None:
            content += [{'type': 'text', 'text': 'Image 3: the same crop just before the sound began.'},
                        {'type': 'image_url', 'image_url': {'url': jpeg_data_url(crop_before)}}]
        reply = self.chat_json(self.image_model, content, SOURCE_SCHEMA, 'sound_source', SOURCE_PROMPT,
                               max_tokens=2000, kind='source', effort='minimal')
        reply['sound_confidence'] = float(min(1.0, max(0.0, reply['sound_confidence'])))
        reply['watch'] = [w for w in reply.get('watch') or [] if w in set(watch)]
        return reply

    def parse_command(self, text):
        """Muse Spark: spoken words -> {action, sound, question, reply}."""
        return self.chat_json(self.text_model, [{'type': 'text', 'text': text}], COMMAND_SCHEMA,
                              'soundscope_command', COMMAND_PROMPT, kind='command')

    def describe_near(self, jpeg, crop_kind='color'):
        """Muse Spark looking at the camera crop around the spot the sound picked -> {near, surface}."""
        note = '' if crop_kind == 'color' else (' The room is dark: this is a depth-camera 3D relief (shapes '
                                                'only, nearer is brighter); name shapes, not colors.')
        content = [{'type': 'text', 'text': 'What is around the crosshair?' + note},
                   {'type': 'image_url', 'image_url': {'url': jpeg_data_url(jpeg)}}]
        return self.chat_json(self.image_model, content, NEAR_SCHEMA, 'near_spot', NEAR_PROMPT,
                              max_tokens=1200, kind='context', effort='minimal')

    def answer_query(self, question, records_table, now_text):
        """Muse Spark over the sound log -> {matches: [ids], answer}."""
        content = [{'type': 'text', 'text': f'Now: {now_text}\nQuestion: {question}\nSounds:\n{records_table}'}]
        return self.chat_json(self.text_model, content, SEARCH_SCHEMA, 'sound_search', SEARCH_PROMPT,
                              max_tokens=1500, kind='search')

    # -- SAM 3.1 -----------------------------------------------------------------------------------

    def segment_at(self, jpeg, width, height, u, v, timeout=45):
        """SAM 3.1 with one positive point at the pixel the sound picked -> list of polygons
        (each an (N, 2) int array in image pixels) whose mask contains that pixel."""
        cost = PRICES.get(self.sam_model, {}).get('image', 0.0025)
        self.budget.check(cost, 'outline')
        prompt = sam_point_prompt(width, height, u, v)
        payload = {'model': self.sam_model, 'stream': True,
                   'input': [{'type': 'message', 'role': 'user',
                              'content': [{'type': 'input_text', 'text': prompt},
                                          {'type': 'input_image', 'image_url': jpeg_data_url(jpeg)}]}]}
        text = []
        with self._open('/responses', json.dumps(payload).encode(), 'application/json',
                        'text/event-stream', timeout) as response:
            self.budget.spend(cost, 'outline')  # accepted: billed per image
            for event in sse_events(response):
                kind = event.get('type', '')
                if kind == 'response.output_text.delta':
                    text.append(event.get('delta', ''))
                elif kind in ('response.failed', 'error'):
                    raise MetaError(f'SAM 3.1 failed: {event.get("error") or event}')
                elif kind == 'response.completed':
                    break
        return sam_polygons(''.join(text), u, v)


def sam_point_prompt(width, height, u, v):
    """SAM's object-prompt grammar for one positive point (same as meta_sam_parser.build_object_prompt)."""
    u, v = int(min(max(u, 0), width - 1)), int(min(max(v, 0), height - 1))
    return f'<0f>1<|point;x={u};y={v};w={int(width)};h={int(height)}|>'


def sse_events(lines):
    """Server-sent events (bytes lines) -> JSON dicts."""
    data = []
    for raw in lines:
        line = raw.decode('utf-8', 'replace').rstrip('\r\n') if isinstance(raw, bytes) else raw.rstrip('\r\n')
        if not line:
            if data:
                chunk = '\n'.join(data)
                data = []
                if chunk.strip() == '[DONE]':
                    return
                try:
                    yield json.loads(chunk)
                except ValueError:
                    continue
        elif line.startswith('data:'):
            data.append(line[5:].lstrip())
    if data:
        try:
            yield json.loads('\n'.join(data))
        except ValueError:
            pass


def sam_polygons(output_text, u=None, v=None):
    """SAM output text -> polygons (image pixels) of masks containing (u, v), using Meta's
    meta-sam-parser package (pip install meta-sam-parser). Masks are box-local rasters."""
    try:
        import cv2
        from meta_sam_parser import (CompletedOutcome, SegmentationMaskRecord, decode_mask_to_raster,
                                     image_segmentation_format)
    except ImportError as exc:
        raise MetaError(f'outline needs meta-sam-parser and OpenCV ({exc}); pip install meta-sam-parser') from None
    if not output_text.strip():
        return []
    parser = image_segmentation_format().create_parser()
    parser.push(output_text)
    result = parser.finish(CompletedOutcome()).result
    polygons = []
    for record in result.records:
        if not isinstance(record, SegmentationMaskRecord):
            continue
        m, b = record.mask, record.bounds
        raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
        left, top = int(math.floor(b.left)), int(math.floor(b.top))
        bw, bh = max(1, int(math.ceil(b.right)) - left), max(1, int(math.ceil(b.bottom)) - top)
        mask = cv2.resize(raster * 255, (bw, bh), interpolation=cv2.INTER_NEAREST)
        if u is not None and v is not None:
            x, y = int(u) - left, int(v) - top
            if not (0 <= x < bw and 0 <= y < bh and mask[y, x]):
                continue
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= 30:
                polygons.append(c.reshape(-1, 2) + np.array([left, top]))
    return polygons


def read_wav(path):
    data = Path(path).read_bytes()
    with wave.open(io.BytesIO(data)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise SystemExit(f'{path}: needs 16 kHz mono 16-bit WAV')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env-file', help='load MODEL_API_KEY from this .env file')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('transcribe').add_argument('wav')
    sub.add_parser('captions').add_argument('wav')
    s = sub.add_parser('match')
    s.add_argument('wav')
    s.add_argument('target')
    sub.add_parser('identify').add_argument('wav')
    sub.add_parser('command').add_argument('text')
    sub.add_parser('budget')
    args = p.parse_args()
    load_env(args.env_file or ENV_FILE)
    if args.cmd == 'budget':
        print(json.dumps(Budget().summary(), indent=1))
        return
    try:
        meta = Meta()
        if args.cmd == 'transcribe':
            print(meta.transcribe(read_wav(args.wav)))
        elif args.cmd == 'captions':
            print(json.dumps(meta.transcribe_turns(read_wav(args.wav)), indent=1))
        elif args.cmd == 'match':
            print(json.dumps(meta.sound_matches(read_wav(args.wav), args.target)))
        elif args.cmd == 'identify':
            print(json.dumps(meta.classify_sound(read_wav(args.wav))))
        else:
            print(json.dumps(meta.parse_command(args.text)))
        print(meta.budget.short())
    except MetaError as exc:
        raise SystemExit(f'Meta: {exc}')


if __name__ == '__main__':
    main()
