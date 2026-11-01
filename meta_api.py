"""Meta Model API calls for SoundScope: Muse Voice Transcribe (speech -> text) and Muse Spark
(what a sound is, and what a spoken command means). Direction always comes from the mics, never
from these models.

  python meta_api.py transcribe command.wav
  python meta_api.py match clip.wav "phone ringing"
  python meta_api.py command "turn toward my phone when it rings"

Muse Spark is shown a spectrogram picture of the sound, not the audio itself: tested Nov 1, 2026,
the API accepts `input_audio` parts on muse-spark-1.2/1.3 but drops them (prompt token count
unchanged; see github.com/meta-models/meta-model-cookbook/issues/59). Image input does work.

Audio must be 16 kHz mono 16-bit WAV. Needs MODEL_API_KEY in meta-soundscope/.env (or pass
--env-file). Request formats follow the Meta_Hack sound_scope server.
"""
import argparse
import base64
import io
import json
import os
import uuid
import wave
from pathlib import Path
from urllib import error, request

import numpy as np

API_BASE = 'https://api.meta.ai/v1'
ENV_FILE = Path(__file__).with_name('.env')

MATCH_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['match', 'heard', 'confidence'],
    'properties': {'match': {'type': 'boolean'}, 'heard': {'type': 'string'},
                   'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}},
}
COMMAND_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['action', 'sound', 'reply'],
    'properties': {'action': {'type': 'string', 'enum': ['follow', 'find', 'stop', 'home', 'ignore']},
                   'sound': {'type': 'string'}, 'reply': {'type': 'string'}},
}
COMMAND_PROMPT = (
    "You control a robot arm whose camera turns toward sounds. Turn the user's words into one action. "
    "follow: keep turning toward sounds (only the named sound, if they name one). "
    "find: turn toward the next occurrence of the named sound once, then hold still. "
    "stop: stop moving and stop following. home: turn back to the starting direction. "
    "ignore: anything that is not a command for the arm. "
    "sound: the sound to track in a few words (for example 'phone ringing'), or '' for any sound. "
    "reply: a short confirmation for the user."
)
MATCH_PROMPT = ('You are the ears of a robot arm that turns toward sounds. You get a spectrogram of a short '
                'sound clip: time runs left to right, frequency goes up to 8 kHz, brighter means louder. '
                'Judge only the sound it shows, and say briefly what the sound most likely is.')


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


def spectrogram_jpeg(wav):
    """16 kHz WAV bytes -> spectrogram picture as a JPEG data URL, with labeled axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with wave.open(io.BytesIO(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float)
    n, hop = 512, 128
    frames = x[np.arange(n) + hop * np.arange(max(1, (len(x) - n) // hop + 1))[:, None]] * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    db = 10 * np.log10(power + power.max() * 1e-9)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.imshow(db.T, origin='lower', aspect='auto', cmap='magma', vmin=db.max() - 80,
              extent=(0, len(x) / 16000, 0, 8000))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close(fig)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


class MetaError(Exception):
    pass


class Meta:
    def __init__(self):
        self.key = os.environ.get('MODEL_API_KEY', '').strip()
        if not self.key:
            raise MetaError('Set MODEL_API_KEY in meta-soundscope/.env (same key as Meta_Hack).')
        self.image_model = os.environ.get('META_IMAGE_MODEL') or 'muse-spark-1.3'
        self.text_model = os.environ.get('META_TEXT_MODEL') or 'muse-spark-1.3'
        self.asr_model = os.environ.get('META_TRANSCRIPTION_MODEL') or 'muse-voice-transcribe-1.0'

    def _post(self, endpoint, body, content_type='application/json'):
        req = request.Request(API_BASE + endpoint, data=body, method='POST',
                              headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': content_type,
                                       'Accept': 'application/json'})
        try:
            with request.urlopen(req, timeout=60) as response:
                return json.loads(response.read())
        except error.HTTPError as exc:
            raise MetaError(f'Meta API answered HTTP {exc.code} on {endpoint}') from None
        except (error.URLError, TimeoutError) as exc:
            raise MetaError(f'could not reach the Meta API ({exc})') from None
        except ValueError:
            raise MetaError(f'unreadable reply from {endpoint}') from None

    def transcribe(self, wav):
        """Muse Voice Transcribe: spoken audio -> text."""
        boundary = 'SoundScope' + uuid.uuid4().hex
        settings = json.dumps({'model': self.asr_model, 'mode': 'PUSH_TO_TALK', 'audioEncoding': 'WAV'})
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
                f'Content-Type: application/json\r\n\r\n{settings}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="command.wav"\r\n'
                'Content-Type: audio/wav\r\n\r\n').encode() + wav + f'\r\n--{boundary}--\r\n'.encode()
        result = self._post('/asr/transcribe', body, 'multipart/form-data; boundary=' + boundary)
        if not isinstance(result, dict) or not isinstance(result.get('transcript'), str):
            raise MetaError('no transcript in the reply')
        return result['transcript']

    def chat_json(self, model, content, schema, name, system=None, max_tokens=2000):
        """Muse Spark with a strict JSON reply. It reasons before answering, so max_tokens must leave
        room for that (about 500-1000 tokens at low effort); low effort answers in ~2-3 s."""
        messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': content}]
        payload = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'reasoning_effort': 'low',
                   'response_format': {'type': 'json_schema',
                                       'json_schema': {'name': name, 'strict': True, 'schema': schema}}}
        result = self._post('/chat/completions', json.dumps(payload).encode())
        try:
            if result['choices'][0].get('finish_reason') == 'length':
                raise MetaError('Muse Spark ran out of tokens before answering; raise max_tokens')
            reply = json.loads(result['choices'][0]['message']['content'])
        except (KeyError, IndexError, TypeError, ValueError):
            raise MetaError('Muse Spark did not return the expected JSON') from None
        if not isinstance(reply, dict) or set(reply) != set(schema['required']):
            raise MetaError('Muse Spark returned JSON with the wrong fields')
        return reply

    def sound_matches(self, wav, target):
        """Muse Spark, reading a spectrogram of the clip: is this the sound we're looking for?"""
        content = [{'type': 'text', 'text': f'Is the main sound in this clip "{target}"?'},
                   {'type': 'image_url', 'image_url': {'url': spectrogram_jpeg(wav)}}]
        return self.chat_json(self.image_model, content, MATCH_SCHEMA, 'sound_match', MATCH_PROMPT)

    def parse_command(self, text):
        """Muse Spark: spoken words -> {action, sound, reply}."""
        return self.chat_json(self.text_model, [{'type': 'text', 'text': text}], COMMAND_SCHEMA,
                              'arm_command', COMMAND_PROMPT)


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
    s = sub.add_parser('match')
    s.add_argument('wav')
    s.add_argument('target')
    sub.add_parser('command').add_argument('text')
    args = p.parse_args()
    load_env(args.env_file or ENV_FILE)
    try:
        meta = Meta()
        if args.cmd == 'transcribe':
            print(meta.transcribe(read_wav(args.wav)))
        elif args.cmd == 'match':
            print(json.dumps(meta.sound_matches(read_wav(args.wav), args.target)))
        else:
            print(json.dumps(meta.parse_command(args.text)))
    except MetaError as exc:
        raise SystemExit(f'Meta: {exc}')


if __name__ == '__main__':
    main()
