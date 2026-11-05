"""Offline tests for the sound memory: python -m unittest test_sound_memory

Synthetic sounds stand in for the mics and a fake stands in for Muse Spark; nothing touches the
network or the real recordings folder.
"""
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import sound_memory as sm
from meta_api import wav_bytes

FS = 16000
RNG = np.random.default_rng(7)


def tone(freq=2000, seconds=1.5, level=0.2):
    t = np.arange(int(seconds * FS)) / FS
    return level * np.sin(2 * np.pi * freq * t)


def clicks(n=4, gap=0.25, level=0.5):
    x = np.zeros(int((n * gap + 0.3) * FS))
    for i in range(n):
        start = int((0.1 + i * gap) * FS)
        x[start:start + 400] += level * RNG.standard_normal(400) * np.exp(-np.arange(400) / 60)
    return x


def voice(seconds=2.0, f0=140):
    """Harmonic 'vowels' switching on and off at a syllable rate: speech-like for the heuristics."""
    t = np.arange(int(seconds * FS)) / FS
    pitch = f0 * (1 + 0.05 * np.sin(2 * np.pi * 1.5 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / FS
    x = sum((1 / k) * np.sin(k * phase) * (1 + 0.8 * np.sin(2 * np.pi * (500 * k) * t / 4000)) for k in range(1, 20))
    syllables = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 4 * t))) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2)
    return 0.1 * x * syllables


def fix(snr=20.0, seconds=1.2, z=30.0):
    return {'snr_db': snr, 'seconds': seconds, 'z': z, 'coherence': 0.3}


class Profiles(unittest.TestCase):
    def test_tone_clicks_noise_and_voice_look_different(self):
        p_tone = sm.acoustic_profile(tone())
        self.assertIn('tonal around 2.0 kHz', p_tone['character'])
        self.assertLess(p_tone['speech_score'], 0.5)
        p_clicks = sm.acoustic_profile(clicks())
        self.assertGreaterEqual(p_clicks['onsets'], 3)
        self.assertIn('quick bursts', p_clicks['character'])
        p_noise = sm.acoustic_profile(0.1 * RNG.standard_normal(FS * 2))
        self.assertIn('noisy', p_noise['character'])
        self.assertLess(p_noise['speech_score'], 0.5)
        p_voice = sm.acoustic_profile(voice())
        self.assertGreater(p_voice['speech_score'], p_tone['speech_score'])
        self.assertGreater(p_voice['speech_score'], p_noise['speech_score'])

    def test_dual_tone_ring_names_both_pitches(self):
        t = np.arange(int(16000 * 2.0)) / 16000
        ring = (np.sin(2 * np.pi * 1400 * t) + np.sin(2 * np.pi * 1800 * t)) * (np.sin(2 * np.pi * 20 * t) > 0)
        self.assertIn('tonal around 1.4 kHz and 1.8 kHz', sm.acoustic_profile(0.2 * ring)['character'])

    def test_words(self):
        self.assertEqual(sm.loudness_words(3), 'faint')
        self.assertEqual(sm.loudness_words(30), 'loud')
        self.assertEqual(sm.direction_words(0), 'straight ahead')
        self.assertEqual(sm.direction_words(-40, 20), 'to the left, above')
        self.assertEqual(sm.direction_words(15), 'slightly right')
        self.assertEqual(sm.direction_words(130), 'behind, to the right')

    def test_too_short_is_safe(self):
        self.assertIn('character', sm.acoustic_profile(np.zeros(10)))


class RealArrayRecordings(unittest.TestCase):
    """Clips captured by the real XVF3800 on Nov 4 (laptop text-to-speech and test tones, ~1.7 m)."""

    def clip(self, name):
        return sm.wav_samples((Path(__file__).with_name('test_data') / name).read_bytes())

    def test_speech_alarm_and_knock(self):
        speech = sm.acoustic_profile(self.clip('array_tts_find_my_keys.wav'))
        alarm = sm.acoustic_profile(self.clip('array_alarm_2900hz.wav'))   # once looked "speech-like"
        knock = sm.acoustic_profile(self.clip('array_knock.wav'))
        self.assertEqual(speech['character'], 'speech-like')
        self.assertLess(alarm['speech_score'], sm.SPEECH_SCORE)
        self.assertIn('tonal around 2.9 kHz', alarm['character'])
        self.assertIn('3 quick bursts', knock['character'])

    def test_examples_tell_them_apart(self):
        bank = sm.ReferenceBank(tempfile.mkdtemp())
        folder = Path(__file__).with_name('test_data')
        bank.add('smoke alarm', (folder / 'array_alarm_2900hz.wav').read_bytes())
        bank.add('knock', (folder / 'array_knock.wav').read_bytes())
        self.assertEqual(bank.nearest(self.clip('array_knock.wav'))[0], 'knock')
        self.assertEqual(bank.nearest(self.clip('array_alarm_2900hz.wav'))[0], 'smoke alarm')


class Records(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.now = time.mktime((2026, 11, 4, 15, 42, 10, 0, 0, -1))

    def record(self, azimuth=-30.0, label=None, near=None, speech=None, when=None, snr=24.0):
        a = math.radians(azimuth)
        view = {'spot': (320, 200), 'point': np.array([0.5 * math.sin(a), 0.1, 1.2]), 'measured': True,
                'quality': {'state': 'candidate'}}
        rec = sm.make_record(fix(snr=snr), [math.sin(a), 0.0, math.cos(a)], profile=sm.acoustic_profile(tone()),
                             view=view, level={'rms_dbfs': -30.0, 'peak_dbfs': -12.0}, now=when or self.now)
        if label:
            rec['sound'].update(label=label, confidence=0.9, source='muse-spark')
        if near:
            rec['near'] = {'text': near}
        if speech:
            rec['speech'] = {'text': speech, 'speaker': 'Speaker 1'}
        return rec

    def test_record_fields_are_measured_facts(self):
        rec = self.record()
        self.assertAlmostEqual(rec['where']['azimuth_deg'], -30.0, delta=0.1)
        self.assertEqual(rec['where']['words'], 'to the left')
        self.assertEqual(rec['where']['pixel'], [320, 200])
        self.assertAlmostEqual(rec['where']['distance_m'], 1.23, delta=0.02)
        self.assertEqual(rec['loudness']['words'], 'moderate')
        self.assertEqual(rec['loudness']['rms_dbfs'], -30.0)
        self.assertFalse(rec['sound']['verified'])

    def test_memory_round_trip_with_patches(self):
        memory = sm.Memory(self.dir.name, now=self.now)
        rec = memory.add(self.record())
        memory.update(rec['id'], {'sound': {'label': 'phone ringing', 'confidence': 0.8}, 'near': {'text': 'a laptop'}})
        again = sm.Memory(self.dir.name, now=self.now)
        loaded = again.get(rec['id'])
        self.assertEqual(loaded['sound']['label'], 'phone ringing')
        self.assertEqual(loaded['near']['text'], 'a laptop')
        self.assertEqual(loaded['where']['words'], 'to the left')
        self.assertEqual(len(again.recent()), 1)
        files = list(Path(self.dir.name).glob('*.jsonl'))
        self.assertEqual(len(files), 1)
        self.assertEqual(len(files[0].read_text(encoding='utf-8').splitlines()), 2)  # append-only

    def test_phrase_says_where_when_how_loud_what_and_near(self):
        rec = self.record(label='phone ringing', near='a laptop on the desk')
        text = sm.phrase(rec, now=self.now + 120)
        for part in ('Phone ringing', '3:42:10 PM', '2 min ago', '1.2 m away', 'to the left', 'moderate',
                     '+24 dB', 'near a laptop on the desk'):
            self.assertIn(part, text)
        unsure = self.record(label='keys jingling')
        unsure['sound']['confidence'] = 0.5
        self.assertTrue(sm.phrase(unsure, now=self.now).startswith('Probably keys jingling'))
        talk = self.record(speech='where are my keys')
        self.assertIn('Speaker 1 said: "where are my keys"', sm.phrase(talk, now=self.now))

    def test_time_windows(self):
        now = self.now
        self.assertEqual(sm.time_window('anything in the last 5 minutes?', now), (now - 300, now))
        self.assertEqual(sm.time_window('the past hour', now), (now - 3600, now))
        start, end = sm.time_window('what happened today', now)
        self.assertLess(start, now)
        self.assertEqual(end, now)
        self.assertIsNone(sm.time_window('when did the phone ring', now))

    def test_local_search_ranks_by_meaning_and_time(self):
        records = [self.record(label='hand clap', when=self.now - 50),
                   self.record(label='phone ringing', near='the shelf', when=self.now - 30),
                   self.record(speech='hello there', when=self.now - 20),
                   self.record(label='phone ringing', when=self.now - 7200)]
        top = sm.local_search('when did my phone ring?', records, now=self.now)
        self.assertEqual(top[0]['sound']['label'], 'phone ringing')
        self.assertEqual(top[0]['epoch'], self.now - 30)  # recent one first
        recent = sm.local_search('phone in the last 10 minutes', records, now=self.now)
        self.assertEqual(len(recent), 1)
        talk = sm.local_search('who was talking', records, now=self.now)
        self.assertEqual(talk[0]['speech']['text'], 'hello there')
        everything = sm.local_search('what happened in the last 2 minutes', records, now=self.now)
        self.assertEqual(len(everything), 3)

    def test_search_uses_muse_spark_but_only_real_ids(self):
        from types import SimpleNamespace
        from unittest import mock
        records = [self.record(label='phone ringing', when=self.now - 30), self.record(label='clap', when=self.now - 10)]
        meta = SimpleNamespace(answer_query=mock.Mock(return_value={
            'matches': ['ev-invented', records[0]['id']], 'answer': 'The phone rang 30 s ago, to your left.'}))
        result = sm.search('when did the phone ring?', records, meta, now=self.now)
        self.assertEqual(result['source'], 'muse-spark')
        self.assertEqual([r['id'] for r in result['matches']], [records[0]['id']])
        self.assertIn('Phone ringing', result['answer'])
        table = meta.answer_query.call_args.args[1]
        self.assertIn(records[0]['id'], table)
        self.assertIn('phone ringing (muse-spark 0.90)', table)

    def test_search_falls_back_locally_when_meta_fails(self):
        from types import SimpleNamespace
        from unittest import mock
        from meta_api import BudgetError
        records = [self.record(label='phone ringing', when=self.now - 30)]
        meta = SimpleNamespace(answer_query=mock.Mock(side_effect=BudgetError('daily Meta budget reached')))
        result = sm.search('phone', records, meta, now=self.now)
        self.assertEqual(result['source'], 'local')
        self.assertEqual(len(result['matches']), 1)
        self.assertIn('budget', result['answer'])
        self.assertEqual(sm.search('anything', [], None)['answer'], 'No sounds recorded yet.')

    def test_phone_history_question_returns_historical_highlight_and_level_units(self):
        rec = self.record(label='phone ringing', when=self.now - 30)
        unrelated = self.record(label='hand clap', when=self.now - 5)
        result = sm.search('was there any point my phone rang?', [rec, unrelated], now=self.now)
        self.assertEqual([r['id'] for r in result['matches']], [rec['id']])
        mark = result['highlights'][0]
        self.assertEqual(mark['record_id'], rec['id'])
        self.assertEqual(mark['pixel'], [320, 200])
        self.assertEqual(mark['point_m'], rec['where']['point_m'])
        self.assertTrue(mark['historical'])
        self.assertFalse(mark['source_verified'])
        self.assertIn('UTC', result['answer'])
        self.assertIn('2026-11-04', result['answer'])
        self.assertIn('dB SNR', result['answer'])
        self.assertIn('-30.0 dBFS', result['answer'])
        self.assertIn('not calibrated dB SPL', result['answer'])
        self.assertRegex(rec['time'], r'T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}$')

    def test_phone_query_does_not_call_unlabelled_tone_a_phone(self):
        self.assertEqual(sm.local_search('did my phone ring?', [self.record()], now=self.now), [])

    def test_model_cannot_invent_hidden_location_time_or_loudness(self):
        from types import SimpleNamespace
        from unittest import mock
        rec = self.record(label='phone ringing', near='the desk', when=self.now - 30)
        meta = SimpleNamespace(answer_query=mock.Mock(return_value={
            'matches': [rec['id'], rec['id']],
            'answer': 'Your phone was under the pillow at midnight and measured 100 dB SPL.'}))
        result = sm.search('where was my phone ringing?', [rec], meta, now=self.now)
        self.assertEqual(len(result['matches']), 1)
        self.assertNotIn('pillow', result['answer'])
        self.assertNotIn('midnight', result['answer'])
        self.assertNotIn('100 dB', result['answer'])
        self.assertIn('near the desk', result['answer'])

    def test_model_answer_without_real_ids_cannot_claim_an_event(self):
        from types import SimpleNamespace
        from unittest import mock
        rec = self.record(label='clap')
        meta = SimpleNamespace(answer_query=mock.Mock(return_value={
            'matches': ['invented'], 'answer': 'The phone definitely rang under the pillow.'}))
        result = sm.search('did my phone ring?', [rec], meta, now=self.now)
        self.assertEqual(result['matches'], [])
        self.assertNotIn('pillow', result['answer'])
        self.assertIn('No matching sound', result['answer'])

    def test_empty_time_window_does_not_widen_or_spend_a_model_call(self):
        from types import SimpleNamespace
        from unittest import mock
        rec = self.record(label='phone ringing', when=self.now - 3600)
        meta = SimpleNamespace(answer_query=mock.Mock())
        result = sm.search('phone ringing in the last 2 minutes', [rec], meta, now=self.now)
        self.assertEqual(result['matches'], [])
        self.assertIn('for that time', result['answer'])
        meta.answer_query.assert_not_called()

    def test_occlusion_context_requires_observed_placement_evidence(self):
        rec = self.record(label='phone ringing', near='under the pillow')
        self.assertNotIn('under the pillow', sm.phrase(rec, now=self.now))
        self.assertNotIn('under the pillow', sm.table([rec], now=self.now))
        rec['spatial_evidence'] = {'verified': True, 'method': 'observed_placement',
                                   'text': 'phone under the pillow'}
        self.assertIn('under the pillow', sm.phrase(rec, now=self.now))

    def test_nonfinite_depth_is_not_saved_as_a_real_distance(self):
        rec = sm.make_record(fix(), [0, 0, 1], view={'point': [0, 0, float('nan')], 'measured': True},
                             now=self.now)
        self.assertIsNone(rec['where']['point_m'])
        self.assertIsNone(rec['where']['distance_m'])


class Examples(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_nearest_example_and_sheet(self):
        bank = sm.ReferenceBank(self.dir.name)
        self.assertEqual(bank.nearest(tone()), (None, 0.0))
        bank.add('phone ringing', wav_bytes(tone(1800) + tone(2400)))
        bank.add('hand clap', wav_bytes(clicks()))
        label, similarity = bank.nearest(tone(1800, level=0.05) + tone(2400, level=0.05))  # quieter: same label
        self.assertEqual(label, 'phone ringing')
        self.assertGreater(similarity, 0.9)
        self.assertEqual(bank.nearest(clicks(n=3))[0], 'hand clap')
        reloaded = sm.ReferenceBank(self.dir.name)
        self.assertEqual(reloaded.labels(), ['hand clap', 'phone ringing'])
        sheet = reloaded.sheet(wav_bytes(tone(1800)))
        self.assertTrue(sheet.startswith('data:image/jpeg;base64,'))


class Talkers(unittest.TestCase):
    def test_names_stick_to_directions(self):
        speakers = sm.SpeakerMap(tolerance_deg=20)
        self.assertEqual(speakers.name_for(-40, now=0), 'Speaker 1')
        self.assertEqual(speakers.name_for(35, now=1), 'Speaker 2')
        self.assertEqual(speakers.name_for(-45, now=2), 'Speaker 1')
        self.assertEqual(speakers.name_for(None, now=3), 'Someone')
        self.assertEqual([s[0] for s in speakers.summary(now=4)], ['Speaker 1', 'Speaker 2'])
        self.assertEqual(speakers.name_for(-40, now=10000), 'Speaker 3')  # forgotten after a while

    def test_turns_take_the_direction_heard_during_them(self):
        track = [(10.0 + 0.1 * i, -30.0 if i < 20 else 40.0, 0.0) for i in range(40)]  # left, then right
        turns = [{'start_s': 0.0, 'end_s': 1.8}, {'start_s': 2.2, 'end_s': 3.9}, {'start_s': 9.0, 'end_s': 9.5}]
        az = sm.turn_directions(turns, track, clip_end_at=14.0, clip_seconds=4.0)
        self.assertEqual(az[0], -30.0)
        self.assertEqual(az[1], 40.0)
        self.assertIsNone(az[2])


if __name__ == '__main__':
    unittest.main()
