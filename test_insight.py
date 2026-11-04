"""Offline tests for the Meta enrichment worker: python -m unittest test_insight

A fake Meta records every call, so these check WHEN money would be spent, not model quality.
"""
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import sound_memory as sm
from insight import Enricher, RateLimit
from meta_api import BudgetError, wav_bytes
from test_sound_memory import clicks, tone, voice


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def fake_meta():
    return SimpleNamespace(
        classify_sound=mock.Mock(return_value={'label': 'hand clap', 'category': 'clap or knock',
                                               'description': 'sharp bursts', 'confidence': 0.85, 'watch': []}),
        source_check=mock.Mock(return_value={'sound': 'phone ringing', 'sound_confidence': 0.8,
                                             'object': 'a black phone', 'action': 'screen lit up',
                                             'consistent': 'yes', 'near': 'a laptop', 'surface': 'desk',
                                             'watch': [], 'summary': 'A phone rang on the desk.'}),
        transcribe_turns=mock.Mock(return_value={'text': 'hi there where are my keys', 'turns': [
            {'start_s': 0.0, 'end_s': 0.9, 'text': 'hi there', 'speaker': 'A'},
            {'start_s': 1.2, 'end_s': 2.0, 'text': 'where are my keys', 'speaker': 'B'}]}),
        describe_near=mock.Mock(return_value={'near': 'a laptop and a mug', 'surface': 'desk'}),
        segment_at=mock.Mock(return_value=[np.array([[1, 1], [9, 1], [9, 9]])]))


class Worker(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.memory = sm.Memory(self.dir.name)
        self.meta = fake_meta()
        self.results = []
        self.clock = Clock()

    def enricher(self, features=('identify', 'captions', 'context', 'outline'), references=None):
        return Enricher(self.memory, self.meta, features, on_result=lambda *r: self.results.append(r),
                        references=references, clock=self.clock)

    def task(self, x, azimuth=-30.0, snr=20.0, **extra):
        a = math.radians(azimuth)
        view = {'spot': (100, 80), 'point': np.array([0.3, 0.0, 1.0]), 'measured': True,
                'quality': {'state': 'candidate'}}
        rec = sm.make_record({'snr_db': snr, 'seconds': len(x) / 16000, 'z': 30}, [math.sin(a), 0, math.cos(a)],
                             profile=sm.acoustic_profile(x), view=view)
        self.memory.add(rec)
        return {'record': rec, 'wav': wav_bytes(x), **extra}

    def test_clear_sound_is_labeled_once(self):
        e = self.enricher()
        e.process(self.task(clicks()))
        self.meta.classify_sound.assert_called_once()
        self.meta.transcribe_turns.assert_not_called()
        rid = self.results[-1][0]
        self.assertEqual(self.memory.get(rid)['sound']['label'], 'hand clap')
        self.assertEqual(self.memory.get(rid)['sound']['source'], 'muse-spark')

    def test_weak_sound_costs_nothing(self):
        e = self.enricher()
        t = self.task(clicks(), snr=2.0)
        t['record']['where']['quality'] = 'weak'
        e.process(t)
        self.meta.classify_sound.assert_not_called()

    def test_speech_gets_captions_named_by_direction(self):
        e = self.enricher()
        x = voice(2.2)
        track = [(10.0 + 0.1 * i, -35.0 if i < 10 else 30.0, 0.0) for i in range(23)]
        e.process(self.task(x, track=track, clip_end_at=12.2, clip_seconds=2.2))
        self.meta.transcribe_turns.assert_called_once()
        self.meta.classify_sound.assert_not_called()  # speech is captioned, not labeled from a picture
        kind, turns = self.results[-1][1], self.results[-1][2]
        self.assertEqual(kind, 'captions')
        self.assertEqual([t['name'] for t in turns], ['Speaker 1', 'Speaker 2'])
        self.assertEqual(turns[0]['where'], 'to the left')
        self.assertEqual(turns[1]['where'], 'to the right')
        rec = self.memory.get(self.results[-1][0])
        self.assertEqual(rec['speech']['text'], 'hi there where are my keys')

    def test_context_is_cached_per_place(self):
        e = self.enricher(features=('context',))
        e.process(self.task(tone(), crop_jpeg=b'jpeg'))
        e.process(self.task(tone(), crop_jpeg=b'jpeg'))   # same place, a moment later: no new request
        self.meta.describe_near.assert_called_once()
        rec = self.memory.get(self.results[-1][0])
        self.assertEqual(rec['near']['text'], 'a laptop and a mug on the desk')
        self.clock.t += 601
        e.process(self.task(tone(), crop_jpeg=b'jpeg'))   # cache expired
        self.assertEqual(self.meta.describe_near.call_count, 2)

    def test_outline_only_when_asked(self):
        e = self.enricher()
        e.process(self.task(tone()))
        self.meta.segment_at.assert_not_called()
        e.process(self.task(tone(), outline=True, frame_jpeg=b'j', frame_size=(640, 400), pixel=(100, 80), scale=0.5))
        self.meta.segment_at.assert_called_once_with(b'j', 640, 400, 100, 80)
        kind, payload = self.results[-1][1], self.results[-1][2]
        self.assertEqual(kind, 'outline')
        self.assertEqual(payload['scale'], 0.5)

    def test_budget_exhaustion_pauses_meta_but_not_memory(self):
        self.meta.classify_sound.side_effect = BudgetError('daily Meta budget reached')
        e = self.enricher()
        e.process(self.task(clicks()))
        self.clock.t += 60
        e.process(self.task(clicks()))
        self.meta.classify_sound.assert_called_once()
        self.assertIn('paused', [r for r in self.results if r[1] == 'notice'][0][2])
        self.assertEqual(len(self.memory.recent()), 2)

    def test_your_examples_label_for_free(self):
        bank = sm.ReferenceBank(tempfile.mkdtemp())
        bank.add('hand clap', wav_bytes(clicks()))
        e = self.enricher(references=bank)
        e.process(self.task(clicks(n=4)))
        rec = self.memory.get(self.results[0][0])
        self.assertEqual(rec['sound']['local_label'], 'hand clap')
        if rec['sound']['local_similarity'] >= 0.92:
            self.meta.classify_sound.assert_not_called()  # trusted local match: no Meta call

    def test_visible_sound_gets_one_audio_visual_check(self):
        e = self.enricher()
        e.watch = ['phone ringing']
        self.meta.source_check.return_value = {**self.meta.source_check.return_value, 'watch': ['phone ringing']}
        e.process(self.task(clicks(), crop_jpeg=b'after', crop_before=b'before'))
        self.meta.source_check.assert_called_once()
        args = self.meta.source_check.call_args.args
        self.assertEqual((args[1], args[2]), (b'after', b'before'))
        self.assertEqual(args[4], ['phone ringing'])
        self.meta.classify_sound.assert_not_called()   # no second call for the label
        self.meta.describe_near.assert_not_called()    # nor for the context
        rid = self.results[0][0]
        rec = self.memory.get(rid)
        self.assertEqual(rec['source']['object'], 'a black phone')
        self.assertEqual(rec['near']['text'], 'a laptop on the desk')
        self.assertEqual(rec['sound']['source'], 'muse-spark audio+visual')
        alerts = [r for r in self.results if r[1] == 'alert']
        self.assertEqual(alerts[0][2]['watch'], 'phone ringing')
        text = sm.phrase(rec)
        self.assertIn('at the spot: a black phone (screen lit up), sound and sight agree', text)

    def test_agreeing_source_gets_outlined_and_old_sounds_cost_nothing(self):
        e = self.enricher()
        e.process(self.task(clicks(), crop_jpeg=b'after', frame_jpeg=b'frame', frame_size=(640, 400),
                            pixel=(100, 80), scale=0.5))
        self.meta.segment_at.assert_called_once_with(b'frame', 640, 400, 100, 80)  # consistent: yes
        self.meta.source_check.reset_mock()
        self.clock.t += 60
        old = self.task(clicks(), crop_jpeg=b'after')
        old['record']['epoch'] -= 120            # a backlog: this sound is two minutes old
        e.process(old)
        self.meta.source_check.assert_not_called()
        self.meta.classify_sound.assert_not_called()

    def test_off_screen_sound_is_labeled_from_the_spectrogram_only(self):
        e = self.enricher()
        e.process(self.task(clicks()))  # no crop: the camera cannot see it
        self.meta.source_check.assert_not_called()
        self.meta.classify_sound.assert_called_once()

    def test_examples_raise_watch_alerts_for_free(self):
        bank = sm.ReferenceBank(tempfile.mkdtemp())
        x = clicks()
        bank.add('hand clap', wav_bytes(x))
        e = self.enricher(features=(), references=bank)
        e.watch = ['clapping hands']
        e.process(self.task(x))
        alerts = [r for r in self.results if r[1] == 'alert']
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0][2]['how'], 'your example')

    def test_rate_limit(self):
        clock = Clock()
        limit = RateLimit(2.0, 3, clock=clock)
        self.assertTrue(limit.allow())
        self.assertFalse(limit.allow())
        for _ in range(2):
            clock.t += 2.5
            self.assertTrue(limit.allow())
        clock.t += 2.5
        self.assertFalse(limit.allow())  # 3 per minute
        clock.t += 60
        self.assertTrue(limit.allow())


if __name__ == '__main__':
    unittest.main()
