"""Possible alerts you confirm or reject, naming that teaches, speech subtitles, the top toolbar:
python -m unittest test_possible   (hardware-free)
"""
import math
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import aim
import scope
import scope_ui
import sound_memory as sm
from insight import Enricher
from meta_api import wav_bytes


def ring(seconds=1.0, f=(1400, 1800)):
    t = np.arange(int(16000 * seconds)) / 16000
    return sum(np.sin(2 * np.pi * fi * t) for fi in f) * 0.3


def noise(seconds=1.2):
    return np.random.default_rng(1).normal(0, 0.2, int(16000 * seconds))


def record(snr=12.0, x=None):
    x = ring() if x is None else x
    rec = sm.make_record({'snr_db': snr, 'seconds': len(x) / 16000, 'z': 30}, [0.3, 0, 1.0],
                         profile=sm.acoustic_profile(x),
                         view={'spot': (100, 80), 'point': np.array([0.3, 0.0, 1.0]), 'measured': True,
                               'quality': {'state': 'candidate'}})
    return rec


class Tiers(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.results = []
        self.enricher = Enricher(sm.Memory(self.dir.name), None, set(), on_result=lambda *r: self.results.append(r),
                                 clock=lambda: 1000.0)
        self.enricher.watch = ['knock on the door', 'smoke alarm']

    def kinds(self):
        return [(r[1], r[2]['watch']) for r in self.results]

    def test_sure_is_an_alert_unsure_is_only_possible(self):
        self.enricher._alerts('a', record(12.0), ['knock on the door'], 'Muse Spark', 0.9)
        self.enricher._alerts('b', record(12.0), ['smoke alarm'], 'Muse Spark', 0.5)
        self.assertEqual(self.kinds(), [('alert', 'knock on the door'), ('possible', 'smoke alarm')])

    def test_faint_or_very_unsure_sounds_are_not_offered_at_all(self):
        self.enricher._alerts('a', record(2.0, noise()), ['knock on the door'], 'Muse Spark', 0.9)
        self.enricher._alerts('b', record(0.5), ['smoke alarm'], 'Muse Spark', 0.9)
        self.enricher._alerts('c', record(12.0), ['knock on the door'], 'Muse Spark', 0.2)
        self.assertEqual(self.results, [])

    def test_a_quiet_steady_ring_is_offered_as_possible(self):
        # Nov 5: a real ring at 1148 Hz measured +1.5 to +3 dB over the room (broadband)
        self.enricher._alerts('a', record(2.0), ['phone ringing'], 'Muse Spark', 0.6)
        self.assertEqual(self.kinds(), [('possible', 'phone ringing')])

    def test_a_sound_you_said_no_to_never_flags_that_watch_again(self):
        self.enricher._alerts('a', record(12.0), ['knock on the door'], 'Muse Spark', 0.95,
                              nearest=('not knock on the door', 0.93))
        self.assertEqual(self.results, [])
        self.enricher._alerts('b', record(12.0), ['knock on the door'], 'Muse Spark', 0.95,
                              nearest=('not knock on the door', 0.6))   # not similar enough: still alerts
        self.assertEqual(self.kinds(), [('alert', 'knock on the door')])


class Answers(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        for p in (mock.patch.object(sm, 'MEMORY_DIR', aim.Path(self.dir.name)),
                  mock.patch.object(sm, 'REFERENCES_DIR', aim.Path(self.dir.name) / 'refs'),
                  mock.patch.object(scope, 'beep')):
            p.start()
            self.addCleanup(p.stop)
        self.cfg = aim.load_config(profile='tripod')
        self.session = scope.Session(SimpleNamespace(memory=True), self.cfg, {'note': ''})
        self.addCleanup(self.session.close)

    def heard(self):
        fix = {'direction': aim.camera_to_array([0.3, 0, 1.0], self.cfg), 'z': 30.0, 'snr_db': 12.0, 'seconds': 1.0}
        view = {'spot': (60, 40), 'point': np.array([0.3, 0.0, 1.0]), 'measured': True, 'quality': {'state': 'candidate'}}
        return self.session.record({'kind': 'sound', 'fix': fix, 'event_id': 1, 'observed_at': time.monotonic(),
                                    'wav': wav_bytes(ring())}, view, np.zeros((80, 120, 3), np.uint8))

    def offer(self, rec, watch='knock on the door'):
        self.session.results.append((rec['id'], 'possible', {'watch': watch, 'how': 'Muse Spark', 'confidence': 0.5}))
        notes = self.session.drain(now=100.0)
        self.assertIn('Possible knock on the door', notes[0])
        self.assertIsNone(self.session.alert)                           # no red banner, no beep
        self.assertEqual(self.session.possible['rid'], rec['id'])

    def test_yes_labels_the_sound_and_teaches_it(self):
        rec = self.heard()
        self.offer(rec)
        reply = self.session.answer_possible(True)
        self.assertIn('Similar sounds will be recognized', reply)
        self.assertEqual(sm.what_it_was(self.session.memory.get(rec['id'])), 'knock on the door')
        self.assertIn('knock on the door', self.session.references.labels())
        self.assertIsNone(self.session.possible)

    def test_no_teaches_a_rejection(self):
        rec = self.heard()
        self.offer(rec)
        reply = self.session.answer_possible(False)
        self.assertIn('will not be flagged', reply)
        self.assertIn('not knock on the door', self.session.references.labels())
        self.assertEqual(self.session.memory.get(rec['id'])['sound']['not'], ['knock on the door'])

    def test_the_card_expires(self):
        rec = self.heard()
        self.offer(rec)
        self.session.drain(now=100.0 + scope.POSSIBLE_S + 1)
        self.assertIsNone(self.session.possible)

    def test_a_typed_name_teaches_and_speaker_names_rename_captions(self):
        rec = self.heard()
        self.assertTrue(self.session.teach(rec['id'], 'my kettle'))
        self.assertIn('my kettle', self.session.references.labels())
        self.session.speakers.name_for(15.0)
        self.session.memory.update(rec['id'], {'speech': {'text': 'hi', 'speaker': 'Speaker 1',
                                                          'turns': [{'name': 'Speaker 1', 'text': 'hi'}]}})
        self.assertEqual(sm.parse_naming('Speaker 1 is Keith'), ('speaker', 'Speaker 1', 'Keith'))
        self.assertEqual(self.session.speakers.rename('speaker one', 'Keith'), 'Speaker 1')
        self.assertEqual(sm.rename_speaker(self.session.memory, 'Speaker 1', 'Keith'), 1)
        self.assertEqual(self.session.memory.get(rec['id'])['speech']['speaker'], 'Keith')


class Claps(unittest.TestCase):
    # Profiles measured on the real board, Nov 4.
    CLAPS = [dict(duration_s=0.14, attack_ms=20, crest_db=21.6, flatness=0.116, centroid_hz=1760, onsets=2,
                  tone_share=0.193, speech_score=0.0, snr=30.1),
             dict(duration_s=0.08, attack_ms=30, crest_db=21.8, flatness=0.097, centroid_hz=1366, onsets=2,
                  tone_share=0.073, speech_score=0.0, snr=14.7)]
    OTHERS = [dict(duration_s=0.33, crest_db=14.2, centroid_hz=677, onsets=1, tone_share=0.424, speech_score=0.0,
                   snr=24.2),                                                   # knock (low)
              dict(duration_s=0.21, crest_db=8.9, centroid_hz=1265, onsets=1, tone_share=0.927, speech_score=0.0,
                   snr=26.2),                                                   # electronic beep (tonal)
              dict(duration_s=0.27, crest_db=12.1, centroid_hz=335, onsets=1, tone_share=0.476, speech_score=0.0,
                   snr=1.6),                                                    # room thump (faint, low)
              dict(duration_s=0.72, crest_db=14.7, centroid_hz=878, onsets=1, tone_share=0.213, speech_score=0.79,
                   snr=13.1)]                                                   # speech

    def test_real_claps_are_claps_and_other_sounds_are_not(self):
        for c in self.CLAPS:
            self.assertTrue(sm.looks_like_clap(c, c['snr']), c)
        for o in self.OTHERS:
            self.assertFalse(sm.looks_like_clap(o, o['snr']), o)

    def test_a_clap_is_labeled_instantly_and_never_sent_to_a_model(self):
        results = []
        enricher = Enricher(sm.Memory(tempfile.mkdtemp()), mock.Mock(), {'identify', 'context'},
                            on_result=lambda *r: results.append(r))
        enricher.watch = ['knock on the door']
        rec = record(30.0)
        rec['sound'].update(label='clap', source='local', confidence=0.85)
        enricher.process({'record': rec, 'wav': wav_bytes(ring(0.1)), 'requested_at': time.time()})
        enricher.meta.classify_sound.assert_not_called()
        enricher.meta.source_check.assert_not_called()
        self.assertEqual(results, [])


class QuietRingsAndYeah(unittest.TestCase):
    """Live, Nov 5: a quiet ring was never named, and hum or a ring became "Yeah" captions."""

    def setUp(self):
        self.results = []
        self.meta = mock.Mock()
        self.meta.classify_sound.return_value = {'label': 'phone ringing', 'category': 'ring or alarm',
                                                 'description': 'a phone ringing', 'confidence': 0.6,
                                                 'watch': ['phone ringing']}
        self.meta.transcribe_turns.return_value = {'text': 'Yeah.', 'turns': [
            {'text': 'Yeah.', 'start_s': 0.0, 'end_s': 0.6, 'speaker': 'A'}]}
        self.enricher = Enricher(sm.Memory(tempfile.mkdtemp()), self.meta, {'identify', 'captions'},
                                 on_result=lambda *r: self.results.append(r), identify_limit=(0.0, 100))
        self.enricher.watch = ['phone ringing']
        self.enricher.listen_for_names = True     # --name Keith

    def run_task(self, rec, x):
        rec['loudness'].setdefault('peak_dbfs', -65.0)   # raw XVF3800 level, as on the real board
        self.enricher.process({'record': rec, 'wav': wav_bytes(x), 'speech_wav': wav_bytes(x),
                               'requested_at': time.time()})

    def test_filler_transcripts(self):
        for text in ('Yeah.', 'yeah', 'Yeah, yeah.', '[Music]', '(upbeat music)', '♪', 'Thank you.', ''):
            self.assertTrue(sm_filler(text), text)
        for text in ("Keith, dinner's ready", 'Can you hear me?', 'Testing one two three', 'Hello'):
            self.assertFalse(sm_filler(text), text)

    def test_a_quiet_ring_is_named_not_transcribed(self):
        self.run_task(record(2.0), ring())
        self.meta.transcribe_turns.assert_not_called()
        self.meta.classify_sound.assert_called_once()
        facts = self.meta.classify_sound.call_args.args[1]
        self.assertEqual(facts['same_tone_in_last_15_s'], 0)
        self.assertIn(('possible', 'phone ringing'), [(r[1], r[2].get('watch')) for r in self.results
                                                       if isinstance(r[2], dict)])
        self.run_task(record(2.0), ring())          # the next ring of the same phone
        self.assertEqual(self.meta.classify_sound.call_args.args[1]['same_tone_in_last_15_s'], 1)

    def test_the_name_listener_skips_sounds_under_3_db(self):
        self.run_task(record(2.0, noise()), noise())
        self.meta.transcribe_turns.assert_not_called()

    def test_a_lone_yeah_is_dropped_and_the_sound_is_named_instead(self):
        rec = record(12.0)
        rec['acoustic'].update(speech_score=0.6, tone_share=0.3, flatness=0.04)   # hum-tinged ring
        self.run_task(rec, ring())
        self.meta.transcribe_turns.assert_called_once()
        self.assertNotIn('captions', [r[1] for r in self.results])
        self.meta.classify_sound.assert_called_once()

    def test_real_speech_is_still_captioned(self):
        self.meta.transcribe_turns.return_value = {'text': "Keith, dinner's ready", 'turns': [
            {'text': "Keith, dinner's ready", 'start_s': 0.0, 'end_s': 1.2, 'speaker': 'A'}]}
        rec = record(8.0, noise())
        rec['acoustic'].update(speech_score=0.6)
        self.run_task(rec, noise())
        self.assertIn('captions', [r[1] for r in self.results])
        self.meta.classify_sound.assert_not_called()


def sm_filler(text):
    from insight import filler_transcript
    return filler_transcript(text)


class Screen(unittest.TestCase):
    def test_possible_card_buttons_are_clickable_where_they_are_drawn(self):
        frame = np.full((720, 1280, 3), 80, np.uint8)
        possible = {'rid': 'r', 'watch': 'knock on the door', 'where': 'to the right', 'when': '00:10:10'}
        with mock.patch.object(scope_ui, '_text', wraps=scope_ui._text) as text:
            scope_ui.render(frame, None, None, '', now=10, viewport_size=(1600, 900), possible=possible)
        labels = [str(c.args[1]) for c in text.call_args_list]
        self.assertIn('Knock on the door?', labels)
        self.assertIn('POSSIBLE', labels)
        rects = scope_ui.control_rects(frame.shape, (1600, 900), possible=True)
        for name in ('possible_yes', 'possible_no', 'possible_inspect'):
            x, y, w, h = rects[name]
            self.assertEqual(scope_ui.control_hit_test(x + w / 2, y + h / 2, frame.shape, (1600, 900), possible=True),
                             name)
        self.assertNotIn('possible_yes', scope_ui.control_rects(frame.shape, (1600, 900)))

    def test_action_buttons_sit_under_the_header_not_at_the_bottom(self):
        frame = np.full((720, 1280, 3), 80, np.uint8)
        rects = scope_ui.control_rects(frame.shape, (1600, 900))
        camera_top = scope_ui.camera_rect(frame.shape, (1600, 900))[1]
        for name in ('voice', 'analyze', 'heatmap', 'fullscreen', 'quit'):
            self.assertLess(rects[name][1] + rects[name][3], camera_top)
        x, y, w, h = scope_ui.camera_rect(frame.shape, (1600, 900))
        self.assertEqual(scope_ui.canvas_to_frame(x + w / 2, y + h / 2, frame.shape, (1600, 900)), (640, 360))


if __name__ == '__main__':
    unittest.main()
