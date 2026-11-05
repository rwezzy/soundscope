"""Offline paid-policy tests; every Meta endpoint is a mock."""
import copy
import math
import os
from pathlib import Path
import time
import unittest
from unittest import mock

import numpy as np

from insight import Enricher, likely_speech
from meta_api import BudgetError
from sound_memory import acoustic_profile, wav_samples
import test_insight as fixtures
from test_sound_memory import tone, voice


class PaidPolicy(unittest.TestCase):
    task = fixtures.Worker.task
    enricher = fixtures.Worker.enricher

    def setUp(self):
        fixtures.Worker.setUp(self)
        env = mock.patch.dict(os.environ, {'SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS': '-50'})
        env.start()
        self.addCleanup(env.stop)

    def visual_task(self, *, explicit=False):
        task = self.task(tone(), crop_jpeg=b'crop', frame_jpeg=b'frame', outline=True,
                         frame_size=(640, 480), pixel=(100, 80))
        if explicit:
            task.update(explicit=True, requested_at=time.time())
        return task

    def assert_no_paid_calls(self):
        for name in ('classify_sound', 'source_check', 'transcribe_turns', 'describe_near', 'segment_at'):
            getattr(self.meta, name).assert_not_called()

    def test_quiet_auto_events_cannot_bypass_floor_through_any_branch(self):
        for features in (('identify',), ('captions',), ('context',), ('outline',),
                         ('identify', 'captions', 'context', 'outline')):
            with self.subTest(features=features):
                worker = self.enricher(features=features)
                worker.listen_for_names = True
                task = self.visual_task()
                task['record']['loudness']['peak_dbfs'] = -59.0
                worker.process(task)
                self.assert_no_paid_calls()
                self.assertIsNotNone(self.memory.get(task['record']['id']))
        worker = self.enricher()
        task = self.task(voice(2))
        task['record']['loudness']['peak_dbfs'] = -59.0
        worker.process(task)
        self.assert_no_paid_calls()

    def test_missing_or_invalid_raw_peak_does_not_use_normalised_profile_peak(self):
        for peak in (None, math.nan, math.inf, -math.inf, 'bad', True):
            with self.subTest(peak=peak):
                task = self.visual_task()
                task['record']['loudness']['peak_dbfs'] = peak
                task['record']['acoustic']['peak_dbfs'] = -1  # normalised WAV; not the input level
                self.enricher().process(task)
                self.assert_no_paid_calls()

    def test_invalid_visual_evidence_never_reaches_camera_models_even_explicitly(self):
        changes = ({'quality': 'ambiguous'}, {'quality': 'weak'}, {'on_screen': False},
                   {'point_m': None}, {'point_m': [0, 0, math.nan]}, {'point_m': [0, 0, -1]},
                   {'distance_m': None}, {'distance_kind': None})
        for change in changes:
            for features in (('context',), ('outline',), ('identify', 'context', 'outline')):
                with self.subTest(change=change, features=features):
                    for name in vars(self.meta):
                        getattr(self.meta, name).reset_mock()
                    task = self.visual_task(explicit=True)
                    task['record']['where'].update(change)
                    self.enricher(features=features).process(task)
                    self.meta.source_check.assert_not_called()
                    self.meta.describe_near.assert_not_called()
                    self.meta.segment_at.assert_not_called()
                    if 'identify' in features:
                        self.meta.classify_sound.assert_called_once()  # audio label remains possible

    def test_fresh_explicit_request_analyzes_old_quiet_record_without_changing_time(self):
        task = self.visual_task(explicit=True)
        task['record']['epoch'] -= 300
        task['record']['loudness']['peak_dbfs'] = -65
        original_epoch = task['record']['epoch']
        self.enricher().process(task)
        self.meta.source_check.assert_called_once()
        self.meta.segment_at.assert_called_once()
        self.assertEqual(task['record']['epoch'], original_epoch)

    def test_explicit_weak_ring_gets_one_audio_label_without_relaxing_source_gate(self):
        task = self.visual_task(explicit=True)
        task['record']['loudness'].update(snr_db=1.0, peak_dbfs=-65.0)
        epoch = task['record']['epoch']
        self.enricher(features=('identify', 'context')).process(task)
        self.meta.classify_sound.assert_called_once()
        self.meta.source_check.assert_not_called()
        self.meta.describe_near.assert_not_called()
        self.meta.transcribe_turns.assert_not_called()
        self.assertEqual(task['record']['epoch'], epoch)
        self.assertIn('audio-only', task['_analysis_status'])

    def test_explicit_request_can_check_a_trusted_local_match_once(self):
        task = self.visual_task(explicit=True)
        task['record']['sound'].update(local_label='old example', local_similarity=0.99)
        self.enricher(features=('identify', 'context')).process(task)
        self.meta.source_check.assert_not_called()
        self.meta.classify_sound.assert_called_once()

    def test_explicit_ambiguous_audio_tries_asr_before_one_audio_only_fallback(self):
        task = self.visual_task(explicit=True)
        task['record']['acoustic'].update(speech_score=0.8, tone_share=0.3)
        task['record']['duration_s'] = 1.0
        task['record']['loudness']['snr_db'] = 1.0
        order = mock.Mock()
        order.attach_mock(self.meta.transcribe_turns, 'asr')
        order.attach_mock(self.meta.classify_sound, 'label')
        self.meta.transcribe_turns.return_value = {'text': '', 'turns': []}
        self.enricher().process(task)
        self.assertEqual([call[0] for call in order.mock_calls], ['asr', 'label'])
        self.meta.source_check.assert_not_called()
        self.meta.describe_near.assert_not_called()
        self.meta.segment_at.assert_not_called()

    def test_explicit_blank_asr_turns_do_not_count_as_a_usable_transcript(self):
        task = self.visual_task(explicit=True)
        task['record']['acoustic'].update(speech_score=0.8, tone_share=0.3)
        task['record']['duration_s'] = 1.0
        self.meta.transcribe_turns.return_value = {
            'text': ' ', 'turns': [{'text': ' ', 'start_s': 0, 'end_s': 1, 'speaker': 'A'}]}
        self.enricher().process(task)
        self.meta.classify_sound.assert_called_once()
        self.assertFalse(any(kind == 'captions' for _, kind, _ in self.results))

    def test_explicit_speech_like_event_can_fall_back_when_captions_are_disabled(self):
        task = self.task(voice(2))
        task.update(explicit=True, requested_at=time.time())
        self.enricher(features=('identify',)).process(task)
        self.meta.transcribe_turns.assert_not_called()
        self.meta.classify_sound.assert_called_once()
        self.assertIn('audio-only', task['_analysis_status'])

    def test_explicit_usable_speech_needs_no_second_label_call(self):
        task = self.task(voice(2))
        task.update(explicit=True, requested_at=time.time())
        self.enricher().process(task)
        self.meta.transcribe_turns.assert_called_once()
        self.meta.classify_sound.assert_not_called()
        self.meta.source_check.assert_not_called()
        self.assertIn('transcript updated', task['_analysis_status'])

    def test_explicit_successful_source_check_needs_no_second_label_call(self):
        task = self.visual_task(explicit=True)
        self.enricher().process(task)
        self.meta.source_check.assert_called_once()
        self.meta.classify_sound.assert_not_called()

    def test_explicit_empty_source_result_falls_back_once(self):
        task = self.visual_task(explicit=True)
        self.meta.source_check.return_value = None
        self.enricher().process(task)
        self.meta.source_check.assert_called_once()
        self.meta.classify_sound.assert_called_once()

    def test_automatic_empty_asr_does_not_add_a_new_paid_fallback(self):
        task = self.task(voice(2))
        self.meta.transcribe_turns.return_value = {'text': '', 'turns': []}
        self.enricher().process(task)
        self.meta.transcribe_turns.assert_called_once()
        self.meta.classify_sound.assert_not_called()

    def test_explicit_feature_rate_and_empty_result_give_specific_status(self):
        task = self.visual_task(explicit=True)
        self.enricher(features=()).process(task)
        self.assert_no_paid_calls()
        self.assertIn('feature is not enabled', task['_analysis_status'])
        worker = self.enricher(features=('identify',))
        self.meta.classify_sound.return_value = None
        worker.process(task)
        self.assertIn('no usable label', task['_analysis_status'])
        worker.process(task)
        self.assertIn('rate limit', task['_analysis_status'])
        self.meta.classify_sound.assert_called_once()

    def test_explicit_asr_budget_failure_cannot_spend_on_fallback(self):
        task = self.task(voice(2))
        task.update(explicit=True, requested_at=time.time())
        self.meta.transcribe_turns.side_effect = BudgetError('daily limit reached')
        self.enricher().process(task)
        self.meta.transcribe_turns.assert_called_once()
        self.meta.classify_sound.assert_not_called()
        self.assertIn('budget is paused', task['_analysis_status'])

    def test_explicit_stale_missing_or_invalid_request_time_cannot_spend(self):
        for requested in (None, time.time() - 60, time.time() + 60, math.nan, math.inf, 'bad'):
            with self.subTest(requested=requested):
                task = self.visual_task(explicit=True)
                task['requested_at'] = requested
                self.enricher().process(task)
                self.assert_no_paid_calls()

    def test_explicit_analysis_does_not_override_budget_or_rate(self):
        task = self.visual_task(explicit=True)
        task['record']['loudness']['peak_dbfs'] = -65
        worker = self.enricher()
        worker.exhausted = 'Budget used'
        worker.process(task)
        self.assert_no_paid_calls()
        worker = self.enricher(features=('identify',))
        worker.process(task)
        worker.process(task)
        self.meta.classify_sound.assert_called_once()

    def test_environment_floor_is_configurable_but_must_be_valid(self):
        with mock.patch.dict(os.environ, {'SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS': '-70'}):
            task = self.visual_task()
            task['record']['loudness']['peak_dbfs'] = -65
            self.enricher(features=('identify',)).process(task)
            self.meta.classify_sound.assert_called_once()
        for value in ('nan', 'inf', 'bad', '1', '-161', ''):
            with self.subTest(value=value), mock.patch.dict(os.environ, {'SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS': value}):
                with self.assertRaisesRegex(ValueError, 'SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS'):
                    Enricher(None, None, ())

    def test_automatic_media_preflight_is_pure_and_cannot_be_explicitly_overridden(self):
        worker = self.enricher()
        record = self.visual_task()['record']
        before = copy.deepcopy(record)
        for _ in range(3):
            self.assertTrue(worker.automatic_media_eligible(record))
        self.assertEqual(record, before)
        self.assert_no_paid_calls()
        self.assertTrue(all(not limit.times for limit in worker.limits.values()))
        for peak in (-65.0, None, math.nan):
            record['loudness']['peak_dbfs'] = peak
            record.update(explicit=True, requested_at=time.time())
            self.assertFalse(worker.automatic_media_eligible(record))
        record['loudness']['peak_dbfs'] = 0.0  # zero dBFS is a valid loud peak, not missing data
        self.assertTrue(worker.automatic_media_eligible(record))
        record['where']['quality'] = 'ambiguous'
        self.assertTrue(worker.automatic_audio_eligible(record))
        self.assertFalse(worker.automatic_media_eligible(record))
        record['where'] = []
        self.assertFalse(worker.automatic_media_eligible(record))
        self.assertFalse(worker.automatic_media_eligible(None))

    def test_modulated_tones_use_one_audio_label_across_pitch_and_cadence(self):
        t = np.arange(4 * 16000) / 16000
        noise = np.random.default_rng(42).normal(0, 0.004, len(t))
        misrouted_before = 0
        for hz in (440, 700, 900, 1400, 2200):
            for cadence in (2, 5, 8):
                with self.subTest(hz=hz, cadence=cadence):
                    for name in vars(self.meta):
                        getattr(self.meta, name).reset_mock()
                    phase = 2 * np.pi * hz * t + 3 * np.sin(2 * np.pi * 1.7 * t)
                    envelope = 0.5 * (1 + np.sign(np.sin(2 * np.pi * cadence * t)))
                    x = 0.1 * np.sin(phase) * envelope + noise
                    task = self.task(x)
                    acoustic = task['record']['acoustic']
                    misrouted_before += acoustic['speech_score'] >= 0.5
                    self.assertFalse(likely_speech(acoustic))
                    self.enricher().process(task)
                    self.meta.classify_sound.assert_called_once()
                    self.meta.transcribe_turns.assert_not_called()
                    facts = self.meta.classify_sound.call_args.args[1]
                    self.assertIn('flatness', facts)
                    self.assertIn('tone_share', facts)
                    self.assertNotIn('character', facts)
                    self.assertNotIn('speech_score', facts)
        self.assertGreaterEqual(misrouted_before, 9)

    def test_tonal_routing_does_not_relax_possible_speech_visual_gate(self):
        task = self.visual_task()
        task['record']['acoustic'].update(speech_score=0.54, tone_share=0.664, flatness=0.021)
        self.enricher().process(task)
        self.meta.classify_sound.assert_called_once()
        self.meta.transcribe_turns.assert_not_called()
        self.meta.source_check.assert_not_called()
        self.meta.describe_near.assert_not_called()
        self.meta.segment_at.assert_not_called()

    def test_real_and_varied_synthetic_speech_still_go_to_transcription(self):
        real = wav_samples((Path(__file__).with_name('test_data') / 'array_tts_find_my_keys.wav').read_bytes())
        for index, x in enumerate([real] + [voice(3, f0=f0) for f0 in (100, 140, 220)]):
            with self.subTest(example=index):
                for name in vars(self.meta):
                    getattr(self.meta, name).reset_mock()
                task = self.task(x)
                self.assertTrue(likely_speech(task['record']['acoustic']))
                self.enricher().process(task)
                self.meta.transcribe_turns.assert_called_once()
                self.meta.classify_sound.assert_not_called()

    def test_speech_routing_requires_both_valid_tonal_measurements(self):
        base = {'speech_score': 0.7, 'tone_share': 0.7, 'flatness': 0.01}
        self.assertFalse(likely_speech(base))
        for patch in ({'tone_share': 0.59}, {'flatness': 0.051}, {'tone_share': None},
                      {'flatness': None}, {'tone_share': math.nan}, {'flatness': math.inf},
                      {'flatness': -0.01}, {'tone_share': 1.5}):
            with self.subTest(patch=patch):
                self.assertTrue(likely_speech({**base, **patch}))


if __name__ == '__main__':
    unittest.main()
