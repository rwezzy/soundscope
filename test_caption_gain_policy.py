"""Low-gain raw microphones must not veto clear processed-speech captions."""
import io
import math
import os
import time
import unittest
import wave
from unittest import mock

import numpy as np

from meta_api import BudgetError, wav_bytes
import test_insight as fixtures
from test_sound_memory import voice


class CaptionGainPolicy(unittest.TestCase):
    task = fixtures.Worker.task
    enricher = fixtures.Worker.enricher

    def setUp(self):
        fixtures.Worker.setUp(self)
        patch = mock.patch.dict(os.environ, {'SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS': '-50'})
        patch.start()
        self.addCleanup(patch.stop)

    def speech_task(self, peak=-64.3, snr=8.9, duration=0.72, score=0.67):
        task = self.task(voice(2), crop_jpeg=b'crop', outline=True, frame_jpeg=b'frame',
                         pixel=(100, 80), frame_size=(640, 480))
        task['record']['loudness'].update(peak_dbfs=peak, snr_db=snr)
        task['record']['duration_s'] = duration
        task['record']['acoustic'].update(speech_score=score, tone_share=0.335, flatness=0.026)
        task['speech_wav'] = wav_bytes(voice(2, f0=140))
        return task

    def assert_no_calls(self):
        for fn in vars(self.meta).values():
            fn.assert_not_called()

    def test_observed_low_raw_level_speech_gets_only_processed_audio_asr(self):
        # Numeric metrics from four real missed events, not stored/private speech.
        examples = [(-64.3, 8.9, .72, .67), (-63.5, 6.8, 3.73, .64),
                    (-63.5, 6.0, 5.01, .50), (-52.4, 8.0, 2.50, .51)]
        for values in examples:
            with self.subTest(metrics=values):
                for fn in vars(self.meta).values():
                    fn.reset_mock()
                worker = self.enricher()
                task = self.speech_task(*values)
                self.assertFalse(worker.automatic_audio_eligible(task['record']))
                self.assertFalse(worker.automatic_media_eligible(task['record']))
                worker.process(task)
                self.meta.transcribe_turns.assert_called_once_with(task['speech_wav'])
                for name in ('source_check', 'classify_sound', 'describe_near', 'segment_at'):
                    getattr(self.meta, name).assert_not_called()
                self.assertEqual(self.results[-1][1], 'captions')
                self.assertTrue(self.memory.get(task['record']['id'])['speech']['text'])

    def test_processed_channel_must_be_present_and_caption_feature_enabled(self):
        for absent in (None, b'', 'not audio'):
            task = self.speech_task()
            task['speech_wav'] = absent
            self.enricher().process(task)
            self.assert_no_calls()
        worker = self.enricher(features=('identify', 'context', 'outline'))
        worker.process(self.speech_task())
        self.assert_no_calls()
        self.assertIn('not enabled', worker.activity_status()['last']['message'])

    def test_actual_synthetic_speech_profile_and_waveform_take_caption_route(self):
        samples = voice(2, f0=140)
        task = self.task(samples, snr=8)
        task['record']['loudness']['peak_dbfs'] = -64
        task['speech_wav'] = wav_bytes(samples)
        worker = self.enricher()
        worker.process(task)
        self.meta.transcribe_turns.assert_called_once_with(task['speech_wav'])
        self.assertEqual(worker.activity_status()['completed'], 1)

    def test_processed_wav_must_be_complete_mono_16k_pcm_with_real_variation(self):
        def pcm(samples, rate=16000, channels=1):
            stream = io.BytesIO()
            with wave.open(stream, 'wb') as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(np.asarray(samples, dtype='<i2').tobytes())
            return stream.getvalue()
        signal = np.sin(np.arange(32000) * .1) * 5000
        valid = pcm(signal)
        invalid = [b'not a wav', pcm(np.zeros(32000)), pcm(np.ones(32000) * 20),
                   pcm(signal, rate=8000), pcm(np.column_stack([signal, signal]), channels=2),
                   valid[:-200], pcm(signal[:100]), pcm(np.tile(signal, 6))]
        for wav in invalid:
            with self.subTest(bytes=len(wav)):
                task = self.speech_task()
                task['speech_wav'] = wav
                self.enricher().process(task)
                self.assert_no_calls()

    def test_noise_tone_short_or_invalid_evidence_cannot_use_exception(self):
        cases = [({'loudness': {'snr_db': 2.99}}, {}),
                 ({'loudness': {'snr_db': math.nan}}, {}),
                 ({'loudness': {'snr_db': None}}, {}),
                 ({'loudness': {'peak_dbfs': math.inf}}, {}),
                 ({'loudness': {'peak_dbfs': -120}}, {}),
                 ({'loudness': {'peak_dbfs': None}}, {}),
                 ({'duration_s': .49}, {}), ({'duration_s': True}, {}),
                 ({}, {'speech_score': .49}),
                 ({}, {'speech_score': .67, 'tone_share': .664, 'flatness': .021})]
        for fields, acoustic in cases:
            with self.subTest(fields=fields, acoustic=acoustic):
                task = self.speech_task()
                for key, value in fields.items():
                    if isinstance(value, dict):
                        task['record'][key].update(value)
                    else:
                        task['record'][key] = value
                task['record']['acoustic'].update(acoustic)
                self.enricher().process(task)
                self.assert_no_calls()

    def test_rate_limit_budget_and_age_still_apply(self):
        worker = self.enricher()
        worker.process(self.speech_task())
        worker.process(self.speech_task())
        self.meta.transcribe_turns.assert_called_once()
        self.assertIn('rate limit', worker.activity_status()['last']['message'])
        self.meta.transcribe_turns.reset_mock()
        task = self.speech_task()
        task['record']['epoch'] = time.time() - 46
        self.enricher().process(task)
        self.meta.transcribe_turns.assert_not_called()
        self.meta.transcribe_turns.side_effect = BudgetError('session limit reached')
        worker = self.enricher()
        worker.process(self.speech_task())
        worker.process(self.speech_task())
        self.meta.transcribe_turns.assert_called_once()
        self.assertTrue(worker.activity_status()['budget_paused'])
        self.meta.classify_sound.assert_not_called()

    def test_caption_only_weak_or_ambiguous_location_does_not_name_speakers_by_track(self):
        cases = [('weak', None), ('ambiguous', None), (None, None),
                 ('candidate', 'where'), ('candidate', 'record'),
                 ('candidate', 'record_fix'), ('candidate', 'task_fix')]
        for quality, flag_at in cases:
            with self.subTest(quality=quality, flag_at=flag_at):
                self.meta.transcribe_turns.reset_mock()
                task = self.speech_task()
                task['record']['where']['quality'] = quality
                if flag_at == 'where':
                    task['record']['where']['directional_ambiguous'] = True
                elif flag_at == 'record':
                    task['record']['directional_ambiguous'] = True
                elif flag_at == 'record_fix':
                    task['record']['fix'] = {'directional_ambiguous': True}
                elif flag_at == 'task_fix':
                    task['fix'] = {'directional_ambiguous': True}
                task.update(track=[(10 + .1 * i, -35 if i < 10 else 30, 0)
                                   for i in range(23)], clip_end_at=12.2, clip_seconds=2.2)
                worker = self.enricher()
                worker.process(task)
                self.meta.transcribe_turns.assert_called_once_with(task['speech_wav'])
                turns = self.results[-1][2]
                self.assertEqual([turn['azimuth_deg'] for turn in turns], [None, None])
                self.assertEqual([turn['name'] for turn in turns], ['Someone', 'Someone'])
                self.assertEqual([turn['where'] for turn in turns], ['unknown direction'] * 2)
                self.assertEqual(worker.speakers.clusters, [])
                self.assertEqual(worker.speakers.count, 0)
                self.assertEqual(self.memory.get(task['record']['id'])['speech']['speaker'], 'Someone')

    def test_caption_only_candidate_track_keeps_directional_labels(self):
        task = self.speech_task()
        task.update(track=[(10 + .1 * i, -35 if i < 10 else 30, 0)
                           for i in range(23)], clip_end_at=12.2, clip_seconds=2.2)
        worker = self.enricher()
        worker.process(task)
        turns = self.results[-1][2]
        self.assertEqual([turn['azimuth_deg'] for turn in turns], [-35, 30])
        self.assertEqual([turn['name'] for turn in turns], ['Speaker 1', 'Speaker 2'])
        self.assertEqual([turn['where'] for turn in turns], ['to the left', 'to the right'])

    def test_nonfinite_caption_directions_never_create_a_speaker_cluster(self):
        for value in (math.nan, math.inf, 'unknown', True):
            with self.subTest(azimuth=value):
                task = self.speech_task()
                task['record']['where']['azimuth_deg'] = value
                worker = self.enricher()
                worker.process(task)
                self.assertEqual([turn['azimuth_deg'] for turn in self.results[-1][2]], [None, None])
                self.assertEqual(worker.speakers.clusters, [])


if __name__ == '__main__':
    unittest.main()
