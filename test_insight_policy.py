"""Offline paid-policy tests; every Meta endpoint is a mock."""
import math
import os
import time
import unittest
from unittest import mock

from insight import Enricher
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


if __name__ == '__main__':
    unittest.main()
