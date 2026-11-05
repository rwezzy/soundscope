"""Model activity is observable without additional API calls or result callbacks."""
import os
import threading
import time
import unittest
from unittest import mock

from insight import Enricher
from meta_api import BudgetError, MetaError
import test_insight as fixtures
from test_sound_memory import clicks, voice


class Activity(unittest.TestCase):
    task = fixtures.Worker.task
    enricher = fixtures.Worker.enricher

    def setUp(self):
        fixtures.Worker.setUp(self)
        patch = mock.patch.dict(os.environ, {'SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS': '-50'})
        patch.start()
        self.addCleanup(patch.stop)

    def test_queued_count_is_not_an_api_call_and_snapshot_is_detached(self):
        worker = self.enricher()
        worker.submit({'record': {'id': 'queued'}})
        status = worker.activity_status()
        self.assertEqual(status['queued'], 1)
        self.assertEqual(status['started'], 0)
        self.assertEqual(status['active'], [])
        status['last']['state'] = 'invented'
        status['active'].append('invented')
        self.assertEqual(worker.activity_status()['last']['state'], 'idle')
        self.assertEqual(worker.activity_status()['active'], [])
        self.assertEqual(self.results, [])
        for fn in vars(self.meta).values():
            fn.assert_not_called()

    def test_all_model_operations_report_real_names_without_callbacks(self):
        worker = self.enricher()
        for method, model in (('source_check', 'Muse Spark'), ('classify_sound', 'Muse Spark'),
                              ('transcribe_turns', 'Muse Voice'), ('describe_near', 'Muse Spark'),
                              ('segment_at', 'SAM 3.1')):
            with self.subTest(method=method):
                worker._call('r', getattr(self.meta, method))
                status = worker.activity_status()
                self.assertEqual(status['last']['operation'], method)
                self.assertIn(model, status['last']['label'])
                self.assertEqual(status['last']['state'], 'complete')
                self.assertEqual(status['active_operations'], [])
        self.assertEqual(status['started'], 5)
        self.assertEqual(status['completed'], 5)
        self.assertEqual(self.results, [])

    def test_concurrent_same_operation_stays_active_until_both_finish(self):
        worker = self.enricher()
        both_started = threading.Barrier(3)
        release = threading.Event()
        def slow():
            both_started.wait(timeout=3)
            release.wait(timeout=3)
            return {'text': 'hello', 'turns': [{'text': 'hello'}]}
        self.meta.transcribe_turns.side_effect = slow
        threads = [threading.Thread(target=worker._call, args=('r', self.meta.transcribe_turns)) for _ in range(2)]
        for thread in threads:
            thread.start()
        try:
            both_started.wait(timeout=3)
            status = worker.activity_status()
            self.assertEqual(status['active_operations'], ['transcribe_turns', 'transcribe_turns'])
            self.assertEqual(status['started'], 2)
            self.assertEqual(status['completed'], 0)
            self.assertEqual(status['last']['state'], 'running')
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())
        self.assertEqual(worker.activity_status()['active'], [])
        self.assertEqual(worker.activity_status()['completed'], 2)

    def test_empty_transcript_or_label_is_not_reported_as_completed(self):
        worker = self.enricher()
        self.meta.transcribe_turns.return_value = {'text': '', 'turns': []}
        worker._call('r', self.meta.transcribe_turns)
        status = worker.activity_status()
        self.assertEqual(status['last']['state'], 'empty')
        self.assertIn('No speech', status['last']['message'])
        self.meta.classify_sound.return_value = {'label': '  '}
        worker._call('r', self.meta.classify_sound)
        self.assertEqual(worker.activity_status()['empty'], 2)
        self.assertEqual(worker.activity_status()['completed'], 0)
        self.assertEqual(self.results, [])

    def test_api_error_and_budget_refusal_clear_active_and_keep_existing_notices(self):
        worker = self.enricher()
        self.meta.classify_sound.side_effect = MetaError('HTTP 503')
        worker._call('r', self.meta.classify_sound)
        status = worker.activity_status()
        self.assertEqual(status['errors'], 1)
        self.assertEqual(status['last']['state'], 'error')
        self.assertEqual(status['active'], [])
        self.assertEqual(self.results, [('r', 'notice', 'Meta call failed: HTTP 503')])
        self.meta.transcribe_turns.side_effect = BudgetError('session limit reached')
        worker._call('r', self.meta.transcribe_turns)
        status = worker.activity_status()
        self.assertTrue(status['budget_paused'])
        self.assertEqual(status['blocked'], 1)
        self.assertEqual(status['last']['state'], 'blocked')
        self.assertEqual(status['active'], [])
        self.assertEqual(len(self.results), 2)

    def test_unexpected_exception_cannot_leave_a_phantom_active_request(self):
        worker = self.enricher()
        self.meta.describe_near.side_effect = ValueError('malformed result')
        with self.assertRaises(ValueError):
            worker._call('r', self.meta.describe_near)
        status = worker.activity_status()
        self.assertEqual(status['last']['state'], 'error')
        self.assertEqual(status['errors'], 1)
        self.assertEqual(status['active'], [])
        self.assertEqual(self.results, [])

    def test_feature_and_rate_blocks_are_truthful_without_spending(self):
        worker = self.enricher(features=('captions',))
        self.assertFalse(worker._meta_ok('identify'))
        status = worker.activity_status()
        self.assertEqual(status['last']['operation'], 'classify_sound')
        self.assertIn('not enabled', status['last']['message'])
        self.assertTrue(worker._meta_ok('captions'))
        self.assertFalse(worker._meta_ok('captions'))
        status = worker.activity_status()
        self.assertIn('rate limit', status['last']['message'])
        self.assertEqual(status['started'], 0)
        self.assertEqual(status['blocked'], 2)
        self.assertEqual(self.results, [])

    def test_quiet_and_stale_skips_are_visible_without_relaxing_gates(self):
        worker = self.enricher()
        quiet = self.task(voice(2))
        quiet['record']['loudness']['peak_dbfs'] = -65
        worker.process(quiet)
        self.assertEqual(worker.activity_status()['last']['state'], 'skipped')
        self.assertIn('recording peak', worker.activity_status()['last']['message'])
        stale = self.task(clicks())
        stale['record']['epoch'] = time.time() - 46
        worker.process(stale)
        self.assertIn('stale', worker.activity_status()['last']['message'])
        self.assertEqual(worker.activity_status()['skipped'], 2)
        self.assertEqual(worker.activity_status()['started'], 0)
        for fn in vars(self.meta).values():
            fn.assert_not_called()
        self.assertEqual(self.results, [])

    def test_caption_result_callback_remains_the_last_result(self):
        worker = self.enricher(features=('captions',))
        task = self.task(voice(2))
        worker.process(task)
        self.assertEqual(self.results[-1][1], 'captions')
        self.assertEqual(worker.activity_status()['last']['operation'], 'transcribe_turns')
        self.assertEqual(worker.activity_status()['completed'], 1)


if __name__ == '__main__':
    unittest.main()
