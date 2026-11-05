"""Offline tests for the Meta API layer: budget, request shapes, reply parsing. No network.

  python -m unittest test_meta_api
"""
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

import meta_api
from meta_api import Budget, BudgetError, Meta, MetaError, wav_bytes


def clip(seconds=2.0):
    return wav_bytes(np.sin(np.arange(int(seconds * 16000)) * 0.3))


class Budgets(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, 'usage.json')

    def test_refuses_before_crossing_either_limit_and_persists_per_day(self):
        day = ['2026-11-04']
        budget = Budget(daily_usd=0.05, session_usd=0.03, path=self.path, today=lambda: day[0])
        budget.check(0.02, 'identify')
        budget.spend(0.02, 'identify')
        with self.assertRaises(BudgetError):
            budget.check(0.02, 'identify')            # session: 0.02 + 0.02 > 0.03
        other = Budget(daily_usd=0.05, session_usd=1.0, path=self.path, today=lambda: day[0])
        other.check(0.03, 'search')                    # day: 0.02 + 0.03 <= 0.05
        with self.assertRaises(BudgetError):
            other.check(0.031, 'search')               # day limit, shared through the file
        self.assertEqual(other.summary()['calls_today'], {'identify': 1})
        day[0] = '2026-11-05'
        other.check(0.04, 'search')                    # a new day starts from zero
        self.assertEqual(other.summary()['today_usd'], 0.0)

    def test_pending_request_counts_against_daily_and_session_limits(self):
        budget = Budget(daily_usd=0.01, session_usd=0.01, path=self.path)
        with budget.reserve(0.007, 'identify') as ticket:
            with self.assertRaises(BudgetError):
                budget.reserve(0.007, 'search')
            self.assertAlmostEqual(budget.summary()['pending_session_usd'], 0.007)
            ticket.settle(0.002)
        self.assertAlmostEqual(budget.summary()['today_usd'], 0.002)
        self.assertEqual(budget.summary()['pending_today_usd'], 0)
        self.assertEqual(budget.summary()['calls_today'], {'identify': 1})
        ticket.settle(0.002)  # duplicate completion cannot bill twice
        self.assertEqual(budget.summary()['calls_today'], {'identify': 1})

    def test_exception_releases_a_reservation(self):
        budget = Budget(daily_usd=0.01, session_usd=0.01, path=self.path)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with budget.reserve(0.01, 'identify'):
                raise RuntimeError('failed')
        with budget.reserve(0.01, 'search'):
            pass
        self.assertEqual(budget.summary()['today_usd'], 0)
        self.assertEqual(budget.summary()['pending_today_usd'], 0)
        self.assertEqual(budget.summary()['calls_today'], {})

    def test_concurrent_threads_and_budget_instances_cannot_overbook(self):
        for shared_instance in (True, False):
            with self.subTest(shared_instance=shared_instance):
                path = self.path + str(shared_instance)
                first = Budget(daily_usd=0.01, session_usd=0.01, path=path)
                second = first if shared_instance else Budget(daily_usd=0.01, session_usd=0.01, path=path)
                barrier = threading.Barrier(2)
                def reserve(budget):
                    barrier.wait(timeout=2)
                    try:
                        return budget.reserve(0.007, 'identify')
                    except BudgetError:
                        return None
                with ThreadPoolExecutor(max_workers=2) as pool:
                    tickets = list(pool.map(reserve, [first, second]))
                self.assertEqual(sum(ticket is not None for ticket in tickets), 1)
                for ticket in tickets:
                    if ticket is not None:
                        ticket.settle(0.007)
                self.assertAlmostEqual(first.summary()['today_usd'], 0.007)

    def test_separate_processes_share_atomic_daily_reservations(self):
        script = '''
import sys, time
from meta_api import Budget, BudgetError
b = Budget(daily_usd=0.01, session_usd=1, path=sys.argv[1])
try:
    with b.reserve(0.007, 'test') as ticket:
        time.sleep(0.15)
        ticket.settle(0.007)
    print('accepted')
except BudgetError:
    print('refused')
'''
        workers = [subprocess.Popen([sys.executable, '-c', script, self.path],
                                    cwd=Path(meta_api.__file__).parent,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                   for _ in range(2)]
        try:
            results = [worker.communicate(timeout=15) for worker in workers]
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait(timeout=3)
        for worker, (_, error) in zip(workers, results):
            self.assertEqual(worker.returncode, 0, error)
        self.assertEqual(sorted(output.strip() for output, _ in results), ['accepted', 'refused'])
        self.assertAlmostEqual(Budget(path=self.path).summary()['today_usd'], 0.007)

    def test_abandoned_reservation_is_not_optimistically_expired(self):
        first = Budget(daily_usd=0.01, session_usd=1, path=self.path)
        ticket = first.reserve(0.01, 'identify')
        later = Budget(daily_usd=0.01, session_usd=1, path=self.path)
        with self.assertRaises(BudgetError):
            later.reserve(0.001, 'search')
        self.assertEqual(later.summary()['pending_today_usd'], 0.01)
        ticket.release()

    def test_corrupt_ledger_fails_closed(self):
        Path(self.path).write_text('{broken', encoding='utf-8')
        with self.assertRaises(BudgetError):
            Budget(path=self.path).reserve(0.001, 'search')


class FakeMeta(Meta):
    """Meta with the network replaced by canned replies; records what would have been sent."""

    def __init__(self, replies, **budget):
        tmp = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {'MODEL_API_KEY': 'test-key'}):
            super().__init__(Budget(path=os.path.join(tmp, 'u.json'), **({'daily_usd': 1, 'session_usd': 1} | budget)))
        self.replies, self.sent = list(replies), []

    def _post(self, endpoint, body, content_type='application/json'):
        self.sent.append((endpoint, body, content_type))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def chat_reply(content, prompt_tokens=1000, completion_tokens=400):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}}


class Requests(unittest.TestCase):
    def test_chat_costs_what_the_reply_reports(self):
        meta = FakeMeta([chat_reply({'label': 'hand clap', 'category': 'clap or knock', 'description': 'sharp',
                                     'confidence': 1.4, 'watch': ['hand clap', 'invented']})])
        reply = meta.classify_sound(clip(), {'onsets': 2}, watch=['hand clap', 'baby crying'])
        self.assertEqual(reply['label'], 'hand clap')
        self.assertEqual(reply['watch'], ['hand clap'])  # only items that were really on the list
        self.assertEqual(reply['confidence'], 1.0)  # clamped
        expected = 1000 * 1.25e-6 + 400 * 4.25e-6
        self.assertAlmostEqual(meta.budget.summary()['session_usd'], round(expected, 4), places=4)
        payload = json.loads(meta.sent[0][1])
        self.assertEqual(payload['model'], 'muse-spark-1.3')
        self.assertEqual(payload['reasoning_effort'], 'minimal')  # automatic background check: fast setting
        self.assertEqual(payload['response_format']['json_schema']['name'], 'sound_label')
        image = payload['messages'][1]['content'][1]['image_url']['url']
        self.assertTrue(image.startswith('data:image/jpeg;base64,'))

    def test_audio_visual_check_sends_three_pictures(self):
        meta = FakeMeta([chat_reply({'sound': 'mug set down', 'sound_confidence': 0.7, 'object': 'a mug',
                                     'action': 'a mug was set down', 'consistent': 'yes', 'near': 'a laptop',
                                     'surface': 'desk', 'watch': [], 'summary': 'A mug was set on the desk.'},
                                    prompt_tokens=5000)])
        reply = meta.source_check(clip(), b'after-jpeg', b'before-jpeg', {'onsets': 1}, watch=['kettle'])
        self.assertEqual(reply['consistent'], 'yes')
        payload = json.loads(meta.sent[0][1])
        parts = payload['messages'][1]['content']
        self.assertEqual(sum(1 for part in parts if part['type'] == 'image_url'), 3)
        self.assertIn('Watch list: ["kettle"]', parts[0]['text'])
        self.assertEqual(payload['response_format']['json_schema']['name'], 'sound_source')
        self.assertIn('ALREADY located', payload['messages'][0]['content'])

    def test_budget_refusal_sends_nothing_and_failures_cost_nothing(self):
        meta = FakeMeta([], session_usd=0.001)
        with self.assertRaises(BudgetError):
            meta.parse_command('find my keys')
        self.assertEqual(meta.sent, [])
        meta = FakeMeta([MetaError('Meta API answered HTTP 500')])
        with self.assertRaises(MetaError):
            meta.parse_command('find my keys')
        self.assertEqual(meta.budget.summary()['session_usd'], 0.0)
        self.assertEqual(meta.budget.summary()['pending_today_usd'], 0.0)

    def test_concurrent_http_call_is_refused_before_sending(self):
        meta = FakeMeta([], session_usd=0.01)
        arrived, release = threading.Event(), threading.Event()
        def request_stub(*_):
            arrived.set()
            self.assertTrue(release.wait(timeout=3))
            return chat_reply({'ok': True}, prompt_tokens=1, completion_tokens=1)
        meta._post = mock.Mock(side_effect=request_stub)
        schema = {'required': ['ok'], 'properties': {'ok': {'type': 'boolean'}}}
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(meta.chat_json, meta.text_model, 'test', schema, 'test')
            try:
                self.assertTrue(arrived.wait(timeout=3))
                with self.assertRaises(BudgetError):
                    meta.chat_json(meta.text_model, 'test', schema, 'test')
                self.assertEqual(meta._post.call_count, 1)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=3), {'ok': True})
        self.assertEqual(meta.budget.summary()['pending_today_usd'], 0)

    def test_accepted_malformed_chat_reply_is_charged_and_reservation_closed(self):
        meta = FakeMeta([{}])
        with self.assertRaises(MetaError):
            meta.parse_command('find my keys')
        self.assertGreater(meta.budget.summary()['session_usd'], 0)
        self.assertEqual(meta.budget.summary()['pending_today_usd'], 0)

    def test_transcription_failure_releases_reservation(self):
        meta = FakeMeta([MetaError('HTTP 500')])
        with self.assertRaises(MetaError):
            meta.transcribe(clip())
        self.assertEqual(meta.budget.summary()['today_usd'], 0)
        self.assertEqual(meta.budget.summary()['pending_today_usd'], 0)

    def test_command_schema_has_search_who_describe(self):
        meta = FakeMeta([chat_reply({'action': 'search', 'sound': '', 'question': 'When did the phone ring?',
                                     'reply': 'Looking.'})])
        cmd = meta.parse_command('when did my phone ring')
        self.assertEqual(cmd['action'], 'search')
        self.assertEqual(json.loads(meta.sent[0][1])['reasoning_effort'], 'low')  # user-asked: more careful
        schema = json.loads(meta.sent[0][1])['response_format']['json_schema']['schema']
        self.assertTrue({'search', 'who', 'describe', 'find', 'follow'} <= set(schema['properties']['action']['enum']))

    def test_diarized_captions(self):
        meta = FakeMeta([{'sessionId': 's', 'transcript': 'where are my keys they are here', 'audioDurationMs': 4000,
                          'turns': [{'turnId': 0, 'startMs': 100, 'endMs': 1900, 'transcript': 'where are my keys',
                                     'speaker': 'A'},
                                    {'turnId': 1, 'startMs': 2200, 'endMs': 3800, 'transcript': 'they are here',
                                     'speaker': 'B'}]}])
        result = meta.transcribe_turns(clip(4.0))
        self.assertEqual([t['speaker'] for t in result['turns']], ['A', 'B'])
        self.assertEqual(result['turns'][1]['start_s'], 2.2)
        endpoint, body, content_type = meta.sent[0]
        self.assertEqual(endpoint, '/asr/transcribe')
        self.assertIn(b'"mode": "DIARIZATION"', body)
        self.assertTrue(content_type.startswith('multipart/form-data; boundary='))
        self.assertAlmostEqual(meta.budget.summary()['session_usd'], round(0.18 * 4 / 3600, 4), places=4)

    def test_turnless_reply_becomes_one_caption(self):
        meta = FakeMeta([{'transcript': 'hello', 'turns': []}])
        self.assertEqual(meta.transcribe_turns(clip())['turns'][0]['text'], 'hello')
        meta = FakeMeta([{'transcript': 'find my keys', 'turns': []}])
        self.assertEqual(meta.transcribe(clip()), 'find my keys')
        self.assertIn(b'PUSH_TO_TALK', meta.sent[0][1])

    def test_contributor_tiers_are_refused(self):
        with mock.patch.dict(os.environ, {'MODEL_API_KEY': 'k', 'META_TEXT_MODEL': 'muse-spark-1.3-contributor'}):
            with self.assertRaises(MetaError):
                Meta()


class Segmentation(unittest.TestCase):
    def test_accepted_sam_stream_charged_once_even_when_stream_reports_failure(self):
        meta = FakeMeta([])
        stream = [b'data: {"type":"response.failed","error":"decode failed"}\n', b'\n']
        with mock.patch.object(meta, '_open', return_value=nullcontext(stream)):
            with self.assertRaises(MetaError):
                meta.segment_at(b'jpeg', 640, 480, 10, 20)
        summary = meta.budget.summary()
        self.assertEqual(summary['calls_today'], {'outline': 1})
        self.assertAlmostEqual(summary['today_usd'], 0.0025)
        self.assertEqual(summary['pending_today_usd'], 0)

    def test_sam_http_failure_releases_reservation(self):
        meta = FakeMeta([])
        with mock.patch.object(meta, '_open', side_effect=MetaError('HTTP 500')):
            with self.assertRaises(MetaError):
                meta.segment_at(b'jpeg', 640, 480, 10, 20)
        self.assertEqual(meta.budget.summary()['today_usd'], 0)
        self.assertEqual(meta.budget.summary()['pending_today_usd'], 0)

    def test_point_prompt_matches_meta_helper(self):
        text = meta_api.sam_point_prompt(1280, 800, 640.7, 900)
        self.assertEqual(text, '<0f>1<|point;x=640;y=799;w=1280;h=800|>')
        try:
            from meta_sam_parser import FramePrompt, PromptObject, PromptPoint, build_object_prompt
        except ImportError:
            self.skipTest('meta-sam-parser not installed')
        official = build_object_prompt(size=(1280, 800), objects=[PromptObject(id=1, prompts=[
            FramePrompt(frame=0, points=[PromptPoint(640, 799)])])])
        self.assertEqual(text, official)

    def test_server_sent_events(self):
        lines = [b'event: x\n', b'data: {"type": "response.output_text.delta", "delta": "<0f>"}\n', b'\n',
                 b'data: {"type": "response.completed"}\n', b'\n', b'data: [DONE]\n', b'\n']
        events = list(meta_api.sse_events(lines))
        self.assertEqual([e['type'] for e in events], ['response.output_text.delta', 'response.completed'])

    def test_empty_segmentation_is_no_outline(self):
        try:
            import meta_sam_parser  # noqa: F401
        except ImportError:
            self.skipTest('meta-sam-parser not installed')
        self.assertEqual(meta_api.sam_polygons(''), [])


if __name__ == '__main__':
    unittest.main()
