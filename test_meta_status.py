"""Meta display-state tests; fake clients only, no paid or sensor calls."""
import collections
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import scope


class MetaStatus(unittest.TestCase):
    def session(self):
        return scope.Session(SimpleNamespace(memory=False), {}, {})

    def client(self, session_limit=.03):
        return SimpleNamespace(budget=SimpleNamespace(
            session=session_limit, daily=.10, short=lambda: 'Meta $0.01/$0.10 today'))

    def test_default_off_and_on_demand_client_does_not_enable_captions(self):
        session = self.session()
        self.assertEqual(session.ai_status()['phase'], 'off')
        session.meta = self.client()
        status = session.ai_status()
        self.assertEqual(status['phase'], 'off')
        self.assertEqual(status['features'], [])
        self.assertIn('No automatic transcripts', status['detail'])

    def test_actual_worker_progress_and_budget_pause(self):
        session = self.session()
        session.features = {'captions', 'identify'}
        session.meta = self.client()
        session.enricher = Mock()
        session.enricher.activity_status.return_value = {
            'active': ['Muse Voice transcript'], 'queued': 0, 'last': {}, 'budget_paused': False}
        status = session.ai_status()
        self.assertEqual(status['phase'], 'processing')
        self.assertIn('Muse Voice', status['text'])
        self.assertIn('Transcripts', status['features'])
        session.enricher.activity_status.return_value = {'budget_paused': True}
        self.assertEqual(session.ai_status()['phase'], 'budget')

    def test_zero_budget_is_visible_even_with_features_selected(self):
        session = self.session()
        session.features = {'captions'}
        session.meta = self.client(0)
        self.assertEqual(session.ai_status()['phase'], 'budget')
        self.assertIn('$0', session.ai_status()['text'])

    def test_real_transcript_remains_available_after_camera_bubble_expires(self):
        session = self.session()
        session.features = {'captions'}
        session.results = collections.deque([('event', 'captions', [
            {'name': 'Speaker 1', 'where': 'unknown direction', 'text': 'This is a real returned transcript.'}])])
        with patch('builtins.print'):
            session.drain(10)
            session.drain(100)
        self.assertEqual(session.captions, [])
        self.assertIn('real returned transcript', session.ai_status()['transcript'])

    def test_voice_command_reports_processing_and_preserves_transcript(self):
        session = self.session()
        client = Mock()
        def transcribe(*args, **kwargs):
            self.assertEqual(session.ai_status()['phase'], 'processing')
            return 'show everything'
        client.transcribe.side_effect = transcribe
        client.parse_command.return_value = {'action': 'home', 'sound': ''}
        hunt = {'target': 'ring', 'mode': 'find', 'found': None}
        with patch('builtins.print'):
            scope.handle_voice(client, b'fake-wav', session.ui, hunt)
        self.assertEqual(hunt['mode'], 'follow')
        self.assertIn('You: show everything', session.ai_status()['transcript'])
        self.assertEqual(session.ai_status()['phase'], 'idle')


if __name__ == '__main__':
    unittest.main()
