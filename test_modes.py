"""Tripod and wrist mounts, and the viewer's new hooks: python -m unittest test_modes

Hardware-free: config files, geometry, the dry-run arm and the drawing code only.
"""
import asyncio
import copy
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import aim
import arm_follow
import scope
import scope_ui
import sound_memory as sm


class Profiles(unittest.TestCase):
    def test_both_mounts_load_over_the_base_config(self):
        self.assertTrue({'tripod', 'wrist'} <= aim.profiles())
        tripod = aim.load_config(profile='tripod')
        self.assertEqual(tripod['camera']['mount'], 'base')
        self.assertEqual(tripod['array']['position_mm'], [0, -11, 72.5])
        self.assertEqual(tripod['safety'], aim.load_config(profile='')['safety'])
        wrist = aim.load_config(profile='wrist')
        self.assertEqual(wrist['camera']['mount'], 'arm')
        self.assertTrue(wrist['array']['on_arm'])
        self.assertTrue(wrist['profile']['has_yaw'])
        self.assertTrue(wrist['profile']['path'].endswith('wrist.toml'))

    def test_unknown_or_unsafe_profiles_are_refused(self):
        with self.assertRaises(SystemExit):
            aim.load_config(profile='ceiling')
        with tempfile.TemporaryDirectory() as folder:
            bad = aim.Path(folder) / 'fast.toml'
            bad.write_text('[safety]\nspeed_deg_s = 300\n', encoding='utf-8')
            with mock.patch.object(aim, 'PROFILES_DIR', aim.Path(folder)), self.assertRaises(SystemExit):
                aim.load_config(profile='fast')

    def test_calibration_saves_into_the_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            path = aim.Path(folder) / 'wrist.toml'
            path.write_text('[array]\nfront_yaw_deg = 0\n', encoding='utf-8')
            with mock.patch.object(scope, 'YAW_FILE', str(path)):
                scope.save_front_yaw(12.5)
            self.assertIn('front_yaw_deg = 12.500', path.read_text(encoding='utf-8'))


class Wrist(unittest.TestCase):
    def setUp(self):
        self.cfg = aim.load_config(profile='wrist')

    def test_errors_are_relative_to_the_camera_view(self):
        for yaw, pitch in ((0, 0), (25, 0), (-40, 10), (10, -20)):
            y, p = math.radians(yaw), math.radians(pitch)
            cam = [math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)]
            got = aim.wrist_errors(aim.camera_to_array(cam, self.cfg), self.cfg)
            self.assertAlmostEqual(got[0], yaw, places=6)
            self.assertAlmostEqual(got[1], pitch, places=6)

    def test_dry_run_turns_toward_the_sound_within_the_gate(self):
        async def go():
            arm = arm_follow.DryRunArm(self.cfg)
            gate = aim.Gate(self.cfg, await arm.joints())
            inbox = asyncio.Queue()
            follower = arm_follow.Follower(self.cfg, arm, gate, self.cfg['look'], None, inbox, None, None, True)
            start = (await arm.joints())[0]
            y = math.radians(20)
            direction = aim.camera_to_array([math.sin(y), 0, math.cos(y)], self.cfg)
            with mock.patch('builtins.print'):
                await follower.on_sound({'fix': {'direction': direction, 'z': 40.0, 'snr_db': 5.0}})
            return start, (await arm.joints())[0]
        start, end = asyncio.run(go())
        self.assertAlmostEqual(end - start, -20.0, places=3)  # wrist_yaw_sign = -1: sound right -> J1 down

    def test_simulated_sounds_are_camera_bearings_on_the_wrist(self):
        posted = []
        with mock.patch('asyncio.sleep', new=mock.AsyncMock()):
            asyncio.run(arm_follow.feed_simulated([30], posted.append, 0, self.cfg))
        yaw, _ = aim.wrist_errors(posted[0]['fix']['direction'], self.cfg)
        self.assertAlmostEqual(yaw, 30, places=6)

    def test_viewer_accepts_the_wrist_mount_only_with_mics_on_it(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['array']['on_arm'] = False
        with self.assertRaises(SystemExit):
            scope.run(SimpleNamespace(no_heatmap=False, diagnostics=False), cfg)


class Viewer(unittest.TestCase):
    def setUp(self):
        printer = mock.patch('builtins.print')
        printer.start()
        self.addCleanup(printer.stop)

    def voice(self, action, question=''):
        meta = SimpleNamespace(transcribe=mock.Mock(return_value='when did the phone ring'),
                               parse_command=mock.Mock(return_value={'action': action, 'sound': '',
                                                                     'question': question, 'reply': 'ok'}))
        ui, hunt = {'note': ''}, {'target': 'keys', 'mode': 'follow', 'found': None}
        ask = mock.Mock(return_value='The phone rang at 3:42 PM, to your left.')
        scope.handle_voice(meta, b'wav', ui, hunt, ask)
        return ui, hunt, ask

    def test_questions_go_to_the_memory_and_leave_the_hunt_alone(self):
        for action in ('search', 'who', 'describe'):
            ui, hunt, ask = self.voice(action, 'When did the phone ring?')
            ask.assert_called_once()
            self.assertEqual(ui['note'], 'The phone rang at 3:42 PM, to your left.')
            self.assertEqual(hunt['target'], 'keys')

    def test_verdict_keeps_the_sound_on_screen(self):
        meta = SimpleNamespace(sound_matches=mock.Mock(return_value={'match': True, 'heard': 'keys', 'confidence': 0.9}))
        view = {'time': scope.time.monotonic() - 3.0}
        scope.check_sound(meta, b'wav', 'keys', view)
        self.assertGreaterEqual(view['expires_after'], 3.0 + scope.AFTER_VERDICT_S - 0.1)
        self.assertFalse(view['rejected'])

    def test_session_records_every_finished_sound(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(sm, 'MEMORY_DIR', aim.Path(folder)), \
                mock.patch.object(sm, 'REFERENCES_DIR', aim.Path(folder) / 'refs'):
            cfg = aim.load_config(profile='tripod')
            session = scope.Session(SimpleNamespace(memory=True), cfg, {'note': ''})
            self.assertTrue(session.active)
            self.assertIsNone(session.meta)  # memory alone never creates a Meta client
            fix = {'direction': aim.camera_to_array([0.5, 0, 0.866], cfg), 'z': 30.0, 'snr_db': 18.0,
                   'seconds': 0.4, 'coherence': 0.3}
            view = {'spot': (50, 40), 'point': np.array([0.4, 0.0, 0.7]), 'measured': True,
                    'quality': {'state': 'candidate'}}
            report = {'kind': 'sound', 'fix': fix, 'event_id': 3, 'observed_at': scope.time.monotonic(),
                      'wav': sm_wav(), 'level': {'rms_dbfs': -31.0, 'peak_dbfs': -14.0}}
            rec = session.record(report, view, np.zeros((80, 120, 3), np.uint8))
            self.assertEqual(rec['where']['words'], 'to the right')
            self.assertEqual(rec['loudness']['rms_dbfs'], -31.0)
            self.assertNotIn('rms_dbfs', rec['acoustic'])
            self.assertEqual(session.ask({'action': 'describe'}, 'what was that'), sm.phrase(rec))
            answer = session.ask({'action': 'search', 'question': 'what happened in the last minute'}, '')
            self.assertIn('to the right', answer)
            session.close()

    def test_watch_alerts_and_spatial_recall(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(sm, 'MEMORY_DIR', aim.Path(folder)),                 mock.patch.object(sm, 'REFERENCES_DIR', aim.Path(folder) / 'refs'),                 mock.patch.object(scope, 'beep'):
            cfg = aim.load_config(profile='tripod')
            session = scope.Session(SimpleNamespace(memory=True), cfg, {'note': ''})
            reply = session.ask({'action': 'watch', 'sound': 'kettle whistling'}, 'tell me when the kettle whistles')
            self.assertIn('kettle whistling', reply)
            self.assertIn('Watching: kettle whistling', session.status({'target': None, 'mode': 'follow', 'found': None}))
            fix = {'direction': aim.camera_to_array([0.0, 0, 1.0], cfg), 'z': 30.0, 'snr_db': 18.0, 'seconds': 0.4}
            view = {'spot': (60, 40), 'point': np.array([0.0, 0.0, 1.0]), 'measured': True,
                    'quality': {'state': 'candidate'}}
            rec = session.record({'kind': 'sound', 'fix': fix, 'event_id': 1, 'observed_at': scope.time.monotonic(), 'wav': sm_wav()},
                                 view, np.zeros((80, 120, 3), np.uint8))
            session.results.append((rec['id'], 'alert', {'watch': 'kettle whistling', 'how': 'Muse Spark'}))
            notes = session.drain(now=100.0)
            self.assertIn('ALERT: kettle whistling (straight ahead)', notes[0])
            self.assertEqual(session.memory.get(rec['id'])['alerts'], ['kettle whistling'])
            self.assertEqual(session.alert[0], 'ALERT: kettle whistling (straight ahead)')
            session.captions.clear()
            session.ask({'action': 'search', 'question': 'what happened in the last minute'}, '')
            recall = [c for c in session.captions if c['kind'] == 'recall']
            self.assertEqual(recall[0]['pixel'], [60, 40])  # shown where it happened
            session.close()

    def test_night_mode_sends_depth_shapes_instead_of_a_black_crop(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(sm, 'MEMORY_DIR', aim.Path(folder)),                 mock.patch.object(sm, 'REFERENCES_DIR', aim.Path(folder) / 'refs'),                 mock.patch.object(scope, 'Meta'), mock.patch.object(scope, 'Enricher') as enricher:
            cfg = aim.load_config(profile='tripod')
            session = scope.Session(SimpleNamespace(memory=True, context=True), cfg, {'note': ''})
            fix = {'direction': aim.camera_to_array([0.0, 0, 1.0], cfg), 'z': 30.0, 'snr_db': 18.0, 'seconds': 0.4}
            view = {'spot': (160, 120), 'point': np.array([0.0, 0.0, 1.2]), 'measured': True,
                    'quality': {'state': 'candidate'}}
            yy, xx = np.mgrid[0:240, 0:320]
            depth = np.where((xx - 160) ** 2 + (yy - 120) ** 2 < 30 ** 2, 1100.0, 1500.0).astype(np.float32)
            report = {'kind': 'sound', 'fix': fix, 'event_id': 1, 'observed_at': scope.time.monotonic(), 'wav': sm_wav()}
            session.record(report, view, np.zeros((240, 320, 3), np.uint8), depth=depth)
            task = enricher.return_value.submit.call_args.args[0]
            self.assertEqual(task['crop_kind'], 'depth')
            self.assertEqual(task['crop_jpeg'][:2], bytes([0xFF, 0xD8]))  # a JPEG
            session.record(report, view, np.full((240, 320, 3), 140, np.uint8), depth=depth)
            task = enricher.return_value.submit.call_args.args[0]
            self.assertNotIn('crop_kind', task)  # lit room: the real camera crop
            session.close()

    def test_late_results_are_saved_but_not_drawn(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(sm, 'MEMORY_DIR', aim.Path(folder)),                 mock.patch.object(sm, 'REFERENCES_DIR', aim.Path(folder) / 'refs'):
            cfg = aim.load_config(profile='tripod')
            session = scope.Session(SimpleNamespace(memory=True), cfg, {'note': ''})
            fix = {'direction': aim.camera_to_array([0.0, 0, 1.0], cfg), 'z': 30.0, 'snr_db': 18.0, 'seconds': 0.4}
            view = {'spot': (60, 40), 'point': np.array([0.0, 0.0, 1.0]), 'measured': True,
                    'quality': {'state': 'candidate'}}
            rec = session.record({'kind': 'sound', 'fix': fix, 'event_id': 1, 'observed_at': scope.time.monotonic(),
                                  'wav': sm_wav()}, view, np.zeros((80, 120, 3), np.uint8))
            session.results.append((rec['id'], 'label', rec))
            session.drain(now=10.0)
            self.assertEqual(len(session.captions), 1)            # fresh: drawn at its pixel
            session.captions.clear()
            session.memory.update(rec['id'], {'epoch': rec['epoch'] - 60})
            session.results.append((rec['id'], 'label', rec))
            session.results.append((rec['id'], 'outline', {'polygons': [np.array([[1, 1], [9, 1], [9, 9]])],
                                                            'scale': 1.0}))
            session.drain(now=11.0)
            self.assertEqual(session.captions, [])                # a minute late: not on the live picture
            self.assertEqual(session.outlines, [])
            session.close()

    def test_assist_mode_watches_household_sounds_and_names(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(sm, 'MEMORY_DIR', aim.Path(folder)),                 mock.patch.object(sm, 'REFERENCES_DIR', aim.Path(folder) / 'refs'),                 mock.patch.object(scope, 'Meta'), mock.patch.object(scope, 'Enricher'),                 mock.patch.object(scope, 'beep') as beep:
            cfg = aim.load_config(profile='tripod')
            session = scope.Session(SimpleNamespace(memory=True, assist=True, name=['Keith'], watch=['dog barking']),
                                    cfg, {'note': ''})
            self.assertEqual(session.features, set(scope.FEATURES))
            for sound in ('smoke alarm', 'doorbell', 'knock on the door', 'baby crying', 'dog barking'):
                self.assertIn(sound, session.watch)
            fix = {'direction': aim.camera_to_array([-0.6, 0, 0.8], cfg), 'z': 30.0, 'snr_db': 18.0, 'seconds': 1.4}
            view = {'spot': (40, 40), 'point': np.array([-0.6, 0.0, 0.8]), 'measured': True,
                    'quality': {'state': 'candidate'}}
            rec = session.record({'kind': 'sound', 'fix': fix, 'event_id': 1, 'observed_at': scope.time.monotonic(),
                                  'wav': sm_wav()}, view, np.zeros((80, 120, 3), np.uint8))
            turn = {'name': 'Speaker 1', 'where': 'to the left', 'text': 'Keith, dinner is ready'}
            session.results.append((rec['id'], 'captions', [turn]))
            notes = session.drain(now=50.0)
            self.assertIn('Speaker 1 said "Keith" (to the left)', notes[0])
            self.assertTrue(session.alert[0].startswith('ALERT: Speaker 1 said "Keith"'))
            self.assertIn('name called', session.memory.get(rec['id'])['alerts'])
            beep.assert_called_once()
            frame = np.full((200, 320, 3), 60, np.uint8)
            banner = scope_ui.render(frame, None, None, 'n', now=1.0, alert=session.alert[0])
            plain = scope_ui.render(frame, None, None, 'n', now=1.0)
            self.assertGreater(int(np.abs(banner.astype(int) - plain.astype(int)).sum()), 10000)
            session.close()

    def test_dashboard_draws_status_captions_and_outline(self):
        frame = np.full((200, 320, 3), 60, np.uint8)
        view = {'time': 10.0, 'final': True, 'spot': (160, 100), 'quality': {'state': 'candidate'},
                'sam_outline': [np.array([[120, 70], [200, 70], [200, 130], [120, 130]])]}
        plain = scope_ui.render(frame, view, None, 'note', now=10.5)
        rich = scope_ui.render(frame, view, None, 'note', now=10.5, status='Finding: keys',
                               captions=[{'text': 'Speaker 1 (to the left): where are my keys', 'pixel': (160, 100),
                                          'kind': 'speech'},
                                         {'text': 'phone ringing', 'pixel': None, 'kind': 'label'}])
        self.assertEqual(plain.shape, rich.shape)
        self.assertGreater(int(np.abs(rich.astype(int) - plain.astype(int)).sum()), 1000)
        held = scope_ui.render(frame, {**view, 'hold': True, 'time': 0.0}, 'keys', 'n', now=100.0)
        self.assertEqual(held.shape, plain.shape)  # a found target never goes stale


def sm_wav():
    from meta_api import wav_bytes
    t = np.arange(6400) / 16000
    return wav_bytes(np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 8))


if __name__ == '__main__':
    unittest.main()
