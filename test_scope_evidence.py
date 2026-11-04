"""Regression checks for actual score selection and saved diagnostic evidence."""
import json
from contextlib import ExitStack
from pathlib import Path
import queue
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

import scope
from test_scope import K, RIGHT_BOX, heard_from, render, scope_config


class Evidence(unittest.TestCase):
    def test_camera_timeout_does_not_drop_drained_completed_report(self):
        reports=queue.Queue();reports.close=mock.Mock()
        reports.put({'kind':'ready'})
        event={'kind':'sound','observed_at':time.monotonic(),'fix':{'z':20,'coherence':.3}}
        reports.put(event)
        process=mock.Mock(pid=123);process.is_alive.side_effect=[True,False]
        camera=mock.Mock();camera.read.return_value=None
        bridge=mock.Mock();bridge.accept_event.return_value=True
        args=SimpleNamespace(target=None,no_heatmap=False,diagnostics=False,record=None,seconds=None,reach=False)
        with ExitStack() as patches:
            patches.enter_context(mock.patch('multiprocessing.Queue',return_value=reports))
            patches.enter_context(mock.patch('multiprocessing.Event',return_value=mock.Mock()))
            patches.enter_context(mock.patch('multiprocessing.Process',return_value=process))
            patches.enter_context(mock.patch('camera.Gemini',return_value=camera))
            patches.enter_context(mock.patch.object(scope,'Localizer',return_value=mock.Mock()))
            for name in ('namedWindow','resizeWindow','setMouseCallback','destroyAllWindows'):
                patches.enter_context(mock.patch.object(scope.cv2,name))
            patches.enter_context(mock.patch.object(scope.cv2,'waitKey',return_value=ord('q')))
            patches.enter_context(mock.patch('builtins.print'))
            scope.run(args,{'camera':{'mount':'base'}},bridge=bridge)
        bridge.observe.assert_called_once()
        self.assertIs(bridge.observe.call_args.args[0],event)
        self.assertIsNone(bridge.observe.call_args.args[2])
        camera.close.assert_called_once()

    def test_completed_sound_uses_matching_buffer_not_newest_frame(self):
        bridge=mock.Mock();bridge.accept_event.return_value=True
        event={'kind':'sound','observed_at':10.,'fix':{'z':20,'coherence':.3}}
        source=np.zeros((12,16,3),np.uint8);depth=np.ones((12,16))
        newest=np.ones_like(source)
        history=[(10.04,source,depth,K),(10.5,newest,depth,K)]
        view={'quality':{'state':'candidate'}}
        with mock.patch.object(scope,'make_view',return_value=view) as build:
            scope.retain_completed([event],history,bridge,10.6,None,{},None)
        build.assert_called_once()
        self.assertIs(build.call_args.args[3],depth)
        self.assertIs(bridge.observe.call_args.args[2],source)
        self.assertEqual(bridge.observe.call_args.kwargs['frame_at'],10.04)

    def test_completed_sound_survives_missing_or_stale_frame_as_audio_only(self):
        event={'kind':'sound','observed_at':10.,'fix':{'z':20,'coherence':.3}}
        frame=np.zeros((12,16,3),np.uint8);depth=np.ones((12,16))
        for history,now,reason in [([],10.6,'No RGB-D'),([(10.3,frame,depth,K)],10.6,'300 ms'),
                                   ([(10.,frame,depth,K)],12.,'after capture')]:
            with self.subTest(reason=reason):
                bridge=mock.Mock();bridge.accept_event.return_value=True
                with mock.patch.object(scope,'make_view') as build:
                    scope.retain_completed([event],history,bridge,now,None,{},None)
                build.assert_not_called();bridge.observe.assert_called_once()
                self.assertIsNone(bridge.observe.call_args.args[2])
                self.assertFalse(bridge.observe.call_args.args[1]['measured'])
                self.assertIn(reason,bridge.observe.call_args.kwargs['pairing_reason'])

    def test_retention_respects_wrist_gate_and_invalid_capture_times(self):
        bridge=mock.Mock();bridge.accept_event.return_value=False
        event={'kind':'sound','observed_at':10.,'fix':{}}
        scope.retain_completed([event],[],bridge,10.5,None,{},None)
        bridge.observe.assert_not_called()
        bridge.accept_event.return_value=True
        for stamp in (float('nan'),float('inf'),11.,None):
            scope.retain_completed([dict(event,observed_at=stamp)],[],bridge,10.5,None,{},None)
        bridge.observe.assert_not_called()

    def test_nearer_surface_cannot_override_better_acoustic_score(self):
        scores = np.array([[0.98, 1.0]])
        points = np.array([[[0, 0, .2], [0, 0, 2.0]]])
        self.assertEqual(scope.best_spot(scores, np.ones_like(scores, bool), points), (0, 1))

    def test_capture_time_survives_expensive_view_processing(self):
        cfg = scope_config()
        depth, _ = render(cfg)
        loc, fix = heard_from((RIGHT_BOX[0] + RIGHT_BOX[1]) / 2, cfg)
        view = scope.make_view({'fix': fix, 'observed_at': 12.0, 'event_id': 7},
                               True, loc, depth, K, cfg)
        self.assertEqual(view['time'], 12.0)
        self.assertEqual(view['event_id'], 7)
        self.assertEqual(view['quality']['state'], 'candidate')

    def test_snapshot_keeps_scored_frame_depth_and_acoustic_evidence(self):
        frame = np.full((40, 64, 3), 90, np.uint8)
        depth = np.full((40, 64), 900, np.float32)
        gcc = np.arange(48).reshape(6, 8)
        loc = mock.Mock()
        loc.gcc.return_value = gcc
        ev = {'observed_at': 123.4, 'fix': {'evidence': (1, 2), 'direction': [0, 0, 1],
                                          'z': 20., 'coherence': .4}}
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(scope, 'SNAPSHOTS', Path(folder)):
            path = scope.save_snapshot(frame, frame, depth, K, {'array': {}}, ev, loc, frame_at=123.6)
            self.assertTrue(path.exists())
            np.testing.assert_array_equal(cv2.imread(str(path)), frame)
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['frame'], frame)
                np.testing.assert_array_equal(saved['depth_mm'], depth)
                np.testing.assert_array_equal(saved['gcc'], gcc)
                self.assertEqual(json.loads(str(saved['intrinsics'])), K)
                self.assertAlmostEqual(float(saved['observed_at']), 123.4)
                self.assertAlmostEqual(float(saved['frame_at']), 123.6)


if __name__ == '__main__':
    unittest.main()
