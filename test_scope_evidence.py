"""Regression checks for actual score selection and saved diagnostic evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import scope
from test_scope import K, RIGHT_BOX, heard_from, render, scope_config


class Evidence(unittest.TestCase):
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
