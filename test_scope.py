"""Offline tests for the camera view: python -m unittest test_scope

A small ray-cast scene (table, two boxes, wall) stands in for the Gemini 2, and simulated mic
signals stand in for the array, so the whole chain camera -> base -> array frame gets checked.
"""
import copy
import math
import unittest

import numpy as np

import aim
import scope
from camera import backproject
from doa import FS, Localizer
from mic_test import simulate

K = {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 200.0}
SHAPE = (400, 640)
TABLE_Z = -0.45                                          # table top, meters below the lens
RIGHT_BOX = (np.array([0.75, -0.20, TABLE_Z]), np.array([0.85, -0.10, TABLE_Z + 0.10]))
LEFT_BOX = (np.array([0.85, 0.15, TABLE_Z]), np.array([0.95, 0.25, TABLE_Z + 0.12]))


def scope_config(mount='facing', yaw=0.0):
    cfg = copy.deepcopy(aim.load_config())
    cfg['camera']['base'].update(position_mm=[0, 0, 0], yaw_deg=0.0, pitch_deg=-25.0)
    cfg['array'].update(mount=mount, position_mm=[0, 0, -80], front_yaw_deg=yaw)
    return cfg


def render(cfg):
    """Depth image (mm) and which object each pixel sees: 0 nothing, 1 table, 2 right box, 3 left box, 4 wall."""
    v, u = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    c = cfg['camera']['base']
    right, down, lens = aim.camera_axes(c['yaw_deg'], c['pitch_deg'])
    ray = backproject(u, v, 1.0, K)            # camera z = 1, so the ray parameter is the depth
    ray = ray[..., :1] * right + ray[..., 1:2] * down + ray[..., 2:3] * lens
    origin = np.asarray(c['position_mm'], float) / 1000
    depth, what = np.full(SHAPE, np.inf), np.zeros(SHAPE, int)

    def take(t, label):
        closer = (t > 0) & (t < depth)
        depth[closer], what[closer] = t[closer], label

    safe = np.where(np.abs(ray) < 1e-12, 1e-12, ray)
    take((TABLE_Z - origin[2]) / safe[..., 2], 1)
    take((2.0 - origin[0]) / safe[..., 0], 4)
    for label, (lo, hi) in ((2, RIGHT_BOX), (3, LEFT_BOX)):
        t1, t2 = (lo - origin) / safe, (hi - origin) / safe
        tmin, tmax = np.minimum(t1, t2).max(axis=-1), np.maximum(t1, t2).min(axis=-1)
        take(np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf), label)
    return np.where(np.isfinite(depth), depth * 1000, 0).astype(np.float32), what


def heard_from(point_base_m, cfg, snr=0, seed=1):
    """A sound fix as the tracker would report it, for a source at a base-frame point."""
    p = aim.base_to_array(np.asarray(point_base_m) * 1000, cfg)
    d = p / np.linalg.norm(p)
    angle, elevation = -math.degrees(math.atan2(d[1], d[0])), math.degrees(math.asin(d[2]))
    x = simulate('keys', snr, angle, np.random.default_rng(seed), elevation=elevation, seconds=3.0)
    loc = Localizer()
    loc.calibrate(x[:FS])
    X = loc.stft(x[FS:])
    evidence = loc.evidence(X, loc.weights(loc.power(X))[0])
    fix = loc.peak(*evidence)
    fix.update(evidence=evidence, snr_db=0.0, seconds=2.0)
    return loc, fix


class View(unittest.TestCase):
    def test_srp_points_far_away_matches_the_direction_map(self):
        cfg = scope_config()
        loc, fix = heard_from((RIGHT_BOX[0] + RIGHT_BOX[1]) / 2, cfg)
        gcc = loc.gcc(*fix['evidence'][:2])
        far = loc.srp_points(gcc, 50 * fix['direction'][None])[0]
        self.assertAlmostEqual(far, fix['coherence'], delta=0.03)
        rotated = aim.rot_z(30) @ fix['direction']
        self.assertGreater(far, loc.srp_points(gcc, 50 * rotated[None])[0])

    def test_finds_and_outlines_the_box_that_made_the_sound(self):
        cfg = scope_config()
        depth, what = render(cfg)
        for box, label in ((RIGHT_BOX, 2), (LEFT_BOX, 3)):
            with self.subTest(box=label):
                loc, fix = heard_from((box[0] + box[1]) / 2, cfg)
                view = scope.make_view({'fix': fix}, True, loc, depth, K, cfg)
                u, v = view['spot']
                self.assertEqual(what[v, u], label, f'spot landed on object {what[v, u]}')
                self.assertIsNotNone(view['contour'])
                filled = np.zeros(SHAPE, np.uint8)
                import cv2
                cv2.drawContours(filled, [view['contour']], -1, 1, -1)
                on_box = np.count_nonzero(filled & (what == label))
                self.assertGreater(on_box / max(1, np.count_nonzero(what == label)), 0.5)  # covers the box
                self.assertGreater(on_box / np.count_nonzero(filled), 0.7)                 # not the table

    def test_printed_rig_orientation_finds_the_box_too(self):
        cfg = scope_config()
        cfg['array'].update(cable_edge='down', position_mm=[-2, 0, -73.5])  # values from cad/rig.py
        axes = aim.array_axes(cfg)
        self.assertAlmostEqual(np.linalg.det(axes), 1.0)                     # a real rotation, not a mirror
        np.testing.assert_allclose(axes[:, 0], [0, 0, -1], atol=1e-9)        # cable edge points down
        depth, what = render(cfg)
        loc, fix = heard_from((RIGHT_BOX[0] + RIGHT_BOX[1]) / 2, cfg)
        u, v = scope.make_view({'fix': fix}, False, loc, depth, K, cfg)['spot']
        self.assertEqual(what[v, u], 2)

    def test_sound_beside_the_camera_points_off_screen(self):
        cfg = scope_config()
        depth, _ = render(cfg)
        loc, fix = heard_from(np.array([0.2, -1.5, TABLE_Z]), cfg)
        view = scope.make_view({'fix': fix}, True, loc, depth, K, cfg)
        self.assertEqual(view['off_screen'][0], 'right')

    def test_click_calibration_recovers_a_wrong_board_direction(self):
        true_cfg, wrong_cfg = scope_config(yaw=0.0), scope_config(yaw=25.0)
        depth, what = render(true_cfg)
        center = (RIGHT_BOX[0] + RIGHT_BOX[1]) / 2
        _, fix = heard_from(center, true_cfg)           # what the board really hears
        vs, us = np.nonzero(what == 2)
        click = (int(np.median(us)), int(np.median(vs)))
        value, problem = scope.calibrate_yaw(wrong_cfg, fix, click, depth, K)
        self.assertIsNone(problem)
        self.assertAlmostEqual(value, 0.0, delta=4.0)


class CalibrationFlow(unittest.TestCase):
    """Clicking works whether the sound came just before the click, after it, or never."""

    def setUp(self):
        import time
        from unittest import mock
        self.now = time.monotonic()
        self.cfg = scope_config(yaw=25.0)
        self.depth, what = render(scope_config(yaw=0.0))
        _, self.fix = heard_from((RIGHT_BOX[0] + RIGHT_BOX[1]) / 2, scope_config(yaw=0.0))
        vs, us = np.nonzero(what == 2)
        self.click = (int(np.median(us)), int(np.median(vs)))
        patcher = mock.patch.object(scope, 'save_front_yaw')
        self.saved = patcher.start()
        self.addCleanup(patcher.stop)
        printer = mock.patch('builtins.print')
        printer.start()
        self.addCleanup(printer.stop)

    def step(self, click_age, heard_age):
        cal = {'active': True, 'click': (*self.click, self.now - click_age)}
        ui = {'note': '', 'last': None if heard_age is None else (self.now - heard_age, self.fix)}
        scope.calibrate_step(cal, ui, self.cfg, self.depth, K)
        return cal, ui

    def test_sound_just_before_the_click(self):
        cal, ui = self.step(click_age=0.5, heard_age=2.0)
        self.assertFalse(cal['active'])
        self.assertAlmostEqual(self.saved.call_args.args[0], 0.0, delta=4.0)

    def test_sound_after_the_click(self):
        cal, ui = self.step(click_age=3.0, heard_age=0.2)
        self.assertFalse(cal['active'])

    def test_waits_then_gives_up_without_sound(self):
        cal, ui = self.step(click_age=4.0, heard_age=None)
        self.assertTrue(cal['active'])
        self.assertIsNotNone(cal['click'])        # still waiting
        cal, ui = self.step(click_age=11.0, heard_age=None)
        self.assertIsNone(cal['click'])
        self.assertIn('heard no sound', ui['note'])
        self.saved.assert_not_called()


class VoiceCommands(unittest.TestCase):
    """What each spoken command does, with a stand-in for the Meta API."""

    def setUp(self):
        from unittest import mock
        printer = mock.patch('builtins.print')
        printer.start()
        self.addCleanup(printer.stop)

    def fake_meta(self, transcript, action='find', sound='keys', error=None):
        from types import SimpleNamespace
        from unittest import mock

        from meta_api import MetaError
        meta = SimpleNamespace(transcribe=mock.Mock(return_value=transcript),
                               parse_command=mock.Mock(return_value={'action': action, 'sound': sound, 'reply': 'ok'}))
        if error:
            meta.transcribe.side_effect = MetaError(error)
        return meta

    def command(self, *args, hunt=None, wav=b'wav', **kwargs):
        hunt = hunt or {'target': None, 'mode': 'follow', 'found': None}
        ui = {'note': ''}
        scope.handle_voice(self.fake_meta(*args, **kwargs), wav, ui, hunt)
        return hunt, ui['note']

    def test_find_and_follow_set_the_target(self):
        hunt, note = self.command('find my keys', 'find', 'keys')
        self.assertEqual((hunt['target'], hunt['mode']), ('keys', 'find'))
        self.assertIn('looking for "keys"', note)
        hunt, _ = self.command('follow the phone', 'follow', 'phone ringing')
        self.assertEqual((hunt['target'], hunt['mode']), ('phone ringing', 'follow'))

    def test_stop_pauses_and_home_shows_everything(self):
        busy = {'target': 'keys', 'mode': 'find', 'found': {'view': 1}}
        hunt, _ = self.command('stop', 'stop', '', hunt=dict(busy))
        self.assertEqual((hunt['target'], hunt['mode'], hunt['found']), (None, 'paused', None))
        hunt, _ = self.command('show everything', 'home', '', hunt=dict(busy))
        self.assertEqual((hunt['target'], hunt['mode']), (None, 'follow'))

    def test_non_commands_and_failures_change_nothing(self):
        for kwargs, expect in ((dict(transcript="what's the weather", action='ignore', sound=''), 'not a SoundScope'),
                               (dict(transcript='   '), "Didn't catch"),
                               (dict(transcript='x', error='HTTP 500'), 'failed'),
                               (dict(transcript='x', wav=None), "Didn't hear")):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                hunt, note = self.command(**kwargs, hunt={'target': 'keys', 'mode': 'follow', 'found': None})
                self.assertEqual(hunt['target'], 'keys')
                self.assertIn(expect, note)

    def test_find_mode_holds_the_first_confirmed_match(self):
        from types import SimpleNamespace
        from unittest import mock
        hunt, ui = {'target': 'keys', 'mode': 'find', 'found': None}, {'note': ''}
        no = SimpleNamespace(sound_matches=mock.Mock(return_value={'match': False, 'heard': 'speech', 'confidence': 0.9}))
        other = {'time': 0}
        scope.check_sound(no, b'wav', 'keys', other, hunt, ui)
        self.assertTrue(other['rejected'])
        self.assertIsNone(hunt['found'])
        yes = SimpleNamespace(sound_matches=mock.Mock(return_value={'match': True, 'heard': 'keys jingling', 'confidence': 0.9}))
        view = {'time': 0}
        scope.check_sound(yes, b'wav', 'keys', view, hunt, ui)
        self.assertIs(hunt['found'], view)
        self.assertTrue(view['hold'] and not view['rejected'])
        self.assertEqual(scope.hunt_status(hunt), 'FOUND: keys')

    def test_screen_draws_in_every_state(self):
        cfg = scope_config()
        depth, _ = render(cfg)
        loc, fix = heard_from((RIGHT_BOX[0] + RIGHT_BOX[1]) / 2, cfg)
        frame = np.zeros((*SHAPE, 3), np.uint8)
        view = scope.make_view({'fix': fix}, True, loc, depth, K, cfg)
        for extra in ({}, {'verdict': None, 'rejected': True}, {'hold': True},
                      {'verdict': {'match': True, 'heard': 'keys', 'confidence': 0.9}, 'rejected': False}):
            for hunt in ({'target': None, 'mode': 'follow', 'found': None},
                         {'target': 'keys', 'mode': 'paused', 'found': None}):
                out = scope.draw(frame, {**view, **extra}, hunt['target'], 'note', scope.hunt_status(hunt))
                self.assertEqual(out.shape, frame.shape)
                self.assertGreater(int(out.sum()), 0)


if __name__ == '__main__':
    unittest.main()
