"""Integration regressions for acoustic projection and the optional robot preview."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import aim
import scope
import studio
from rig_geometry import RigGeometry
from test_scope import K, RIGHT_BOX, heard_from, render, scope_config


class StudioProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = scope_config()
        cls.cfg['camera']['mount'] = 'base'
        cls.depth, _ = render(cls.cfg)
        cls.loc, cls.fix = heard_from((RIGHT_BOX[0] + RIGHT_BOX[1]) / 2, cls.cfg)
        cls.gcc = cls.loc.gcc(*cls.fix['evidence'][:2])

    def test_srp_scores_and_candidate_match_legacy_exactly(self):
        rig=RigGeometry.from_legacy_config(self.cfg)
        old=scope.score_view(self.loc,self.gcc,self.depth,K,self.cfg)
        new=scope.score_view(self.loc,self.gcc,self.depth,K,self.cfg,rig=rig)
        for a,b in zip(old,new):np.testing.assert_allclose(a,b,atol=1e-10)
        event={'fix':self.fix,'observed_at':100.,'event_id':1}
        a=scope.make_view(event,True,self.loc,self.depth,K,self.cfg)
        b=scope.make_view(event,True,self.loc,self.depth,K,self.cfg,rig=rig)
        self.assertEqual(a['spot'],b['spot'])
        self.assertEqual(a['quality']['state'],b['quality']['state'])
        np.testing.assert_allclose(a['point'],b['point'])
        self.assertNotIn('reach',b)  # Raw legacy offsets must not imply verified robot reach.

    def test_wrist_relative_projection_does_not_read_robot_or_legacy_frame(self):
        rig=RigGeometry.from_legacy_config(self.cfg)
        rig.mode='wrist'
        a=scope.score_view(self.loc,self.gcc,self.depth,K,self.cfg,rig=rig)
        b=scope.score_view(self.loc,self.gcc,self.depth,K,{},rig=rig)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y)
        self.assertEqual(scope.off_screen(self.fix['direction'],{},K,self.depth.shape,rig=rig),
                         scope.off_screen(self.fix['direction'],self.cfg,K,self.depth.shape,rig=rig))

    def test_offscreen_all_directions_agrees_with_legacy_in_rotated_frames(self):
        rng=np.random.default_rng(81)
        for mount in ['flat','facing']:
            cfg=copy.deepcopy(self.cfg)
            cfg['array'].update(mount=mount,position_mm=[210,-70,60],front_yaw_deg=33)
            cfg['camera']['base'].update(position_mm=[180,-100,130],yaw_deg=-24,pitch_deg=12)
            rig=RigGeometry.from_legacy_config(cfg)
            for direction in rng.normal(size=(40,3)):
                direction/=np.linalg.norm(direction)
                old=scope.off_screen(direction,cfg,K,self.depth.shape)
                new=scope.off_screen(direction,cfg,K,self.depth.shape,rig=rig)
                if old is None:self.assertIsNone(new)
                else:
                    self.assertEqual(old[0],new[0]);self.assertAlmostEqual(old[1],new[1],places=10)

    def test_saving_yaw_rebuilds_projection_without_robot_calibration(self):
        bridge=studio.Studio.__new__(studio.Studio)
        bridge.mode='wrist'
        cfg=copy.deepcopy(self.cfg)
        before=bridge.refresh_legacy_rig(cfg)
        cfg['array']['front_yaw_deg']+=22
        after=bridge.refresh_legacy_rig(cfg)
        self.assertEqual(after.mode,'wrist')
        self.assertFalse(after.robot_calibrated)
        self.assertGreater(np.linalg.norm(before.array_direction_to_camera([0,0,1])-
                                          after.array_direction_to_camera([0,0,1])),.1)
        old=scope.score_view(self.loc,self.gcc,self.depth,K,cfg)
        new=scope.score_view(self.loc,self.gcc,self.depth,K,cfg,rig=after)
        np.testing.assert_allclose(old[0],new[0],atol=1e-10)

    def test_observation_persists_camera_point_and_time_for_preview(self):
        rig=RigGeometry.from_legacy_config(self.cfg)
        # This is a deliberately synthetic calibration, solely for the offline test.
        rig=RigGeometry(rig.T_camera_from_array,T_base_from_camera=np.eye(4),
                        robot_calibrated=True,calibration_id='synthetic-test-only')
        with tempfile.TemporaryDirectory() as tmp, mock.patch('urllib.request.urlopen',side_effect=AssertionError('No network')):
            bridge=studio.Studio(tmp,rig=rig)
            try:
                now=time.monotonic()
                ev={'fix':self.fix,'observed_at':now,'event_id':99,'level_dbfs':-37.5}
                view={'quality':{'state':'candidate'},'measured':True,'point':np.array([.1,-.1,.8]),'spot':(20,30)}
                frame=np.zeros((80,100,3),np.uint8)
                bridge.observe(ev,view,frame,now,self.cfg,rig)
                saved=bridge.store.list_events()[0]
                np.testing.assert_allclose(saved['point_camera_m'],[.1,-.1,.8])
                self.assertEqual(saved['observed_at'],now)
                self.assertAlmostEqual(saved['range_m'],np.sqrt(.66))
                self.assertFalse(saved['source_verified'])
                self.assertEqual(saved['level_dbfs'],-37.5)
                result=bridge.preview(saved['id'])
                self.assertEqual(result['status'],'preview')
                self.assertFalse(result['motion_enabled'])
                self.assertTrue(result['dry_run'])
                bridge.observe(ev,view,frame,now,self.cfg,rig)
                self.assertEqual(len(bridge.store.list_events()),1)
            finally:bridge.close()

    def test_weak_evidence_cannot_create_depth_target(self):
        rig=RigGeometry.from_legacy_config(self.cfg)
        with tempfile.TemporaryDirectory() as tmp:
            bridge=studio.Studio(tmp,rig=rig)
            try:
                now=time.monotonic()
                ev={'fix':self.fix,'observed_at':now,'event_id':100}
                view={'quality':{'state':'weak'},'measured':True,'point':np.array([.1,0,.8]),'spot':(20,30)}
                bridge.observe(ev,view,np.zeros((80,100,3),np.uint8),now,self.cfg,rig)
                saved=bridge.store.list_events()[0]
                self.assertIsNone(saved['point_camera_m'])
                self.assertIsNone(saved['range_m'])
                self.assertIsNone(saved['pixel'])
                self.assertEqual(bridge.preview(saved['id'])['status'],'blocked')
            finally:bridge.close()

    def test_wrist_event_gate_rejects_audio_spanning_movement(self):
        bridge=studio.Studio.__new__(studio.Studio)
        bridge.mode='wrist';bridge.stable=True
        bridge.motion=studio.MotionGate(settle_s=.8)
        bridge.motion.unstable_at=100
        self.assertFalse(bridge.accept_event({'onset_at':100.5,'observed_at':101.5}))
        self.assertFalse(bridge.accept_event({'observed_at':101.5}))
        self.assertTrue(bridge.accept_event({'onset_at':100.9,'observed_at':101.5}))
        bridge.stable=False
        self.assertFalse(bridge.accept_event({'onset_at':100.9,'observed_at':101.5}))

    def test_completed_sound_preview_uses_matching_historical_pose(self):
        # An event's quiet tail delays delivery, so the latest pose is too new.
        now=time.monotonic()
        historical=dict(captured_at=now-.7,moving=False,settled_since=now-2,
                        T_base_from_tool=np.eye(4).tolist())
        latest=dict(historical,captured_at=now,status='ready',samples=[historical])
        rig=RigGeometry(np.eye(4),mode='wrist',T_tool_from_camera=np.eye(4),
                        robot_calibrated=True,calibration_id='synthetic-only')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pose.json';path.write_text(json.dumps(latest),encoding='utf-8')
            bridge=studio.Studio(tmp,rig=rig,mode='wrist',pose_file=path)
            try:
                event=bridge.store.add(dict(id='delayed',timestamp='2026-11-04T00:00:00Z',
                    quality='candidate',point_camera_m=[.1,0,1],observed_at=now-.72,
                    measurement_frame_at=now-.7,source_verified=False))
                result=bridge.preview(event['id'])
                self.assertEqual(result['status'],'preview')
                self.assertFalse(result['motion_enabled'])
                latest.update(status='blocked',moving=True,T_base_from_tool=None)
                path.write_text(json.dumps(latest),encoding='utf-8')
                self.assertEqual(bridge.preview(event['id'])['status'],'blocked')
            finally:bridge.close()


if __name__=='__main__':unittest.main()
