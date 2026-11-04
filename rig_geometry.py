"""Rigid camera/microphone geometry, with optional guarded robot-frame previews.

Every translation is in METRES. T_destination_from_source maps column vectors
from source into destination. Camera axes are right/down/forward; array axes
match doa.MIC_XYZ. This module has no device connections or motion commands.

A rigid pair has the same acoustic geometry whether on a desk or a wrist.
Robot/world coordinates additionally require a measured calibration and, for a
wrist, a synchronized stationary tool pose. A CAD offset alone is not calibration.
"""
import json
import math
import time
from pathlib import Path

import numpy as np


def validate_transform(value, name='transform'):
    """Return a copied, read-only SE(3) matrix; reject malformed or millimetre inputs."""
    try:
        t = np.array(value, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a finite 4x4 matrix in metres') from exc
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError(f'{name} must be a finite 4x4 matrix')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError(f'{name} must have homogeneous bottom row [0,0,0,1]')
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError(f'{name} rotation must be orthonormal and right-handed')
    if np.linalg.norm(t[:3, 3]) > 20:
        raise ValueError(f'{name} translation exceeds 20 m; check metre versus millimetre units')
    t.setflags(write=False)
    return t


def _points(value):
    p = np.asarray(value, dtype=float)
    if p.ndim == 0 or p.shape[-1] != 3 or not np.isfinite(p).all():
        raise ValueError('points must be finite vectors with final dimension 3, in metres')
    return p


def transform_points(t, points):
    p = _points(points)
    return p @ t[:3, :3].T + t[:3, 3]


def _blocked(reason):
    return {'ok': False, 'reason': reason, 'point_base_m': None, 'frame': 'base'}


def _finite_time(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


class RigGeometry:
    def __init__(self, T_camera_from_array, *, mode='fixed', T_base_from_camera=None,
                 T_tool_from_camera=None, robot_calibrated=False, calibration_id=None,
                 max_age_s=1.2, max_sync_error_s=0.15, settle_s=0.3):
        if mode not in ('fixed', 'wrist'):
            raise ValueError('mode must be fixed or wrist')
        if not all(math.isfinite(x) and x > 0 for x in (max_age_s, max_sync_error_s, settle_s)):
            raise ValueError('timing limits must be positive and finite')
        self.mode = mode
        self.T_camera_from_array = validate_transform(T_camera_from_array, 'T_camera_from_array')
        self.T_base_from_camera = (None if T_base_from_camera is None else
                                  validate_transform(T_base_from_camera, 'T_base_from_camera'))
        self.T_tool_from_camera = (None if T_tool_from_camera is None else
                                  validate_transform(T_tool_from_camera, 'T_tool_from_camera'))
        if not isinstance(robot_calibrated, bool):
            raise ValueError('robot_calibrated must be a JSON boolean')
        self.robot_calibrated = robot_calibrated
        self.calibration_id = calibration_id
        self.max_age_s, self.max_sync_error_s, self.settle_s = max_age_s, max_sync_error_s, settle_s

    @classmethod
    def from_legacy_config(cls, cfg):
        """Preserve old acoustic projection exactly; do not certify its robot transform."""
        import aim
        if cfg['camera']['mount'] != 'base':
            raise ValueError('legacy arm settings do not define a measured rigid sensor transform; use a rig JSON file')
        c = cfg['camera']['base']
        bc = np.eye(4)
        bc[:3, :3] = np.column_stack(aim.camera_axes(c['yaw_deg'], c['pitch_deg']))
        bc[:3, 3] = np.asarray(c['position_mm']) / 1000
        ba = np.eye(4)
        ba[:3, :3] = aim.array_axes(cfg)
        ba[:3, 3] = np.asarray(cfg['array']['position_mm']) / 1000
        return cls(np.linalg.inv(bc) @ ba, T_base_from_camera=bc, robot_calibrated=False)

    @classmethod
    def from_dict(cls, data):
        if data.get('units') != 'm':
            raise ValueError('rig configuration must explicitly specify units: m')
        return cls(data['T_camera_from_array'], mode=data.get('mode', 'fixed'),
                   T_base_from_camera=data.get('T_base_from_camera'),
                   T_tool_from_camera=data.get('T_tool_from_camera'),
                   robot_calibrated=data.get('robot_calibrated', False),
                   calibration_id=data.get('calibration_id'))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @property
    def array_origin_camera_m(self):
        return self.T_camera_from_array[:3, 3].copy()

    def camera_points_to_array(self, points_m):
        return (_points(points_m) - self.array_origin_camera_m) @ self.T_camera_from_array[:3, :3]

    def array_direction_to_camera(self, direction):
        d = _points(direction)
        if d.shape != (3,) or np.linalg.norm(d) < 1e-12:
            raise ValueError('direction must be one nonzero finite 3-vector')
        return self.T_camera_from_array[:3, :3] @ (d / np.linalg.norm(d))

    def camera_point_to_base(self, point_m, pose_packet=None, *, observed_at=None, now=None):
        """Guarded conversion for a *candidate surface point*, never authorization to move.

        Times use time.monotonic() on the same host, not wall-clock/UTC timestamps.
        Wrist packets need T_base_from_tool, captured_at, moving=False, and
        settled_since. The entire observation must occur after the settling interval.
        """
        now = time.monotonic() if now is None else now
        try:
            p = _points(point_m)
        except (ValueError, TypeError):
            return _blocked('invalid camera point')
        if p.shape != (3,) or p[2] <= 0 or np.linalg.norm(p) > 20:
            return _blocked('camera point must be a positive-depth point in metres')
        if not self.robot_calibrated or not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            return _blocked('robot extrinsic calibration has not been verified and identified')
        if not _finite_time(now) or not _finite_time(observed_at):
            return _blocked('a monotonic observation timestamp is required')
        if not 0 <= now - observed_at <= self.max_age_s:
            return _blocked('observation is stale or from the future')
        if self.mode == 'fixed':
            if self.T_base_from_camera is None:
                return _blocked('T_base_from_camera is missing')
            t = self.T_base_from_camera
        else:
            if self.T_tool_from_camera is None:
                return _blocked('measured T_tool_from_camera is missing')
            if not isinstance(pose_packet, dict):
                return _blocked('synchronized wrist pose is missing')
            captured = pose_packet.get('captured_at')
            settled = pose_packet.get('settled_since')
            if not _finite_time(captured) or not 0 <= now - captured <= self.max_age_s:
                return _blocked('wrist pose is stale or from the future')
            if abs(captured - observed_at) > self.max_sync_error_s:
                return _blocked('wrist pose and observation are not synchronized')
            if pose_packet.get('moving') is not False:
                return _blocked('wrist motion state is moving or unknown')
            if not _finite_time(settled) or observed_at - settled < self.settle_s:
                return _blocked('wait for wrist movement and motor noise to settle')
            try:
                bt = validate_transform(pose_packet.get('T_base_from_tool'), 'T_base_from_tool')
            except ValueError as exc:
                return _blocked(str(exc))
            t = bt @ self.T_tool_from_camera
        return {'ok': True, 'reason': 'calibrated coordinate preview only',
                'point_base_m': transform_points(t, p).tolist(), 'frame': 'base',
                'calibration_id': self.calibration_id}

    def status(self):
        return {'mode': self.mode, 'units': 'm', 'array_origin_camera_m': self.array_origin_camera_m.tolist(),
                'robot_calibrated': self.robot_calibrated and bool(self.calibration_id),
                'calibration_id': self.calibration_id, 'motion_enabled': False}


class RobotPreview:
    """A read-only optional application; emits no trajectory, joint command or grasp."""
    def __init__(self, rig):
        self.rig = rig

    def plan(self, event, pose_packet=None, *, now=None, action='point'):
        result = {'dry_run': True, 'motion_enabled': False, 'action': action,
                  'status': 'blocked', 'point_base_m': None, 'source_verified': False}
        if action != 'point':
            result['reason'] = 'grasping and live motion are not implemented; only pointing previews are available'
            return result
        quality = event.get('quality')
        if isinstance(quality, dict):
            quality = quality.get('state')
        if quality != 'candidate':
            result['reason'] = 'weak or ambiguous acoustic evidence'
            return result
        point = event.get('point_camera_m')
        if point is None or event.get('measured') is False:
            result['reason'] = 'a measured visible-surface candidate is required'
            return result
        mapped = self.rig.camera_point_to_base(point, pose_packet,
                                              observed_at=event.get('observed_at'), now=now)
        result.update(reason=mapped['reason'], point_base_m=mapped['point_base_m'])
        if mapped['ok']:
            result.update(status='preview', reason='candidate surface only; source and collision-free reach are unverified')
        return result
