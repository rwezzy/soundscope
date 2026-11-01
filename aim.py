"""Point the Lite 6 at a sound: geometry and safety rules only (no hardware, no network).

Frames: the arm base frame is UFACTORY's and Viam's "world": origin at the base center on the
table, +X where the arm reaches at J1 = 0, +Y to its left, +Z up. Millimeters and degrees.
"""
import math
import tomllib
from pathlib import Path

import numpy as np

CONFIG_FILE = Path(__file__).with_name('arm_config.toml')

# Lite 6 joint ranges (UFACTORY Lite6 hardware manual). We stay LIMIT_MARGIN inside them.
JOINT_LIMITS = [(-360, 360), (-150, 150), (-3.5, 300), (-360, 360), (-124, 124), (-360, 360)]
LIMIT_MARGIN = 10.0
# Ceilings no config value can exceed.
CAPS = {'speed_deg_s': 30, 'accel_deg_s2': 200, 'max_step_deg': 45,
        'yaw_window_deg': 170, 'pitch_window_deg': 30}


def load_config(path=CONFIG_FILE):
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    s = cfg['safety']
    for key in ('speed_deg_s', 'accel_deg_s2', 'max_step_deg'):
        if not 0 < s[key] <= CAPS[key]:
            raise SystemExit(f'arm_config.toml: safety.{key} = {s[key]} must be above 0 and at most {CAPS[key]}.')
    for key in ('yaw_window_deg', 'pitch_window_deg'):
        lo, hi = s[key]
        if not -CAPS[key] <= lo <= 0 <= hi <= CAPS[key]:
            raise SystemExit(f'arm_config.toml: safety.{key} must include 0 and stay within ±{CAPS[key]}.')
    if cfg['camera']['mount'] not in ('base', 'arm'):
        raise SystemExit('arm_config.toml: camera.mount must be "base" or "arm".')
    return cfg


def fingertip_clearance(cfg):
    """How far in front of the arm-mounted lens the gripper fingertips sit, along the lens axis (mm)."""
    m = cfg['camera']['arm']
    lens = np.asarray(m['lens_direction'], float)
    tip = np.array([0.0, 0.0, m['gripper_tip_mm']])
    return float((tip - np.asarray(m['offset_mm'], float)) @ (lens / np.linalg.norm(lens)))


def warnings(cfg):
    """Setup problems worth printing before anything runs."""
    out = []
    gap = fingertip_clearance(cfg)
    if cfg['camera']['mount'] == 'arm' and gap < cfg['camera']['min_depth_mm']:
        out.append(f'the fingertips are only {gap:.0f} mm in front of the arm-mounted lens, but depth starts '
                   f'at {cfg["camera"]["min_depth_mm"]} mm. Move the camera back on its mount.')
    if math.hypot(*cfg['array']['position_mm'][:2]) < 120:
        out.append('the mic array is within 12 cm of the arm base; the arm may block or bump it.')
    return out


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sound_point(direction, cfg, distance_mm=None):
    """Likely sound position in the base frame: out from the array along its measured direction.

    direction: unit vector in the array's own frame (x = front edge, y = left, z = up), as doa.py gives it.
    """
    a = cfg['array']
    d = rot_z(a['front_yaw_deg']) @ np.asarray(direction, float)
    return np.asarray(a['position_mm'], float) + (distance_mm or cfg['aim']['assumed_distance_mm']) * d


def j1_to_face(target, pointer):
    """J1 (deg) that turns a forward-looking pointer (camera or gripper) to face `target`, or None.

    pointer: reach_mm, lateral_mm and heading_deg describe it with J1 = 0 (see [look] in the config).
    It sits at (reach, lateral) and looks `heading` to the left of +X; J1 swings it about the Z axis.
    """
    psi = math.radians(pointer['heading_deg'])
    k = pointer['reach_mm'] * math.sin(psi) - pointer['lateral_mm'] * math.cos(psi)
    r = math.hypot(target[0], target[1])
    if r <= abs(k):
        return None
    j1 = math.atan2(target[1], target[0]) - psi + math.asin(k / r)
    # Of the two lines through the target, keep the one where the target is in front of the pointer.
    pos = rot_z(math.degrees(j1))[:2, :2] @ np.array([pointer['reach_mm'], pointer['lateral_mm']])
    look = np.array([math.cos(j1 + psi), math.sin(j1 + psi)])
    if (np.asarray(target[:2], float) - pos) @ look <= 0:
        return None
    return math.degrees(j1)


def pitch_to_face(target, pointer, j1):
    """Tilt (deg, + = up) the pointer needs, relative to its starting tilt, to face `target` from J1 = j1."""
    pos = rot_z(j1)[:2, :2] @ np.array([pointer['reach_mm'], pointer['lateral_mm']])
    flat = math.hypot(*(np.asarray(target[:2], float) - pos))
    return math.degrees(math.atan2(target[2] - pointer['height_mm'], flat)) - pointer['pitch_deg']


def camera_axes(yaw_deg, pitch_deg):
    """Base-frame directions of a level camera's right (x), down (y) and lens (z) axes."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    lens = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    return right, np.cross(lens, right), lens


def camera_to_base(point, cfg):
    """Point from the camera beside the base (mm; x right, y down, z out of the lens, as depth
    cameras report it) -> arm base frame. An arm-mounted camera goes through Viam's frame system."""
    c = cfg['camera']['base']
    right, down, lens = camera_axes(c['yaw_deg'], c['pitch_deg'])
    return np.asarray(c['position_mm'], float) + point[0] * right + point[1] * down + point[2] * lens


def view_position(target, cfg):
    """Where a base-frame point lands across the base camera's image: 0 = left edge, 1 = right edge,
    outside 0..1 = off screen, None = behind the camera."""
    c = cfg['camera']['base']
    right, _, lens = camera_axes(c['yaw_deg'], c['pitch_deg'])
    v = np.asarray(target, float) - np.asarray(c['position_mm'], float)
    if v @ lens <= 0:
        return None
    return 0.5 + (v @ right) / (v @ lens) / (2 * math.tan(math.radians(cfg['camera']['hfov_deg'] / 2)))


def reach_check(target, cfg):
    """(within grasp?, horizontal distance from the base axis in mm)."""
    r = cfg['reach']
    radius = math.hypot(target[0], target[1])
    ok = (r['min_radius_mm'] <= radius <= r['max_radius_mm']
          and r['min_height_mm'] <= target[2] <= r['max_height_mm'])
    return ok, radius


class Gate:
    """Decides whether and how far to move. Only J1, and J5 when pitch is enabled, ever change."""

    def __init__(self, cfg, start_joints):
        self.s = cfg['safety']
        self.start = [float(v) for v in start_joints]
        self.ranges = {}
        for j, key in ((0, 'yaw_window_deg'), (4, 'pitch_window_deg')):
            lo, hi = self.s[key]
            lo = max(self.start[j] + lo, JOINT_LIMITS[j][0] + LIMIT_MARGIN)
            hi = min(self.start[j] + hi, JOINT_LIMITS[j][1] - LIMIT_MARGIN)
            if lo > hi:
                raise SystemExit(f'J{j + 1} starts at {self.start[j]:.0f}°, too close to its limit. '
                                 'Move it toward the middle of its range in UFACTORY Studio first.')
            self.ranges[j] = (lo, hi)
        self.last_move = -math.inf

    def accept(self, fix):
        """None if a sound fix is trustworthy enough to move for, else the reason it isn't."""
        if fix['z'] < self.s['min_confidence_z']:
            return f"not sure enough (z {fix['z']:.0f}, needs {self.s['min_confidence_z']})"
        if fix['snr_db'] < self.s['min_snr_db']:
            return f"too faint (SNR {fix['snr_db']:+.0f} dB)"
        return None

    def _toward(self, j, current, wanted):
        """(where this step ends, where the joint is finally allowed to go)."""
        lo, hi = self.ranges[j]
        if j == 0:  # J1 can go all the way around: take the equivalent angle nearest the current one
            options = [wanted + 360 * k for k in range(-2, 3)]
            wanted = min([a for a in options if lo <= a <= hi] or options, key=lambda a: abs(a - current))
        goal = min(max(wanted, lo), hi)
        step = max(-self.s['max_step_deg'], min(self.s['max_step_deg'], goal - current))
        return current + step, goal, goal != wanted

    def next_joints(self, current, want_j1, want_j5=None, now=0.0, continuing=False):
        """One safe step toward the wanted angles: (joints, note), or (None, reason not to move)."""
        if not continuing and now - self.last_move < self.s['min_interval_s']:
            return None, 'too soon after the last move'
        new = [float(v) for v in current]
        moving, clipped, left = False, False, 0.0
        for j, wanted in ((0, want_j1), (4, want_j5)):
            if wanted is None:
                continue
            end, goal, hit_limit = self._toward(j, new[j], wanted)
            if abs(goal - new[j]) >= self.s['deadband_deg']:
                new[j], moving = end, True
                clipped |= hit_limit
                left = max(left, abs(goal - end))
        if not moving:
            return None, 'already facing it'
        note = (' (stopping at the allowed turn limit)' if clipped else '') + \
               (f' ({left:.0f}° more to go)' if left >= 0.5 else '')
        return new, note

    def moved(self, now):
        self.last_move = now


def viam_arm_attributes(cfg):
    s = cfg['safety']
    return {'host': 'ARM_IP_FROM_THE_BASE_LABEL', 'speed_degs_per_sec': s['speed_deg_s'],
            'acceleration_degs_per_sec_per_sec': s['accel_deg_s2'],
            'collision_sensitivity': s['collision_sensitivity']}


def viam_frames(cfg):
    """Frame-system JSON for the Viam app, built from the same numbers this program uses."""
    def frame(parent, xyz, direction=(0, 0, 1), th=0.0):
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        return {'parent': parent,
                'translation': {k: round(float(v), 1) for k, v in zip('xyz', xyz)},
                'orientation': {'type': 'ov_degrees',
                                'value': {**{k: round(float(v), 4) for k, v in zip('xyz', d)}, 'th': th}}}

    base = cfg['camera']['base']
    m = cfg['camera']['arm']
    return {'arm': frame('world', (0, 0, 0)),
            'camera_beside_base': frame('world', base['position_mm'],
                                        camera_axes(base['yaw_deg'], base['pitch_deg'])[2]),
            'camera_on_arm': frame(cfg['viam']['arm_name'], m['offset_mm'], m['lens_direction'], m['roll_deg'])}
