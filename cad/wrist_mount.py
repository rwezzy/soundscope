"""SoundScope wrist mount: camera and mic board ride on the Lite 6 gripper, looking the same way as the
suction cup. One printed part, no holes used on the board or the gripper.

  pip install trimesh manifold3d matplotlib scipy
  python cad/wrist_mount.py    writes wrist_mount.stl (print as-is, flat side down) and wrist_mount.png

- A split ring clamps around the gripper body (77.5 mm). One spare M6x10 screw closes the split: it
  goes through the counterbored ear and cuts its own thread in the other ear. The second screw is a spare.
- The Gemini 2 slides in sideways (along its length) into a sleeve on one side, lenses toward the
  cup. Crush ribs make it a press fit; both ends stay open for the Type-C plug.
- The reSpeaker slides edge-first into a C-shaped groove on the other side, LED face toward the cup,
  USB edge outward. Notches leave the mic spots open; two small bumps keep it from sliding out.

Measured with calipers, checked against the Gemini 2 datasheet (90 x 30 x 25 mm +-0.5):
camera 89.2 long, 29.9 deep, 24.0 tall. Board 1.2 thick, ~100 mm across (Seeed drawing: 100).
Gripper body 77.5. Spare screws: head 5.9 tall, shaft 9.9 long -> M6x10 socket head.

Frame: Z = gripper axis, the cup looks toward -Z; z = 0 is the bed when printing.
"""
from pathlib import Path

import numpy as np
import trimesh

from rig import box, cylinder, overhangs

HERE = Path(__file__).resolve().parent

# Gripper clamp
GRIP_D = 77.5
RING_ID, RING_T, RING_H = GRIP_D + 0.3, 4.0, 18.0
SPLIT = 3.0                      # closes when the screw is tightened
EAR_T, EAR_OUT = 10.0, 17.0
CLEAR_D, PILOT_D, HEAD_D, UNDER_HEAD = 6.6, 4.9, 10.5, 2.0
SCREW_Z = (5.0, 13.0)

# Camera (Gemini 2), lying with its height radial, length along Y, depth along Z
CAM_L, CAM_D, CAM_H = 89.2, 29.9, 24.0
FIT = 0.15
SLEEVE_L, SLEEVE_WALL, CAM_LIP, LIP_T = 50.0, 3.0, 1.5, 2.0
RIB, RIB_W = 0.3, 1.2            # crush ribs: 0.3 mm interference

# Board (reSpeaker XVF3800), flat, LED face down
BOARD_R, BOARD_T = 50.0, 1.2
GROOVE_R, GROOVE_W = BOARD_R + 0.4, BOARD_T + 0.25
BOARD_LIP_R = BOARD_R - 1.7      # lips overlap the rim by 1.7 mm; mic spots are ~3.3 mm in
ARC_WALL, ARC_H = 4.0, 6.0
GROOVE_Z = 2.0
MIC_R = 46.7                     # 4 mics on a 66 mm square

R_OUT = RING_ID / 2 + RING_T
CAM_X = -(R_OUT + SLEEVE_WALL - 0.5 + (CAM_H / 2 + FIT))      # camera center
CAM_Z0 = LIP_T                                                  # camera front face
BOARD_X = R_OUT + GROOVE_R + ARC_WALL - 1.5                     # board center


def ring():
    solids = [cylinder((0, 0, RING_H / 2), 2 * R_OUT, RING_H, sections=128)]
    y0 = R_OUT - 4
    solids += [box((SPLIT / 2, y0, 0), (SPLIT / 2 + EAR_T, R_OUT + EAR_OUT, RING_H)),
               box((-SPLIT / 2 - EAR_T, y0, 0), (-SPLIT / 2, R_OUT + EAR_OUT, RING_H))]
    cuts = [cylinder((0, 0, RING_H / 2), RING_ID, RING_H + 2, sections=128),
            box((-SPLIT / 2, 0, -1), (SPLIT / 2, R_OUT + EAR_OUT + 1, RING_H + 1))]
    sy = R_OUT + EAR_OUT / 2 + 1
    for z in SCREW_Z:
        x_head = SPLIT / 2 + UNDER_HEAD
        cuts += [rot_x(cylinder((0, 0, 0), CLEAR_D, EAR_T + 2), (SPLIT / 2 + EAR_T / 2, sy, z)),
                 rot_x(cylinder((0, 0, 0), HEAD_D, EAR_T), (x_head + EAR_T / 2, sy, z)),
                 rot_x(cylinder((0, 0, 0), PILOT_D, EAR_T - 1), (-SPLIT / 2 - (EAR_T - 1) / 2, sy, z))]
    return solids, cuts


def rot_x(m, at):
    """Cylinder made along Z, turned to lie along X, centered at `at`."""
    m = m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    m.apply_translation(at)
    return m


def sleeve():
    hx, hy = CAM_H / 2 + FIT, SLEEVE_L / 2
    z1 = CAM_Z0 + CAM_D + 2 * FIT
    x_in, x_out = CAM_X + hx, CAM_X - hx
    solids = [box((x_out - SLEEVE_WALL, -hy, 0), (x_in + SLEEVE_WALL, hy, z1 + LIP_T)),
              box((x_in, -hy - 4, 0), (-R_OUT + 1.5, hy + 4, RING_H))]            # bridge to the ring
    cuts = [box((x_out, -hy - 1, CAM_Z0), (x_in, hy + 1, z1)),                    # camera body
            box((x_out + CAM_LIP, -hy - 1, -1), (x_in - CAM_LIP, hy + 1, CAM_Z0 + 0.01))]  # lens window; the back stays closed
    ribs = []
    for x, s in ((x_out, 1), (x_in, -1)):
        for y in (-hy / 2, 0, hy / 2):
            ribs.append(box((min(x, x + s * (RIB + FIT)), y - RIB_W / 2, CAM_Z0 + 3),
                            (max(x, x + s * (RIB + FIT)), y + RIB_W / 2, z1 - 3)))
    return solids, cuts, ribs


def board_arc():
    cx = BOARD_X
    solids = [cylinder((cx, 0, ARC_H / 2), 2 * (GROOVE_R + ARC_WALL), ARC_H, sections=180),
              box((R_OUT - 3, -22, 0), (cx - BOARD_LIP_R + 2, 22, ARC_H))]        # bridge to the ring
    cuts = [cylinder((cx, 0, ARC_H / 2), 2 * BOARD_LIP_R, ARC_H + 2, sections=180),
            cylinder((cx, 0, GROOVE_Z + GROOVE_W / 2), 2 * GROOVE_R, GROOVE_W, sections=180),
            box((cx - 4.4, -80, -1), (cx + 80, 80, ARC_H + 1))]                       # open side: slide in here
    for a in (135, 225):                                                               # mic spots stay open
        t = np.radians(a)
        cuts.append(cylinder((cx + 49 * np.cos(t), 49 * np.sin(t), ARC_H / 2), 10, ARC_H + 2))
    bumps = [cylinder((cx + (GROOVE_R + 0.8) * np.cos(np.radians(a)), (GROOVE_R + 0.8) * np.sin(np.radians(a)),
                       GROOVE_Z + GROOVE_W / 2), 2.4, GROOVE_W + 0.02, sections=16) for a in (97, 263)]
    return solids, cuts, bumps


def build():
    rs, rc = ring()
    ss, sc, ribs = sleeve()
    bs, bc, bumps = board_arc()
    part = trimesh.boolean.union(rs + ss + bs, engine='manifold')
    part = trimesh.boolean.difference([part] + rc + sc + bc, engine='manifold')
    # keep the gripper bore and camera pocket clear of the bridges
    part = trimesh.boolean.difference([part, cylinder((0, 0, RING_H / 2), RING_ID, 200, sections=128)], engine='manifold')
    return trimesh.boolean.union([part] + ribs + bumps, engine='manifold')


def preview(part, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    t = np.linspace(0, 2 * np.pi, 90)
    zb = GROOVE_Z + GROOVE_W / 2
    disc = np.stack([BOARD_X + BOARD_R * np.cos(t), BOARD_R * np.sin(t), np.full_like(t, zb)], 1)
    grip = trimesh.creation.cylinder(radius=GRIP_D / 2, height=40, sections=64)
    grip.apply_translation((0, 0, 10))
    cam = trimesh.creation.box(extents=(CAM_H, CAM_L, CAM_D),
                               transform=trimesh.transformations.translation_matrix((CAM_X, 0, CAM_Z0 + CAM_D / 2)))
    fig = plt.figure(figsize=(14, 6))
    for k, (elev, azim, title) in enumerate(((25, -60, 'three-quarter'), (90, -90, 'top (as printed)'),
                                             (-20, -90, 'from the cup side'))):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection(part.vertices[part.faces], facecolor='#4c9be8', edgecolor='none', alpha=0.9))
        ax.add_collection3d(Poly3DCollection(grip.vertices[grip.faces], facecolor='#999999', alpha=0.15))
        ax.add_collection3d(Poly3DCollection(cam.vertices[cam.faces], facecolor='#333333', alpha=0.35))
        ax.add_collection3d(Poly3DCollection([disc], facecolor='#2e7d32', alpha=0.35))
        ax.set_xlim(-100, 160), ax.set_ylim(-130, 130), ax.set_zlim(-60, 60)
        ax.set_box_aspect((260, 260, 120))
        ax.view_init(elev, azim)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle('SoundScope wrist mount (blue) - gripper (gray), camera (dark), board (green)')
    fig.savefig(path, dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    part = build()
    part.apply_translation((0, 0, -part.bounds[0][2]))
    part.export(HERE / 'wrist_mount.stl')
    preview(part, HERE / 'wrist_mount.png')
    area, where = overhangs(part)
    print(f'{part.bounds[1][2]:.1f} mm tall; overhangs past 45 deg: {area:.1f} mm2')
    for z in sorted({round(float(p[2]), 1) for p in where}):
        near = where[np.abs(where[:, 2] - z) < 0.05]
        print(f'   at height {z:5.1f} mm: {len(near)} faces around x {near[:, 0].mean():6.1f}, y {near[:, 1].mean():6.1f}')
    ext = part.bounds[1] - part.bounds[0]
    print(f'watertight={part.is_watertight} bodies={part.body_count} size {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm '
          f'volume {part.volume / 1000:.1f} cm3')
    print(f'board center from camera center: {BOARD_X - CAM_X:.1f} mm sideways, '
          f'{(GROOVE_Z + GROOVE_W / 2) - (CAM_Z0 + CAM_D / 2):.1f} mm along the view axis')
