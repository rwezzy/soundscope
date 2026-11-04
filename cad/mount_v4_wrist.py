"""SoundScope wrist mount v4: a one-piece saddle that hugs the Lite 6 vacuum gripper's real (oval) body.

  python cad/mount_v4_wrist.py     writes cad/v4/ (print file, part, previews)

Why v4: the older wrist rings (and v3's) assumed a round 77.5 mm gripper body. UFACTORY's own model of the
Vacuum Gripper Lite (xArm-Developer/xarm_ros2, xarm_description/meshes/lite6/visual/vacuum_gripper_lite.stl)
shows the body is an OVAL, 52.1 x 75.0 mm, with a slight draft, under a 3.5 mm tall 77.5 mm disc. Only that
disc and the arm's metal end ring (the ring with the tool I/O port and the manual-mode button) are round
77.5. A round ring therefore touched nothing and could not clamp. v4 hugs the oval and stays 12 mm clear of
the arm's ring, so the port, its plug and the button are free whatever way the plug points.

Sizes (manufacturer first):
  - Gripper body, UFACTORY mesh in the link_eef frame (z = 0 at the flange face, +z toward the cup):
    narrow sides r 26.5 at z 4-12, drafting to 26.0 by z 30; wide ends r 37.7 at z 12 to 37.5 from z 20;
    straight to z 39, then the dome; suction cup tip at z 61 (the URDF TCP). Its 2 x M6 screws (PCD 63,
    UFACTORY flange drawing) sit under the disc on the narrow sides. Gripper Lite has the same body.
  - Arm end ring (UFACTORY link6 mesh + flange drawing): 77.5 round, z -19.5..-1; the M8 8-pin tool I/O
    socket on it sits over one WIDE end of the gripper (+y), 9 mm behind the flange face, sticking out 7.4 mm.
    Keith: the plugged-in cable reaches 2-4 cm out; the manual-mode button is on the same ring.
    Keith's tape (24.5-24.7 cm around) and caliper (77.5) agree with the official 77.5 ring.
  - Camera (Orbbec drawing) and mic board (Seeed drawing): see mount_v3.py. The v3 camera collar shape and
    the v3 board frame, clips and pins are reused unchanged (same fit-checked numbers).

Frame G (in use): origin at the flange face centre; +Z toward the cup = the camera's view direction;
X across the gripper's narrow sides (camera at +X, mic board at -X); Y along its wide ends, +Y = the end
under the cable port (left open).
"""
from pathlib import Path

import numpy as np
import trimesh

import mount_v3 as m3
from rig import box, cylinder, overhangs

HERE = Path(__file__).resolve().parent
OUT = HERE / 'v4'

# --- gripper body (official mesh) and fit --------------------------------------------------------------
FIT = 0.25
TAPER = ((12.0, 26.50, 37.70), (24.0, 26.06, 37.50))   # (z, narrow-side r, wide-end r) of the body
Z0, Z1 = 12.0, 52.0                 # saddle back face (top as printed) and front face (on the bed)
RING_T = 4.5                        # saddle wall around the oval
SLOT_HALF, SLOT_Y0 = 17.6, 20.0     # cable-port end left open: the body is 34.7 wide where the slot starts
Z_FRONT = Z1 - m3.LIP               # camera and board front faces (50): 11 mm behind the cup tip
CAM_X0 = 29.9                       # camera bottom face (its tripod side faces the gripper)
Y_CAM = -6.0                        # camera slid 6 mm away from the port end: its corner stays out of the
                                    # zone a right-angle plug can swing through (y 47-62, 45 mm round the port)
X_ST = 35.9                         # outer face of the board-frame spine (mirror of the camera side + 6)
SPINE_Y = m3.SPINE_W / 2
SET_Z = 25.0                        # M6 set screw: middle of the straight part of the body
BOSS_D, PILOT_D = 13.0, m3.M6_PILOT
PUCK_D, PUCK_T, PUCK_POCKET = 11.0, 3.2, 3.5

A_IN = TAPER[1][1] + FIT            # bore across the narrow sides (straight part)
B_IN = TAPER[1][2] + FIT            # bore along the wide ends (straight part)
A_OUT, B_OUT = A_IN + RING_T, B_IN + RING_T


def ellipse_pts(a, b, z, n=144):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([a * np.cos(t), b * np.sin(t), np.full(n, z)])


def oval_prism(a, b, z0, z1, n=144):
    return trimesh.convex.convex_hull(np.vstack([ellipse_pts(a, b, z0, n), ellipse_pts(a, b, z1, n)]))


def bore():
    """The body plus FIT, following its draft near the disc so the saddle seats where it was designed."""
    (za, ra, rb), (zb, sa, sb) = TAPER
    # near the disc the body is fuller than an ellipse on its diagonals (+0.4 mm at z 12): scale the top
    # ellipse up 1.2 % so every angle keeps >= 0.18 mm there (checked against the official mesh)
    taper = trimesh.convex.convex_hull(np.vstack([ellipse_pts(1.012 * (ra + FIT + 0.03), 1.012 * (rb + FIT + 0.01), za - 1.0),
                                                  ellipse_pts(sa + FIT, sb + FIT, zb)]))
    return m3.union([taper, oval_prism(A_IN, B_IN, zb - 0.01, Z1 + 1.0)])


def frame_to_g():
    """v3 board-frame coordinates (X right, Y forward, Z up from the spine) -> G: the frame stands out of the
    saddle's -X face, the board faces the cup, its USB edge points away from the gripper."""
    return np.array([[0, 0, -1, m3.SPINE_TOP - X_ST],
                     [-1, 0, 0, 0],
                     [0, 1, 0, Z_FRONT],
                     [0, 0, 0, 1]], float)


def collar_to_g():
    """v3 camera coordinates (X right seen from behind, Y forward, Z up from the camera's lowest point) -> G:
    camera on the +X side, its bottom (tripod thread) against the saddle, long axis along the wide ends."""
    return np.array([[0, 0, 1, CAM_X0],
                     [1, 0, 0, Y_CAM],
                     [0, 1, 0, Z_FRONT],
                     [0, 0, 0, 1]], float)


def moved(mesh, T):
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return mesh


def camera_collar():
    """The v3 collar (same walls, lips, crush ribs and cross pin) without the spine on top, in v3 coords."""
    x0, x1 = m3.IN_W / 2, m3.IN_W / 2 + m3.WALL
    solids = [box((-x1, m3.COVER, m3.IN_H), (x1, m3.LIP, m3.TOP_OUT)),              # top wall (outward)
              box((-x1, m3.COVER, -m3.WALL), (x1, m3.LIP, 0.0))]                    # bottom wall (saddle side)
    for sx in (-1, 1):
        lo, hi = sorted((sx * x0, sx * x1))
        solids += [box((lo, m3.BACK, -m3.WALL), (hi, m3.LIP, 5.0)),
                   box((lo, m3.BACK, m3.IN_H - 5.0), (hi, m3.LIP, m3.TOP_OUT)),
                   box((lo, m3.BACK, -m3.WALL), (hi, m3.PIN_Y + 6.0, m3.TOP_OUT))]
    for z_face, inward in ((0.0, 1), (m3.IN_H, -1)):
        tri = [(z_face, 0.0), (z_face, m3.LIP), (z_face + inward * m3.LIP, m3.LIP)]
        solids.append(trimesh.convex.convex_hull(np.array([(x, y, z) for x in (-x0, x0) for z, y in tri], float)))
    body = m3.union(solids)
    pin = cylinder((0, m3.PIN_Y, m3.PIN_Z), m3.PIN_HOLE, 2 * x1 + 4, axis='y').apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1), (0, m3.PIN_Y, m3.PIN_Z)))
    body = m3.diff(body, [pin])
    ribs = [box((x - 0.8, m3.COVER + 1.5, m3.IN_H - m3.RIB_H), (x + 0.8, -2.0, m3.IN_H + 0.01)) for x in (-30.0, 0.0, 30.0)]
    return m3.union([body] + ribs)


def along(axis, d, p0, p1, sections=48):
    """Cylinder of diameter d from p0 to p1 along a main axis ('x' or 'y')."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c = cylinder(tuple((p0 + p1) / 2), d, float(np.linalg.norm(p1 - p0)), sections=sections)
    rot = (0, 1, 0) if axis == 'x' else (1, 0, 0)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, rot, (p0 + p1) / 2))
    return c


def saddle():
    """One piece: oval ring + camera collar (+X) + board-frame spine (-X) + set-screw boss (-X, below the spine)."""
    ring = trimesh.boolean.intersection([oval_prism(A_OUT, B_OUT, Z0, Z1),
                                         box((-80.0, -80.0, Z0 - 1), (CAM_X0, 80.0, Z1 + 1))], engine='manifold')
    cam_side = box((0.0, -34.0, Z0), (CAM_X0, 34.0, Z1))             # fills under the camera's bottom wall
    board_side = box((-CAM_X0, -22.0, Z0), (0.0, 22.0, Z1))
    spine_z0 = Z_FRONT + (m3.BOARD_Y0 - m3.BOARD_T - m3.FRAME_D) - 3.0
    spine = box((-X_ST, -SPINE_Y, spine_z0), (-CAM_X0 + 0.9, SPINE_Y, Z1))
    boss = along('x', BOSS_D, (-X_ST - 0.6, 0, SET_Z), (-CAM_X0 + 1.0, 0, SET_Z))
    collar = moved(camera_collar(), collar_to_g())
    body = m3.union([ring, cam_side, board_side, spine, boss, collar])

    cuts = [bore(), box((-SLOT_HALF, SLOT_Y0, Z0 - 1), (SLOT_HALF, 80.0, Z1 + 1))]
    # dovetail groove for the v3 board frame: open toward the cup, stopped where the frame's back face sits
    t, b = m3.DOVE_TOP / 2 + m3.DOVE_FIT, m3.DOVE_BOT / 2 + m3.DOVE_FIT
    x_open, x_deep = -X_ST - 0.1, -X_ST + m3.DOVE_H + m3.DOVE_FIT
    z_stop = Z_FRONT + (m3.BOARD_Y0 - m3.BOARD_T - m3.FRAME_D)
    pts = [(x, y, z) for z in (z_stop, Z1 + 1) for x, y in ((x_open, -t), (x_open, t), (x_deep, -b), (x_deep, b))]
    cuts.append(trimesh.convex.convex_hull(np.array(pts, float)))
    lock_x, lock_z = -X_ST + m3.DOVE_H / 2, Z_FRONT - 6.5
    cuts.append(along('y', 4.15, (lock_x, -SPINE_Y - 1, lock_z), (lock_x, SPINE_Y + 1, lock_z)))
    # set screw: pilot through the boss and wall, pocket for the printed puck on the bore side
    cuts.append(along('x', PILOT_D, (-X_ST - 2, 0, SET_Z), (-A_IN + 0.5, 0, SET_Z)))
    cuts.append(along('x', PUCK_D + 0.6, (-A_IN - PUCK_POCKET, 0, SET_Z), (-A_IN + 1.0, 0, SET_Z)))
    return m3.diff(body, cuts)


def puck():
    """Soft-faced pad under the set screw (printed flat): spreads the screw's push on the gripper."""
    return cylinder((0, 0, PUCK_T / 2), PUCK_D, PUCK_T, sections=48)


def gripper_model():
    """Approximate official body for previews (the fit checks use UFACTORY's mesh itself)."""
    disc = cylinder((0, 0, 1.75), 77.5, 3.5, sections=120)
    body = trimesh.convex.convex_hull(np.vstack([ellipse_pts(26.5, 37.7, 3.5), ellipse_pts(26.0, 37.5, 39.0)]))
    dome = trimesh.convex.convex_hull(np.vstack([ellipse_pts(26.0, 37.5, 39.0), ellipse_pts(8.0, 8.0, 52.0, 48)]))
    cup = cylinder((0, 0, 56.5), 16.0, 9.0, sections=48)
    ring = cylinder((0, 0, -10.0), 77.5, 19.0, sections=120)
    port = along('y', 10.0, (0, 38.0, -9.0), (0, 46.0, -9.0))
    return m3.union([disc, body, dome, cup]), m3.union([ring, port])


def front_down(mesh):
    """Print orientation: the front face (z = Z1) on the bed."""
    mesh = mesh.copy()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0)))
    mesh.apply_translation(-mesh.bounds[0] * np.array([0, 0, 1]))
    return mesh


def preview(parts, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    Tf, Tc = frame_to_g(), collar_to_g()
    t = np.linspace(0, 2 * np.pi, 120)
    disc_f = np.stack([m3.BOARD_R * np.cos(t), np.full_like(t, m3.BOARD_Y0 - m3.BOARD_T / 2),
                       m3.BOARD_ZC + m3.BOARD_R * np.sin(t), np.ones_like(t)], 1)
    disc = (Tf @ disc_f.T).T[:, :3]
    mics = np.array([(Tf @ np.array([sx * 33, 0.0, m3.BOARD_ZC + sz * 33, 1]))[:3] for sx in (-1, 1) for sz in (-1, 1)])
    lens = (Tc @ np.array([m3.RGB_X, 0.0, m3.LENS_Z, 1]))[:3]
    cam = moved(m3.camera_box(), Tc)
    grip, arm = gripper_model()
    fig = plt.figure(figsize=(16, 6))
    views = ((22, -125, 'three-quarter (cup end)'), (90, -90, 'looking down the gripper axis from the arm'),
             (0, -90, 'side'))
    for k, (elev, azim, name) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        for mesh, color, alpha in [(grip, '#bbbbbb', 0.35), (arm, '#888888', 0.35)] + parts + [(cam, '#333333', 0.4)]:
            ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolor=color, edgecolor='none', alpha=alpha))
        ax.add_collection3d(Poly3DCollection([disc], facecolor='#2e7d32', alpha=0.35))
        ax.scatter(*mics.T, color='red', s=25, depthshade=False)
        ax.scatter(*lens, color='yellow', edgecolor='black', s=45, depthshade=False)
        ax.set_xlim(-150, 80), ax.set_ylim(-115, 115), ax.set_zlim(-40, 90)
        ax.set_box_aspect((230, 230, 130))
        ax.view_init(elev, azim)
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle('Wrist mount v4: saddle (blue) on the gripper (light gray, oval body), arm end ring + port (gray), '
                 'v3 board frame (orange), clips (purple), camera (dark), board (green)')
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    part = saddle()
    Tf = frame_to_g()
    frame, clips = moved(m3.board_frame(), Tf), [moved(m3.clip(s), Tf) for s in ('left', 'right')]
    printed = front_down(part)
    m3.report('wrist_saddle', printed)
    printed.export(OUT / 'part_wrist_saddle.stl')
    pk = puck()
    pk.export(OUT / 'part_set_screw_puck.stl')
    # one plate: the saddle and two pucks (one spare); the frame, clips and pins come from the v3 plate
    plate = m3.plate([printed, pk, pk.copy()])
    plate.export(OUT / 'PRINT_wrist_v4.stl')
    # a complete wrist set for when the stand keeps its own frame
    v3_small = [m3.back_down(m3.board_frame()), m3.front_down(m3.clip('left')), m3.front_down(m3.clip('right'))]
    v3_small += [m3.pin(m3.CLIP_PIN_D, m3.CLIP_D + m3.FRAME_D + 1.0, 9.0, 2.0) for _ in range(5)]
    v3_small += [m3.pin(m3.PIN_D, m3.IN_W + 2 * m3.WALL + 2.0, 10.0, 3.0), m3.pin(4.0, m3.SPINE_W + 1.0, 7.0, 2.0)]
    full = m3.plate([printed, pk, pk.copy()] + v3_small)
    full.export(OUT / 'PRINT_wrist_v4_full_set.stl')
    print(f'PRINT_wrist_v4.stl: {plate.extents[0]:.0f} x {plate.extents[1]:.0f} x {plate.extents[2]:.0f} mm, '
          f'{plate.volume / 1000:.1f} cm3; full set {full.volume / 1000:.1f} cm3')
    preview([(part, '#4c9be8', 0.9), (frame, '#e8a24c', 0.9), (clips[0], '#9b59b6', 0.9), (clips[1], '#9b59b6', 0.9)],
            OUT / 'preview_wrist_v4.png')
    Tc = collar_to_g()
    lens = (Tc @ np.array([m3.RGB_X, 0.0, m3.LENS_Z, 1]))[:3]
    board = (Tf @ np.array([0.0, m3.BOARD_Y0 - m3.BOARD_T / 2, m3.BOARD_ZC, 1]))[:3]
    d = board - lens
    # camera axes in G: right = +Y, forward = +Z, up = +X
    print(f'color lens in the flange frame (x, y, z) mm: {np.round(lens, 1).tolist()}')
    print(f'board centre from the color lens [forward, left, up] mm: [{d[2]:.1f}, {-d[1]:.1f}, {d[0]:.1f}]  (cable edge down)')
    ext = np.vstack([part.bounds, frame.bounds])
    print(f'assembly span across the narrow sides: {part.bounds[1][0] - (board[0] - m3.BOARD_R):.0f} mm '
          f'(saddle + camera {part.bounds[1][0]:.0f} mm out on one side, board rim {-(board[0] - m3.BOARD_R):.0f} mm on the other)')
