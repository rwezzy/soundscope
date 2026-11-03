"""SoundScope rig v2: no screws, no tripod. One printed part that stands on the table.

  pip install trimesh manifold3d matplotlib
  python cad/rig_v2.py         writes soundscope_rig_v2_print.stl (print this one, as-is),
                               soundscope_rig_v2.stl (as used) and soundscope_rig_v2.png (preview)

The Gemini 2 slides into a sleeve from the back (lenses forward) and stops on small front lips.
The floor is open in the middle, so the camera screws straight onto its own tripod as usual; the left wall
is open at the back for the Type-C plug.
The reSpeaker board stands on the sleeve's roof, LED face forward, cable edge UP: its bottom rim
drops into two curved seats and its two holes (opposite the USB port) go over two printed pegs.
Nothing covers the mics. The board center sits straight above the camera's color lens.

Print soundscope_rig_v2_print.stl as it comes (lying on its back, sleeve opening down): no supports.
PrusaSlicer + NJIT Makerspace printer preset, PLA, 0.2 mm layers, 20% infill, supports off.

Sources: Orbbec Gemini 2 datasheet v1.0 (90 wide, 30 deep, 25 tall; Type-C on the left end; color lens 11 mm from the right
IR camera, seen from the front), Seeed reSpeaker XVF3800 drawing (100 mm disc, holes 42.71 mm apart,
24.5 mm from center).

Frame (in use): X = camera's right seen from behind, Y = forward, Z = up; origin = middle of the
camera's bottom front edge.
"""
from pathlib import Path

import numpy as np
import trimesh

from rig import box, cylinder, overhangs

HERE = Path(__file__).resolve().parent

# Camera (Gemini 2)
CAM_L, CAM_W, CAM_H = 90.0, 30.0, 25.0  # width, depth, height (datasheet side view: 30 deep, 25 tall)
RGB_X, LENS_Z = -14.0, 12.5
TRIPOD_OPEN = 40.0             # floor gap for the tripod head under the camera's 1/4"-20 thread
FIT = 0.3                      # clearance per side in the sleeve

# Board (reSpeaker XVF3800)
BOARD_R, BOARD_T = 50.0, 1.2
HOLE_DX, HOLE_DY = 42.71 / 2, 24.5
MIC_XY = 33.0
PEG_D, SLOT = 1.9, 0.3         # peg through the board holes; slot clearance per side

WALL = 3.0
LIP = 2.0
SEAT_W, SEAT_ABOVE_RIM = 10.0, 4.0
POST_D, POST_GAP = 6.0, 3.0    # round standoff behind each hole, clear of parts on the board's back

IN_X = CAM_L / 2 + FIT
IN_TOP = CAM_H + FIT
BACK = -CAM_W - FIT
ROOF = IN_TOP + WALL
BOARD_FRONT = -0.4
BOARD_BACK = BOARD_FRONT - BOARD_T
BOARD_CENTER = np.array([RGB_X, (BOARD_FRONT + BOARD_BACK) / 2, ROOF + BOARD_R + 1.0])


def lip(x0, x1, y0, inward_axis, at, sign):
    """45 deg wedge along X (floor) or Z (sides): zero width at y0, LIP wide at y0 + LIP."""
    tri = [(y0, 0.0), (y0 + LIP, 0.0), (y0 + LIP, LIP)]
    if inward_axis == 'z':      # floor lip, grows up from z = at
        v = [(x, y, at + w) for x in (x0, x1) for y, w in tri]
    else:                       # side lip, grows inward in X from x = at, spans Z x0..x1
        v = [(at + sign * w, y, z) for z in (x0, x1) for y, w in tri]
    return trimesh.convex.convex_hull(np.array(v))


def build():
    front = LIP + 0.5
    solids = [box((-IN_X - WALL, BACK, -WALL), (IN_X + WALL, front, ROOF))]
    cuts = [box((-IN_X, BACK - 1, 0), (IN_X, front + 1, IN_TOP))]
    lips = [lip(-IN_X, IN_X, 0.5, 'z', 0.0, 1)]
    for sx in (-1, 1):
        lips.append(lip(0.0, IN_TOP, 0.5, 'x', sx * IN_X, -sx))
    # Type-C is on the camera's left end (seen from behind), plug out sideways: open that wall to the back
    cuts.append(box((-IN_X - WALL - 1, BACK - 1, 2), (-IN_X + 0.5, -6, IN_TOP - 2)))
    cuts.append(box((IN_X - 0.5, -24, 4), (IN_X + WALL + 1, -6, IN_TOP - 4)))  # sync port / air on the right end
    # open floor under the middle: the tripod head screws straight onto the camera, full thread,
    # and the camera carries the rig. The two floor pads at the ends still stand on a table.
    cuts.append(box((-TRIPOD_OPEN / 2, BACK - 1, -WALL - 1), (TRIPOD_OPEN / 2, front + 1, 0.01)))
    for sx in (-1, 1):
        x = RGB_X + sx * HOLE_DX
        rim_z = BOARD_CENTER[2] - np.sqrt(BOARD_R ** 2 - HOLE_DX ** 2)
        hole_z = BOARD_CENTER[2] - HOLE_DY
        solids += [
            box((x - SEAT_W / 2, BACK, ROOF - 0.01), (x + SEAT_W / 2, BOARD_FRONT + SLOT + LIP, rim_z + SEAT_ABOVE_RIM)),
            box((x - POST_D / 2, BACK, ROOF - 0.01), (x + POST_D / 2, BOARD_BACK - POST_GAP, hole_z + POST_D / 2)),
            cylinder((x, BOARD_BACK - POST_GAP / 2, hole_z), POST_D, POST_GAP + 0.02, axis='y'),
            cylinder((x, BOARD_BACK + 1.2, hole_z), PEG_D, 2.4, axis='y', sections=24),
        ]
    board_slot = cylinder((BOARD_CENTER[0], BOARD_CENTER[1], BOARD_CENTER[2]), 2 * (BOARD_R + SLOT),
                          BOARD_T + 2 * SLOT, axis='y', sections=180)
    part = trimesh.boolean.union(solids + lips, engine='manifold')
    part = trimesh.boolean.difference([part] + cuts, engine='manifold')
    # the slot only bites the seats; the pegs stay
    pegs = trimesh.boolean.union([cylinder((RGB_X + sx * HOLE_DX, BOARD_BACK + 1.2, BOARD_CENTER[2] - HOLE_DY),
                                           PEG_D, 2.4, axis='y', sections=24) for sx in (-1, 1)], engine='manifold')
    part = trimesh.boolean.difference([part, board_slot], engine='manifold')
    return trimesh.boolean.union([part, pegs], engine='manifold')


def for_printing(part):
    """On its back (in-use -Y down), so the sleeve, seats and pegs all grow straight up."""
    printed = part.copy()
    printed.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    printed.apply_translation((0, 0, -printed.bounds[0][2]))
    return printed


def preview(part, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    t = np.linspace(0, 2 * np.pi, 90)
    c = BOARD_CENTER
    disc = np.stack([c[0] + BOARD_R * np.cos(t), np.full_like(t, c[1]), c[2] + BOARD_R * np.sin(t)], 1)
    mics = np.array([[c[0] + sx * MIC_XY, c[1], c[2] + sz * MIC_XY] for sx in (-1, 1) for sz in (-1, 1)])
    cam = trimesh.creation.box(extents=(CAM_L, CAM_W, CAM_H),
                               transform=trimesh.transformations.translation_matrix((0, -CAM_W / 2, CAM_H / 2)))
    fig = plt.figure(figsize=(14, 6))
    for k, (elev, azim, title) in enumerate(((18, -60, 'front three-quarter'), (5, -90, 'front'), (5, 0, 'side'))):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection(part.vertices[part.faces], facecolor='#4c9be8', edgecolor='none', alpha=0.9))
        ax.add_collection3d(Poly3DCollection(cam.vertices[cam.faces], facecolor='#555555', alpha=0.35))
        ax.add_collection3d(Poly3DCollection([disc], facecolor='#2e7d32', alpha=0.35))
        ax.scatter(*mics.T, color='red', s=40, depthshade=False)
        ax.scatter(RGB_X, 0, LENS_Z, color='yellow', edgecolor='black', s=60, depthshade=False)
        ax.set_xlim(-70, 70), ax.set_ylim(-80, 60), ax.set_zlim(-5, 140)
        ax.set_box_aspect((140, 140, 145))
        ax.view_init(elev, azim)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle('SoundScope rig v2 (blue) - camera (gray), board (green), mics (red), color lens (yellow)')
    fig.savefig(path, dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    part = build()
    part.export(HERE / 'soundscope_rig_v2.stl')
    printed = for_printing(part)
    printed.export(HERE / 'soundscope_rig_v2_print.stl')
    preview(part, HERE / 'soundscope_rig_v2.png')
    area, where = overhangs(printed)
    print(f'print file: {printed.bounds[1][2]:.1f} mm tall; overhangs past 45 deg: {area:.1f} mm2')
    for z in sorted({round(float(p[2]), 1) for p in where}):
        near = where[np.abs(where[:, 2] - z) < 0.05]
        print(f'   at height {z:5.1f} mm: {len(near)} faces around x {near[:, 0].mean():6.1f}, y {near[:, 1].mean():6.1f}')
    ext = part.bounds[1] - part.bounds[0]
    print(f'watertight={part.is_watertight}  size {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm  '
          f'volume {part.volume / 1000:.1f} cm3 (~{part.volume / 1000 * 1.24:.0f} g PLA solid)')
    d = BOARD_CENTER - np.array([RGB_X, 0.0, LENS_Z])
    print(f'arm_config [array]: cable_edge = "up", position_mm = [{d[1]:.1f}, {-d[0]:.1f}, {d[2]:.1f}]')
