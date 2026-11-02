"""SoundScope rig: one 3D-printed part holding the Gemini 2 on top and the reSpeaker board under it.

  pip install trimesh manifold3d matplotlib
  python cad/rig.py            writes cad/soundscope_rig.stl and cad/soundscope_rig.png

In use the board hangs right under the camera's color lens, its LED face flush with the camera's
front, cable edge pointing DOWN. It is held only by the two small holes on the edge opposite the
USB port, so the mics (4 spots 3 mm from the rim, on a 66 mm square) and the whole lower half
stay open on both faces. A 1/4"-20 nut at the back takes a tripod.

Print upside down (camera plate on the bed), no supports. PLA or PETG, 4 walls, 30% infill.
Hardware: 2x M3x6 (camera), 2x M2x10 self-tapping (board), 1x 1/4"-20 hex nut (tripod).

Sources: Seeed reSpeaker XVF3800 2D drawing (100 mm disc; holes 42.71 mm apart, 24.5 mm from
center), Orbbec Gemini 2 datasheet v1.0 (90 x 25 x 30 mm; 2x M3 45 mm apart on the bottom; color
lens 11 mm from the right IR camera, seen from the front).

Frame (in use): X = camera's right seen from behind, Y = forward, Z = up; origin = the middle
of the camera's bottom face.
"""
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent

# Camera (Gemini 2)
CAM = (90.0, 30.0, 25.0)        # length (X), depth (Y), height (Z)
RGB_X = -14.0                   # color lens: 25 - 11 mm left of center, seen from behind
LENS_Z = 12.5
M3_SPACING, M3_Y = 45.0, -15.0  # bottom threads; holes are slotted +-3 mm in Y for tolerance

# Board (reSpeaker XVF3800)
BOARD_R, BOARD_T = 50.0, 1.2
HOLE_DX, HOLE_DY = 42.71 / 2, 24.5  # the two holes opposite the USB port (up, since the cable points down)
MIC_XY = 33.0
GAP_ABOVE_BOARD = 6.0

# Part
PLATE_T = 5.0
PLATE_BACK = -72.0
STANDOFF, STANDOFF_LEN = 7.0, 12.0
FINGER_W, FINGER_T = 9.0, 6.0
TRIPOD_Y, BOSS_D, BOSS_H = -54.0, 26.0, 8.0
NUT_AF, NUT_DEPTH = 11.3, 5.8   # 1/4"-20 hex nut (7/16" across flats) + clearance
PILOT_D = 1.8                   # M2 self-tapping; open to 2.2 for M2.5

BOARD_CENTER = np.array([RGB_X, -1.0 - BOARD_T / 2, -PLATE_T - GAP_ABOVE_BOARD - BOARD_R])


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def cylinder(center, d, h, axis='z', sections=64):
    m = trimesh.creation.cylinder(radius=d / 2, height=h, sections=sections)
    if axis == 'y':
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    m.apply_translation(center)
    return m


def rib(x0, x1, corner, back, down):
    """Triangular stiffening rib in the YZ plane, from x0 to x1: right angle at `corner` (y, z),
    one leg running `back` mm toward -Y along the plate, the other `down` mm toward -Z along a finger."""
    y, z = corner
    tri = [(y, z), (y - back, z), (y, z - down)]
    v = np.array([(x, a, b) for x in (x0, x1) for a, b in tri])  # A B C, then A' B' C'
    f = [(0, 2, 1), (3, 4, 5), (0, 1, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2), (1, 2, 5), (1, 5, 4)]
    return trimesh.Trimesh(v, f)


def build():
    back_face = BOARD_CENTER[1] - BOARD_T / 2
    holes_z = BOARD_CENTER[2] + HOLE_DY
    solids = [box((-CAM[0] / 2 - 1, PLATE_BACK, -PLATE_T), (CAM[0] / 2 + 1, 0, 0)),
              cylinder((0, TRIPOD_Y, -PLATE_T - BOSS_H / 2), BOSS_D, BOSS_H)]
    cuts = [cylinder((0, TRIPOD_Y, -PLATE_T - BOSS_H / 2), 6.9, PLATE_T + BOSS_H + 2),
            cylinder((0, TRIPOD_Y, -NUT_DEPTH / 2 + 0.01), NUT_AF / np.cos(np.pi / 6), NUT_DEPTH, sections=6)]
    for sx in (-1, 1):  # camera screws: slotted through-holes with head counterbores from below
        x = sx * M3_SPACING / 2
        cuts += [box((x - 1.7, M3_Y - 3, -PLATE_T - 1), (x + 1.7, M3_Y + 3, 1)),
                 box((x - 3.25, M3_Y - 4.5, -PLATE_T - 1), (x + 3.25, M3_Y + 4.5, -PLATE_T + 2.5))]
    finger_front = back_face - STANDOFF_LEN
    for sx in (-1, 1):  # fingers down to the board's two holes
        x = RGB_X + sx * HOLE_DX
        solids += [
            box((x - FINGER_W / 2, finger_front - FINGER_T, holes_z - 6), (x + FINGER_W / 2, finger_front, -PLATE_T)),
            box((x - STANDOFF / 2, finger_front, holes_z - STANDOFF / 2), (x + STANDOFF / 2, back_face, holes_z + STANDOFF / 2)),
        ]
        solids.append(rib(x - 2.0, x + 2.0, (finger_front - FINGER_T + 0.5, -PLATE_T + 0.5), 14.0, 16.0))
        cuts.append(cylinder((x, (finger_front - FINGER_T + back_face) / 2, holes_z), PILOT_D,
                             back_face - finger_front + FINGER_T + 2, axis='y', sections=24))
    part = trimesh.boolean.union(solids, engine='manifold')
    return trimesh.boolean.difference([part] + cuts, engine='manifold')


def preview(part, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    t = np.linspace(0, 2 * np.pi, 90)
    disc = np.stack([BOARD_CENTER[0] + BOARD_R * np.cos(t), np.full_like(t, BOARD_CENTER[1]),
                     BOARD_CENTER[2] + BOARD_R * np.sin(t)], 1)
    mics = np.array([[BOARD_CENTER[0] + sx * MIC_XY, BOARD_CENTER[1], BOARD_CENTER[2] + sz * MIC_XY]
                     for sx in (-1, 1) for sz in (-1, 1)])
    cam = trimesh.creation.box(extents=CAM, transform=trimesh.transformations.translation_matrix((0, -CAM[1] / 2, CAM[2] / 2)))
    fig = plt.figure(figsize=(14, 6))
    for k, (elev, azim, title) in enumerate(((18, -60, 'front three-quarter'), (5, -90, 'front'), (5, 0, 'side'))):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection(part.vertices[part.faces], facecolor='#4c9be8', edgecolor='none', alpha=0.95))
        ax.add_collection3d(Poly3DCollection(cam.vertices[cam.faces], facecolor='#555555', alpha=0.25))
        ax.add_collection3d(Poly3DCollection([disc], facecolor='#2e7d32', alpha=0.35))
        ax.scatter(*mics.T, color='red', s=40, depthshade=False)
        ax.scatter(RGB_X, 0, LENS_Z, color='yellow', edgecolor='black', s=60, depthshade=False)
        ax.set_xlim(-70, 70), ax.set_ylim(-80, 60), ax.set_zlim(-115, 30)
        ax.set_box_aspect((140, 140, 145))
        ax.view_init(elev, azim)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle('SoundScope rig (blue) - camera (gray), board (green), mics (red), color lens (yellow)')
    fig.savefig(path, dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    part = build()
    part.export(HERE / 'soundscope_rig.stl')
    preview(part, HERE / 'soundscope_rig.png')
    ext = part.bounds[1] - part.bounds[0]
    print(f'watertight={part.is_watertight}  size {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm  '
          f'volume {part.volume / 1000:.1f} cm3 (~{part.volume / 1000 * 1.24:.0f} g PLA solid)')
    lens_to_board = BOARD_CENTER - np.array([RGB_X, 0.0, LENS_Z])
    print(f'board center from the color lens: forward {lens_to_board[1]:.1f}, '
          f'left {-lens_to_board[0]:.1f}, up {lens_to_board[2]:.1f} mm')
