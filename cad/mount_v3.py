"""SoundScope mounts v3: a standing (tripod) version and a wrist (Lite 6 gripper) version.

  python cad/mount_v3.py      writes the print plates, separate parts and previews into cad/v3/

Sizes come from the manufacturers first:
  - Orbbec Gemini 2 / 335 CAD V1.0 and V2.0 (2D drawings, 2024-07-19 / 2024-11-07): width 89.46 +-0.30,
    depth 30.00 +-0.30, height 25.00 +-0.30; the outer ~9.7 mm of each end is only 20.5 deep (the
    finned back block is 70 wide); 1/4-20 thread on the bottom, centred, 20.5 mm behind the front,
    5 mm max insertion; color lens 11 mm from the centre; V1.0 has two rubber feet under the front
    housing (inside the 25 mm), V2.0 does not.
  - Seeed reSpeaker XVF3800 2D drawing: disc 100.00, PCB 1.20; mics on a 66 x 66 square; USB-C and
    the audio jack at the bottom edge; small edge parts at 3 and 9 o'clock; parts up to ~4.8 mm high
    on the LED side, up to ~6.2 mm overall.
  - UFACTORY: both Lite 6 grippers bolt on with 2 x M6. The gripper body diameter (77.5) is Keith's
    caliper reading; nothing official was found for it.
Keith's caliper readings only fill gaps or confirm: camera width 89.24 (official 89.46 +-0.3) and a
body height of 23.96 without the feet (V1.0: 25 with feet).

Parts (cad/v3/):
  PRINT_stand.stl   one plate: standing collar + board frame + 2 clips + pins (6 clip, 2 cross, 2 lock)
  PRINT_wrist.stl   one plate: wrist collar (ring clamp) + board frame + 2 clips + the same pins
  PRINT_wrist_collar_only.stl   just the wrist collar: the frame, clips and pins move over from the stand
  part_*.stl        each part on its own, in print orientation
  preview_*.png     the assemblies

Frame (in use): X = right seen from behind the camera, Y = forward (the scene), Z = up.
Origin: the camera's front face (y = 0), its centre line (x = 0), its lowest point (z = 0, the feet).
"""
from pathlib import Path

import numpy as np
import trimesh

from rig import box, cylinder, overhangs

HERE = Path(__file__).resolve().parent
OUT = HERE / 'v3'

# --- camera (Orbbec drawing) and fits ----------------------------------------------------------
CAM_W, CAM_D, CAM_H = 89.46, 30.0, 25.0
END_D, FIN_W = 20.5, 70.0          # the ends are only 20.5 deep; the finned back block is 70 wide
RGB_X, LENS_Z = -11.0, 12.5         # color lens, seen from behind; lens height (mid body)
TRIPOD_Y = -20.5                    # 1/4-20 thread position (behind the front face)
IN_W = CAM_W + 0.6                  # 90.06: fits the +0.30 tolerance with 0.15 to spare per side
IN_H = 25.35                        # 25.00 +0.30 tolerance, + a little; crush ribs take up the rest
IN_D = CAM_D + 0.4                  # where the cross pin sits behind the back face
RIB_H = 0.45                        # top crush ribs: 24.90 between ribs -> light press on a 25.0 camera
WALL, LIP = 3.5, 2.0
BACK = -38.0                        # collar's back end (prints on this face)
COVER = -22.0                      # top/bottom walls cover the 20.5 mm main body; printed front-down they
                                    # still grow from the bed, and only the end rails reach back for the pin
PIN_D, PIN_HOLE = 6.0, 6.25         # cross pin behind the camera (snug: printed holes come out small)
PIN_Y = -(IN_D + 0.2 + PIN_D / 2)   # pin centre: just behind the camera
PIN_Z = 12.7

# --- board (Seeed drawing) ---------------------------------------------------------------------
BOARD_R, BOARD_T = 50.0, 1.20
BOARD_Y0 = 0.0                      # board front face flush with the camera front (y = 0)
PAD_ANGLE = 20.0                    # pads at +-20 deg from horizontal, both sides: clear of the mics
                                    # (45 deg), edge parts (0/180), USB (top) and jack (~123 deg)
FRAME_D = 8.0                       # frame thickness (front-back), all of it prints on the bed
POST_X0, POST_X1 = 54.5, 62.5       # posts: x between these (both sides)
SPINE_W, SPINE_H = 32.0, 6.0        # spine on the collar top that takes the dovetail
DOVE_TOP, DOVE_BOT, DOVE_H = 14.0, 20.0, 5.0
DOVE_FIT = 0.25
BRIDGE_H = 8.0
CLIP_D = 4.5                        # clip bar thickness (in front of the posts)
CLIP_PIN_D, CLIP_PIN_HOLE, CLIP_PIN_LOOSE = 5.0, 5.0, 5.3

TOP_OUT = IN_H + WALL                # collar top (outer)
SPINE_TOP = TOP_OUT + SPINE_H
BOARD_ZC = SPINE_TOP + BRIDGE_H + 2.0 + BOARD_R   # board centre height
PAD_DZ = BOARD_R * np.sin(np.radians(PAD_ANGLE))  # 17.1
PAD_X = BOARD_R * np.cos(np.radians(PAD_ANGLE))   # 47.0
RIM_IN = PAD_X - 1.5                 # pads and lips reach in to here: bare rim only (parts start inside r 47)
STOP_R = BOARD_R + 0.3              # edge stops in the board's plane, just outside the rim
CLIP_PIN_DZ = 26.5                   # clip pins above and below the board centre, just past the lips
POST_TOP = BOARD_ZC + CLIP_PIN_DZ + 6.0

# --- wrist ring (Keith's gripper measurement; 2 x M6 like the gripper itself) ------------------
GRIP_D = 77.5
RING_ID, RING_T, RING_Y = GRIP_D + 0.4, 5.5, (LIP - 20.0, LIP)   # at the front, so it prints from the bed
SPLIT, EAR_T, EAR_H = 3.0, 10.0, 15.0
M6_CLEAR, M6_PILOT, M6_HEAD, UNDER_HEAD = 6.6, 5.0, 11.0, 2.5


def prism(points_xz, y0, y1):
    """Convex prism: a polygon in the XZ plane, extruded along Y."""
    pts = [(x, y, z) for y in (y0, y1) for x, z in points_xz]
    return trimesh.convex.convex_hull(np.array(pts, float))


def union(parts):
    return trimesh.boolean.union(parts, engine='manifold')


def diff(a, cuts):
    return trimesh.boolean.difference([a] + list(cuts), engine='manifold')


# --- collar --------------------------------------------------------------------------------------

def collar(stand):
    """Camera collar. Slides on from the back; front lips stop it; a cross pin behind locks it."""
    x0, x1 = IN_W / 2, IN_W / 2 + WALL
    solids = [
        box((-x1, COVER, IN_H), (x1, LIP, TOP_OUT)),                      # top wall
        box((-SPINE_W / 2, COVER, TOP_OUT - 0.01), (SPINE_W / 2, LIP, SPINE_TOP)),  # spine for the frame
        box((-x1, COVER, -WALL), (x1, LIP, 0.0)),                         # bottom wall
    ]
    for sx in (-1, 1):
        lo, hi = sorted((sx * x0, sx * x1))
        solids += [box((lo, BACK, -WALL), (hi, LIP, 5.0)),               # end rails (bottom)
                   box((lo, BACK, IN_H - 5.0), (hi, LIP, TOP_OUT)),      # end rails (top)
                   box((lo, BACK, -WALL), (hi, PIN_Y + 6.0, TOP_OUT))]   # cheeks that take the cross pin
    # front lips, 45 deg, along the top and bottom margins of the front face (clear of the lenses)
    for z_face, inward in ((0.0, 1), (IN_H, -1)):
        tri = [(z_face, 0.0), (z_face, LIP), (z_face + inward * LIP, LIP)]
        pts = [(x, y, z) for x in (-x0, x0) for z, y in tri]
        solids.append(trimesh.convex.convex_hull(np.array(pts, float)))
    body = union(solids)
    cuts = [cylinder((0, PIN_Y, PIN_Z), PIN_HOLE, 2 * x1 + 4, axis='y').apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1), (0, PIN_Y, PIN_Z)))]
    # dovetail groove in the spine, open to the back, stopped where the frame's front face belongs
    t, b = DOVE_TOP / 2 + DOVE_FIT, DOVE_BOT / 2 + DOVE_FIT
    cuts.append(prism([(-t, SPINE_TOP + 0.1), (t, SPINE_TOP + 0.1), (b, SPINE_TOP - DOVE_H - DOVE_FIT),
                       (-b, SPINE_TOP - DOVE_H - DOVE_FIT)], COVER - 1, BOARD_Y0 - BOARD_T + 0.2))
    # lock pin across the spine and the tongue (along X)
    lock = cylinder((0, -6.5, SPINE_TOP - DOVE_H / 2), 4.15, SPINE_W + 2, axis='y')
    lock.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1), (0, -6.5, SPINE_TOP - DOVE_H / 2)))
    cuts.append(lock)
    if stand:
        # the tripod head stays screwed to the camera: open the floor in the middle, from the back
        cuts.append(box((-21.0, COVER - 1, -WALL - 1), (21.0, -4.0, 0.5)))
    body = diff(body, cuts)
    # crush ribs under the top wall (they make the fit snug on either camera version)
    ribs = [box((x - 0.8, COVER + 1.5, IN_H - RIB_H), (x + 0.8, -2.0, IN_H + 0.01)) for x in (-30.0, 0.0, 30.0)]
    return union([body] + ribs)


def ring_clamp():
    """Split ring around the gripper body, under the collar, with two M6 clamp screws."""
    r_out = RING_ID / 2 + RING_T
    cz = -WALL - r_out + 2.0
    y0, y1 = RING_Y
    solids = [cylinder((0, (y0 + y1) / 2, cz), 2 * r_out, y1 - y0, axis='y', sections=160),
              box((-30.0, y0, cz + 0.55 * r_out), (30.0, y1, -WALL + 0.01))]  # saddle into the collar floor
    for sx in (-1, 1):                                                  # gussets to the collar's ends
        solids.append(prism([(sx * 28.0, -WALL + 0.01), (sx * (IN_W / 2 + WALL), -WALL + 0.01),
                             (sx * 28.0, cz + 0.45 * r_out)], y0, y1))
    ez = cz - r_out
    solids += [box((SPLIT / 2, y0, ez - EAR_H + 4), (SPLIT / 2 + EAR_T, y1, ez + 6)),
               box((-SPLIT / 2 - EAR_T, y0, ez - EAR_H + 4), (-SPLIT / 2, y1, ez + 6))]
    body = union(solids)
    cuts = [cylinder((0, (y0 + y1) / 2, cz), RING_ID, y1 - y0 + 2, axis='y', sections=160),
            box((-SPLIT / 2, y0 - 1, ez - EAR_H), (SPLIT / 2, y1 + 1, cz))]
    sz = ez - EAR_H / 2 + 4
    for y in (y0 + 6.5, y1 - 6.5):
        def along_x(d, x_from, x_to):
            c = cylinder(((x_from + x_to) / 2, y, sz), d, abs(x_to - x_from), axis='y')
            c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1), ((x_from + x_to) / 2, y, sz)))
            return c
        cuts += [along_x(M6_CLEAR, SPLIT / 2 - 1, SPLIT / 2 + EAR_T + 1),
                 along_x(M6_HEAD, SPLIT / 2 + UNDER_HEAD, SPLIT / 2 + EAR_T + 1),
                 along_x(M6_PILOT, -SPLIT / 2 + 0.5, -SPLIT / 2 - EAR_T + 1.0)]
    return diff(body, cuts), cz


# --- board frame, clips and pins ---------------------------------------------------------------------

def board_frame():
    """Two posts and a bridge, one flat piece; the bridge's dovetail slides into the collar spine."""
    yb, yf = BOARD_Y0 - BOARD_T - FRAME_D, BOARD_Y0 - BOARD_T            # behind the board's back face
    z0 = SPINE_TOP
    stops = []
    solids = [box((-POST_X1, yb, z0), (POST_X1, yf, z0 + BRIDGE_H))]       # bridge under the board
    for sx in (-1, 1):
        lo, hi = sorted((sx * POST_X0, sx * POST_X1))
        solids.append(box((lo, yb, z0), (hi, yf, POST_TOP)))
        for sz in (-1, 1):                                                 # pads behind the board rim
            zc = BOARD_ZC + sz * PAD_DZ
            plo, phi = sorted((sx * RIM_IN, sx * POST_X0 + sx * 0.01))
            solids.append(box((plo, yb, zc - 5.0), (phi, yf, zc + 5.0)))
            # edge stop: a crescent in the board's own plane, just outside the rim (locates the disc)
            elo, ehi = sorted((sx * (STOP_R - 3.0), sx * (POST_X0 - 0.2)))
            stop = box((elo, yf - 0.01, zc - 5.0), (ehi, BOARD_Y0 - 0.05, zc + 5.0))
            stops.append(diff(stop, [cylinder((0, BOARD_Y0 - BOARD_T / 2, BOARD_ZC), 2 * STOP_R, 4.0,
                                              axis='y', sections=200)]))
        # corner gussets between bridge and posts
        solids.append(prism([(sx * POST_X0, z0 + BRIDGE_H), (sx * (POST_X0 - 14.0), z0 + BRIDGE_H),
                             (sx * POST_X0, z0 + BRIDGE_H + 14.0)], yb, yf))
    t, b = DOVE_TOP / 2, DOVE_BOT / 2
    solids.append(prism([(-t, z0 + 0.01), (t, z0 + 0.01), (b, z0 - DOVE_H), (-b, z0 - DOVE_H)], yb, yf))
    body = union(solids)
    cuts = []
    lock = cylinder((0, -6.5, z0 - DOVE_H / 2), 4.15, DOVE_BOT + 2, axis='y')
    lock.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1), (0, -6.5, z0 - DOVE_H / 2)))
    cuts.append(lock)
    for sx in (-1, 1):
        for sz in (-1, 1):
            cuts.append(cylinder((sx * (POST_X0 + POST_X1) / 2, (yb + yf) / 2, BOARD_ZC + sz * CLIP_PIN_DZ),
                                 CLIP_PIN_HOLE, FRAME_D + 2, axis='y', sections=40))
    # keep the board's own outline (and its edge parts at 3/9 o'clock) clear of the posts
    disc = cylinder((0, (yb + yf) / 2, BOARD_ZC), 2 * (BOARD_R + 2.5), FRAME_D + 4, axis='y', sections=180)
    for sx in (-1, 1):  # windows in the bridge between the dovetail and the gussets
        lo, hi = sorted((sx * (DOVE_BOT / 2 + 4.0), sx * (POST_X0 - 16.0)))
        cuts.append(box((lo, yb - 1, z0 + 2.5), (hi, yf + 1, z0 + BRIDGE_H - 2.5)))
    body = diff(body, cuts)
    # carve the disc clearance only outside the pad zones
    keep = union([box((sorted((sx * (RIM_IN - 0.5), sx * POST_X0))[0], yb - 2, BOARD_ZC + sz * PAD_DZ - 5.0),
                      (sorted((sx * (RIM_IN - 0.5), sx * POST_X0))[1], yf + 2, BOARD_ZC + sz * PAD_DZ + 5.0))
                  for sx in (-1, 1) for sz in (-1, 1)])
    clearance = diff(disc, [keep])
    return union([diff(body, [clearance])] + stops)


def clip(side):
    """Bar in front of a post with two lips over the board's front rim; two printed pins hold it."""
    sx = 1 if side == 'right' else -1
    lo, hi = sorted((sx * POST_X0, sx * POST_X1))
    y0, y1 = BOARD_Y0 - BOARD_T, BOARD_Y0 - BOARD_T + CLIP_D       # bar spans the board's thickness
    solids = [box((lo, y0, BOARD_ZC - CLIP_PIN_DZ - 5.5), (hi, y1, BOARD_ZC + CLIP_PIN_DZ + 5.5))]
    for sz in (-1, 1):
        zc = BOARD_ZC + sz * PAD_DZ
        plo, phi = sorted((sx * RIM_IN, sx * POST_X0 + sx * 0.01))
        solids.append(box((plo, BOARD_Y0 + 0.05, zc - 4.0), (phi, y1, zc + 4.0)))   # lip over the rim
    body = union(solids)
    cuts = [cylinder((sx * (POST_X0 + POST_X1) / 2, (y0 + y1) / 2, BOARD_ZC + sz * CLIP_PIN_DZ),
                     CLIP_PIN_LOOSE, CLIP_D + 2, axis='y', sections=40) for sz in (-1, 1)]
    return diff(body, cuts)


def pin(d, length, head_d, head_t):
    """A pin lying along X (strongest way to print it), head at x = 0."""
    shaft = cylinder((0, 0, 0), d, length, sections=48)
    head = cylinder((0, 0, -length / 2 - head_t / 2 + 0.01), head_d, head_t, sections=48)
    p = union([shaft, head])
    p.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    # flat on the bed: shave 0.6 mm off one side so it does not roll and prints without supports
    p = diff(p, [box((-length - head_t - 5, -head_d, -head_d), (length + 5, head_d, -d / 2 + 0.6))])
    p.apply_translation(-p.bounds[0])
    return p


# --- print orientation and plates ----------------------------------------------------------------------

def back_down(m):
    """Rotate so the in-use back (-Y) sits on the bed."""
    m = m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    m.apply_translation(-m.bounds[0] * np.array([0, 0, 1]))
    return m


def front_down(m):
    m = m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    m.apply_translation(-m.bounds[0] * np.array([0, 0, 1]))
    return m


def place(m, x, y):
    m = m.copy()
    m.apply_translation(np.array([x, y, 0]) - m.bounds[0] * np.array([1, 1, 0]))
    return m


def plate(parts):
    """Lay parts out left to right in rows on a 240 x 210 mm area."""
    out, x, y, row = [], 0.0, 0.0, 0.0
    for m in parts:
        w, d = m.extents[:2]
        if x + w > 240:
            x, y, row = 0.0, y + row + 6, 0.0
        out.append(place(m, x, y))
        x, row = x + w + 6, max(row, d)
    return trimesh.util.concatenate(out)


def camera_box():
    body = box((-CAM_W / 2, -END_D, 0), (CAM_W / 2, 0, CAM_H))
    fins = box((-FIN_W / 2, -CAM_D, 0.5), (FIN_W / 2, -END_D, CAM_H - 0.5))
    return union([body, fins])


def preview(parts, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    t = np.linspace(0, 2 * np.pi, 120)
    disc = np.stack([BOARD_R * np.cos(t), np.full_like(t, BOARD_Y0 - BOARD_T / 2), BOARD_ZC + BOARD_R * np.sin(t)], 1)
    mics = np.array([[sx * 33, BOARD_Y0, BOARD_ZC + sz * 33] for sx in (-1, 1) for sz in (-1, 1)])
    cam = camera_box()
    fig = plt.figure(figsize=(15, 6))
    lim = max(abs(np.vstack([p.bounds for p, _ in parts])).max(), 120)
    for k, (elev, azim, name) in enumerate(((20, -60, 'three-quarter'), (2, -90, 'front'), (2, 0, 'side'))):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        for m, color in parts:
            ax.add_collection3d(Poly3DCollection(m.vertices[m.faces], facecolor=color, edgecolor='none', alpha=0.9))
        ax.add_collection3d(Poly3DCollection(cam.vertices[cam.faces], facecolor='#333333', alpha=0.35))
        ax.add_collection3d(Poly3DCollection([disc], facecolor='#2e7d32', alpha=0.35))
        ax.scatter(*mics.T, color='red', s=30, depthshade=False)
        ax.scatter(RGB_X, 0, LENS_Z, color='yellow', edgecolor='black', s=50, depthshade=False)
        ax.set_xlim(-lim / 1.6, lim / 1.6), ax.set_ylim(-lim / 1.6, lim / 1.6), ax.set_zlim(-lim * 0.75, lim * 0.85)
        ax.set_box_aspect((1, 1, 1.25))
        ax.view_init(elev, azim)
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def report(name, m):
    area, where = overhangs(m)
    print(f'  {name:16s} {m.extents[0]:5.0f} x {m.extents[1]:5.0f} x {m.extents[2]:4.0f} mm, '
          f'{m.volume / 1000:5.1f} cm3, watertight={m.is_watertight}, bodies={m.body_count}, '
          f'overhangs past 45 deg: {area:6.1f} mm2')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    stand = collar(stand=True)
    wrist_collar = collar(stand=False)
    ring, ring_cz = ring_clamp()
    wrist = union([wrist_collar, ring])
    frame = board_frame()
    clips = [clip('left'), clip('right')]
    pins = [pin(CLIP_PIN_D, CLIP_D + FRAME_D + 1.0, 9.0, 2.0) for _ in range(6)]
    cross = [pin(PIN_D, IN_W + 2 * WALL + 2.0, 10.0, 3.0) for _ in range(2)]
    locks = [pin(4.0, SPINE_W + 1.0, 7.0, 2.0) for _ in range(2)]

    printed = {'stand_collar': front_down(stand), 'wrist_collar': front_down(wrist), 'board_frame': back_down(frame),
               'clip_left': front_down(clips[0]), 'clip_right': front_down(clips[1]),
               'clip_pin': pins[0], 'cross_pin': cross[0], 'lock_pin': locks[0]}
    print('parts (print orientation):')
    for name, m in printed.items():
        report(name, m)
        m.export(OUT / f'part_{name}.stl')
    small = [printed['clip_left'], printed['clip_right']] + pins + cross + locks
    plates = {'stand': [printed['stand_collar'], printed['board_frame']] + small,
              'wrist': [printed['wrist_collar'], printed['board_frame']] + small,
              'wrist_collar_only': [printed['wrist_collar']]}          # frame, clips and pins are shared
    for name, parts in plates.items():
        p = plate(parts)
        p.export(OUT / f'PRINT_{name}.stl')
        print(f'PRINT_{name}.stl: {p.extents[0]:.0f} x {p.extents[1]:.0f} x {p.extents[2]:.0f} mm, '
              f'{p.volume / 1000:.1f} cm3 total')
    preview([(stand, '#4c9be8'), (frame, '#e8a24c'), (clips[0], '#9b59b6'), (clips[1], '#9b59b6')],
            'Standing mount v3 (camera stays on its tripod): collar (blue), board frame (orange), clips (purple)',
            OUT / 'preview_stand.png')
    preview([(wrist, '#4c9be8'), (frame, '#e8a24c'), (clips[0], '#9b59b6'), (clips[1], '#9b59b6')],
            'Wrist mount v3: collar + ring clamp (blue), board frame (orange), clips (purple)', OUT / 'preview_wrist.png')
    print(f'board centre from the color lens [forward, left, up] mm: '
          f'[{BOARD_Y0 - BOARD_T / 2:.1f}, {-(0 - RGB_X):.1f}, {BOARD_ZC - LENS_Z:.1f}]  (cable edge up)')
    print(f'gripper axis below the camera bottom: {-ring_cz:.1f} mm')
