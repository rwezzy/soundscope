"""SoundScope standing mount v5: fits any Gemini 2 within tolerance, and any XVF3800 board.

  python cad/mount_v5_stand.py      writes cad/v5_stand/PRINT_stand_v5.stl, parts and a preview, runs the checks

Keith's arrangement from the parallel version's compact_shared_mount (camera on its own tripod, mic
board standing above it, LED side forward, USB edge up), but the geometry is set by DATUM faces and
printed flexure springs push each part onto them, so the camera-to-board offset no longer depends on
the camera's or the board's exact size (the old mount was loose: 91.6 x 26.6 opening, no clamp).

Camera datums: its LEFT face (seen from behind) on the left wall, its TOP face under the top bridge,
its FRONT face on two end lips. Springs: two floor leaves push it up, one side leaf pushes it left,
a removable back clip (two leaves) pushes it forward. Board datums: two lower rim seats at 20 deg
below horizontal and front lugs; springs: two upper rim clips and two rear fingers.

Tolerances covered (Orbbec drawing V1.0 with feet / V2.0, Keith's calipers 89.24 x 29.89 x 23.96):
camera width 89.0-89.8, height 23.66-25.3, end depth 20.2-20.8; board 99-101 diameter, 1.0-1.6 thick.

Frame: X = right seen from behind, Y = forward (scene), Z = up. Origin: camera front face (y = 0),
its left face at x = -45 (datum), its top face at z = 25.3 (datum). Printed upright, bed at z = -3.
"""
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
OUT = HERE / 'v5_stand'

# --- camera range and datums --------------------------------------------------------------------
W_MIN, W_MAX = 89.0, 89.8
H_MIN, H_MAX = 23.66, 25.3
END_MIN, END_MAX = 20.2, 20.8            # the ends are only ~20.5 deep; the finned block behind is 70 wide
FIN_W, FIN_D = 70.6, 30.3
X_LEFT = -45.0                           # datum: camera left face
Z_TOP = 25.3                             # datum: camera top face (bridge underside)
BED = -3.0
RGB_FROM_CENTER, LENS_FROM_TOP = -11.0, 12.5

# --- board range and datums ---------------------------------------------------------------------
R_MIN, R_MAX, R_NOM = 49.5, 50.5, 50.0
T_MIN, T_MAX = 1.0, 1.6
ZC = 81.0                                # board centre for a 100 mm board (moves +-1.46 mm over 99-101)
SEAT_DEG = 20.0                          # lower seats this far below horizontal (clear of mics at 45 deg,
                                         # the edge parts at 3/9 o'clock and the USB/jack at the top)
PLA_STRAIN = 0.025                       # keep every flexure well under PLA's ~3-4 % yield strain


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def polar_box(deg, r0, r1, t0, t1, y0, y1, center=(0.0, ZC)):
    """Box in a board-centred local frame: r = radial, t = tangential (toward larger angle), y."""
    a = np.radians(deg)
    u, t, y = np.array([np.cos(a), 0, np.sin(a)]), np.array([-np.sin(a), 0, np.cos(a)]), np.array([0, 1.0, 0])
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = u, y, t
    mid = u * (r0 + r1) / 2 + t * (t0 + t1) / 2 + y * (y0 + y1) / 2
    m[:3, 3] = mid + np.array([center[0], 0, center[1]])
    return trimesh.creation.box(extents=(r1 - r0, y1 - y0, t1 - t0), transform=m)


def union(parts):
    return trimesh.boolean.union(parts, engine='manifold')


def mirror_x(mesh):
    m = mesh.copy()
    m.apply_transform(np.diag([-1.0, 1, 1, 1]))
    return m


SPRINGS = []  # (name, thickness, free length, travel needed) for the strain table


def spring(name, t, length, travel):
    SPRINGS.append((name, t, length, travel))


def frame():
    """Fixed body (datums, walls, posts) and spring bodies, right half + left half."""
    fixed = [
        box((-58, -36, BED), (X_LEFT, 2, Z_TOP + 3)),          # left wall: its inner face is the side datum
        box((48.6, -36, BED), (58, 2, Z_TOP + 3)),             # right wall (springs bridge the gap)
        box((-58, -20, Z_TOP), (58, 2, Z_TOP + 3)),            # top bridge: underside = top datum
        box((-58, 0, 4), (-40, 2, 14)),                         # front lips: front datum, ends only
        box((40, 0, 4), (58, 2, 14)),
    ]
    springs = []
    for side in (1, -1):
        s = []
        # floor leaf: rooted in the wall, lies on the bed, bump pushes the camera bottom up
        inner = 48.6 if side > 0 else 45.0
        s.append(box((22, -15, BED), (inner + 1, -5, BED + 1.4)))
        s.append(box((24, -14, BED + 1.4), (28, -6, Z_TOP - H_MIN + 0.4)))   # free top 2.04
        spring('floor leaf', 1.4, inner - 28, (Z_TOP - H_MIN + 0.4) - 0.0)
        # back-clip slots are cut later; posts carry the board
        fixed.append(box((53, -6, Z_TOP + 3), (58, 2, ZC + 26)))
        # lower rim seat (datum face at r = 50), its front lug and a rear finger
        a = -SEAT_DEG
        fixed.append(polar_box(a, R_NOM, 58, -4, 4, -6, 2))
        fixed.append(polar_box(a, 47.5, R_NOM + 1, -4, 4, 0, 2))           # front lug
        s.append(polar_box(a, 47.6, 51, -18, -3, -3.2, -2.2))               # rear finger leaf (root fused at r>50)
        s.append(polar_box(a, 47.6, 49.4, -18, -14, -2.2, -T_MIN + 0.2))    # bump, free at y = -0.8
        spring('board rear finger', 1.0, 14.0, (T_MAX - T_MIN) + 0.2)
        # upper front lug at 12 deg (fixed) and the upper rim clip at 20 deg (radial spring)
        fixed.append(polar_box(12, 47.5, 57, -3, 3, 0, 2))
        s.append(polar_box(SEAT_DEG, 51.2, 52.4, -20, 3, -3.5, -0.3))      # leaf, flexes radially
        s.append(polar_box(SEAT_DEG, 48.9, 51.2, -3, 3, -3.5, -0.3))       # bump, free at r = 48.9
        s.append(polar_box(SEAT_DEG, 51.2, 57, -24, -17, -3.5, 2))          # root into the post
        spring('board upper clip', 1.2, 15.0, 2 * R_MAX - R_NOM - 48.9)
        if side > 0:
            # side leaf: vertical, rooted at the bed, bump pushes the camera onto the left datum
            s.append(box((46.0, -20, BED), (47.4, -16, 22)))
            s.append(box((X_LEFT + W_MIN - 0.5, -19.5, 14), (46.0, -16.5, 20)))   # free face x = 43.5
            s.append(box((46.0, -20, BED), (inner + 1, -16, BED + 1.4)))           # foot into the wall
            spring('side leaf', 1.4, 22 - (BED + 1.4) - 3, (X_LEFT + W_MAX) - (X_LEFT + W_MIN - 0.5))
        springs += s if side > 0 else [mirror_x(m) for m in s]
        if side > 0:
            right_fixed = fixed[5:]
            fixed += [mirror_x(m) for m in right_fixed]
    # slots for the back clip (open at the top, behind the camera's finned block)
    body = union(fixed + springs)
    slots = [box((-56.5, -34.2, 7.8), (X_LEFT + 0.01, -30.8, Z_TOP + 3.1)),
             box((48.59, -34.2, 7.8), (56.5, -30.8, Z_TOP + 3.1))]
    body = trimesh.boolean.difference([body] + slots, engine='manifold')
    fixed_body = trimesh.boolean.difference([union(fixed)] + slots, engine='manifold')
    return body, fixed_body


def back_clip():
    """Removable: drops into the wall slots behind the fin block; two leaves push the ends forward."""
    parts = [box((-56.3, -34.0, 8.0), (56.3, -31.0, 16.0))]               # bar (rests on the slot floors)
    for side in (1, -1):
        p = [box((36.0, -31.0, 8.0), (37.2, -22.4, 16.0)),                 # stem beside the fin block
             box((36.0, -23.4, 8.0), (46.0, -22.4, 16.0)),                 # leaf along x, flexes in y
             box((42.0, -22.4, 9.0), (45.0, -END_MIN + 0.2, 15.0))]        # bump, free face y = -20.0
        parts += p if side > 0 else [mirror_x(m) for m in p]
    spring('back clip leaf', 1.0, 8.8, END_MAX - END_MIN + 0.2)
    return union(parts)


def camera(w, h, end, fin=True):
    x0, z0 = X_LEFT, Z_TOP - h
    parts = [box((x0, -end, z0), (x0 + w, 0, Z_TOP))]
    if fin:
        c = x0 + w / 2
        parts.append(box((c - FIN_W / 2, -FIN_D, z0), (c + FIN_W / 2, -end + 0.01, Z_TOP)))
    return union(parts)


def board(r, t):
    zc = ZC + (r - R_NOM) / np.sin(np.radians(SEAT_DEG))
    disc = trimesh.creation.cylinder(radius=r, height=t, sections=128)
    disc.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    disc.apply_translation((0, -t / 2, zc))
    return disc, zc


def overlap(a, b):
    i = trimesh.boolean.intersection([a, b], engine='manifold')
    return float(i.volume) if i is not None and len(i.faces) else 0.0


def checks(body, fixed, clip):
    print('Fit checks (mm3 of overlap; datum parts must be 0):')
    ok = True
    big_cam = camera(W_MAX, H_MAX, END_MAX)
    for name, part in (('fixed frame vs largest camera', fixed), ('back clip bar/stems vs largest camera',
                       trimesh.boolean.difference([clip, box((38, -23.5, 0), (46.5, -19, 20)),
                                                   box((-46.5, -23.5, 0), (-38, -19, 20))], engine='manifold'))):
        v = overlap(part, big_cam)
        ok &= v < 0.01
        print(f'  {name}: {v:.3f}')
    for r, t in ((R_MAX, T_MAX), (R_MIN, T_MAX), (R_MAX, T_MIN)):
        disc, zc = board(r, t)
        v = overlap(fixed, disc)
        ok &= v < 0.01
        print(f'  fixed frame vs board {2 * r:.0f} mm x {t} (centre z {zc:.2f}): {v:.3f}')
    small = camera(W_MIN, H_MIN, END_MIN)
    v = overlap(body, small)
    print(f'  springs preloaded on the smallest camera (want > 0): {v:.1f}')
    ok &= v > 0
    print('Flexures (strain = 1.5 t d / L^2 at full travel; limit %.1f %%):' % (PLA_STRAIN * 100))
    for name, t, length, travel in SPRINGS:
        strain = 1.5 * t * travel / length ** 2
        ok &= strain <= PLA_STRAIN
        print(f'  {name:18s} t {t:.1f}  L {length:5.1f}  travel {travel:4.2f}  strain {strain * 100:4.2f} %')
    print('ALL CHECKS PASS' if ok else 'CHECK FAILED')
    return ok


def main():
    OUT.mkdir(exist_ok=True)
    body, fixed = frame()
    clip = back_clip()
    ok = checks(body, fixed, clip)
    plate_clip = clip.copy()
    plate_clip.apply_translation((0, -40 - plate_clip.bounds[1][1] + plate_clip.bounds[0][1] + 34, -8.0 + BED))
    plate_clip.apply_translation((0, -12, 0))
    for mesh in (body, plate_clip):
        mesh.apply_translation((0, 0, -BED))
    body.export(OUT / 'part_stand_frame.stl')
    plate_clip.export(OUT / 'part_back_clip.stl')
    plate = union([body, plate_clip])
    plate.export(OUT / 'PRINT_stand_v5.stl')
    ext = plate.bounds[1] - plate.bounds[0]
    print(f'plate {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm, watertight {plate.is_watertight}, '
          f'{plate.volume / 1000:.1f} cm3 solid')
    lens = np.array([X_LEFT + (W_MIN + W_MAX) / 4 + RGB_FROM_CENTER, 0.0, Z_TOP - LENS_FROM_TOP])
    b = np.array([0.0, -(T_MIN + T_MAX) / 4, ZC])
    d = b - lens
    print(f'board centre from the color lens: forward {d[1]:.1f}, left {-d[0]:.1f}, up {d[2]:.1f} mm')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
