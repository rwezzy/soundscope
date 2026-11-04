# Mounts v3: standing and wrist

Sizes from the manufacturers (Orbbec Gemini 2 CAD V1.0/V2.0, Seeed XVF3800 drawing, UFACTORY: 2 x M6
grippers); caliper readings only where nothing official exists (gripper body 77.5 mm).
Source: `cad/mount_v3.py`. Fit checks: camera at max tolerance, board with its edge parts, USB plug,
jack and mics, tripod head and gripper - all clear except the intended crush ribs.

> **Wrist: use v4 instead** (`cad/v4/`). The v3 wrist ring assumed a round 77.5 mm gripper body; UFACTORY's model shows it is an oval (52 x 75 mm), so that ring can't clamp. The standing mount below is unchanged.

## What to print

| File | What | 0.28mm DRAFT, PLA (sliced) |
|---|---|---|
| `PRINT_stand.stl` | standing collar + board frame + 2 clips + all pins | 1 h 16 min, 56 g |
| `PRINT_wrist_collar_only.stl` | wrist collar with ring clamp (reuses the stand's frame, clips, pins) | 1 h 2 min, 52 g |
| `PRINT_wrist.stl` | wrist collar + its own frame, clips, pins | 1 h 46 min, 85 g |

PrusaSlicer: printer **CORE One L HF0.4 - Makerspace**, print **0.28mm DRAFT** (or 0.20mm SPEED for a
nicer finish, about 30 min longer), **PLA - Makerspace** (PETG if a PETG printer is free: tougher),
**3 perimeters, 15% infill, supports off**. Import the file as-is: every part is already laid in its
strongest direction (layers run along the loads).

## Standing (camera stays on its tripod)

1. Leave the camera screwed to its tripod. Slide the collar over it **from behind**, lenses forward,
   until the camera's front face stops against the lips. The tripod neck passes through the slot in
   the collar's floor. Push the **cross pin** through the two side holes behind the camera.
2. Slide the **board frame's** dovetail into the groove on top of the collar from the back until it
   stops; push the **lock pin** through the side of the groove.
3. Lay the board on the four pads, **LED side forward, USB edge up**; its edge drops into the four
   small edge stops. Put a **clip** on each side and press two **clip pins** through each clip into
   the frame (they are a press fit; if one is too tight, sand it or twist it in; spares included).
4. `python scope.py --mount stand_v3`, press **C** and calibrate once.

## Wrist

1. Slide the ring over the gripper body (the ring's flat front faces the suction cup end) and tighten
   the two **M6 x 10** spare screws until snug - they cut their own thread in the plastic.
2. Camera into the collar from behind, lenses toward the cup, **tripod thread toward the gripper**;
   cross pin behind it. Frame, lock pin, board and clips exactly as on the stand.
3. `python scope.py --mount wrist_v3` (view only), then `python arm_follow.py --mount wrist_v3 listen`
   (dry run) and check the printed turns go toward a clap before anything moves.

The Type-C cable leaves the camera's end through the open side of the collar; the board's USB cable
goes straight up/outward. Nothing covers the four mics.
