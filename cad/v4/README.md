# Wrist mount v4

**Why a new wrist mount:** the old wrist ring and v3's were both sized for a round 77.5 mm gripper body.
UFACTORY's own 3D model of the Vacuum Gripper Lite (in their `xarm_ros2` repo) shows the body is an **oval,
52 x 75 mm**. Only the gripper's thin top disc and the arm's metal end ring are round 77.5 mm. A round
ring could not grip the oval, so it felt loose and weak. Your tape reading (about 78 mm around the metal
ring) agrees with the official ring size.

v4 hugs the real oval body. It stays 12 mm clear of the arm's metal ring, so these stay free whichever
way the cable plug turns:
- the tool I/O port
- its plug and cable
- the manual-mode button

Source: `cad/mount_v4_wrist.py`.

## What it checks against (UFACTORY's meshes plus the Orbbec and Seeed drawings)

- No contact with the gripper body. The bore follows the body's draft and clears it by at least 0.15 mm.
- No contact with the arm ring and port, or the gripper's 2 screw heads.
- No contact with a plug zone that is 4 cm out and swings 45 mm round the port.
- No contact with the button area all round the ring.
- No contact with any board part: the disc, parts on both faces, the mics or the USB plug.
- The camera touches only its 3 crush ribs, at any tolerance (V1.0 with feet or V2.0).
- Nothing of the gripper, frame or clips appears in the color or depth view. The camera's front sits
  11 mm behind the suction cup tip.

## Print

| File | What | 0.28mm DRAFT, PLA, 4 perimeters, 20% infill |
|---|---|---|
| `PRINT_wrist_v4.stl` | saddle + 2 set-screw pucks (uses the v3 board frame, clips and pins from the stand plate) | **1 h 05 min, 63 g** |
| `PRINT_wrist_v4_full_set.stl` | the same plus its own board frame, clips and pins | 1 h 49 min, 94 g |

PrusaSlicer settings:
- Printer: **CORE One L HF0.4 - Makerspace**
- Print: **0.28mm DRAFT**
- Filament: **PLA - Makerspace** (PETG if a PETG printer is free)
- **Perimeters 4, infill 20%, supports off.**

Import the file as it is. The saddle prints front face down.

## Fit it (about 5 minutes, no tools except the hex key for one M6 screw)

1. Camera into the saddle's pocket from behind:
   - lenses toward the open front
   - its **tripod-thread side against the saddle** (Type-C end away from the open slot)
   - push the cross pin through behind it
2. Slide the v3 board frame's dovetail into the groove from the front until it stops, then add the lock pin.
3. Lay the board on the pads, LED side forward, USB edge pointing away from the gripper. Add both clips and
   their pins.
4. Drop a puck into the round pocket inside the board-side wall. Start the **M6 x 10** spare screw into the
   hole from outside, under the board.
5. **Slide the whole thing up over the suction cup onto the gripper body, with the open slot facing the arm's
   cable port.** It only fits two ways (the body is oval). It stops by itself where the body widens near the
   top.
6. Tighten the M6 screw until it's snug. The puck presses the gripper against the camera side.
7. Run `python scope.py --mount wrist_v4` to view. Then run `python arm_follow.py --mount wrist_v4 listen`
   for a dry run, and check the printed turns go toward a clap before anything moves.

**Check before you print:** the arm's cable port should sit above one of the gripper's two **rounder ends**,
not above a flat side where its screws are. That's what UFACTORY's model shows. If yours is different,
adjust the slot position to match the measured fit.
