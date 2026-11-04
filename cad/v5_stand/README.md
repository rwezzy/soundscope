# Standing mount v5: fits any Gemini 2 and any XVF3800

Source: `cad/mount_v5_stand.py` (run it to rebuild and re-check). Profile: `python scope.py --mount stand_v5`.
Same layout as the parallel version's compact mount (camera on its own tripod, board above it, LED
forward, USB up), but fixed **datum faces** set the geometry and printed **spring leaves** push the
parts onto them, so camera and board size no longer matter.

Covers: camera 89.0-89.8 wide, 23.66-25.3 high (V1.0 with feet, V2.0, 23.96 caliper), ends 20.2-20.8
deep; board 99-101 mm, 1.0-1.6 mm thick. No holes, glue or extra hardware.

## Print
`PRINT_stand_v5.stl`: one plate (frame + back clip), already oriented. PrusaSlicer: **CORE One L
HF0.4 - Makerspace**, **0.28mm DRAFT**, **PLA - Makerspace**, 3 perimeters, 15 % infill, **supports
off**. About 1 h 15 min, 35 g (estimated: same-settings slice is 4 % faster than v3's 1 h 16 min plate).
The top bridge spans ~94 mm: a little sag on its underside is fine (springs take it up).

## Fit
1. Camera on its tripod. Slide the frame on **from behind**, lenses forward, until the camera's front
   stops on the two end lips. The floor leaves and the right side leaf flex as it slides in.
2. Push the camera's left side against the left wall; the leaves keep it there and up against the top bridge.
3. Drop the **back clip** into the two wall slots behind the camera; its leaves press the camera forward.
4. Lay the board on the two lower seats, **LED side forward, USB edge up**, behind the front lugs;
   press the two upper clips outward to let it in. The rear fingers hold it against the lugs.
5. `python scope.py --mount stand_v5`, then press C to calibrate yaw.

## Checks (run by the script)
Fixed frame vs largest camera and 99/101 mm boards: no overlap. Springs preload the smallest camera.
Every leaf stays under 2.5 % strain at full travel (worst 1.7 %). Not checked: tripod heads wider than
~46 mm (floor bumps sit 24-28 mm from centre); test-fit once.
