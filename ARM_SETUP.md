# Arm setup: Lite 6 + Viam + Meta

The arm turns (J1) toward each sound the mic array finds. **Dry run by default: nothing moves.**

## Files
- `arm_follow.py`: run this. Listens, asks Meta models, turns the arm.
- `arm_config.toml`: every offset and limit you measure or tune.
- `aim.py`: the geometry and safety rules. `meta_api.py`: Meta calls. `test_arm.py`: offline tests.

## What uses Meta
| Job | How |
|---|---|
| Spoken command → text | Muse Voice Transcribe |
| Text → action (follow, find X, stop, home) | Muse Spark 1.3 |
| "Is this the phone ringing?" | Muse Spark 1.3 reads a spectrogram picture of the sound |
| Where the sound is | No model: the mic array (physics) |

- No image detection anywhere.
- **Meta's API drops audio sent to Muse Spark** (tested Nov 1: accepted, never reaches the model; [issue #59](https://github.com/meta-models/meta-model-cookbook/issues/59)). Hence the spectrogram. Your `Meta_Hack` "analyze" audio has the same problem.

## Safety built in
- Moves only with all three: `live = true` in the config, `--live`, and typing `MOVE`.
- Sets the arm to 15°/s before the first move. Refuses configs above 30°/s. Stops the arm if any move ran faster than allowed.
- Only J1 turns (J5 optional): at most 30° per move, ±90° around the start pose, 10° inside the Lite 6 limits.
- Ignores unsure sounds (z < 12) and its own motor noise.
- First real turn checks J1 swings the expected way, else stops.
- Ctrl+C or any error → `arm.stop()`. Viam collision sensitivity 4.
- **Your part:** E-stop in hand, area clear, arm parked in a safe pose (UFACTORY Studio) before starting.

## Try it now (arm off)
1. Copy `.env.example` to `.env` and paste `MODEL_API_KEY` from `Meta_Hack\.env`.
2. Run:
```bash
python arm_follow.py simulate 40 -30 170
```
```bash
python arm_follow.py listen
```
```bash
python arm_follow.py listen --target "phone ringing"
```
```bash
python arm_follow.py listen --voice
```
- `--voice`: press Enter, then say e.g. "find my keys", "follow", "stop", "go back".
- `--target` sends each sound clip's spectrogram to Meta.

## Viam setup (when the arm is powered)
1. **viam-server:** Viam app → add a machine → Windows setup wizard (viam-agent). The UFactory module has a Windows build.
2. **Network:** Ethernet from the laptop to the arm's control box. Windows: Settings → Network & internet → Ethernet → IP assignment → Edit → Manual → IPv4 `192.168.1.10`, mask `255.255.255.0` (same subnet as the IP on the arm's label). Then `ping <arm IP>`.
3. **Arm:** CONFIGURE → **+** → Configuration block → search `lite6` (module `viam:ufactory`) → name it `arm`. Paste the attributes from:
```bash
python arm_follow.py viam-config
```
4. **Frames:** paste the arm frame, and camera frame (a) beside the base for now. Check the 3D SCENE tab.
5. **Camera (optional, for viewing in Viam):** `webcam` camera named `camera` → pick "Orbbec Gemini 2 RGB Camera". No Viam module for Gemini 2 depth was found, so depth stays local (pyorbbecsdk2).
6. **Keys:** CONNECT → API keys → put the address, key and key ID in `.env`.
7. **Read-only check** (no motion), then compare with `[look]` in the config:
```bash
python arm_follow.py check
```
8. **Live:** set `live = true`, park the arm in a safe looking pose in Studio, then run `python arm_follow.py listen --live` and type `MOVE`.

## Measuring the offsets (`arm_config.toml`)
Base frame: origin = base center at table level; **+X = where the arm reaches at J1 = 0** (UFACTORY Studio's 3D view shows it); +Y = left; +Z = up.

- **`[array]`:** tape-measure base center → board center, forward (+X) and left (+Y). `front_yaw_deg` = the board front edge's angle from +X, counterclockwise from above.
- **`[camera.base]` (now):** lens center position, plus the way it faces (`yaw_deg`, `pitch_deg`).
  - Check it: put an object 1 m straight in front of the lens, then run `python arm_follow.py where 0 0 1000`. The printed x, y should match your tape measure.
- **`[look]`:** for dry runs, roughly where the pointer is at J1 = 0. The pointer is the gripper now, the camera once mounted. Live runs measure it through Viam (`check`).
- **`[camera.arm]` (after the 3D-printed mount):**
  - `offset_mm` = lens position from the flange center in the flange frame (+Z out of the flange face), from your mount's CAD.
  - **20 cm rule:** the fingertips must be ≥ 200 mm in front of the lens, i.e. `gripper_tip_mm − offset_z ≥ 200`. Example: a 75 mm gripper → lens ≥ 125 mm behind the flange face. `viam-config` prints this "Mount check".
  - Keep the lens off the gripper's axis (e.g. 70 mm to the side or above) so the fingers don't fill the view.
  - Weigh the camera + mount and enter the payload in UFACTORY Studio, so collision detection stays accurate.
  - Then set `mount = "arm"` and use frame (b) in Viam.

## Base frame check
If the first live turn stops with "J1 swung the arm the opposite way": make sure the arm's Viam frame has no rotation and the 3D scene shows the arm reaching along +X at J1 = 0.

## Not built yet
- **Grasping:** the code only reports "within reach". Reaching needs the mounted camera + depth (Viam motion service with obstacles).
- **Distance:** `assumed_distance_mm` until Gemini 2 depth is wired in.

Sources: [Viam UFactory module](https://github.com/viam-modules/viam-ufactory-xarm), [Viam arm + camera frames](https://docs.viam.com/motion-planning/frame-system/arm-gripper-camera/), [Lite 6 specs](https://docs.lite6.ufactory.cc/9.technical_specifications.html)
