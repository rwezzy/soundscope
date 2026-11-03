# SoundScope

Find where a sound is coming from, and turn a robot arm's camera toward it.

- **Where:** a reSpeaker XVF3800 4-mic array. Direction comes from physics (noise-robust SRP-PHAT), not a model.
- **What:** Meta's Muse Spark judges which sound it is, from a spectrogram picture of it.
- **Voice:** in the camera view press `v` and say "find my keys": Muse Voice Transcribe writes it
  down, Muse Spark turns it into what to look for, and only matching sounds get highlighted.
- **Arm:** UFACTORY Lite 6 through Viam. Dry run by default, with slow, limited moves.
- **Camera view:** the depth camera's 3D points are scored against what the mics heard, so the
  sound picks the spot; the outline comes from depth alone. No camera image detection.

## Files
| File | Purpose |
|---|---|
| `scope.py` | Live camera view: heat map, marker, distance and outline where the sound is; `v` for voice |
| `camera.py` | Orbbec Gemini 2 color + aligned depth |
| `doa.py` | Direction finding from the 4 raw mics |
| `mic_test.py` | Mic checks and the accuracy test (`trial`) |
| `arm_follow.py` | Turns the arm toward sounds |
| `aim.py`, `arm_config.toml` | Geometry, safety limits, measured offsets |
| `meta_api.py` | Meta Model API calls |
| `test_arm.py`, `test_scope.py` | Offline tests |
| `ARM_SETUP.md` | Arm, Viam and offset setup |
| `cad/` | 3D-printed rig holding the camera and mic board (`soundscope_rig_print.stl`, made by `rig.py`) |

## Quick start
```
pip install -r requirements.txt
python mic_test.py selftest
python arm_follow.py simulate 40 -30 170
python -m unittest test_arm test_scope
python scope.py
```
The mic array needs the 16 kHz 6-channel firmware (steps at the top of `mic_test.py`).
For Meta features, copy `.env.example` to `.env` and add your key.

## Mount
Print `cad/soundscope_rig_print.stl` as it comes (camera plate flat on the bed): no supports needed.
PLA, 0.2 mm layers, 3 perimeters, 25% infill. Push the tripod screw through the thin skin over the nut pocket.
Hardware: 2x M3x6 (camera), 2x M2x10 self-tapping (board), 1x 1/4"-20 hex nut (tripod).
The board hangs under the camera's color lens, LED side forward, cable down; it is held only by
the two holes opposite its USB port, so the mics stay uncovered. Then set `cable_edge = "down"` and
`position_mm = [-2, 0, -73.5]` under `[array]` in `arm_config.toml`.
