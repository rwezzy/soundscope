# SoundScope

Find where a sound is coming from, and turn a robot arm's camera toward it.

- **Where:** a reSpeaker XVF3800 4-mic array. Direction comes from physics (noise-robust SRP-PHAT), not a model.
- **What:** Meta's Muse Spark judges which sound it is. **Commands:** Muse Voice Transcribe + Muse Spark.
- **Arm:** UFACTORY Lite 6 through Viam. Dry run by default, with slow, limited moves.
- **Camera view:** the depth camera's 3D points are scored against what the mics heard, so the
  sound picks the spot; the outline comes from depth alone. No camera image detection.

## Files
| File | Purpose |
|---|---|
| `scope.py` | Live camera view: heat map, marker, distance and outline where the sound is |
| `camera.py` | Orbbec Gemini 2 color + aligned depth |
| `doa.py` | Direction finding from the 4 raw mics |
| `mic_test.py` | Mic checks and the accuracy test (`trial`) |
| `arm_follow.py` | Turns the arm toward sounds |
| `aim.py`, `arm_config.toml` | Geometry, safety limits, measured offsets |
| `meta_api.py` | Meta Model API calls |
| `test_arm.py`, `test_scope.py` | Offline tests |
| `ARM_SETUP.md` | Arm, Viam and offset setup |

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
