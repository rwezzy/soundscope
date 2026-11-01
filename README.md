# SoundScope

Find where a sound is coming from, and turn a robot arm's camera toward it.

- **Where:** a reSpeaker XVF3800 4-mic array. Direction comes from physics (noise-robust SRP-PHAT), not a model.
- **What:** Meta's Muse Spark judges which sound it is. **Commands:** Muse Voice Transcribe + Muse Spark.
- **Arm:** UFACTORY Lite 6 through Viam. Dry run by default, with slow, limited moves.
- No camera image detection.

## Files
| File | Purpose |
|---|---|
| `doa.py` | Direction finding from the 4 raw mics |
| `mic_test.py` | Mic checks and the accuracy test (`trial`) |
| `arm_follow.py` | Turns the arm toward sounds |
| `aim.py`, `arm_config.toml` | Geometry, safety limits, measured offsets |
| `meta_api.py` | Meta Model API calls |
| `test_arm.py` | Offline tests |
| `ARM_SETUP.md` | Arm, Viam and offset setup |

## Quick start
```
pip install -r requirements.txt
python mic_test.py selftest
python arm_follow.py simulate 40 -30 170
python -m unittest test_arm
```
The mic array needs the 16 kHz 6-channel firmware (steps at the top of `mic_test.py`).
For Meta features, copy `.env.example` to `.env` and add your key.
