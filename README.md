# SoundScope

Estimate where a sound is coming from using four microphones and aligned camera depth.

## SoundScope Studio

The integrated workspace adds a browser dashboard, searchable sound history,
optional Meta analysis, and separate standalone/wrist geometry. Start here:

```powershell
python studio.py --demo
```

Open **http://127.0.0.1:8765**. This is an explicitly synthetic UI demonstration;
it uses no devices, Meta credits, or robot connection. Stop with Ctrl+C before
starting real sensors:

```powershell
python studio.py --live
```

Close other camera/microphone viewers first. The original OpenCV window remains
available for calibration; the browser adds the timeline and search. Keep both
sensors stationary and stay quiet for the initial two seconds. Meta is **off by
default**, even with an API key configured. See [STUDIO_GUIDE.md](STUDIO_GUIDE.md)
for the bounded three-model workflow, data retention, wrist mode and calibration.

The standalone application does not require the arm or a printed mount. A rigid
mount is required to preserve the camera/microphone relationship when moving
them together. Wrist mode currently supplies relative geometry, a motion veto
and guarded robot-coordinate **previews**, not automatic pickup.

## Run the viewer

Close other SoundScope instances and Orbbec Viewer first so only one program
uses the devices. Keep the camera and array fixed. From the cloned repository
folder, run:

```powershell
python scope.py
```

Stay quiet during the initial two-second noise measurement. This command makes
**no Meta requests and does not connect to or move the robot**. It uses the
existing four-channel acoustic localizer, not image recognition.

- **C:** begin calibration. Play a sound from a phone or speaker at the left of
  the image; click the actual source while it is sounding. Stop the sound for
  at least one second, move to the centre, repeat, then do the right. Keep the
  source stationary for each click. The three bearings must be separated.
- **Enter:** save only when the three-position fit is ready. **Esc:** cancel.
  A fourth position after saving is a useful independent check.
- **H:** toggle the acoustic tint. **D:** show diagnostic scores.
- **Space:** pause/resume the view. Calibration is disabled while paused.
- **S:** save a dashboard PNG and, when available, a paired `.npz` evidence file
  under `recordings/snapshots/`. The data contains the original frame/depth used
  for the latest estimate, acoustic cross-correlations, intrinsics, configuration,
  and capture times. It contains no API keys. These files are Git-ignored.
- **Q:** quit and stop the microphone worker.

`python scope.py --record demo.mp4` records the dashboard at 15 fps with elapsed
time preserved. The recording has **no audio track**. Screen-record with audio
separately when checking the timing of claps/snaps.

## How the display works

The dashboard fits the actual window and preserves the camera aspect ratio when
resized or maximized. The graded cyan-to-amber heatmap shows relative acoustic
agreement, including weak measurements; it does not mean a source is confirmed.
The status labels distinguish **candidate**, **ambiguous**, and **weak**.
Weak measurements have no target bracket; expired measurements clear the overlay.
Both ongoing and completed estimates expire 1.2 seconds after their last active audio samples.
The queue is bounded, old reports are discarded, and startup/shutdown have
timeouts instead of waiting indefinitely. Windows WDM-KS ADC timestamps are mapped
independently to the monotonic clock: its ADC and `currentTime` values can have
different origins and must not be subtracted blindly. The independent clock mapping keeps fresh microphone reports available to the
viewer and calibration workflow.

The displayed distance belongs to a **visible surface that fits the sound**.
It does not prove that this surface emitted the sound. Several objects along a
similar direction, or a reflection, can still produce a wrong candidate. The
quality thresholds are conservative heuristics, not confidence probabilities.
No artificial smoothing or visual object recognition moves the estimate onto a
plausible-looking object.

Calibration checks multiple positions before saving, using fresh sound
events and valid depth at the exact clicked pixel. It adjusts **yaw only**
(horizontal orientation). It does not estimate position, tilt, microphone order,
or a robot-to-camera transform. Current `arm_config.toml` still assumes an upright
array, USB edge up, with its centre **80 mm below the camera lens**. Verify those
offsets for the real mount.

The legacy `python scope.py` entry point supports the fixed
`camera.mount = "base"` setup. Studio supplies an optional rigid sensor-pair
transform for wrist-relative viewing. Neither entry point implements autonomous
pickup. Existing `arm_follow.py` commands remain a separate application.

Validation includes simulated audio, offline regression tests, Windows window
resizing and calibration-click checks, and live Gemini 2 / reSpeaker input. A
Windows live check reached approximately 30 FPS. These checks verify operation;
physical localization accuracy still needs an independent measured benchmark.

- **Where:** a reSpeaker XVF3800 4-mic array. Direction comes from physics (noise-robust SRP-PHAT), not a model.
- **What (optional):** Muse Spark evaluates a spectrogram image against a named
  sound via `--target "ringing" --enable-meta --meta-max-calls 5`; this is an
  experimental hypothesis, not a validated native audio classifier. **Commands:**
  Muse Voice Transcribe + Muse Spark in the separate arm follower.
- **Arm:** UFACTORY Lite 6 through Viam. Dry run by default, with slow, limited moves.
- **Camera view:** visible 3D points are scored against microphone timing evidence.
  A depth contour is available in diagnostics; it is not semantic segmentation.

## Files

| File | Purpose |
|---|---|
| `scope.py` | Fixed-camera viewer, calibration workflow, recording and evidence snapshots |
| `studio.py`, `studio_web/` | Integrated local dashboard and explicit Meta actions |
| `sound_events.py` | Local sound history, phrases and concept/time/direction search |
| `semantic_meta.py` | Capped Muse Voice, Muse Spark and SAM requests |
| `rig_geometry.py`, `calibrate_rig.py` | Rigid sensor geometry and offline fixed-camera calibration |
| `pose_export.py` | Optional read-only Viam tool-pose history for wrist previews |
| `scope_ui.py` | Dashboard and image-to-canvas coordinate mapping |
| `calibration.py` | Robust multi-position yaw fitting |
| `localization_quality.py` | Weak/ambiguous/candidate display heuristics |
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
python -m unittest discover
python scope.py
```
The mic array needs the 16 kHz 6-channel firmware (steps at the top of `mic_test.py`).
For Meta features, copy `.env.example` to `.env` and add your key.
