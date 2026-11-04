# SoundScope Studio

SoundScope estimates direction from four microphone signals, then compares that
evidence against visible depth surfaces. The browser application adds a local
event history and optional interpretation. The robot is an optional consumer of
measurements; standalone listening and search do not depend on it.

## First run

From this repository in PowerShell:

```powershell
python studio.py --demo
```

Open http://127.0.0.1:8765. The synthetic scene exercises the interface, timeline
and search without hardware or paid calls. Try **ringing on the left** and select
a result to inspect its timestamp, direction, level and original snapshot. This
demonstrates software behavior, not measured sound-localization accuracy.

Stop with Ctrl+C, close any other SoundScope/Orbbec viewers, then run:

```powershell
python studio.py --live
```

1. Connect the Gemini 2 and reSpeaker with the existing six-channel firmware.
2. Keep their positions fixed relative to each other. Leave the robot still.
3. Stay quiet during the two-second background measurement.
4. Use the OpenCV window's **C** calibration workflow if the physical sensor
   positions still match `arm_config.toml`. Three left/centre/right samples are
   required; Enter saves a valid fit. This estimates yaw only.
5. Make a sound, then leave a quiet gap so the event completes. Inspect it in the
   browser. If it is weak, lacks valid depth, or cannot be paired with a recent
   frame, the application must not invent a precise location.

The current default config assumes the array is upright, USB edge up, with its
centre 80 mm below the camera. **The printed holder may change this.** Measure
the actual arrangement and update/calibrate it before trusting overlays. Changing
only yaw cannot correct an incorrect position or tilt.

The browser uses letterboxing to preserve the camera image's aspect ratio.
The original OpenCV keys, evidence snapshots and optional recording remain
available. Close the OpenCV window or press Q to stop the live server; Ctrl+C
also stops it. Only one instance should use the devices at a time.

## Search and descriptions

Local search uses sound concepts/synonyms plus time, direction and level filters.
Examples include `ringing on the left`, `claps in the last five minutes`, and
`quiet sound near the keyboard`. It is not an embedding search over arbitrary
unlabelled audio. A semantic sound name must already exist in an event label;
unanalysed live events initially have measured descriptions such as a tonal
sound or a brief broadband transient.

Each result produces a phrase containing available evidence: time, possible
sound label, direction, candidate surface range, recorded input level and nearby
visible objects. Unknown information stays unknown. History directions refer
to the camera **when the event occurred**, not a permanent room map.

- **dBFS** measures the digital recording level. It is not calibrated physical
  loudness in dB SPL. Local `loud`/`quiet` filters use digital thresholds, not
  hearing-safety categories or comparisons across different microphones.
- The reported range is the distance to a visible surface compatible with the
  acoustic estimate. It does not establish that the surface made the sound.
- `near` describes proximity in the snapshot's image unless separately measured.
  Visual context never changes the acoustically estimated source position.

## Three optional Meta models

Install the optional SAM response decoder once, if needed:

```powershell
python -m pip install -r requirements-semantic.txt
```

With the existing key in `.env`, enable a small manual allowance:

```powershell
python studio.py --live --enable-meta --meta-max-calls 5
```

| Button | Model | Purpose and limit |
|---|---|---|
| Speak a search | Muse Voice Transcribe | Records five seconds from the browser-selected microphone, transcribes once, then searches locally |
| Interpret with Meta | Muse Spark 1.3 | Rephrases a natural-language search; does not execute robot commands |
| Sound hypothesis | Muse Spark 1.3 | Examines a spectrogram of the captured sound; experimental, unverified, not a validated audio classifier |
| Nearby objects | Muse Spark 1.3 | Describes visible objects near the acoustic pixel; does not decide which object emitted sound |
| SAM outline | SAM 3.1 | Segments the original snapshot using the acoustic candidate as a point prompt; keeps the outline on that snapshot |

Every button makes at most one request. No action automatically chains further
paid requests. There are no automatic retries. Failed attempts count toward the
cap. Successful repeated inputs are cached. The default limit is five attempted
requests per process, with a persistent maximum of twenty per UTC day for the
same data directory. Text responses are capped at 1200 tokens. These limits count
requests; they cannot guarantee a dollar amount or inspect the account balance.

The API adapters were checked against official Meta documentation and tested
with mocked responses. Account access, real model output quality and speech
accuracy still need a small live test. No contributor model is selected. See
[META_FEATURES_NOTES.txt](META_FEATURES_NOTES.txt) for documented sources and
implementation limits. The older separate `arm_follow.py` application does not
inherit Studio's credit controls.

## Standalone and wrist configurations

**Standalone:** camera and microphone are fixed beside the arm or used without
any robot. There is no arm connection in Studio.

**Wrist:** both sensors must move together as one rigid assembly. Their relative
transform remains fixed, so acoustic projection into the camera image does not
need a generative model to guess new calibration after every movement.

```powershell
python studio.py --live --rig-mode wrist
```

This provides camera-relative viewing with the existing nominal sensor offsets.
It vetoes events when image motion is detected and waits for settling. The image
check is a heuristic, not a reliable measurement of joint motion. Capture and
compare sounds while the arm is stationary. Do not use this command as evidence
that the new mount is calibrated or the robot can move safely.

For measured sensor geometry, supply a JSON file based on `rigs.example.json`:

```powershell
python studio.py --live --rig-mode wrist --rig-config measured_rig.json
```

The example file is deliberately marked uncalibrated. Translation units are
metres. `T_camera_from_array` maps microphone coordinates to camera coordinates;
camera axes are right, down, forward. `T_tool_from_camera` additionally describes
the camera relative to the robot's tool. Do not replace missing measurements
with identity matrices or mark calibration complete just to bypass a guard.

The optional **Robot preview** computes a dry-run target only when appropriate
measured transforms and fresh evidence are present. A wrist preview can read a
pose packet from `--pose-file pose.json`. That file must contain
`T_base_from_tool`, `captured_at` on the same host monotonic clock, `moving`, and
`settled_since`. The pose must match the image measurement within 150 ms and be
fresh. No robot-pose producer or motor-control loop is started by Studio.

A separate read-only exporter is prepared for a configured Viam arm:

```powershell
python pose_export.py --connect-read-only --output studio_data/wrist_pose.json
```

It needs a compatible `viam-sdk` with its documented `spatialmath` conversion,
plus the existing Viam address and credentials in `.env`. The SDK bridge is
checked before connecting; it is not installed in the current system Python.
Install the optional robot dependency with `python -m pip install viam-sdk` when
setting up that connection. The exporter reads position and motion state only;
it never enables, moves, stops or reconfigures the arm. No physical connection
was made during implementation.

It writes a bounded two-second history (at most forty poses), so Studio can
match the event frame even though sound completion arrives after a quiet tail.
Movement, slow/erroring reads and shutdown invalidate the packet/history. The
calibration must refer to the **same end-effector frame** reported by Viam, not
an assumed suction-tip or flange frame. In another terminal, the later wrist
preview command is:

```powershell
python studio.py --live --rig-mode wrist --rig-config measured_rig.json --pose-file studio_data/wrist_pose.json
```

RGB-D frames are paired with completed audio reports using a bounded recent
frame buffer. The recorded timing is camera delivery time, not hardware trigger
synchronization. It is appropriate to evaluate stationary observations; it does
not validate localization during robot movement.

`calibrate_rig.py --help` provides an offline fixed-camera-to-base fit from paired
3D points, with held-out validation. It does not measure a wrist hand-eye transform
and never enables live motion automatically.

Still required before physical actuation: mount fit/retention and cable-clearance
checks, payload configuration, measured calibration, independent position checks,
and an explicit bounded control application. Autonomous suction pickup, obstacle
removal, continuous world tracking and speaker diarization are not implemented
in this dashboard.

## Local data and verification

`studio_data/live/` contains metadata for the latest 500 events and the Meta
request ledger/cache. Synthetic events use a separate `studio_data/demo/` folder.
The last twenty event audio clips and camera snapshots live in RAM only and
expire on exit. Metadata and returned transcripts/context can still be personal
data. The data folders are Git-ignored. Selected media is sent to Meta only for
an explicitly requested model action. The server binds to localhost.

The integrated offline suite can be run without devices or paid calls:

```powershell
python -m unittest test_scope_runtime test_scope_ui test_scope_evidence test_listener_runtime test_calibration test_scope test_localization_quality test_sound_events test_semantic_meta test_rig_geometry test_studio_geometry test_calibrate_rig test_studio test_pose_export -q
```

These checks cover existing localization/viewer behavior, event history/search,
budget and cache enforcement, parsing, geometry, stale-pose rejection, and the
local HTTP application. They do not establish real-room accuracy or successful
physical pickup.
