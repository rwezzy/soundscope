# SoundScope

**Hear where a sound came from, see the object that made it, and ask about it later.**

Four microphones find *where* a sound is (physics, not a model). A depth camera shows *the spot*.
Meta models then add *what* it was, *who said what*, *what object is there*, and a memory you can
search by voice: "when did the phone ring?" → "about 3:42 PM, 1.2 m to your left, near the laptop."
The arm demo turns a UFACTORY Lite 6 toward sounds with the camera and mics on its gripper.

**Hard rule: no image detection decides where a sound came from.** The microphones pick the spot
first; the camera and the vision models only describe, outline or double-check *that* spot.

## What Meta does here

| Model | What SoundScope uses it for | Checked live (Nov 4) |
|---|---|---|
| **Muse Spark 1.3** | Labels each clear sound from a **spectrogram picture** (the API drops raw audio), next to your own recorded examples when you have some | "phone ringing", 0.92, from a dual-tone ring |
| **Muse Spark 1.3** | **Sound object detection**: one call gets the spectrogram plus camera crops of the spot the mics picked, *before and after* the sound, and returns the sound, the object at the spot, what visibly happened, and whether sound and sight agree | "The phone rang and its screen lit up blue" |
| **Muse Spark 1.3** | **Night mode**: in a dark room the crop is the depth camera's 3D relief instead, so shapes still show | saw "a round device" from depth alone |
| **Muse Spark 1.3** | Voice commands: find, follow, **watch** ("tell me when the kettle whistles"), stop, search, who, describe | all 4 test phrases parsed right |
| **Muse Spark 1.3** | Answers questions about the sound memory, only from the stored facts (invented record ids are dropped) | correct two-ring answer |
| **Muse Voice Transcribe** | Spoken commands (with keyword biasing), and **captions with speaker turns** (DIARIZATION mode); each turn's direction from the mics names the talker ("Speaker 1, to the left") | two voices → turns A and B with timings |
| **SAM 3.1** | Outlines the object at the spot the sound picked, with a **point prompt** (never a text search for objects) | exact outline of the test object |

Background checks use Muse Spark's `minimal` reasoning setting (about half the wait of `low`);
spoken commands and searches use `low`.

**Budget guard:** every call is priced before it is sent and refused if it would cross
`META_DAILY_LIMIT_USD` (default $1.00, shared by all SoundScope programs) or
`META_SESSION_LIMIT_USD` (default $0.50). Typical costs: a sound label or object check
≈ $0.003–0.01, a caption ≈ $0.0003, an outline $0.0025. All of the live checks above cost ≈ $0.10.
Contributor model tiers (which train on your data) are never used.

## Run it

```
pip install -r requirements.txt
python scope.py --mount tripod                 # live view, sound memory on, no Meta requests
python scope.py --mount tripod --meta          # + labels, captions, object check, outlines
python scope.py --mount wrist --meta           # camera + mics clamped on the arm's gripper
python soundscope.py search "when did the phone ring?"
python soundscope.py digest --minutes 30       # Muse Spark summarizes what happened
python soundscope.py map                       # picture: where every sound came from
python soundscope.py enroll "phone ringing"    # record your own example (free labels + few-shot)
python soundscope.py listen --meta             # mics only, no camera
python soundscope.py budget                    # today's estimated Meta spend
python arm_follow.py listen --mount wrist      # arm turns toward sounds (dry run unless --live)
python -m unittest discover                    # 166 offline tests, no hardware or network
```

Meta features need `MODEL_API_KEY` in `meta-soundscope/.env` (or `--env-file path/to/.env`).
The mic array needs the 16 kHz 6-channel firmware (steps at the top of `mic_test.py`).

**In the viewer:** **V** = voice command, **L** = last sounds, **C** = calibrate (three positions,
**Enter** saves), **H** = heat map, **D** = diagnostics, **Space** = pause, **S** = snapshot, **Q** = quit.
Say: "find my keys", "follow the phone ringing", "tell me when the kettle whistles", "stop",
"show everything", "when did the phone ring?", "who is talking?", "what was that?".
A search also marks on the picture where the matching sounds happened (tripod mount).

## Mounts

Both mounts work on their own; the arm is one application, not a requirement.

| | Tripod / desk | Wrist (arm demo) |
|---|---|---|
| Print | Meta_Hack `compact_shared_mount`, or `cad/soundscope_rig_v2_print.stl` | `cad/wrist_mount.stl` (one part, clamps on the 77.5 mm gripper with one spare M6 screw) |
| Profile | `profiles/tripod.toml` | `profiles/wrist.toml` |
| Viewer | `scope.py --mount tripod` | `scope.py --mount wrist` (arm can stay off) |
| Arm | not needed | `arm_follow.py listen --mount wrist` turns until the sound is centered in the camera view; no camera-to-arm calibration needed because the mics ride with the camera |

Profiles hold only geometry (board position from the camera's color lens, orientation); speed and
turn limits stay in `arm_config.toml`. Calibration (C) saves into the active profile.

## How it decides

- **Where:** SRP-PHAT over the 4 raw mics, weighted by each bin's rise above the noise floor, with
  echo rejection; visible 3D points from the depth camera are scored against the mic evidence, so
  the sound picks a surface in the picture. The noise floor uses minimum statistics: a heater or
  fan switching on after start-up becomes background within ~3 s instead of one endless "sound"
  (found and fixed in the live test on Nov 4; `test_doa_noise.py` reproduces it).
- **What:** local measurements first (free): loudness over the room, duration, pitch, bursts,
  speech likelihood, and the nearest of your recorded examples. Muse Spark only for clear sounds,
  rate-limited, one request at a time in the background.
- **Memory:** every clearly located sound becomes a record (`recordings/memory/*.jsonl`, local and
  git-ignored): when, where (direction, distance, pixel), how loud, what, who said what, the object
  at the spot. Answers are written from those stored facts.

## Measured on the real hardware (Nov 4, laptop speaker ~1.7 m to the camera's left)

- Separate sounds stay separate: 3 noise bursts → 3 records; a ring → 1 record ("tonal around
  1.8 kHz"); a spoken sentence → 2 speech records with captions; 2 claps → "two clicks" at +2 dB.
- All placed 37–42° left at 1.7 m, consistently; earlier mic trial: 5.4° mean error at 1 m.
- The room's own faint blips (z < 7 or below the room's average level) are not saved.
- Limits: labels from spectrograms can be wrong ("dog bark" for a noise burst), which is why they
  say "probably" below 0.75 confidence; transcription of laptop speech across a room drops words;
  depth gives the visible surface along the direction, not proof of the source.

## Files

| File | Purpose |
|---|---|
| `scope.py`, `scope_ui.py` | Live viewer: heat map, candidate spot, calibration, voice, captions, memory |
| `sound_memory.py` | Records, local acoustic measurements, search, your examples, talker names |
| `insight.py` | Background Meta worker: labels, captions, object check, context, outlines, watch alerts |
| `meta_api.py` | Meta Model API calls and the budget guard |
| `soundscope.py` | Command line: search, digest, map, log, enroll, listen, budget |
| `doa.py`, `mic_test.py` | Direction finding; mic checks and accuracy trial |
| `camera.py` | Orbbec Gemini 2 color + aligned depth |
| `calibration.py`, `localization_quality.py` | Multi-position yaw fit; weak/ambiguous/candidate labels |
| `arm_follow.py`, `aim.py`, `arm_config.toml` | Arm: safety gate, base and wrist modes, Viam |
| `profiles/` | Mount geometry: tripod, wrist |
| `cad/` | Printed mounts (`rig.py`, `rig_v2.py`, `wrist_mount.py` and their STLs) |
| `test_*.py` | Offline tests |
| `ARM_SETUP.md` | Arm, Viam and offset setup |

## Privacy

Recordings, the sound memory, your examples and the Meta usage log stay on this computer and are
git-ignored. Captions only run with `--meta` or `--captions`. Context and object checks are not
sent for speech (the crop would show the person talking).
