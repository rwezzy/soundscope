# SoundScope

**See where a sound came from, what it was, and who said what.**

Many people miss sounds that matter: someone who is Deaf or hard of hearing, someone wearing
headphones, anyone in another room. Even when a sound is noticed, it is hard to tell *where* it came
from. A smoke alarm, the doorbell, a knock, a phone, a baby, someone calling your name.

SoundScope puts sound on a screen. Four microphones find **where** a sound came from (physics, not a
model) and a depth camera shows **the spot**. Meta's models add **what it was**, **who said what**,
**what object is there**, and a memory you can ask later: *"did someone knock while I was out?"*
It is a sound-awareness aid, not a medical device.

```
python scope.py --mount carrier_v3 --assist --name Keith
```

| You get | How |
|---|---|
| **Alerts** for a smoke alarm, doorbell, knock, phone, baby crying, kettle, and **your name**: a red banner, a beep and the direction ("ALERT: doorbell (to the left)") | Muse Spark (and your own recorded examples, for free) |
| **Captions placed at the talker**: "Speaker 1 (to the left): dinner is ready" | Muse Voice Transcribe (speaker turns) + the mics' direction |
| **The object that made the sound**, checked against the picture before and after it | Muse Spark (spectrogram + camera) and SAM 3.1 outline |
| **A memory you can ask**: when, where, how loud, what was near | Muse Spark answers only from stored facts |
| **Voice control**: "find my keys", "tell me when the kettle whistles", "who is talking?" | Muse Voice Transcribe + Muse Spark |

**Hard rule: no image detection decides where a sound came from.** The microphones pick the spot
first; the camera and the vision models only describe, outline or double-check *that* spot.

## Works standalone

SoundScope does not need the robot or a particular mount. Each setup is just a geometry profile
(where the mic board sits relative to the camera); calibration (**C**, three claps) fixes the rest.

| Setup | Run |
|---|---|
| **Current rig:** camera on its tripod, board in the adjustable ring carrier v3 (printed screws; fits boards 96-102 mm and cameras ~88-91 mm wide; designed in the parallel SoundScope build, print files not in this repo yet) | `python scope.py --mount carrier_v3 --assist` |
| Camera on its tripod, board on the printed compact stand (on a desk, shelf or the floor) | `python scope.py --mount tripod --assist` |
| Size-tolerant printed stand (`cad/v5_stand/`) or the v3 stand (`cad/v3/`) | `--mount stand_v5` / `--mount stand_v3` |
| **No camera at all**: just the mic board, anywhere | `python soundscope.py listen --assist --name Keith` |
| Optional robot demo: camera + mics on the Lite 6 gripper (`cad/v4/`) | `python scope.py --mount wrist_v4`, `python arm_follow.py --mount wrist_v4 listen` |

Camera tilted up or down (on the floor, on a shelf)? Put the angle in the profile's `pitch_deg`.
Nothing about the localization depends on the arm; the arm only consumes the same directions.

## What Meta does here

| Model | Use in SoundScope | Checked live (Nov 4) |
|---|---|---|
| **Muse Spark 1.3** | Labels each clear sound from a **spectrogram picture** (the API drops raw audio), next to your own recorded examples when you have some; reports which watched sounds it is | "phone ringing", 0.92, from a dual-tone ring |
| **Muse Spark 1.3** | **Sound object check**: one call gets the spectrogram plus camera crops of the spot the mics picked, before and after the sound, and returns the object there, what visibly happened, and whether sound and sight agree | "The phone rang and its screen lit up blue" |
| **Muse Spark 1.3** | **Night mode**: in a dark room the crop is the depth camera's 3D relief instead | saw "a round device" from depth alone |
| **Muse Spark 1.3** | Voice commands (find, follow, watch, stop, search, who, describe); answers about the memory; `soundscope.py digest` summaries | all test phrases parsed right; correct grounded answers |
| **Muse Voice Transcribe** | Commands (keyword-biased) and **captions with speaker turns** (DIARIZATION); each turn's direction names the talker; your name in a caption raises an alert | two voices → turns A and B with timings; commands word for word through the board |
| **SAM 3.1** | Outlines the object at the spot the sound picked (**point prompt**, never a text search), automatically when sound and sight agree | exact outline of the test object |

**Budget guard:** every call is priced before it is sent and refused if it would cross
`META_DAILY_LIMIT_USD` (default $1.00 a day, shared by all SoundScope programs) or
`META_SESSION_LIMIT_USD` ($0.50). A label or object check costs about $0.003-0.01, a caption about
$0.0003, an outline $0.0025. Background checks use Muse Spark's fast `minimal` reasoning;
commands and searches use `low`. Contributor model tiers (they train on your data) are never used.
Without `--assist`/`--meta`, nothing is sent unless you press **V**.

## Run it

```
pip install -r requirements.txt
python scope.py --mount carrier_v3 --assist --name Keith   # sound awareness, everything on
python scope.py --mount carrier_v3 --assist --private     # same, but no pictures leave the computer
python scope.py --mount carrier_v3                        # live view + local memory, no Meta requests
python scope.py --watch "dog barking" --meta           # add your own alerts
python soundscope.py enroll "doorbell"                 # record your doorbell: free, instant label
python soundscope.py search "did the doorbell ring?"   # ask the memory
python soundscope.py digest --minutes 60               # what happened while you were away
python soundscope.py map                               # where every sound came from
python soundscope.py listen --assist                   # mics only, no camera
python soundscope.py budget                            # today's Meta spend
python -m unittest discover                            # 182 offline tests, no hardware or network
```

Meta features need `MODEL_API_KEY` in `meta-soundscope/.env` (or `--env-file path/to/.env`).
The mic array needs the 16 kHz 6-channel firmware (steps at the top of `mic_test.py`).

**In the viewer:** **V** = voice command, **L** = last sounds, **C** = calibrate (three positions,
**Enter** saves), **H** = heat map, **D** = diagnostics, **Space** = pause, **S** = snapshot, **Q** = quit.
A search also marks on the picture where the matching sounds happened.

## How it decides

- **Where:** SRP-PHAT over the 4 raw mics, weighted by each frequency's rise above the room's noise,
  with echo rejection; the depth camera's visible 3D points are scored against the mic evidence, so
  the sound picks a surface in the picture.
- **Steady noise:** the noise floor follows minimum statistics, so a heater or fan switching on
  becomes background within ~3 s instead of swallowing later sounds; any sound that never stops is
  reported every 8 s. Tuned on real recordings with a loud vent.
- **What:** free local measurements first (loudness over the room, length, pitch, bursts, speech
  likelihood, nearest of your examples); Muse Spark only for sounds at least +3 dB over the room,
  three at a time, newest first, rate-limited.
- **Alerts:** only when worth interrupting: at least +6 dB over the room and a label at least 60% sure
  (40% at +20 dB or louder, so a loud alarm is not missed); the same alert at most every 20 s.
- **Memory:** each clearly located sound becomes a record (`recordings/memory/*.jsonl`, local,
  git-ignored): when, where, how loud, what, who said what, the object at the spot, alerts.

## Measured on the real hardware (Nov 4, laptop speaker ~40° left, ~1 m)

- Separate sounds stay separate: 3 noise bursts → 3 records, a ring → 1 ("tonal around 1.4 and
  1.8 kHz"), a sentence → 2 speech records with captions, 2 claps → 1 record ("two clicks").
- Direction is consistent: 37-42° left on every test sound; earlier accuracy trial 5.4° mean error at 1 m.
- Loud vent recording: all 3 claps found separately and on target (a 3 s-chunking alternative found 1).
- Voice commands through the board: "find my keys", "when did the phone ring", "tell me when the
  kettle whistles" transcribed word for word and understood.
- Your examples: after 2 takes each, a smoke alarm and a knock were labeled correctly for free, and
  the smoke-alarm watch raised its alert.
- Wrist mode (dry run): a knock 38° left → the arm would turn J1 0° → 30° → 38°.
- Limits: spectrogram labels can be wrong ("dog bark" for a noise burst), so uncertain ones say
  "probably"; laptop speech across a room loses words; depth gives the visible surface along the
  direction, not proof of the source. Captions were tested mostly with computer voices.

**Ring holder, assist mode (Nov 4, live):**
- A person saying "Hi, my name is Keith" → `ALERT: Speaker 2 said "Keith" (slightly left)`.
- Smoke alarm → `ALERT: smoke alarm (straight ahead)`; knock → `ALERT: knock on the door`.
- A faint doorbell (+5 dB) was labeled "Doorbell" without an alert; faint blips no longer raise
  false knocks.
- Results arrived about 6-13 s after each sound (10-44 s before, with one worker).

## Files

| File | Purpose |
|---|---|
| `scope.py`, `scope_ui.py` | Live viewer: heat map, spot, calibration, voice, captions, alerts, memory |
| `sound_memory.py` | Records, local measurements, search, your examples, talker names |
| `insight.py` | Background Meta worker: labels, object check, captions, outlines, watch alerts |
| `meta_api.py` | Meta Model API calls and the budget guard |
| `soundscope.py` | Command line: search, digest, map, log, enroll, listen (mics only), budget |
| `doa.py`, `mic_test.py` | Direction finding; mic checks and accuracy trial |
| `camera.py` | Orbbec Gemini 2 color + aligned depth |
| `calibration.py`, `localization_quality.py` | Multi-position yaw fit; weak/ambiguous/candidate labels |
| `arm_follow.py`, `aim.py`, `arm_config.toml` | Optional arm: safety gate, base and wrist modes, Viam |
| `profiles/` | Setup geometry: carrier_v3 (ring holder), tripod (compact stand), stand_v3, stand_v5, wrist_v4 |
| `cad/` | Printed mounts: `v3/`, `v4/` (wrist), `v5_stand/` (size-tolerant stand) |
| `test_*.py`, `test_data/` | Offline tests; real array clips (text-to-speech and test tones) |
| `ARM_SETUP.md`, `DEMO.md` | Arm setup; demo-video shot list |

## Privacy and safety

Recordings, the sound memory, your examples and the Meta usage log stay on this computer and are
git-ignored. Captions run only with `--assist`, `--meta` or `--captions`.

- Pictures are never sent for speech-like sounds, and people are never outlined.
- Faces in any picture that is sent are blurred first. This is best effort: OpenCV's face finder
  can miss an angled or dim face, so a picture of the sound's spot can still show a person.
- `--private` sends no pictures at all (labels and captions only).

The arm is dry-run by default, moves slowly within fixed limits, and only after `live = true` and
typing MOVE.
