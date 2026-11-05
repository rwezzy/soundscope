# SoundScope

**See a sound's direction. Ask what happened. Inspect the recorded evidence.**

A phone rings while you are wearing headphones. Someone calls from another part of the room.
You notice too late and want to know what happened. SoundScope combines a live acoustic map with
a searchable local sound history: likely sound labels, captions, time, direction, nearby visual
context, and saved evidence you can revisit.

Four microphones measure acoustic direction. A depth camera supplies visible surfaces consistent
with that direction. Meta models interpret and retrieve those observations. A highlighted surface
is a **candidate**, not proof that the object made the sound. This is an experimental sound-awareness
prototype, not an emergency alarm or a validated accessibility or medical device.

```powershell
python scope.py --mount carrier_v3 --assist
```

| Feature | What it provides |
|---|---|
| **Live sound map** | Acoustic agreement over the camera view, with weak/ambiguous/candidate states |
| **Evidence history** | Saved event metadata and, when available, the captured WAV and scene JPEG; explicit audio replay |
| **Semantic search** | Questions such as “When did the phone ring?” retrieve stored events; answers render their recorded facts |
| **Captions and watch alerts** | Speech transcription, approximate direction, and likely watched-sound matches |
| **Nearby context and outlines** | Models describe or outline the visible candidate selected by acoustic evidence |
| **Voice commands** | Spoken search and watch commands, through transcription and intent parsing |

**No image detector selects the sound direction.** Acoustic time differences supply the spatial
evidence. Visible depth surfaces constrain candidate locations; vision models describe those
candidates afterward. Neither that agreement nor an outline establishes the true sound source.

## Standalone setup

The main application uses a **reSpeaker XVF3800** with raw four-microphone capture and an **Orbbec
Gemini 2** with aligned color and depth. It needs no robot. Keep both sensors rigidly fixed relative
to each other. The microphone needs the **16 kHz six-channel firmware**; see `mic_test.py`.

| Setup | Command or profile |
|---|---|
| Current adjustable ring carrier, camera on its tripod | `python scope.py --mount carrier_v3 --assist` |
| Compact printed tripod stand | `python scope.py --mount tripod --assist` |
| Other existing stand designs | `--mount stand_v5` or `--mount stand_v3` |
| Microphone only, terminal output | `python soundscope.py listen --assist` |

The current `carrier_v3` profile assumes the board is upright, USB edge up, LED side facing the same
scene as the camera, with its centre approximately **130 mm above the color lens** and `mirror=false`.
Other offsets remain physical/CAD estimates. Select the profile matching the actual assembly.

**C** starts a three-position yaw calibration; **Enter** saves the fit. It corrects horizontal
angular alignment, not every dimension of the rig. It cannot repair wrong microphone orientation,
translation, camera tilt, depth alignment, or a flexible mount. Set camera pitch in the profile and
check left, centre and right with actual sounds after moving or reassembling the sensors.

Optional arm experiments remain in `ARM_SETUP.md`; they are separate from the standalone demo.
A wrist rig needs its own calibration and hardware verification. Autonomous pickup is not a
validated capability of this build.

## Run it

```powershell
pip install -r requirements.txt
python scope.py --mount carrier_v3                         # local viewer and memory; no automatic Meta calls
python scope.py --mount carrier_v3 --assist                # captions, labels, context and eligible outlines
python scope.py --mount carrier_v3 --assist --private      # no camera-image uploads
python scope.py --mount carrier_v3 --watch "dog barking" --meta
python soundscope.py enroll "doorbell"                    # record examples for local matching
python soundscope.py search "did the doorbell ring?"
python soundscope.py digest --minutes 60
python soundscope.py map                                  # remembered directions
python soundscope.py budget
python -m unittest discover                              # offline verification; no hardware or paid APIs
```

Meta features require `MODEL_API_KEY` in the project's `.env`, or `--env-file path/to/.env`.
Never commit that file. In Windows, the camera should use a USB 3 port. Keep other applications
that claim the camera or microphone closed.

For isolated recording takes with an explicit spending limit, see [the standalone video guide](DEMO_VIDEO.md).
`demo.ps1` keeps each take's memory separate without deleting previous experiments.

| Key | Action |
|---|---|
| **F** | Fullscreen |
| **M** / **H** | Change heatmap palette / show or hide it |
| **T**, then **Enter** | Type and submit a history question; Esc cancels |
| **B** | Browse saved events; Up/Down selects, B or Esc returns to live view |
| **E** | Explicitly replay the selected event's saved audio, when available |
| **A** in history | Request Meta analysis of the selected saved event; requires Meta enabled and available budget |
| **V** | Record a spoken command; this can use Meta credits |
| **L** | Show recent remembered sounds |
| **C** / **D** | Calibrate / diagnostics |
| **Space** / **S** / **Q** | Pause / save a diagnostic snapshot / quit cleanly |

The history view shows a **saved scene**, not a live camera fix. Its timestamp, scene/audio timing
information and availability matter. A scene JPEG is one captured frame, not an event video clip.
A replay may be normalized: playback volume does not reproduce the original sound pressure.

## What Meta contributes

| Model | Role | Limit |
|---|---|---|
| **Muse Spark 1.3** | Likely sound label from a spectrogram image, optionally alongside recorded reference examples | This application uses image input for sound interpretation; classifications can be wrong |
| **Muse Spark 1.3** | Nearby scene context and a comparison of the acoustic event with camera crops | Visual agreement is a model judgment, not source verification or proof of a hidden object |
| **Muse Spark 1.3** | Command interpretation and semantic retrieval of event IDs | History search discards invented IDs and renders factual answers from stored records |
| **Muse Voice Transcribe** | Spoken commands and captions with within-clip speaker turns | Speaker labels are not verified identities; spatial naming is approximate and can change |
| **SAM 3.1** | Point-prompt outline of an eligible visible candidate | Outlines an object boundary; does not locate sound independently |

Local reference matching labels resemblance to **your recorded examples**. It is not a trained,
universal sound recognizer and needs testing on different takes and competing sounds. A user-entered
speaker name or familiar example label does not establish a person's identity or a source's identity.

Meta is optional for localization. Semantic interpretation enriches the event history while the
microphone/depth pipeline continues locally. Search falls back to local retrieval when Meta is
unavailable. A missing match means “not found in recorded events,” not “that sound never occurred.”

## Estimated spending limits

Every API path reserves an **estimated** request cost before sending it. Pending reservations count
against both `META_SESSION_LIMIT_USD` (default **$0.50**) and `META_DAILY_LIMIT_USD` (default **$1.00**).
Daily accounting is shared across processes using the same usage ledger, with atomic file locking.
Accepted requests settle at reported usage when available; failed requests release their hold.
A reservation left by a crashed process remains held through that accounting day.

These are application estimates, **not an exact ceiling on Meta's invoice**: input-token estimates
and provider billing can differ. Typical development estimates were roughly $0.003–0.01 for a
label/context call, $0.0003 for a short caption and $0.0025 per outline; duration, tokens and model
pricing affect cost. Inspect `python soundscope.py budget` and the provider dashboard.

Background checks use `minimal` reasoning; commands and searches use `low`. Contributor tiers are
refused. Without Meta feature flags there are no automatic model calls; **V** explicitly requests a
voice interaction. `demo.ps1` sets zero limits for a free rehearsal.

Automatic enrichment also requires a recorded raw audio peak of at least **-50 dBFS** by default
(`SOUNDSCOPE_AUTO_META_MIN_PEAK_DBFS`). This is a credit-saving heuristic for the current microphone
gain, not a calibrated physical loudness threshold. Quiet accepted events can still be saved and
matched against local examples. In history, **A** explicitly requests analysis of a saved event,
bypassing that automatic peak filter while retaining the spending limits and its original timestamp.

## What is measured, and what is inferred

- **Direction:** SRP-PHAT over four raw microphones, weighted by frequency evidence above estimated
  room noise. Echoes, narrow tones, multiple sources and weak sounds can produce broad or wrong peaks.
- **Distance:** depth of a visible candidate surface. It is not an independent acoustic range measurement.
- **Loudness:** estimated signal-to-noise ratio or recorded dBFS. This rig is not calibrated in dB SPL.
- **Labels and nearby objects:** model or reference-example interpretations, with their confidence and provenance.
- **Alerts:** model matches require at least +6 dB estimated SNR and 0.75 reported confidence, or
  0.65 confidence at +20 dB or louder. The same watched alert is suppressed for 20 seconds. These
  thresholds are heuristics; model confidence is not a measured probability of correctness.
- **Memory:** accepted events, not every sound in the room. Filtering, event grouping, a busy worker,
  or unavailable evidence can omit events or media. Missing media are shown as unavailable.

A phone under a pillow may be audible, but seeing a pillow along the sound direction does not prove
“the phone is under this pillow.” History search suppresses unsupported hidden-object relationships.
It answers about past observations; it does not track a silent object after the recording.

## Earlier development observations

The following are **archived development observations from November 4**, with earlier geometry,
calibration and test conditions. They are not fresh accuracy measurements for the current assembly:

- A laptop speaker placed roughly 40° left at about 1 m produced reported directions of 37–42°.
  Earlier notes reported 5.4° mean angular error in a separate 1 m trial.
- Three noise bursts were recorded as three events; a ringtone as one; a sentence as two speech
  events; two claps as one grouped event. Event boundaries depend on pauses and the detector.
- Recorded ring, knock and name examples produced labels or alerts in some live trials. False
  labels also occurred, including noise interpreted as barking and claps interpreted as alarms.
- Some label/caption results arrived 6–13 seconds after a sound. Service latency and backlog vary;
  those numbers are not a real-time guarantee. Caption trials included synthesized voices.

Re-check current left/centre/right behavior on the assembled rig before presenting accuracy numbers.
The repository's automated tests cover software behavior and synthetic/recorded cases; they do not
prove general room-scale localization or sound classification accuracy.

## Files

| File | Purpose |
|---|---|
| `scope.py`, `scope_ui.py` | Live view, heatmap, calibration, voice and event integration |
| `history_ui.py`, `event_evidence.py` | Saved-event display and local WAV/JPEG evidence |
| `sound_memory.py` | Event records, grounded search, local examples and approximate speaker labels |
| `insight.py` | Background labeling, captions, visual context, outlines and watch alerts |
| `meta_api.py` | Model calls and atomic estimated-budget reservations |
| `soundscope.py` | Terminal search, digest, map, enrollment and microphone-only mode |
| `doa.py`, `mic_test.py` | Acoustic direction estimation and microphone checks |
| `camera.py`, `calibration.py`, `localization_quality.py` | Aligned depth/color, yaw fitting and uncertainty states |
| `profiles/`, `cad/` | Geometry profiles and available printable mount designs |
| `arm_follow.py`, `arm_target.py`, `aim.py` | Separate experimental robot adapters; not required for the demo |
| `test_*.py`, `test_data/` | Offline regressions and test fixtures |
| `DEMO_VIDEO.md`, `DEMO.md` | Standalone recording guide and short presentation checklist |
| `SUBMISSION.md` | Draft project description, Meta integration explanation and 90-second voiceover |

## Privacy

Local event metadata, saved audio/images, reference examples and the usage ledger are kept under
recording/data directories and are git-ignored. Meta-enabled features upload selected input needed
for that feature. **`--private` blocks camera-image uploads**; spectrogram images and requested
speech audio can still be sent for labeling and transcription.

Speech-like events skip camera enrichment, and recognized people are excluded from outlines.
These checks are heuristic. Camera images selected for upload receive best-effort face blurring;
angled or dim faces can be missed. Inspect the scene and choose private/local mode as appropriate.
No saved audio plays automatically; replay is explicit with **E**.
