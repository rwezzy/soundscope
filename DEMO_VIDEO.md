# Standalone submission video

**Story: a sound happened, you missed it, and now you can ask what happened and inspect the evidence.**
Use the Gemini 2 and reSpeaker on their fixed holder. The robot is not part of this take.
Keep the sensors still so the sound map and saved scene share a consistent reference frame.

## Start a separate take

Open PowerShell in this folder. Free rehearsal:

```powershell
.\demo.ps1 -Record
```

Meta-enabled take, with the project's existing key file:

```powershell
.\demo.ps1 -UseMeta -Record -EnvFile '.env'
```

Use the actual path to your existing `.env` if it is elsewhere. Do not put a key into the command
or show the file in the recording. Each launch creates a separate history without deleting earlier
takes. The key file is read locally; its contents are not copied.

The Meta take sets an **estimated $0.10 session limit and $1 daily limit**. Free rehearsal sets both
to zero. Pending requests reserve part of that allowance; the estimate is not a guaranteed provider
billing cap. If credits are exhausted, demonstrate local localization and existing evidence rather
than increasing the cap during a take.

The built-in MP4 is **silent**. Capture a phone wide shot or screen-recorder microphone track for the
real sounds and narration. Separately, event history can save the actual captured **WAV audio and a
scene JPEG**. A JPEG is a still frame; the event browser does not promise a recorded video clip.
Press **Q** to close and finish saving cleanly.

## Prepare the scene

1. Use `carrier_v3` only with the actual matching assembly: board upright, USB edge up, LED side
   toward the scene, centre about 13 cm above the color lens. Its current profile has **mirror=false**.
2. Keep the camera/microphone holder stable. Check actual sounds at left, centre and right. If the
   horizontal alignment is wrong, **C** collects three positions and **Enter** saves yaw. This does
   not calibrate every offset, tilt or depth error.
3. Put a visible phone or Bluetooth speaker to one side. Leave several ordinary objects stationary
   in view. Keep this first demonstration free of overlapping sounds and hidden-source claims.
4. Press **F** for fullscreen. Pick a readable palette with **M**, a style with **X**, and opacity
   with **[ / ]** or the on-screen buttons. Classic outline + gradient is the default; contours leave more of the image
   visible. These settings do not improve measured accuracy. Ensure the source, overlay, event
   labels and text are visible in the recording. Use the phone wide shot to show the physical rig.
5. Rehearse one genuine ringtone event and inspect it with **B**. Confirm the saved scene and audio
   exist before relying on them in the presentation. Missing evidence must remain labelled missing.

The decisive contrast is **same visible scene, different sound direction**. Show the map responding
to the sound starting, changing location and stopping, rather than suggesting that a recognizable
phone in the picture caused the program to choose it.

## 90-second sequence

| Time | Do | Suggested narration |
|---|---|---|
| **0–10 s** | Show the fixed microphone/camera rig, then the live view. | “A sound happens while you are distracted. Where did it come from, and what did you miss?” |
| **10–30 s** | With the scene stationary, clap on the left; then play a ringtone from the visible source on the right for 3–5 seconds. Stop it. | “Four microphones measure acoustic direction. The camera adds visible surfaces and depth. The overlay shows agreement with the sound.” |
| **30–45 s** | Leave the source still while its likely label or caption arrives. Show the event time and uncertainty. | “Meta adds a likely interpretation and nearby context. This surface is a candidate; an outline alone cannot prove it made the sound.” |
| **45–70 s** | **T**: `When did the phone ring?` → Enter. Open the matching event with **B**, inspect the saved scene, and press **E** for a short audio replay if available. | “Now I can ask about the past and check the actual recording: what was heard, when, and where the evidence pointed.” |
| **70–82 s** | If no doorbell event exists in this take, ask `Was there a doorbell in the last five minutes?` | “It can also say there is no matching recorded event. It does not have to invent an answer.” |
| **82–90 s** | Return to the live view or hold the evidence screen. | “Local acoustic localization, Meta interpretation, and a memory you can verify. SoundScope works on its own.” |

A search may already select or expose the matching event; use **B** to enter history if necessary.
**Up/Down** selects an event. Saved candidate pixels belong to the saved scene, not to whatever is
currently in front of the camera.

Allow real response time. Earlier development calls sometimes took 5–15 seconds; this is not a
latency guarantee. Trim waits for the final edit without changing the event/answer pairing. Do not
place a successful overlay over an unrelated sound. If a label is wrong, show the uncertainty or
make a new honest attempt. An optional SAM outline or voice query is useful only if it actually works
in the take; the evidence-backed history is the central demonstration.

## Controls during the take

If recorded sound is unclear, close the viewer and run `python audio_check.py --seconds 900`.
Press Space, keep quiet for two seconds, then play the test sound for ten seconds. Press **1**
to replay one selected raw microphone and **2** for the old unaligned channel average. Both
use the same playback gain. The six-channel original and diagnostics stay in the ignored
`recordings/audio_checks/` folder; this check makes no API calls.

The latest simple-ring check produced “telephone ringing” from a saved spectrogram with numeric
measurements. An earlier call on the same clip returned speech when given a misleading local
“speech-like” hint; those hints are now excluded. This is one debugging example, not a measured
recognition rate. Rehearse the **live event → saved evidence → search** sequence separately.
If the event is saved without automatic analysis because its original peak is below the digital
threshold, **A** in history requests analysis under the existing budget. Show that interaction
honestly rather than presenting an explicit request as an automatic result.

- **F:** fullscreen. **M:** palette. **H:** heatmap visibility.
- **T:** type a history question; Enter submits, Esc cancels. With Meta disabled, retrieval is local.
- **B:** saved-event history; Up/Down selects; B or Esc returns to live view.
- **E:** explicitly play the selected saved audio when available. Playback loudness may be normalized.
- **A** in history: explicitly request Meta analysis of that saved event (requires Meta enabled).
  Automatic analysis skips recordings below the configured raw-peak threshold, initially -50 dBFS
  for this microphone gain. This is a credit-saving heuristic, not a physical loudness threshold.
  Quiet accepted events are still saved, and A lets you request analysis without changing their timestamps.
- **V:** spoken command, using transcription and model interpretation when credits permit.
- **L:** recent sounds. **S:** diagnostic evidence snapshot. **Q:** finish and save cleanly.

## Keep these distinctions visible

- **Direction versus source:** the acoustic map estimates direction; depth belongs to a visible
  candidate surface. Neither is proof that a particular object produced the sound.
- **History versus tracking:** search retrieves recorded events. It does not locate a silent missing
  phone or establish its current position after it has moved.
- **Saved media versus live video:** history uses captured WAV audio and a scene JPEG, with timing
  and availability metadata. Do not describe a scene frame as an event video clip or exact synchronization.
- **Labels versus certainty:** Meta confidence and personal-example similarity are not validated
  recognition probabilities. Speaker letters/names are approximate associations, not identity verification.
- **Loudness:** SNR and dBFS are not calibrated dB SPL. Replay volume is not the original sound level.
- **No match:** “no matching recorded event” does not establish that nothing happened.
- **Privacy:** `--private` blocks camera-image uploads; spectrogram images and speech can still be
  uploaded for enabled features. Keep consented subjects and suitable surroundings in the shot.
