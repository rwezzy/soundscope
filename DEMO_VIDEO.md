# Standalone submission video

Use the Gemini 2 and reSpeaker on their fixed holder. This demo does not connect to the robot.
Keep the stand still throughout the take so saved sound positions refer to the same scene.

## Start

Open PowerShell in this folder. For a free rehearsal:

```powershell
.\demo.ps1 -Record
```

For the real take with sound labels, nearby visual context, captions and eligible SAM outlines:

```powershell
.\demo.ps1 -UseMeta -Record -EnvFile 'C:\Users\danny\Developer\hackathons-fall-2026\meta-soundscope\.env'
```

Each launch creates a separate history without deleting previous recordings. The Meta take uses
an estimated $0.20 session limit and $1 daily limit. Free rehearsal sets both to zero. The key file
is read locally; its contents are not copied. Press Q to close the recording cleanly.

The built-in MP4 is **silent**. Film a wide shot with your phone, or use a screen recorder with
microphone capture, to include the actual ringtone and narration. Do not put a prerecorded successful
overlay over an unrelated sound. Cuts between real tests are fine; preserve the actual results.

## 75–90 second sequence

1. **0–10 s: problem and rig.** Show the camera and four microphones. Say: “A sound can matter even
   when you miss it. SoundScope shows where sounds may be coming from and lets you ask what happened.”
2. **10–25 s: live localization.** Make two separate sounds at visible left and right positions. Show
   the live acoustic overlay. Press M once to show another palette. Say: “The microphones measure the
   direction. The camera provides visible surfaces and depth; the colour shows acoustic agreement.”
3. **25–45 s: phone ring.** Put a visible phone or Bluetooth speaker at one position and play a ringtone
   for 3–5 seconds. Stop and leave it still while the label arrives. Say: “Meta adds a likely sound
   label and nearby visual context to the measured event.” If the label is wrong, retry honestly;
   don't present it as a successful identification.
4. **45–65 s: search.** Press T, type `When did my phone ring?`, then Enter. Show the history result
   and remembered position. Say: “I can ask about past sounds, with their recorded time and location.”
   Optional: press V and ask the same question aloud; allow time for transcription and the answer.
5. **65–85 s: distinction and close.** Say: “The localization runs locally. Muse Spark helps interpret
   and retrieve events; Muse Voice Transcribe handles speech; SAM outlines a candidate when sound and
   image evidence agree. The robot is an optional future application of the same spatial output.”

Use only features actually visible in your recording. An outline or caption is optional; don't spend
the deadline waiting for every model to appear in one take. Allow roughly 5–15 seconds for a model
result, but actual latency varies. The recording can be cut for length without changing the outcome.

## Controls

- M: cycle amber, turbo, inferno, viridis, plasma. H: show/hide the heatmap.
- T: type a history question. Enter submits; Esc cancels. Questions are local unless a Meta session
  is enabled; Meta failure falls back to local retrieval.
- V: record a spoken command. L: last remembered sound. S: evidence snapshot. Q: finish.
- C: three-position calibration. Only change calibration if left/right is wrong; the imported ring
  profile includes the working version's mirrored orientation. Saving yaw does not calibrate all geometry.

## Claims to keep accurate

- Search reports saved observations, not a current search for a silent missing phone.
- A depth reading belongs to a visible surface consistent with the sound. It is not acoustic range.
- A phone ringing under a pillow can be detected acoustically, but the system cannot claim “under
  this pillow” merely because a pillow is visible. For the deadline, demonstrate a visible ringing source.
- Loudness is relative to noise or digital full scale; this setup is not calibrated in dB SPL.
- When evidence is weak or broad, show that state. Do not turn uncertainty into an exact source claim.
