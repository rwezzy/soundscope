# SoundScope presentation checklist

The current demonstration is **standalone**: Gemini 2, reSpeaker and the laptop. Use the
[90-second recording guide](DEMO_VIDEO.md) for commands, controls and the full shot list.
No robot integration is required for this presentation.

## One clear claim

> SoundScope turns acoustic events into a visual, searchable history, so you can ask what happened
> and inspect the captured sound and scene behind the answer.

The microphone array supplies directional evidence. Camera depth identifies a compatible visible
surface. Meta adds likely labels, captions, visual context and semantic retrieval. Keep those roles
distinct: the camera is not used to guess which recognizable object must be making noise.

## Before recording

- Check the actual fixed rig and profile: the current carrier is forward-facing, USB edge up,
  microphone centre about 13 cm above the lens, `mirror=false`.
- Test left/centre/right once. Three-position calibration adjusts yaw only; do not assume it fixes
  incorrect dimensions, camera pitch, orientation or room reflections.
- Start a separate take with `demo.ps1`; earlier recordings are preserved. Use a free rehearsal first.
- Make one clear ringtone event. Open history with **B** and verify saved WAV audio and scene JPEG
  availability. Replay is explicit with **E**; missing media must not be presented as captured evidence.
- Screen video is silent. Record the real sounds and narration with a phone or screen-recorder audio.
- Keep the measured scene, sounds, overlays and history results from the same genuine test.

## The sequence

1. **Problem and rig:** “A sound happened while I was distracted. What did I miss?”
2. **Live evidence:** clap left, then ring the visible source on the right. The scene remains still
   while the acoustic map responds. Show a weak/ambiguous state honestly if one occurs.
3. **Interpretation:** allow the likely label or caption to arrive. Model latency varies.
4. **Question:** press **T**, ask “When did the phone ring?”, and submit.
5. **Verification:** show the matching event's time, saved scene and audio replay. The highlighted
   location is historical and the visible surface remains an unverified source candidate.
6. **Grounded absence:** if appropriate for the actual recording, ask about a sound that has no
   matching event. Distinguish “no record” from “never happened.”
7. **Close:** local localization, model interpretation, and evidence you can revisit.

Use **F** for fullscreen, **M** for heatmap colors, **B** for history, **E** for explicit replay,
**T** for typed search, **V** for spoken commands and **Q** to finish cleanly.

## Optional additions

If already working reliably, show a caption with approximate talker direction, a watched-sound
alert, or a SAM outline. They do not all need to appear. A model label is tentative; a speaker label
is not verified personal identity; an outline is not proof of the source. Recorded reference
examples demonstrate similarity to those examples, not universal recognition.

Model alerts currently require +6 dB estimated SNR and 0.75 reported confidence, or 0.65 at +20 dB
or louder, with a 20-second repeat suppression. These are prototype heuristics, not a life-safety
alarm guarantee. Avoid making an emergency-detection claim in the pitch.

## When something is unavailable

- **Direction is wrong:** inspect orientation/profile and calibration; do not hide it with a guessed label.
- **No event:** reduce competing noise and repeat a visible, clearly audible source at a reasonable level.
- **Slow labels:** wait or trim the real waiting time in editing. Do not substitute another event's result.
- **Budget pause:** preserve localization and local search. Check estimated spend and pending reservations;
  do not automatically raise limits. A complete demo has no guaranteed fixed API cost.
- **Missing camera:** microphone-only terminal mode remains available, but it has no saved scene or depth.
- **No saved media:** show the metadata as such; do not claim a captured clip or frame that is absent.

For the Meta-enabled take, `demo.ps1` sets estimated limits of $0.10 per session and $1 per day.
These are application estimates, not an exact provider billing cap. `--private` means no camera-image
uploads; enabled labeling/transcription may still send spectrogram images or speech audio.
