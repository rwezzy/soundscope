# SoundScope submission draft

**SoundScope — a searchable memory for the sounds you missed.**

Draft text for review and the submission form. Nothing in this file has been submitted or published.

## Project description

A phone rings while you are wearing headphones. Someone calls from across the room. By the time
you notice, the sound has stopped. SoundScope helps you see where an audible event came from and
later ask what happened.

A four-microphone array estimates acoustic direction, while a depth camera supplies visible
surfaces along that direction. A live heatmap shows the acoustic evidence. Meta models add likely
sound labels, speech captions and nearby visual context. Accepted events become searchable local
records containing when they happened, their estimated direction and level, and, when available,
the actual captured audio and a scene image.

Ask “When did the phone ring?” and inspect the matching record: its saved scene, time and replayable
sound. The design connects a physical measurement to an explanation you can check. It works as a
standalone sound-awareness system; a robot is not required.

## How Meta AI is used

- **Muse Spark 1.3** interprets spectrogram images to propose likely sound labels, adds context from
  eligible camera crops, understands commands and retrieves relevant event IDs for natural-language
  search. Search answers render facts from the retrieved records rather than accepting invented events.
- **Muse Voice Transcribe** turns spoken commands and captured speech into text for interaction and
  captions.
- **SAM 3.1**, optionally, outlines the visible candidate selected by the acoustic/depth pipeline.
  It helps explain the selected region; it does not choose the sound direction.

The application supplies Spark with a **visual representation of sound**, not native audio input.
Local signal processing supplies direction; Meta supplies interpretation and retrieval. Automatic
requests use an estimated spending budget, and saved events can be analyzed on demand.

## Why it matters

Most sound disappears before we can examine it. A visual indication helps in the moment, but a
memory helps answer the next question: “What did I miss?” SoundScope joins those two experiences.
Its useful distinction is the connection between the answer and the original recording: users can
inspect evidence instead of relying only on a generated label.

The motivation includes people who miss sounds while concentrating or wearing headphones, with
potential accessibility applications for people who cannot reliably hear their surroundings.
That is an intended direction for evaluation with users, not a claim of accessibility validation.
It aligns with SDG 10's aim of inclusion by exploring another way to access everyday information.

## What was built

The prototype combines a reSpeaker XVF3800, an Orbbec Gemini 2 and a rigid printed holder with a
Python live viewer. Its software includes acoustic direction estimation, depth-based candidate
mapping, uncertainty displays, event grouping, local WAV/JPEG evidence, semantic search, explicit
audio replay, captions, optional outlines and estimated API-budget reservations.

The main engineering challenge is keeping measurement, interpretation and memory consistent.
An outline is not proof that an object made the sound, and a saved location belongs to the saved
scene. The interface keeps those distinctions available while making the evidence easy to explore.

## 90-second voiceover and screen plan

| Time | Show | Voiceover |
|---|---|---|
| 0–10 s | The physical microphone/camera rig, then the live view. | “A phone rings while you're wearing headphones. By the time you notice, it has stopped. Where did it come from, and what did you miss?” |
| 10–28 s | Keep visible objects still. Clap on the left, then play a ringtone on the right. | “This is SoundScope. Four microphones measure acoustic direction. A depth camera maps visible surfaces along that direction. Watch the heatmap respond to the sound while the scene stays the same.” |
| 28–43 s | Show the actual returned label or caption and event time. | “Meta turns those observations into a likely sound description and nearby context. The highlighted surface is a candidate, rather than a claim that the camera can hear.” |
| 43–67 s | Press T; ask “When did the phone ring?” Open the matching saved event with B and replay with E. | “But the useful part is what happens afterward. I can ask about a past sound, open its saved scene, and replay the actual captured audio. The answer points to a record I can inspect.” |
| 67–80 s | If no doorbell event exists, ask about one; otherwise show another genuine query. | “Search uses recorded events. When it finds no match, it says so. It doesn't need to invent a sound or pretend that a silent object is still in the same place.” |
| 80–90 s | Hold on the evidence screen, then the physical rig. | “SoundScope combines local acoustic measurement with Meta interpretation: a visual, searchable memory for the sounds you missed.” |

Use results actually captured in the take. Leave room for model latency, or shorten the narration
when replaying audio. The saved evidence is **WAV audio plus a scene JPEG**, not an event video clip.
The recording guide and launch commands are in [DEMO_VIDEO.md](DEMO_VIDEO.md).

## Review before submitting

- The live rehearsal verified saved-image history, audio replay, a Meta transcript and a search
  response. Single-microphone replay was reported clearer than the old channel average. A song
  used as a ringtone was classified as speaking; this did not establish ringtone recognition.
  Simple ringtone classification and a positive ringtone-history search remain unverified.
  Verify those features in the final take before claiming them.
- Report any accuracy measurements with their actual test conditions. Do not present archived
  measurements as a benchmark for the current sensor assembly.
- Keep the standalone system as the demonstrated scope. Autonomous robotic pickup and reliable
  identification of an obscured sound source are not demonstrated claims.
