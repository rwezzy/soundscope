# Demo video: 2-3 minutes

The story: **sounds you can't hear, made visible.** One idea per shot. Screen-record the viewer
(it records no audio) and film the room with a phone at the same time, so the sounds are heard.

## Before recording (10 min)

- **Lights on**, camera and mic board on a stand (the compact stand: `--mount tripod`; the new
  size-tolerant v5 stand: `--mount stand_v5`), on a desk or the floor, facing the room. The commands
  below use `--mount tripod`; swap in your stand's name. New v5 print: test-fit it on your tripod
  once (heads wider than ~46 mm were not checked against its floor bumps).
  Plug both into the laptop (the camera into a USB 3 port). If SoundScope says "No Orbbec camera
  is plugged in", replug it.
- Calibrate once: `python scope.py --mount tripod`, press **C**, clap left / center / right and click
  each spot while it sounds, **Enter**. Quit with **Q**.
- Optional, makes alerts instant and free: `python soundscope.py enroll "doorbell"` (play your doorbell
  or a doorbell sound after each Enter). Same for "smoke alarm".
- Start fresh: rename today's file in `recordings/memory/` if it has test sounds in it.
- `python soundscope.py budget`: a full demo uses well under $0.50 of the $1 daily cap.

## Shots

| # | Time | Do | Say (or caption) | Screen shows |
|---|---|---|---|---|
| 1 | 0:00-0:15 | The rig, then the viewer | "If you can't hear a sound, you also can't tell where it came from. SoundScope shows you." | The stand, the dashboard |
| 2 | 0:15-0:35 | `python scope.py --mount tripod --assist --name Keith`. Clap on the left, then the right. | "Four microphones find the direction with physics, not a model; the depth camera shows the spot." | Heat map jumps to each clap, distance in meters |
| 3 | 0:35-0:55 | Play a smoke alarm sound from a phone on one side of the room. | "Muse Spark recognizes it from its spectrogram and SoundScope says where it is." | Red banner **ALERT: smoke alarm (to the left)**, beep |
| 4 | 0:55-1:20 | Someone says "Keith, dinner's ready" from the other side. | "Muse Voice Transcribe writes it down; the mics say who and where. Your name raises an alert." | Caption at the talker, **ALERT: Speaker 1 said "Keith" (to the right)** |
| 5 | 1:20-1:40 | A phone rings on the table. | "Muse Spark looks at the sound and at the spot the mics picked, before and after; SAM 3.1 outlines the object." | "phone ringing: a phone (sound + sight agree)", outline around the phone |
| 6 | 1:40-2:00 | Press **V**: "Did anyone knock while I was away?" | "Every sound is remembered: when, where, how loud. Ask in plain words." | Answer in Field Notes, marker where it happened |
| 7 | 2:00-2:15 | Terminal: `python soundscope.py digest --minutes 10`, then `python soundscope.py map` | "A summary of what happened, and a map of where." | Digest text, sound map |
| 8 | 2:15-2:35 | (Optional) Arm with the wrist mount: `python arm_follow.py --mount wrist_v4 listen` (dry run, or `--live` with the E-stop in hand). Clap to one side. | "The same directions can turn a robot toward a sound." | Arm turning, or the printed turn plan |
| 9 | 2:35-2:50 | Back to the viewer | "No image detection decides where a sound came from. Physics picks the spot; Meta models explain it." | Closing frame |

## If something goes wrong

- Nothing appears when you clap: clap louder or closer. A fan or heater switching on is absorbed as
  background within ~3 s.
- Labels arrive a few seconds late: Muse Spark takes 3-8 s per sound. Pause between sounds.
- "Meta features paused": the daily budget is used up; raise `META_DAILY_LIMIT_USD` in `.env`.
- No camera at hand: `python soundscope.py listen --assist --name Keith` does alerts and captions
  with the mics alone (terminal output, no picture).
