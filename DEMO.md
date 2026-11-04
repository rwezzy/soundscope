# Demo video: 2-3 minutes

One idea per shot. Screen-record the viewer (it has no audio track) and film the room with a phone
at the same time, so claps and voices are heard.

## Before recording (10 min)

- **Lights on.** The color camera needs light; in the dark SoundScope switches to depth shapes
  ("night mode"), which works but looks less impressive.
- Tripod mount on the table, camera facing the area you will act in. Calibrate once:
  `python scope.py --mount tripod`, press **C**, play a sound left / center / right and click it,
  **Enter**.
- Record 2-3 examples of a sound you will use, so it gets a free, instant label:
  `python soundscope.py enroll "phone ringing"` (play the ring after each Enter).
- Start fresh: move or rename today's file in `recordings/memory/` if it has test sounds in it.
- Check the budget: `python soundscope.py budget` (a full demo uses well under $0.50).

## Shots

| # | Time | Do | Say (or caption) | Shows |
|---|---|---|---|---|
| 1 | 0:00-0:15 | Wide shot of the rig, then the viewer | "SoundScope finds where a sound came from, shows the object, and remembers it." | The rig, the dashboard |
| 2 | 0:15-0:40 | `python scope.py --mount tripod --meta --watch "phone ringing"`. Clap on the left, then the right. | "Four microphones find the direction with physics, not a model. The depth camera shows the spot." | Heat map jumps to each clap, distance in meters |
| 3 | 0:40-1:05 | Ring a phone lying on the table. | "Muse Spark looks at the sound's spectrogram AND at the spot the mics picked, before and after. It names the object and checks that sound and sight agree." | Caption "phone ringing: a phone (sound + sight agree)", SAM 3.1 outline around the phone, **ALERT: phone ringing** |
| 4 | 1:05-1:30 | Two people talk from different sides (or one person moves). | "Muse Voice Transcribe writes down who said what; the mics name each talker by where they stand." | Captions "Speaker 1 (to the left): ...", "Speaker 2 (to the right): ..." |
| 5 | 1:30-1:55 | Press **V**: "When did the phone ring?" | "The memory answers from what was measured: when, where, how loud, what was near." | The answer appears in Field Notes, with a marker on the picture where it rang |
| 6 | 1:55-2:15 | Terminal: `python soundscope.py digest --minutes 5` and `python soundscope.py map` | "A summary of everything heard, and a map of where it came from." | Digest text, sound map picture |
| 7 | 2:15-2:40 | (Arm, if printed) `python arm_follow.py --mount wrist listen` (dry run, or `--live` with E-stop in hand). Clap to one side. | "On the robot, the camera and mics ride on the gripper. It turns toward the sound with no camera calibration." | Arm turning, or the printed J1 plan |
| 8 | 2:40-2:55 | Back to the viewer | "No image detection decides where a sound came from. Physics picks the spot; Meta models describe it." | Closing frame |

## If something goes wrong

- Nothing on screen when you clap: clap louder or closer; a fan or heater right after start-up is
  absorbed within ~3 s.
- Labels late: Muse Spark takes 3-8 s per sound; do one sound at a time with a pause.
- "Meta features paused": the daily budget is used up; raise `META_DAILY_LIMIT_USD` in `.env`.
- Voice not understood: speak toward the board, short commands ("find my keys").
