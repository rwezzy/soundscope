# SoundScope (Meta hackathon)

**Idea:** ask for a sound ("find the jingling keys"). Live video shows where it's coming from, the object gets outlined, and you can play the isolated sound.

**Due: Wed Nov 4, 9 PM ET.** Aim to submit by 6 PM. Needs a public repo and a 3-minute narrated video. All code must be new.

## Pipeline
1. **Capture:** the XVF3800 4-mic array records 4 raw mic signals on one shared clock.
2. **Pick the sound:** SAM Audio separates the sound you asked for (a text prompt). It's hosted on fal; check it's allowed and counts as Meta tooling before relying on it.
3. **Localize:** signal processing on the *original* raw signals (GCC-PHAT / SRP-PHAT), guided by the separated sound. Never take direction from AI-generated audio.
4. **Show:** a direction band over live video, with SAM 3.1 outlining the object inside the band.
5. **Optional extras:**
   - Muse Spark 1.2 audio: "I hear …" suggestions with no training
   - Muse Voice Transcribe: ask out loud
   - the Lite 6 points at the source (stretch goal)

**Plan B** (if SAM Audio isn't allowed or doesn't work): steer the array toward each direction with plain signal processing, send each direction's audio to Muse Spark 1.2 and ask "is this keys?", and take the best match. This uses only provided models, and it's also the baseline to beat.

**Checkpoint (Nov 1):** the model-guided direction must beat the plain, unguided direction. If it doesn't, ship Plan B.

## Hardware
- reSpeaker XVF3800 USB 4-Mic Array (bare board). Lay it flat near the webcam, with one edge facing the way the camera faces.
- Flash the 6-channel firmware `respeaker_xvf3800_usb_dfu_firmware_6chl_v2.0.x.bin` with `dfu-util -a 1`. Use the USB-C port near the 3.5 mm jack.
- Channel map: 0–1 are processed, **2–5 are raw mics 0–3**. Audio is 16 kHz, 32-bit.
- The chip also reports its own direction estimate (a fallback).
- Wiki: https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/

## Schedule (shift as needed)
| Day | Build | Done when |
|---|---|---|
| Before the array arrives | Direction math on fake signals and phone recordings; video overlay; test SAM Audio on a phone clip of keys over music | Overlay points the right way on fake data |
| Array day | Flash the firmware; tap each mic to confirm the channel order; capture one sound source | Direction matches where you put the source |
| Next day | Combine separation + localization (the checkpoint) | Beats or ties the baseline, or Plan B chosen |
| Nov 3 | Voice ask, SAM 3.1 outline, numbers (angle error, hit rate) | Numbers in the README |
| Nov 4 | Video, repo check, submit by 6 PM ET | Submitted |

## FPGA
Not in this build. Its one real job here would be recording many mics on one shared clock, which the XVF3800 already does. Saved for v2: 8 PDM mics on a Tang Nano 9K with a wider spread.

## Credit
- Prior work on text-queried sound localization (EUSIPCO 2024)
- UW research on sound awareness for deaf and hard-of-hearing people
- Acoustic cameras in general
