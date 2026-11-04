# Validation — 2026-11-04

## Later refinement check

The subsequent refinement suite ran **226 tests: 222 passed, four optional Viam
tests skipped**. JavaScript syntax and browser interaction checks passed.
The browser's Inferno selection and 65% opacity were confirmed in server state
and the rendered synthetic view; no model requests were made.

New tests reproduce two confidence failures: a thin full-height support region
could previously pass the area threshold, and a 6 kHz pure tone could have a
strong peak while similarly strong alternatives lay over 50 degrees away.
Per-axis support extent and whole-hemisphere angular-support checks now mark
these cases ambiguous. This changes eligibility for a precise target, not the
underlying peak direction. Calibration and SAM point selection reject unresolved
directions. No unmeasured direction is shifted toward the centre of the image.

A later real sensor check had healthy capture/HTTP samples and no runtime error,
but closed early after 27.7 seconds including setup; it did not complete the
requested 45 seconds. The check harness now distinguishes an early close from a
completed run. A user-positioned centred-ringtone check also collected numerical
diagnostics. Vertical scatter and background-surface selection remained; these
are **not** solved by the new uncertainty labels. No new measured localization
accuracy claim is made. All refinement checks used zero Meta requests.

Numerical live diagnostics now retain at most 5,000 updates locally, without
audio, imagery or transcripts. The older results below describe the preceding
checkpoint and remain useful historical validation.

These checks distinguish software operation from real-world localization accuracy.
The connected Gemini 2 and reSpeaker XVF3800 were exercised without robot motion
or audio playback. The original repository was not modified.

## Hardware integration

| Check | Measured result |
|---|---|
| Gemini 2 capture | 233 aligned RGB-D frames in about eight seconds, no missing frames, approximately 29 FPS |
| Resolution | 1280 × 720 color and aligned depth |
| Valid depth in this scene | Mean 77.3% of pixels; not a general accuracy specification |
| reSpeaker input | Six-channel WDM-KS stream at 16 kHz; all four raw microphone channels produced distinct samples |
| First integrated check after fixes | 35-second capture, clean exit, 12 retained sound segments |
| Final integrated check | 120-second capture, clean exit, 42 retained sound segments |
| Final dashboard monitoring | 60 ready samples, all passed the JPEG/freshness/bounded-media criteria |
| Three-request local HTTP round trip | Median 33.1 ms, 95th percentile 54.6 ms; this is not end-to-end localization latency |
| Maximum sampled frame age | 1.41 seconds across startup/run/shutdown monitoring |
| Meta calls during sensor checks | Zero |

The final run took 127.2 seconds including setup and cleanup. Its local report is
`studio_data/hardware_checks/20261104T071942Z/report.json`. Diagnostic data is
Git-ignored. Event counts are **sound segments, not correctly identified objects**.
Continuous background sound can produce multiple segments.

## Problems found and fixed

- Continuous sound previously kept updating the live view without ever reaching
  searchable history. It now finishes bounded segments about every three seconds.
- Completed localization uses bounded recent evidence and a matching audio clip.
  Quiet tails and later callback samples are excluded from that clip.
- Camera timeouts no longer discard completed audio events. Without a suitably
  timed RGB-D frame, history retains audio/direction only and omits spatial claims.
- Model buttons now check whether the required audio or original image still
  exists. Segmentation also requires a valid acoustic/depth candidate.
- A stopped server no longer leaves a stale LIVE badge and broken image. The
  dashboard shows DISCONNECTED, preserves the last good image and disables model
  actions. This was checked in the browser with the synthetic demo.
- Microphone discovery distinguishes limited process/device visibility from
  unknown firmware state, instead of recommending a firmware change prematurely.

## Three bounded live Meta checks

Exactly three requests were made under a three-request limit, using generated
speech and a generated image. No room audio, camera imagery or personal material
was uploaded. Attempts were recorded in the normal local daily budget ledger.

| Model | Input and observed result | Response time |
|---|---|---|
| Muse Voice Transcribe | Generated spoken search was transcribed correctly | 1.87 s |
| Muse Spark 1.3 | Converted that command into a find/search query | 4.30 s |
| SAM 3.1 | Point prompt selected the intended circle in a two-shape image; one mask returned | 2.30 s |

These are single integration checks, not quality benchmarks. Spectrogram sound
identification remains experimental and unverified. Nearby-object descriptions
and segmentation never prove which object emitted a sound. All Meta features
remain off by default, manually triggered when enabled, cached and request-capped.

## Regression checks

`python -m unittest discover -q`: **208 tests run, 204 passed, four skipped**.
The skips are legacy tests requiring the optional Viam SDK, which is not installed
in the tested system Python. JavaScript syntax checking also passed.

Coverage includes synthetic acoustic localization, long/moving sound segments,
audio timing/ring-buffer wrap, missing frames, stale evidence, search, credit
guards, response parsing, coordinate transforms and the local HTTP application.

## Not established by these tests

- Bearing error, source-identification accuracy or performance on quiet rings in
  the real room: a visible source at known independent positions is still needed.
- Correct yaw/translation/tilt for the final printed rig, or its mechanical fit.
- Real-room transcription or sound-label accuracy.
- Successful robot communication, safe motion, wrist hand-eye calibration,
  autonomous pickup or obstacle removal. Studio currently provides guarded robot
  coordinate previews only.

All capture processes were stopped after these checks. No unattended monitoring
or background model requests are left running.
