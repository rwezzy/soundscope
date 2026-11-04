# Gemini 2 / reSpeaker stationary sensor carrier v2

**Stationary tripod prototype. Physical fit and retention remain unverified.
This is not a wrist mount or a moving-robot attachment.**

This carrier holds an Orbbec Gemini 2 and a bare Seeed reSpeaker XVF3800 together
on the camera's original tripod. It uses that tripod's existing metal screw and
printed tightening screws and retainers; no additional purchased fasteners are
specified. The earlier fixed-clearance holder failed its fit check. That does not
establish that this revised design fits: support the devices while checking it.

## Print project

Open **SoundScope_SV06_PLA_FULL.3mf** as a project in PrusaSlicer. It contains all
twelve pieces, layout and settings for the **original Sovol SV06, 0.4 mm nozzle,
PLA**. Follow [PRINT_ON_SV06.txt](PRINT_ON_SV06.txt), confirming the actual printer,
nozzle and filament before slicing. The saved settings are not universal presets.

The checked slice estimates **3 h 30 min 29 s and 31.4 g**. That is a slicer
estimate, not measured print time. The supplied model footprint is approximately
205.40 × 204.82 mm; the saved 5 mm brim fits within the 220 × 220 mm bed.

Alternatively, `print_files/FULL_CARRIER_SV06_220MM.stl` contains the same twelve
pieces without printer settings. Slice it with the actual machine's profile at
100% scale. **Do not print both the 3MF and STL.** No machine G-code is distributed.

## Assembly and verification

Follow [READ_BEFORE_PRINTING.txt](READ_BEFORE_PRINTING.txt) for assembly, and see
the labelled [assembly diagram](assembled_tripod_diagram.png). Tighten printed
screws gently by hand. They use custom printed threads, not standard metal
fasteners. Neither sensor should wobble, slide or contact glass, microphone
inlets, electronic components or USB plugs. Actual USB overmolds and tripod-head
clearance still need checking. A mesh-intersection check cannot establish grip,
plastic strength or safe operation.

Both sensors must remain rigid relative to one another. Measure their actual
positions and redo microphone-to-camera calibration after assembly; the print
does not update SoundScope's software offsets automatically.

## What this package establishes

- `sv06_packing_checks.json` records unchanged part geometry and bed placement.
- `sv06_slice_checks.json` records the checked 3MF hash, estimated time and slice
  bounds, with `physical_fit_tested: false`.
- The camera still mounts directly to its original tripod screw. There are no
  robot-flange holes in this carrier.

This is an **output-only print package**. Generator scripts are not included
because they currently depend on separately obtained manufacturer CAD meshes,
component-bound data and local geometry helpers. This package is sufficient to
open and slice the delivered geometry, but does not claim a self-contained CAD
rebuild. Older holders, redundant plates, fit coupons and obsolete fourth-clip
variants are deliberately excluded.

The independent wrist prototype is not released here. Its bolt engagement,
sensor retention, payload balance and moving-joint clearance remain unresolved.
