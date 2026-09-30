# Isopod-model trilobite

A print-in-place trilobite built from three things:

- **The head:** the crescent head, layered onto the giant isopod's head piece.
- **The body:** the generator's default body, built the isopod way, with plates lying over the next piece's lip on an air gap.
- **The joints:** the isopod's own joints, copied 1:1 from its STL. That means its middle-bottom strip: column, ball with its slit and neck, socket, and prongs.

It prints as one file of 11 pieces (head, 9 thorax pieces, tail), in place, with no supports under the joints. The isopod STL is the one Claire uploaded on 27 Sep 2026, byte for byte (`assets/isopod/`). It is used with its owner's permission; see "Before sharing" below.

![overview](photos/overview.png)

## What's in the folder

| Path | What it is |
|---|---|
| `stl/trilobite_isopod.stl` | **The model.** All 11 pieces in place, in print mm, ready to slice. |
| `stl/pieces/NN_name.stl` | The same pieces one per file, for inspection. Print the whole file, not these: they are in place and interlock. |
| `photos/overview.png` | The model: 3/4 view, top, side, underside. |
| `photos/curl.png` | The model curled 0, 15 and 22° per joint. |
| `photos/joint_vs_isopod.png` | Our joint beside the isopod's, from the asset itself: from below, pulled apart, cut down the middle. |
| `photos/joint_in_context.png` | Two whole pieces beside the isopod's two. The parts outside the copied strip are ours and look different. |
| `checks.json` | Everything `check.py` measured (below). |
| `source/crescent_head_piece.stl` | The accepted crescent head ("perfect!", 27 Sep 2026): the isopod's head piece with the crescent layered on. Isopod size, work frame (midline x = 0, bed z = 0). |
| `source/isopod_crescent_head.json`, `source/isopod_poses_cache.npz` | The isopod joints' free ranges and pose grid. `crescent_head.py` uses them to keep the horns clear of the body. |
| `build.py` | One command builds everything here. |
| `crescent_head.py` | Makes the head piece (was `scripts/isopod_head.py`). |
| `body.py` | Builds the body and pastes in the isopod joints (was `scripts/iso_body.py`). |
| `check.py` | The checks. |
| `render.py` | The photos. |
| `presets.py` | Builds any preset and runs the enrollment test on it. |
| `heads_sheet.py` | The head alone for several presets, side by side (`photos/heads.png`). |
| `enroll.py` | The enrollment test on this model (instrument 2.1's procedure, this model's joints). |
| `readings/` | One reading per preset, plus `summary.json`. |

## Build

```
python isopod_model/build.py                  # ~7 min: stl/, checks.json, photos/
python isopod_model/build.py --rebuild-head   # also remakes source/crescent_head_piece.stl first
python isopod_model/check.py                  # re-run the checks on stl/pieces
python isopod_model/render.py                 # re-render the photos from stl/pieces
```

`render.py` is fast with `numba` installed and slow without it (plain Python).

## How it's made

### Size

The joints print at the isopod's own size: 18 mm between joints, ball radius 5.75 mm, centred 7.5 mm above the bed. The body is the generator's default animal, scaled so its segment pitch is 18 mm. That makes the model 173 × 318 × 46 mm.

### Body (`body.py`)

Each piece is drawn once, whole, from the generator's own anatomy surface:

- **Plate:** behind its rear joint, a piece is a plate 2 model-mm thick (4.5 mm printed).
- **Lip:** the next piece's front lies under that plate on an air gap. The gap is computed by swinging the lip about the joint to 30° and keeping it at least 0.5 model-mm (1.1 mm printed) under the plate the whole way.
- **Columns:** below the lip, the piece ahead ends in a column face, and the piece behind starts on the isopod's column face, 6 mm behind the ball.

### Joints

This is the isopod's middle-bottom strip, copied from the segment piece (piece 3): 15 mm either side of the midline, up to 20 mm above the bed. Each piece gets the socket half at its rear and the ball half at its front, placed on the joint. The ball therefore sits in the socket exactly as it does in the isopod. The isopod's 0.5 mm shorter pitch is absorbed in the plain column between joints.

### Head (`crescent_head.py`, then `body.py`)

- **The head piece:** the crescent head's relief is layered on the isopod head piece's own top surface. The horns ride at rim level beside the body.
- **Scale:** across (length and width) it is scaled 1.95×, so it stands to our first thorax piece as it stood to the isopod's. Its height is set separately (0.85 × the tallest thorax piece; see Parameters), 39 mm on the default.
- **Socket:** its socket is replaced by the same 1:1 isopod socket every other piece has, since the scaled one would be twice the ball's size. Nothing else on the head changes.

## Checks (`check.py`, latest in `checks.json`)

- **Pieces:** all 11 are watertight single bodies.
- **At rest:** nothing touches.
- **Curl:** with every joint bent the same angle, nothing touches up to 24° per joint. At 25° the tail, having come all the way round, meets the underside of the head.
- **Joints against the isopod file:** checked at all 10 joints, lined up on least-squares sphere fits to the balls. Every surface point inside the copied strip lies within 0.002 mm of the isopod's surface, measured both ways, on both the socket and the ball side. That is the STL format's own precision. Ball radius is 5.7500 mm in both, centred 7.5001 mm above the bed.
- **Range with whole pieces** (measured 28 Sep 2026):

| | Curl | Back-bend | Side to side | Twist |
|---|---|---|---|---|
| Isopod pieces 3/4 | 33° | 3° | 2° | 1° |
| Ours, thorax pieces 2/3 | 33° | 1° | 1° | 5° |

The curl is set by the isopod's columns, so it matches. The other limits come from our plates.

## Parameters: any preset builds

The model reads the same parameter set the generator does (`schema`, no schema change, no hash change). The body, the joint spacing and the print scale follow the preset. The head is always the crescent head, but these follow the preset (28 Sep 2026):

| On the model | Preset parameters | How |
|---|---|---|
| Face: eye bump, glabella | `eyeSize`, `eyePos`, `eyeHeight`, `eyeLat`, `glabInflate`, `glabRise` | The crescent's detail skin for that face (`head_crescent.FACE` was set from the default preset, so the default gives the approved face). The face sits nose to notch, as in the face and eye tests. |
| Eye solids | `eyeSolid`, and the eye keys `eyeArc`, `eyeSlope`, `eyeHeight`, lens size and gap, `eyeStalk`, `eyeShade`, `eyeLean` | The generator's own eye solid on the crescent's eye, at full size (never squashed). Sessile eyes sit at the lowest point under them; stalked eyes stand on their stalk. |
| Dome curve | `headDomeExp` | Heights above the rim are remapped: rim + (crown − rim) · t^(1.5 / exp). A larger exponent gives a fuller dome, a smaller one a peaked dome. The rim, the horns and the crown height do not move. |
| Head height | `headRelief` | The head's crown is `HEAD_HEIGHT` (0.85) × the tallest thorax piece × `headRelief`. The isopod's own head is 0.80 of its body. It was 1.24 before this change. |
| Spines on top | `occipitalSpine`, `headProngs` (head), `axialSpine` (every thorax ring, the spine that grows from the back), `termSpine`, `tailProngs` (tail) | The generator's own spine and prong solids, where the generator puts them, on the model's surfaces. The spines go on after the clearances, so nothing ever trims a spine: a spine that meets another piece through the curl is what stops the curl. A back spine stands on its ring's open top. Where the generator's spot is under the plate ahead it moves back along the ring, and a ring that is entirely under the plate ahead gets none; this happens to the first thorax ring, under the crescent head, and is recorded in the reading as `skipped_ornaments`. |

Behind its rear face the head is now a plate, like every other piece. The isopod head piece's cheek walls reached down to the bed behind its column, where our first segment's column stands, so everything behind the rear face lower than `WALL` under the head's own top is cut away. The horns, which are thinner than `WALL`, stay.

- **Pitch:** the joints stay 1:1 isopod, so the print scale is 18 / pitch.
- **Build a preset:** `python isopod_model/presets.py <name> [key=value ...] [--stl]`, for example `odontopleurida axialSpine=2.5`. The overrides are schema parameters set on top of the preset, and the reading is saved as `<name>+key=value`. Presets are `textured` (the generator's default) and the ten orders in `presets/*.json`, loaded as `sweep.py` loads them. `python isopod_model/presets.py` builds them all.
- **Clearance:** after the plates and lips are drawn, every piece is kept clear of the piece ahead, and of the head through the whole curl (horns). This only removes material where two pieces would meet. Any piece that ends up in more than one body keeps its largest body, and the rest is recorded as "dropped flakes".

All 11 build as watertight single pieces with nothing touching at rest (28 Sep 2026). One preset loses anatomy: in redlichiida the horns' clearance cuts the pleural spine tips off thorax piece 3 (86 model-mm³ dropped).

## The enrollment test on this model (`enroll.py`)

This runs instrument 2.1's own procedure and constants, unchanged: the rest baseline and censor, the 2.5° forward scan, 0.1° bisection, a collision as more than 0.5 mm³ of overlap growth on any pair, closure under 3 mm, and `classify()` verbatim.

Two inputs are the model's own rather than the pin build's:
- **Kinematics:** each joint turns about its ball centre, uniform flexion at every joint.
- **Geometry for s_tail:** the head's length (the crescent head's, not `cephFrac × length`) and the tail margin point (the tail's rear edge on the midline).

`instrument.py`, the pre-registration and the frozen references are untouched. These readings are separate and say so in every row (`kinematics = "isopod ball (isopod_model)"`). They live in `readings/<preset>.json` and `readings/summary.json`, with photos in `photos/presets/` and `photos/presets_all.png`.

| preset | segs | isopod model: limited by / class / θ per joint / total / s_tail | pin instrument (frozen reference) |
|---|---|---|---|
| textured | 9 | closed / spiral / 24.22 / 242.2 / −0.157 | none frozen |
| agnostida | 2 | anatomy / open / 40.16 / 120.5 / 0.632 (the ball joints' own stop) | closed / discoidal / 21.41 / 64.2 / 1.559 |
| asaphida | 8 | closed / spiral / 26.48 / 238.4 / −0.500 | anatomy / open / 20.70 / 186.3 / −0.023 |
| corynexochida | 8 | closed / spiral / 26.48 / 238.4 / −1.067 | closed / sphaeroidal / 22.58 / 203.2 / −0.016 |
| harpetida | 9 | closed / spiral / 27.89 / 278.9 / −0.114 | closed / double / 30.31 / 303.1 / 0.906 |
| lichida | 10 | closed / spiral / 22.03 / 242.3 / −0.601 | none frozen |
| odontopleurida | 9 | closed / double / 22.81 / 228.1 / 0.095 | none frozen |
| phacopida | 11 | closed / spiral / 19.92 / 239.1 / −0.187 | invalid (unsane parts) |
| proetida | 9 | closed / spiral / 24.22 / 242.2 / −0.271 | closed / double / 22.81 / 228.1 / 0.258 |
| ptychopariida | 12 | closed / sphaeroidal / 19.84 / 258.0 / 0.037 | invalid (unsane parts) |
| redlichiida | 14 | closed / double / 20.16 / 302.3 / 0.378 | invalid (unsane parts) |

Every closed reading is stopped by the tail meeting the head. Most classify as "spiral" (the tail margin ends past the head's front) because the crescent head is shorter than the schema's head: 30 model-mm here against about 45 for `cephFrac × length`.

## Before sharing

`assets/isopod/README.md` still needs the source link and the exact terms of the owner's permission: attribution wording, and whether derivatives and publication are covered. Record them before this model or its photos leave the lab.
