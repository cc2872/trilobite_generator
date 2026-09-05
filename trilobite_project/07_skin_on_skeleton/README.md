# Skin on skeleton — and the "would Claire hate this" pass

**Architecture:** your `schema.py`+`fields.py` are the skeleton (structure + every parameter); the
furrow-cut Olenoides is laid on it as skin by your `wrap.py` law, segments resampled to `segCount`.
One animal. `skin_on_skeleton.py` (Python, exports STL) and `skin_viewer.html` (your `index.html`
chain, live) share the same logic.

## What I got wrong on the first pass, weakness-first
1. **Dropped the legs.** Restored: real legs split per segment, ride with their plate.
2. **Overclaimed "P0 reproduces the sculpt unchanged."** Head/tail are exact; the thorax is re-spaced
   **0.63–1.40×** because the sculpt's furrows are unequal and your skeleton is uniform-pitch by
   design. Stated, not hidden. (The 14 mm segment is likely a missed furrow — a cut issue.)
3. **`hinge_z` sat above the shell.** I double-counted `axisRise`; the sculpt's relief already includes
   its ring. P0 now uses `axisRise=0` → hinge 34.6 mm inside a 40.8 mm shell.
4. **Magic number** `overlap·0.35` stretched the skin. Removed; the shingle flap is a hinge-stage feature.
5. **P0 outline was schema defaults.** Now fitted from the sculpt's own profile (your `P0_from`).
6. **Instrument never run.** Now run — and it found the real bug (below).

## The instrument found the original false verdict — and its cause
First run: `e_max = 0.0`, `touching_at_zero = True`, every adjacent pair colliding — the *same* false
"cannot enroll" you started with. Diagnosis (measured, not assumed): pieces **interleaved ±6 mm across
every seam**, because adjacent pieces share one curved furrow seam but I remapped each piece's y with
its *own* scale, so the two copies of the seam diverged at the sides. Fix = your `resample_segments`
rule ("neighbours must not interleave; front wins") + your `clearance`, applied vertex-wise so the gap
is real. Contact map: `contact_where.png`.

## Result — your `instrument.measure`, unmodified (3-line `trilobite.py` shim)
| legs | clearance | e_max | free curl | closure gap | class   | stopped by | touching at 0 | print valid |
|-----:|----------:|------:|----------:|------------:|---------|-----------:|:-------------:|:-----------:|
| off  | 0.3 mm    | 0.078 | 11.2°/144°| 54.7 mm     | partial | 3–4        | no | yes |
| off  | 0.8 mm    | 0.184 | 26.4°/144°| 47.7 mm     | partial | 0–1        | no | yes |
| on   | 0.3 mm    | 0.078 | 11.2°/144°| 54.3 mm     | partial | 3–4        | no | yes |
| on   | 0.8 mm    | 0.184 | 26.4°/144°| 46.4 mm     | partial | 0–1        | no | yes |

Across the two clearances tested, e_max rises with clearance, and the limiter at default clearance is the
mid-thorax pair **3–4** — the same pair your very first screenshot named, now on whole spines with no
rest contact. Magnitude is clearance-dependent and is reported as such. Legs don't change the verdict at either clearance (four runs).

## Second pass against the "hate" list (this revision)
- **Reinvention:** I had re-implemented `P0_from`, `profiles`, `transforms`, `collisions`, `e_max`,
  `measure`. Deleted. Your `wrap.py` and `instrument.py` now run unmodified via a 3-line shim; your
  `print_validity` runs too (passes). What remains mine: segment tiling + open-shell trim.
- **Extras:** removed the procedural body, the old viewers, the 3-species bakes, the superseded rig.
- **Overclaims:** "robust" softened to "two clearances tested"; legs claim now backed by four runs.

## Third pass: connected + parametric (verified, not asserted)
**Connectivity.** The head is **1 component** (4028 faces) after dropping 3 debris faces
(`clean_piece`, slivers < 8 faces). Segments carry 3–7 components each: their pleural spine tips are
*separate shells in the purchased sculpt* — verified by decimating at 40k instead of 16k (same
detachment) and by the raw welded source itself having 19 components. They belong to their segment
and move rigidly with it; stitching them would mean editing the artist's topology.

**Parametric.** Every skeleton knob moves exactly its own quantity (P0 = 9 parts, 130 × 100.5 mm,
head 44.2 mm, pitch 8.8):
| knob            | parts | length | width | head  | relief | pitch |
|-----------------|------:|-------:|------:|------:|-------:|------:|
| segCount 5      | 7     | 130    | 100.3 | 44.2  | 40.8   | 12.3  |
| segCount 14     | 16    | 130    | 100.5 | 44.2  | 40.8   | 4.4   |
| length 180      | 9     | 180    | 100.5 | 61.2  | 40.8   | 12.2  |
| cephFrac 0.42   | 9     | 130    | 95.0  | 63.8  | 40.8   | 6.4   |
| width 70        | 9     | 130    | 70.2  | 44.2  | 40.8   | 8.8   |
| relief 25       | 9     | 130    | 100.5 | 44.2  | 25.0   | 8.8   |

**Robustness.** After the debris cleanup, `e_max`, `stopped_by`, `touching_at_zero` and
`print_valid` are unchanged; only the closure-gap sample moved ~2 mm.

## Still true
- Parts are open shells, not thickened solids; collision uses your FCL fallback. Thicken + hinge
  (your `skin.py` stage) before print, and rerun — Manifold overlap volumes then replace FCL depth.
- The viewer is untested in a browser here (syntax-checked; math verified in Python).
