# Elevation sketch — design notes (2026-09-13)

Decisions from the session, not yet implemented in web/index.html.

## Layout
- Keep the existing vertical plan (head up). Add a narrow elevation strip to its RIGHT,
  rotated 90°: z runs horizontally, dorsal → right. Both views share the plan's y-scale
  and y-origin verbatim; the 50 mm rail on the left is the shared length ruler.
- Same px/mm for z as for x/y. No z-exaggeration. Relief 17 mm ≈ 45 px at the current scale.
- Move the SEG stepper below the plan to free the column.

## Profile lines
- Solid: axial section (zfun at x = 0 plus rings) — this IS section A–A on the sheet.
- Dashed: pleural margin at the fulcrum (gives marginHeight a grab point).

## Handle ownership (one owner per parameter — never editable in both views)
- Plan owns: stations (eyePos, widthMaxPos, prong split), widths, arcs, splay, genal path, eyeLat.
- Elevation owns (horizontal drags only): relief, headRelief, tailRelief, marginHeight,
  glabRise, eyeHeight + eyeSlope (the eye bar), stalk length (grip drags left to shorten),
  headProngCurl (top of strip).
- Read-only ticks: hinge z (always), e@stop arc (dashed, greyed until Measure runs).

## Fork (head prongs) — plan-view handles to add
- Split point on the axis: drag along y → headProngStem; drag sideways → headProngSplay.
- Root width handle at the base → headProngWidth. Tine count stays a +/− stepper.

## Eyes
- Plan: disc at (eyePos, eyeLat), radius grip = eyeSize, pie grip = eyeArc (exists).
- Elevation: bar = eyeHeight / eyeSlope; stalk grip below it.
- Packing: NOT handles. A ~60 px lens swatch with a three-way selector
  (holochroal / schizochroal / abathochroal) that sets the tuple
  {lensD, lensGap, lensRise, lattice, style, grad}; sliders under MORE for fine control.

## Text
- Zero labels at rest. Hover: one word + dashed link line to the twin handle in the other
  view. Drag: the value only. Hinge tick and stop arc are never labelled (the sheet has them).

## Schema cost (not done)
- New keys, all identity-defaulted so frozen anchors don't move:
  eyeStalk, eyeStalkR, eyeLeanOut, eyeLeanFwd, eyeKidney, eyeKy, eyeTaper, eyeLensStyle,
  eyeLattice, lensGrad. → schema 6.1. eyes2.eye2 replaces eye_solid behind eyeSolid=1
  with a compat test that old defaults reproduce the old mesh within noise.
- Instrument untouched: eyes are outside the enrolment chain.
