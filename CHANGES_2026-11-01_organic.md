# CHANGES 1 Nov 2026 (later) — the organic protocol (schema 6.8)

Branch `feat/organic` off `feat/genal-shapes` (a8ac904). One set of rules, in one module (`anatomy/organic.py`), that every
builder applies to the thin parts of an animal, so nothing comes out as a wire, a needle, a sawn-off blade or a hard
corner. One new parameter, `organic` (Body, 0..1, default 1): 1 = on, 0 = the geometry exactly as built before.

## The rules (`anatomy/organic.py`; the constants are at the top of the file)

1. **Slenderness.** A spine's root radius is at least its length / 40 (`SLENDER`). A long spine is a horn, not a wire.
   A spine drawn in a plan (pleural spines, the genal piece) is at least its length / 24 in half-width (`SLENDER_PLAN`).
2. **Taper and tip.** The radius falls on a full, convex profile (thick for longer, then easing off; `TAPER_EXP`), not a
   straight cone, and ends in a round tip at least 0.4 × the root (`TIP_FRAC`). The old "last fifth sharpened to a
   point" is off. A lobe drawn on a margin (the tail's) ends round too: its apex is eased (`LOBE_ROUND`).
3. **Root.** A spine standing on a plate grows out of it through a collar 1.5 × its radius (`FLARE`). Spines that
   continue a blade drawn in the plan (pleural horns, tail lobes running on as spines) keep their blade-shaped root.
4. **Section.** A spine drawn in a plan is a rounded ridge: its crest stands 0.6 × its half-width above its edges
   (`SECTION`), and it eases into the blade it leaves. (The segment's front strip, lowered to tuck under the plate ahead,
   no longer flattens a pleural spine narrower than that strip.)
5. **Bends.** A kinked genal piece bends through an arc (radius ≥ 1.5 × its width, `BEND_W`; shrunk to fit short legs),
   not two capsules meeting at a corner.
6. **Edges.** In the studio's parts, every free edge is rolled: the top falls to the edge through a quarter round of up
   to 1.6 mm (`EDGE_R`; never more than a third of the width there, so a spine is not flattened). A plate lifted off the
   bed gets a bullnose. Beside the joint column the roll eases out, so the column's faces are as before.

## Where each rule is applied

| builder | rules |
|---|---|
| `anatomy/common.curved_spine`, `spine_solid`, `prong` (every tube: occipital, axial, frontal, genal horns, prongs and tines, tail spines, the studio's free spines) | 1, 2, 3 |
| `anatomy/head.sphere_eye` (the eye stalk: one smooth tube, not a chain of cones and balls) | 1, 2, 3 |
| `anatomy/thorax.plan` (pleural spines in the plan) | 1, 2, 4 |
| `anatomy/tail.plan` (margin lobes) | 2 |
| `anatomy/genal.Piece` (both heads' genal piece) | 1, 2, 5 — widened on its outer side only, so a prolongation beside the thorax does not grow into it |
| `anatomy/head.plan` (the pin head's genal piece) | 4 (the studio's piece already had a rounded ridge) |
| `isopod_model/body.build_part` (studio parts) | 6 |

The strength reaches the builders from `P` (`assemble.part`, `body.build`, and each `ornaments()` set it), or from
`organic=` arguments.

## What stays at 0

- **The instrument** (`instrument.py`, `ORGANIC = 0.0`): enrolment is measured on the animal as pre-registered. Set it
  to 1.0 to measure the rounded animal the site prints by default.
- **Landmarks** (`landmarks.cephalon`): the genal tip (LM16) is the design's point, not the rounded finish, so the fits
  are unchanged.
- **Frozen-reference tests** (`test_head`, `test_parts`, `test_regression`, `test_schema`'s whole-animal reading): their
  helpers set `organic = 0`.

## Checked

- Full test suite (`tests/`, less the agent, recorder, site-password and printjoint2 modules): 240 passed, 4 skipped.
  New `tests/test_organic.py` (each rule's guarantee; the instrument and the landmarks at 0). `test_curved_spines`'s arc
  test builds the arc at organic 0 (it checks the arc, not the finish).
- Every preset's pin parts at organic 0 and 1 (bevel 45): watertight; rest overlap between neighbours under 0.5 mm³ on
  every preset (the instrument's budget is 1 mm³); parts with more than one body are the same at 0 and 1 (lichida,
  odontopleurida, phacopida, as before this change). Harpetida's prolongation, widened on both sides, ran 5.9 mm³ into
  segment 0 (over the budget); widened outward only it is 0.17 mm³.
- Studio previews of the spiny prototype and four presets at 0 and 1 (renders in the delivery).

## Not done

- The pin builds' plates keep square edges (rule 6 is the studio's parts only): there the plate edge is the joint's
  bearing surface and the instrument's geometry.
- The head's own outline (margin, rim) is unchanged; it already has a rolled rim.
- The tail lobes get a round apex, not a slenderness floor (their width is set as an angle on the margin).

Not pushed.
