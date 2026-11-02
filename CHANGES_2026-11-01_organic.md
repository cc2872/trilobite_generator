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
7. **Curves** (added after the first delivery: "we should be trying to make everything a curve"). Nothing is a straight
   line:
   - a spine asked for straight (bend 0) bends by 20° over its length (`CURVE`), its tip turning toward the rear; one
     pointing straight back or forward turns outward, or up on the midline. A bend the user set is kept as set.
   - a prong's stem takes its share of that curve, and each tine curves away from the middle of its fork.
   - an eye stalk asked for straight bends by the same 20°.
   - the genal piece's centreline is one curve. Unkinked, it is an arc on the same chord (the tip end stays where it
     was put) turning 30° (`CURVE_PLAN`), bowed so its tip turns toward the rear and in. Kinked, it is a smooth curve
     leaving the root along the first leg and reaching the tip along the second, passing near the kink (`KINK_EASE`).
     When either would bend tighter than the piece's half-width (a hairpin), rule 5's fillet is used.
   - in the studio, where the genal piece climbs over the body, its rise is one smooth curve along its length (the
     lift's upper envelope over ~10 mm, smoothed) and eases out over 7 mm past the body's reach instead of 2.
   - where a pleural spine leaves its blade, the two outlines meet through a smooth maximum, not a crease.

## Where each rule is applied

| builder | rules |
|---|---|
| `anatomy/common.curved_spine`, `spine_solid`, `prong` (every tube: occipital, axial, frontal, genal horns, prongs and tines, tail spines, the studio's free spines) | 1, 2, 3, 7 |
| `anatomy/head.sphere_eye` (the eye stalk: one smooth tube, not a chain of cones and balls) | 1, 2, 3, 7 |
| `anatomy/thorax.plan` (pleural spines in the plan) | 1, 2, 4, 7 |
| `anatomy/tail.plan` (margin lobes) | 2 |
| `anatomy/genal.Piece` (both heads' genal piece) | 1, 2, 5, 7 — widened on its outer side only, so a prolongation beside the thorax does not grow into it |
| `anatomy/head.plan` (the pin head's genal piece) | 4 (the studio's piece already had a rounded ridge) |
| `isopod_model/body.build_part` (studio parts) | 6 |
| `isopod_model/body.genal_piece_solids` (the studio's genal piece over the body) | 7 |

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

- Full test suite (`tests/`, less the agent, recorder, site-password and printjoint2 modules), rules 1-6: 240 passed,
  4 skipped. With rule 7: 241 passed, 4 skipped, 1 failed -- the head with a trident was not mirror-symmetric, from a
  bug in `prong()` (the tine loop's index shadowed the strength; it also changed a fork's root ball at organic 0 in the
  first delivery). Fixed; the modules that build prongs (`test_schema`, `test_head`, `test_curved_spines`,
  `test_organic`, `test_head_contract`, `test_free_spines`) then 66 passed. At organic 0, `prong` and `curved_spine`
  give the same volumes as 630405f to 4 decimals.
  New `tests/test_organic.py` (each rule's guarantee, rule 7's directions; the instrument and the landmarks at 0).
  `test_curved_spines`'s arc test builds the arc at organic 0 (it checks the arc, not the finish).
- Every preset's pin parts at organic 0 and 1 (bevel 45; rerun with rule 7): watertight; rest overlap between neighbours under 0.5 mm³ on
  every preset (the instrument's budget is 1 mm³); parts with more than one body are the same at 0 and 1 (lichida,
  odontopleurida, phacopida, as before this change). Harpetida's prolongation, widened on both sides, ran 5.9 mm³ into
  segment 0 (over the budget); widened outward only it is 0.17 mm³.
- Studio previews of the spiny prototype and four presets at 0 and 1 (renders in the delivery).

## Not done

- The pin builds' plates keep square edges (rule 6 is the studio's parts only): there the plate edge is the joint's
  bearing surface and the instrument's geometry.
- The head's own outline (margin, rim) is unchanged; it already has a rolled rim. Its rear margin is the joint with
  segment 0, so it is not bowed.
- The axial ring's front and rear edges stay straight across: they are the joint lines.
- Spines that continue a blade (pleural horns, tail lobes run on as spines) keep the turn their own controls give them.
- A spine pointing along the axis turns outward or up (a choice); tell me if a family should curve another way.
- The tail lobes get a round apex, not a slenderness floor (their width is set as an angle on the margin).

Not pushed.

## 2 Nov 2026: hooked spines (Ceratarges' frontal horns)

A spine was always a circular arc (even curvature), so a horn that turns hard near its root and then runs straight back
could not be drawn. New `hook` (-1..1) on `curved_spine`, `spine_tip` and the studio's free spines (a slider in the
free-spine panel; `rows.FREE_SPINE_RANGE`): where along the spine the turn happens. 0 = the even arc as before (every
saved body unchanged); +1 = the turn at the root, then a straight run; -1 = a hooked tip. `anatomy/common.hook_turn` /
`hook_path`; the studio's drawing (`arcPts`) follows the same curve.

- The spiny prototype's frontal horns are fitted to the Ceratarges drawing (`fit_horns.py`, outside the repo): from
  the front of the glabella, up and out, round, and back over the cheeks to just behind the head. x 10.7, y -32.7,
  len 78, rise 155, roll -87, yaw 115, hook 0.2.
- `body._spine_clear` asks its inside-the-plate question in batches of 128 points: one query over a long hooked tube
  ran to 6 GB and was killed.
- `test_curved_spines`' prong equality is relative (1e-9 x volume), not absolute: two identical calls differed by
  float noise in a longer run. New hook test in `test_organic.py`.
- (later) A free spine's leave angle and its turn were one number (`rise`: a 155° turn forced a 54° leave), so a horn
  that leaves low and loops round could not be drawn. New `bend` on free spines (deg, -360..360, default 0): extra turn
  in the roll's plane, on top of the rise's; the leave angle stays. Studio slider and drawing follow it.
- The horns refitted to the outline traced on the render: over the head's front, down its sides, ending at the rear
  corners. x 10.7, y -33.0, len 116, rise 12.6, roll -50, yaw -173.5, hook 0.23, bend 204.
- (2 Nov) `turn_at` on curved spines and free spines (0..1; -1 = off, the default): the turn is gathered round that
  point -- straight, a bend, straight, a hairpin -- with `hook` then setting how tight. Studio slider and drawing follow.
  Dicranurus' horns are fitted with it: back alongside the body, a round turn, forward up the outside.
