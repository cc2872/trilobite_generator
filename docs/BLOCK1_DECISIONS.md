# Block 1 — freeze the ruler (13 Sep 2026)

Working copy on top of a8984dd. Everything below is evidence plus a proposed `[DECISION]`; the decisions are Claire's.
The order matters: 4 and 5 can be decided from the numbers in this file; 6 needs one more look at the old build;
7 is done in code; 8 is a script that must not be run with `--write` until 4–7 are resolved; 9 is the tag.

## Item 4 — spines in the collision set

The instrument already excludes the genal spine from the *closure* test (`head_body` is the cephalon rebuilt with
`genalSpine = 0`), but every spine solid is in the *collision* test (`collisions_deg` uses the full meshes). The pilot
CSV never recorded the limiting pair, so `scripts/block1_spines.py` re-reads the four anatomy-limited pilot animals
with the pair recorded, then with every spine family zeroed. Results: `docs/block1_spines.json` / `.log`.

`[DECISION]` proposed: if no anatomy reading in the pilot is spine-limited, leave the collision set as is and state
in methods that spines are rigid and in the set. If any is, exclude spine solids from `collisions_deg` (build the
collision meshes with all spine keys zeroed, the way `head_body` is built) — the closure test already works that way.

## Item 5 — pleural facet + ventral doublure: gate-before-scaling

Schema 6.2 adds four identity-defaulted keys: `facetExtent` (0 = flat stub), `facetAngle`, `doublureWidth` (0 =
none), `vincular`. `parts.pleural_facet` chamfers the top-front edge of the outer `facetExtent` of each blade at
`facetAngle` from vertical, pivoting 0.6 wall under the lowest top along the facet so narrow tips survive (first
version severed harpetida seg7; fixed). `parts.ventral_doublure` adds the rim plate under head and tail margins,
`doublureWidth × width` inboard, one wall thick, with a band cleared at the hinge edge so the neighbouring joint
can swing (first version ran the plate into seg0 and read proetida at 2.9°; fixed — zero collisions to 15°).

Gate: `scripts/block1_gate.py` reads each anchor OFF (= 6.1 geometry) and ON (`facetExtent 0.6, facetAngle 45,
doublureWidth 0.11, vincular 0.8`). Load-bearing = |Δθ| > 2 × 0.1° floor, or class/limited_by changes.
Results: `docs/block1_gate.json` / `.log`.

First pass (before the facet tip cap; doublure already fixed):

| anchor | OFF | ON | Δ |
|---|---|---|---|
| proetida | 22.81 closed double | 22.73 closed double | −0.08° |
| corynexochida | 22.66 closed sphaeroidal | 22.50 closed sphaeroidal | −0.16° |
| harpetida | 30.39 closed double | invalid (facet severed tips → bevel 25) | construction bug, fixed |

Second pass (all five anchors, chamfer construction with the tip cap — `docs/block1_gate.json`):

| anchor | OFF | ON | Δθ | s_tail off→on |
|---|---|---|---|---|
| proetida | 22.81 closed double | 22.73 closed double | −0.08° | 0.258 → 0.252 |
| corynexochida | 22.66 closed sphaeroidal | 22.50 closed sphaeroidal | −0.16° | −0.014 → −0.019 |
| harpetida | 30.39 closed double | 29.30 closed double | −1.09° | 0.910 → 0.858 |
| asaphida | 20.70 anatomy open (seg6–tail) | 20.70 anatomy open (seg6–tail) | 0.00° | — |
| agnostida | 21.41 closed discoidal | 21.41 closed discoidal | 0.00° | 1.559 (facet has no effect on 2 mm blades) |

Reading: the structures move closure readings by 0.1–1.1°, always toward *earlier* closure (the doublure gives the
tail a surface to meet sooner), and never change a class or a limiting pair. The anatomy anchor is untouched:
asaphida's stop is seg6 against the tail, not pleura against pleura, so the facet is irrelevant there. Harpetida is
the only one over threshold, and only because its doublure is large relative to a small animal (11 % of width).

`[DECISION]` proposed: keep the keys at identity for the sweep. Direction claims are unaffected (every delta has
the same sign and no class flips); magnitude claims carry a stated ≤ ~1° downward bias from the missing doublure.
State it in methods: pleurae are flat stubs with a print bevel, no doublure, readings are lower bounds on the stop.

## Item 6 — agnostida frozen reference

Frozen (BREP builder, instrument 2.1): `invalid / rest_interference` at bevel 35, because the 45° wedge severed
seg1's pleural tips (3 bodies) and the 35° build interfered at rest. Current (mesh builder + 12 Sep overspan fix):
seg1 is ONE body at 45, no rest interference, reads `closed / discoidal`, 21.41°/joint, 64° total.

Evidence for reading (a) "the old overlap was the orphan tip inside its neighbour":
- `tests/test_parts.py::test_agnostida_bevel_severs_tips_like_the_brep_builder` asserts 3 bodies at 45 and now
  FAILS on a8984dd — the test encodes the defect, not the anatomy.
- `rest_overlaps` on the current 45° build: empty (block1_gate.json, agnostida OFF).
- With the PREREG §3 overlap term restored (item 7), agnostida stays `discoidal`: 64° total and s_tail = 1.56 ≥ 0.5
  (the pygidium passes well under the head). 21.41°/joint is the reading to freeze if (a) is accepted.

`[DECISION]` proposed: accept (a). Delete or invert the severed-tips test (it tests the old builder's bug), re-freeze
agnostida from the current build, and record in PREREG Amendments that the reference changed because the
orphan-tip censor was a builder defect (CHANGES_2026-09-12).

## Item 7 — discoidal overlap term (done)

`instrument.classify` now implements PREREG §3 verbatim: `discoidal` requires `total_deg < 150` AND
`s_tail ≥ DISC_OVERLAP` (0.5). Short-curl closures without the overlap fall through to the s_tail rules.
`DISC_OVERLAP`, `DISC_DEG`, `CLASS_TOL` are recorded in every result. `[DECISION]` DISC_OVERLAP = 0.5 (PREREG's
proposed value; nothing was tuned).

## Item 8 — re-freeze

`scripts/refreeze_references.py` (dry run by default) prints frozen vs current for every reference. Run it once
after 4–7 are decided; then `--write`; then the full test suite; then commit. Expected diffs: agnostida (item 6)
and possibly the default-animal-derived tests if any reference used `eyeSolid` unset (none do — all presets are
fully stamped).

## Item 9 — tag

After the re-freeze commit: resolve every `[DECISION]` in PREREG_v1_sweep.md, add the Amendments entries below,
commit, `git tag prereg-v1`, Zenodo.

### Item 4 result
`docs/block1_spines.json`: idx 0 (agnostida 24.61°, seg1–tail), 5 (agnostida 25.47°, seg2–tail), 11 (asaphida 12.42°,
seg7–tail), 20 (corynexochida 12.66°, seg1–seg2…) read IDENTICALLY with every spine family zeroed. No pilot
anatomy reading is spine-limited. `[DECISION]` proposed: leave the collision set as is; one sentence in methods.

### Item 8 dry run (refreeze_references.py, current tree)
| reference | frozen | current |
|---|---|---|
| proetida | 22.81 closed double | 22.81 closed double — same |
| corynexochida | 22.58 closed sphaeroidal | 22.66 — same (within 0.2) |
| harpetida | 30.31 closed double | 30.39 — same |
| agnostida | invalid rest_interference @35 | 21.41 closed discoidal @45 — DIFFERS (item 6) |
| asaphida | invalid rest_interference @45 | 20.70 anatomy open — DIFFERS (BREP defect; freeze the real reading) |
| redlichiida | invalid unsane seg2 | builds sane now; freeze the real reading |
| phacopida | invalid unsane seg0,seg7 | STILL unsane: seg4–seg9 split in the mesh builder — a live builder bug, Block 2 |
| ptychopariida | invalid unsane seg9 | not re-checked (time); run the dry run |

### PREREG Amendments to add
1. 13 Sep 2026 — `classify` had dropped the §3 overlap term; restored. No sweep rows had been analysed.
2. 13 Sep 2026 — agnostida reference re-frozen: the BREP builder's rest interference was a severed-tip artefact
   (CHANGES_2026-09-12); the mesh builder has no such defect.
3. 13 Sep 2026 — schema 6.2 adds facet and doublure keys at identity; not swept (gate result in block1_gate.json).
4. 13 Sep 2026 — spines: [fill from item 4].

## Not Block 1, but found while doing it
- Pilot censoring: 29 of 53 rows invalid (9 rest_interference, 15 unsane parts, 2 bound). That rate gates the sweep
  more than anything above; it is Block 2 item 10 (the parameter audit) and should be closed before item 13.
- `safe_expr` is still `eval`. Block 0 item 1.
