# trilobite generator, schema 6.3, instrument 2.1, by Claire Choi the trilobiter
<img width="1815" height="1287" alt="sheet_isopod_none (4)" src="https://github.com/user-attachments/assets/b1ca3f45-d5f6-4a96-b0c5-54285340ca22" />

A parametric generator for articulated dorsal plate chains, trilobites first, with three outputs from one parameter
table: a **printable animal** (print-in-place, ball joints), a **fixed-ruler enrolment instrument** (pin joints,
pre-registered), and a **blueprint sheet** for either. No OpenCascade, numpy height fields triangulated directly,
Manifold booleans, mirror-symmetric by construction. Site: volvation.org (unpublished research: do not share).

    pip install -r requirements.txt flask
    python build.py                        # the printable animal -> isopod_model/stl/trilobite_isopod.stl, checks, photos
    python build.py --preset lichida       # one preset
    python web/app.py                      # the site on :8765
    python -m pytest tests -q -m "not slow"   # ~9 min on 2 cores; add -m slow for the full builds

Python 3.12, `manifold3d==3.0.1` pinned (PREREG §9: other versions' booleans shed slivers and move readings).

## Two animals, one parameter table

| | printable animal: the **isopod model** | measured animal: the **pin model** |
|---|---|---|
| where | `isopod_model/` | `anatomy/` + `joints/pin.py` + `assemble.py` |
| head | the crescent head on the giant isopod's head piece, or a guide-style **shaped head** (`shaped_head.py`, 24 styles with page citations) | the classic head (`anatomy/head.py`) |
| body | the preset's plan, or the **rows model** (`rows.py`): one list of rows head to tail, short rows fuse into one piece, the pygidium = the trailing fused rows | thorax rows + pygidium from `fields.py` curves |
| joint | the isopod's own ball-and-socket, copied 1:1 from the asset (`check.joint_deviation`: 0.002 mm) | interleaving knuckles on a pin with a ventral bevel stop |
| what it is for | printing. Every part is cleaned and validated (watertight, one body, no rest contacts); a failed part refuses the build | the reading. Changing anything here is a new instrument version |
| on the site | the default; the studio, head editor and body editor all build it | `joint = pin` on `/generator`; the MEASURE mode runs the instrument |

The two share `schema.py` (the 109 parameters, presets, `coerce()`, `migrate()`, CELLS, the 3×3 contract),
`fields.py` and `mesh.py`. Nothing in `isopod_model/` is imported by the instrument.

## Layout

```
schema.py           the parameters, presets, coerce/coerce_report/migrate, CELLS
fields.py           quantities that vary along the body (width, spines, furrows) as curves
mesh.py             height-field shells, envelopes, Manifold booleans, primitives, the pin hinge cut, mirror/symmetry
anatomy/            the parts, with NO joints                    swap a head here
  head.py           the classic head: plan, shell/solid, ornaments (arms, eyes, occipital spine, prongs), ports, cells
  head_crescent.py  the crescent head (27 Sep): the rear edge is one U that runs on as the horns' inner edge
  heads.py          get(name): classic | crescent. A head is a build option, not a parameter
  thorax.py         segment i;  tail.py  the pygidium;  port.py  what a part tells a joint;  common.py  pitch, primitives, safe_expr
joints/             every way two parts connect, one file each   swap a joint here
  base.py           the template (cut / overhang / pose / pivot / stop / fits)
  pin.py            the MEASURED joint.  flexi.py  Thingiverse #3839472 form.  ball.py  flexi with spheres.
assemble.py         the only file that imports both anatomy and joints: plate -> joint at each port -> ornaments
instrument.py       the ruler: build with joints/pin at a fixed bevel, probe, sweep to first interference or closure, classify, row
isopod_model/       the printable animal (its own README)
  body.py           build(P, head, face, outline, rows): plans, cutters, clearances, the kit pasted in, clean + validate
  rows.py           the rows model: rows_from_P, boring (the plain start), validate_list, groups (fusion), plans, diagram, macro, problems
  crescent_head.py  the head piece;  shaped_head.py  the parametrised silhouette;  head_styles.py  the 24 guide styles
  check.py / render.py / enroll.py / presets.py     checks, photos, the enrolment test on this model, every preset
  source/, stl/, photos/, readings/                 assets, outputs
segment/            two standalone one-piece programs (not used by the site): segment.py (a tergite, partial fill),
                    dome.py (the isopod-style dome shingle with its joint). --copies N poses a chain; each prints its own checks
printfill.py        print only: thicken pin-jointed shells, leaving the joint's keep-out clear
printjoint2.py, parts.py   compatibility names; they forward to assemble
sweep.py            presets batch; design / run / one for the pre-registered morphospace sweep
analyze_sweep.py    the pre-registered analysis in the pre-registered order (K1 to K5)
blueprint.py        the sheet: sheet() for the pin model, isopod_sheet() for the printable animal; gallery mode for a batch
web/                Flask app + the pages (below); static/p5.min.js self-hosted
tests/              see below
presets/            the ten orders, two variants (harpetid, phacopid) and two dev presets (reference, textured) as JSON
assets/             the isopod STL (used with its owner's permission, see assets/isopod/README.md), Earth
docs/               BLOCK1_DECISIONS.md, recording.md, UI notes
legacy/             the OpenCascade builder, the v5 site, the flat files this layout replaced
PREREG_v1_sweep.md  the pre-registration; tag prereg-v1 after its [DECISION] lines are resolved
CHANGES_*.md        every finding in order, with what was and was not run
```

## The site

| page | what |
|---|---|
| `/` | **studio**: head + body on one drawing (p5). Click a handle, drag, build. Rows + / −; short rows fuse (orange); save a head or a body by name; BUILD swaps the drawing for the sheet, with VIEW 3D |
| `/head`, `/body` | the head editor and the body editor on their own (the studio is the two combined) |
| `/generator` | the advanced / instrument page (retired as the home, 13 Oct 2026; the studio is the front page): the 3×3 sketch control and the full parameter set, MEASURE (the instrument / ruler), the share link and TPS landmarks (TriloMorph protocol, `landmarks.py`). The studio's VIEW 3D embeds this page's viewer (`?only3d=1`), so the route is kept as a backend even though it is no longer presented as a co-equal editor |
| `/tree` | the family tree |

API, all JSON: `POST /api/build {P, joint, head, body, fill}` (joint = `isopod` default | `pin` | `flexi` | `ball`;
head = a style name, a saved head's name, or `{style, outline, face}`; body = a rows spec or a saved body's name) →
manifest with per-part GLBs, cached by parameter hash + builder-source hash; `POST /api/sheet` the PNG;
`GET /api/stl/<key>` one flat STL in print mm; `POST /api/measure {P}` the reading; `GET/POST /api/head_diagram`,
`/api/body_diagram` the editors' drawings from the builder's own geometry; `GET/POST/DELETE /api/heads`, `/api/bodies`
saved designs (`web/heads/`, `web/bodies/`, git-ignored); `POST /api/macro` projects a body/head onto schema
parameters; `/llms.txt` and `/api/agent/make` for programs (token-gated when `TRILO_AGENT_TOKEN` is set).

Environment: `TRILO_AGENT_TOKEN` (or `_FILE`), `TRILO_AGENT=0` removes the agent routes; `TRILO_RECORD=0` turns
recording off, `TRILO_RECORD_URL` / `_SECRET` stream readings to a receiver, `TRILO_RECORD_DIR` keeps them locally;
`TRILO_WORKERS` forked cutter workers (default: cores, up to 4); `TRILO_HEAD_CACHE` where built heads are cached.
See `DEPLOY.md` and `.env.example`.

## The reading
`instrument.read(P)` builds every part at a 45° ventral bevel (dropping to 35 / 25 only if a wide wedge would sever
a pleural tip, recorded), sweeps uniform flexion per joint to the first interference or head-tail closure, and
returns `theta_joint_deg`, `total_deg`, the gap, `limited_by` (closed / anatomy / bound / invalid), `enroll_class`
(sphaeroidal / double / spiral / discoidal / open / censored), `s_tail`, the limiting pair, the bevel actually built,
the instrument version and the parameter hash. `bound` and `invalid` are censoring, never measurements. Only
`joints/pin.py` is measured: a change there moves the readings, so it needs a new instrument version and a re-freeze
of `tests/references/`.

## Provenance of the numbers
`tests/references/` holds readings and meshes frozen from the OpenCascade builder on 10 Sep 2026. The mesh builder
reproduces them: proetida 22.81 / 22.81, corynexochida 22.66 / 22.58, harpetida 30.39 / 30.31°. Where the two
builders disagree on a surface (domed heads, up to 1.6 mm), the spline fit was the biased one; where they disagreed
on a body count (harpetida seg7), the spline's retry loop had hidden a real bevel-band bug, fixed in the pin joint's
band rule (`joints/pin._band`). Noise floor: proetida reads 22.81 at every grid, scan step and bisection setting
tried; the floor is the 0.1° resolution. The isopod model's default build is guarded the same way: every speed-up
since 11 Oct is checked part for part against the reference (volumes and bounds equal). See `CHANGES_2026-09-10.md`
onward.

## Swapping a joint or a head
A joint: write `joints/<name>.py` with the functions in `joints/base.py`, register it in `joints/__init__.py`, run
`pytest tests/test_joint_contract.py`. A head: a module returning the same things as `anatomy/head.py` (`plan`,
`shell`, `solid`, `ornaments`, `ports`, `cells`), registered in `anatomy/heads.py`, run `pytest tests/test_head_contract.py`.
Nothing else changes. For the printable animal, a new head silhouette is a style in `isopod_model/head_styles.py`
(outline + face keys, with the guide page it comes from).

## Tests
`tests/`: contracts every joint and head must pass (`test_joint_contract`, `test_head_contract`); mesh, parts,
head, schema, web, print joint; regression of whole animals against the frozen references; the editors' endpoints,
the rows model's fusion rule and the macro projection (`test_head_editor`); the segment programs
(`test_segment_programs`, slow); sanity animals and noise floor (`sanity_animals.py`, `noise_floor.py`, run by hand).
Suite on 13 Oct 2026: 207 passed, 4 skipped, 5 slow. The 11 Oct stress battery (19 extreme bodies) is in
`CHANGES_2026-10-09.md`; the guards it bought (`rows.problems`, `OUTLINE_RANGE`, point budgets) are what refuse a
body before a build.

## Status (13 Oct 2026)
Block 1 (freeze the ruler) is closed in the tree; `prereg-v1` is not yet tagged; `tests/references/` and the
presets must be re-reconciled first (see `claude/software-review-2026-10-13.md` in the project, §B). The printable
animal, the editors and the sheet are current. The site has no access gate yet: saved heads and bodies are
server-wide and unauthenticated.

## Reference data: TriloMorph

`landmarks.py` reads the TriloMorph landmark protocol (Serra et al. 2023: 16 cephalic + 7 pygidial landmarks, 4 + 3
semilandmark curves) off the head and tail plans, so a generated animal can be superimposed with the photographed ones
in geomorph (`/api/tps`, `python landmarks.py`). Their 5-95 % ranges set `axisFrac`, `genalSpine`, `preglabField` and
the `eyeLat` default (schema 6.3, 20 Oct 2026). The instrument is untouched by this: TriloMorph enters the generator's
anatomy and the parameter box only.

Cite: Serra F., Balseiro D., Monnet C., Randolfe E., Bignon A., Rustán J.J., Bault V., Muñoz D.F., Vaccari N.E.,
Martinetto M., Crônier C. & Waisfeld B.G. (2023). A dynamic and collaborative database for morphogeometric information of
trilobites. *Scientific Data* 10: 841. doi:10.1038/s41597-023-02724-9 — and the TriloMorph stable release used
(doi:10.5281/zenodo.8033445, CC BY 4.0; contributors in its `contributors.csv`). Their images are third-party figures
and are not redistributed here.
