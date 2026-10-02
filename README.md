# trilobite generator 6.1, instrument 2.1 by Claire Choi the trilobiter
<img width="1815" height="1287" alt="sheet_isopod_none (2)" src="https://github.com/user-attachments/assets/20b9d050-8d7a-4a14-af05-db476dc755d4" />

A parametric generator for articulated dorsal plat
e chains trilobites first with a fixed-ruler enrolment
instrument, a printable animal, a labeled blueprint sheet, and a website. No OpenCascade: numpy height fields and
Manifold booleans, mirror-symmetric by construction.

## The default build is the isopod model

The generator's default animal is the **isopod model**: the crescent head made on the giant isopod's head piece, on
the isopod-style body with the isopod's own ball joints, at the isopod's print size. It lives in `isopod_model/`.

    pip install -r requirements.txt
    python build.py                     # the isopod model -> isopod_model/stl/trilobite_isopod.stl, checks, photos
    python build.py --preset lichida    # one preset
    python web/app.py                   # the website — the isopod model is the default build

See `isopod_model/README.md` for the model (how it is made, the parameters, the checks). The website serves the
isopod model by default; the classic parametric heads/joints are still selectable there.

## The classic parametric generator

The classic pipeline (anatomy/ + assemble.py + joints/pin) is still here and runs the frozen enrollment instrument
unchanged:

    pip install -r requirements.txt flask
    python -m pytest tests -q              animals
    python sweep.py presets 
    out/presets.csv
    python blueprint.py gallery out           
    python web/app.py                        

## Layout: parts, joints, and the one place they meet

The animal is built in three layers. Each layer only sees the one below it, so a head or a joint can be swapped
by replacing one file.

```
anatomy/            the parts, with NO joints                    swap a head here
  head.py           plan, shell/solid, ornaments (arms, eyes, occipital spine, prongs), ports, cells, the eye
  thorax.py         segment i: plan, shell/solid, ornaments (axial spine), ports, cells
  tail.py           pygidium: plan, shell/solid, ornaments (terminal spine, prongs), ports, cells
  port.py           Port: the only thing a part tells a joint (joint plane, side, width, kind, shingle)
  common.py         pitch, ring_top, surface primitives, spine_solid, prong, safe_expr, grids
joints/             every way two parts connect, one file each   swap a joint here
  base.py           the template every joint follows (cut / overhang / pose / pivot / stop / fits)
  pin.py            the MEASURED joint: knuckles, bore, bevel stop. Changing it = a new instrument version.
  flexi.py          print-in-place joint (Thingiverse #3839472 form, 0.4 mm faces). Never measured.
assemble.py         the only file that imports both: plate -> joint at each port -> ornaments; the overhang pass
```

| file | what |
|---|---|
| `schema.py` | the 98 parameters, presets, `coerce()`, `coerce_report()`, `migrate()`, and *CELLS*, the 3x3 contract |
| `fields.py` | quantities that vary along the body (width, spines, furrows) as curves |
| `mesh.py` | height-field shells, envelopes, Manifold booleans, primitives, the pin hinge cut, mirror/symmetry |
| `anatomy/`, `joints/`, `assemble.py` | see above |
| `instrument.py` | the ruler: `assemble` with `joints/pin` at a fixed bevel, probe, sweep to first interference or closure, classify, row |
| `printfill.py` | print only: thicken pin-jointed shells, leaving the joint's keep-out clear |
| `sweep.py` | `presets` batch; `design` / `run` / `one` for the pre-registered morphospace sweep |
| `analyze_sweep.py` | the pre-registered analysis in the pre-registered order (K1 to K5 verdicts) |
| `blueprint.py` | the labeled sheet for one animal; `gallery` mode for a batch |
| `web/` | Flask app + the 3x3 site; builds through `assemble` with any registered joint |
| `parts.py`, `printjoint2.py` | compatibility names only: they forward to `assemble` (old scripts and tests keep working) |
| `tests/` | contracts every joint and head must pass (`test_joint_contract`, `test_head_contract`); mesh, parts, head, schema, web, print joint; regression against frozen references; sanity animals; noise floor |
| `legacy/` | the OpenCascade builder, the v5 site, and the flat `parts_v6_flat.py` / `printjoint2_v3_flat.py` this layout replaced |
| `PREREG_v1_sweep.md` | the pre-registration; tag the repo after its `[DECISION]` lines are resolved |

### Swapping a joint
Write `joints/<name>.py` with the functions in `joints/base.py`, register it in `joints/__init__.py`, run
`python -m pytest tests/test_joint_contract.py`. Nothing else changes. Only `joints/pin.py` is measured: a change
there moves the readings, so it needs a new instrument version and a re-freeze of `tests/references/`.

### Swapping a head
Replace `anatomy/head.py` with a module that returns the same things (`plan`, `shell`, `solid`, `ornaments`,
`ports`, `cells`), run `python -m pytest tests/test_head_contract.py`. Every joint fits it through its Port.

## The reading
`instrument.read(P)` builds every part at a 45 deg ventral bevel (dropping to 35 / 25 only if a wide wedge would sever
a pleural tip — recorded), sweeps uniform flexion per joint to the first interference or head–tail closure, and returns
`theta_joint_deg`, `total_deg`, the gap, `limited_by` (closed / anatomy / bound / invalid), `enroll_class`
(sphaeroidal / double / spiral / discoidal / open / censored), `s_tail`, the limiting pair, the bevel actually built,
the instrument version and the parameter hash. `bound` and `invalid` are censoring, never measurements.

## Provenance of the numbers
`tests/references/` holds readings and meshes frozen from the OpenCascade builder on 10 Sep 2026. The mesh builder
reproduces them: proetida 22.81 / 22.81, corynexochida 22.66 / 22.58, harpetida 30.39 / 30.31 deg. Where the two
builders disagree on a surface (domed heads, up to 1.6 mm), the spline fit was the biased one; where they disagreed on
a body count (harpetida seg7), the spline's retry loop had hidden a real bevel-band bug, fixed in the pin joint's band rule (`joints/pin._band`).
Noise floor: proetida reads 22.81 at every grid, scan step and bisection setting tried, the floor is the 0.1 deg
resolution. See `CHANGES_2026-09-10.md` for every finding in order.
