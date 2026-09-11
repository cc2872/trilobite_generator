# trilobite generator 6.0 · instrument 2.1

A parametric generator for articulated dorsal plate chains trilobites first with a fixed-ruler enrolment
instrument, a printable animal, a labeled blueprint sheet, and a website. No OpenCascade: numpy height fields and
Manifold booleans, mirror-symmetric by construction.

    pip install -r requirements.txt flask
    python -m pytest tests -q              animals
    python sweep.py presets 
    out/presets.csv
    python blueprint.py gallery out           
    python web/app.py                        

## Layout (95 parameters, 8 modules)
| file | what |
|---|---|
| `schema.py` | the 86 parameters, presets, `coerce()`, `coerce_report()`, `migrate()`, and **CELLS** — the 3x3 contract (three primary keys per cell) |
| `fields.py` | outline and lobe fields (the sculpture's numpy) |
| `mesh.py` | height-field shells, envelopes, Manifold booleans, primitives, the hinge, mirror/symmetry |
| `parts.py` | cephalon / segment rows / pygidium on `mesh.py`, their a·b·c cells, spines, prongs, eye solid |
| `instrument.py` | the ruler: build at a fixed bevel, probe, sweep to first interference or closure, classify, row |
| `sweep.py` | `presets` batch; `design` / `run` / `one` for the pre-registered morphospace sweep |
| `analyze_sweep.py` | the pre-registered analysis in the pre-registered order (K1–K5 verdicts) |
| `blueprint.py` | the labeled sheet for one animal; `gallery` mode for a batch |
| `web/` | Flask app + the 3x3 site |
| `tests/` | mesh, parts, head, schema, web, regression against frozen references, sanity animals, noise floor |
| `legacy/` | the OpenCascade builder and the v5 site (reference only; needs build123d) |
| `PREREG_v1_sweep.md` | the pre-registration; tag the repo after its `[DECISION]` lines are resolved |

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
a body count (harpetida seg7), the spline's retry loop had hidden a real bevel-band bug, fixed in `parts._hinge_geometry`.
Noise floor: proetida reads 22.81 at every grid, scan step and bisection setting tried — the floor is the 0.1 deg
resolution. See `CHANGES_2026-09-10.md` for every finding in order.
