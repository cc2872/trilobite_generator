# Pre-registration — v1 dorsal enrolment sweep

**Status:** DRAFT. Lines marked `[DECISION]` are Claire's calls; everything else is proposed.
**Lock:** when the `[DECISION]` lines are resolved, commit this file, tag the commit, and deposit
the tagged tree on Zenodo. The sweep runs from that tag. Nothing below changes after the lock
without a dated amendment section at the bottom.

## 0. Claim under test

The maximum enrolment of a trilobite dorsal exoskeleton is limited by self-interference of the
dorsal plates, and that limit is set by a small number of geometric parameters. Consequently the
region of dorsal morphospace in which complete enrolment is possible is a proper subset of the
buildable region, and realized taxa lie inside it.

Three sub-claims, each with its own kill condition (§7):

- **H1 (sensitivity).** Closure depends on the six sweep axes (§2) with predicted signs (§5).
- **H2 (feasibility).** The feasible region is a proper subset — not "nearly everything."
- **H3 (occupation).** Realized taxa (fitted from specimens, §6) fall inside the feasible region
  at a rate above chance.

Scope limit, stated up front: the instrument is dorsal-only with a single revolute axis per
joint. It does not model coaptative devices (vincular furrows, doublure fit) or ventral anatomy.
Predictions are therefore **ordinal** (class, can/cannot close), and the known failure mode is
under-prediction of enrollability in taxa with coaptative structures. A miss in such a taxon is
recorded as a miss, and is the v2 problem, not evidence for H2.

## 1. Instrument (locked before the sweep)

- `instrument2.py`, version stamp `INSTRUMENT_VERSION = "2.1"` written into every row.
- Fixed measurement bevel `BOUND_DEG` per joint, identical for all animals, **not** passed
  through the morphology clamp (`maxAngle.hi`). Recorded value == built value.
  `[DECISION]` BOUND_DEG = 45 (proposed) — must exceed the largest per-joint angle any preset
  reaches, else `bound` censoring dominates.
- Per-joint sweep in degrees over `[0, BOUND_DEG]`: coarse forward scan at `SCAN_STEP_DEG`
  (`[DECISION]` proposed 2.5) to the first collision or closure, then bisection inside the
  bracketing interval to `RES_DEG` (`[DECISION]` proposed 0.1, the code's RES_DEG).
- Collision = exact Manifold overlap volume above `OVERLAP_TOL` after subtracting the rest
  baseline for that pair.
- Rest baseline: per-pair overlap at θ = 0, keyed on a **content hash** of the mesh set. If any
  pair's rest overlap exceeds `REST_BUDGET_MM3` (`[DECISION]` proposed 1.0), the animal is
  `invalid:rest_interference` and not measured.
- Output per animal: `theta_joint_deg`, `total_deg`, `gap_mm`, `gap_over_length`,
  `limited_by ∈ {closed, anatomy, bound}`, `limiting_pair`, `enroll_class` (§3), `valid` (§4),
  `instrument_version`, `param_hash`, `preset`, `build_seconds`.

## 2. Sweep axes

Six axes. All other parameters sit at the order's preset value.

| axis | schema key(s) | range policy |
|---|---|---|
| A1 segment count | `segCount` | preset ±3, integer |
| A2 pleural overlap | `overlap` (shingle overlap, rear flap / pitch) | 0.2–0.8 |
| A3 cephalon:pygidium | `cephFrac`, `pygFrac` | preset ±30 % each |
| A4 vault / relief | `relief` (and `vault` if separate) | preset ±40 % |
| A5 width taper | `taper` | preset ±40 % |
| A6 spine length | pleural + genal spine length dials | 0 to preset ×1.5 |

`[DECISION]` sweep.py's `overlap` upper bound: 0.8 hit a build-cost cliff at 0.6 in testing; 0.6 may be the practical ceiling.

**Null axis (control):** `eyeSize`. Swept over its full range on a subset (`[DECISION]` 50
animals). Prediction: no effect on `theta_joint_deg`, `gap_mm`, or `enroll_class`.

Design: Latin hypercube within each order's ranges, `N_PER_ORDER` animals
(`[DECISION]` proposed 150; 10 orders → 1,500). Seed recorded. One process per animal.

## 3. Enrolment class — definition

Computed from the pose at the end of the sweep. Quantities in the sagittal plane at that pose:
`s_tail` = position of the pygidial posterior margin along the closed body's arc relative to
the cephalic anterior margin; positive = passes beyond (under) the head margin.

| class | rule |
|---|---|
| `sphaeroidal` | `limited_by == closed`, `|s_tail| ≤ CLASS_TOL`, `total_deg ≥ 150` |
| `double` | `limited_by == closed`, `s_tail > CLASS_TOL` (tail margin inside the head's footprint — tucked under the cephalon) |
| `spiral` | `limited_by == closed`, `s_tail < -CLASS_TOL` (tail margin past the front of the head) |
| `discoidal` | `limited_by == closed`, `total_deg < 150`, closure achieved by cephalon/pygidium overlap ≥ `DISC_OVERLAP` of head length |
| `open` | `limited_by == anatomy`; report `gap_over_length` |
| `censored` | `limited_by == bound` or `valid == False`; excluded from H1–H3, counted in the censoring rate (§7) |

`[DECISION]` `CLASS_TOL` (proposed 0.05 × body length), `DISC_OVERLAP` (proposed 0.5), the
150° split (from Esteve 2017's ~200° sphaeroidal totals with margin). These are fixed here so
the classes are not tuned to the sweep.

## 4. Validity gates (every build)

A row that fails any gate is `valid = False` with a `reason`, never a numeric zero.

1. Boolean sanity: after each boolean, `|V_result − V_expected| / V_expected < 0.5` where
   `V_expected` is the sum/difference of operand volumes. (build123d returns the tool solid
   silently on failure.)
2. Watertight: every part mesh is watertight and has `≤ MAX_COMPONENTS` connected components
   (`[DECISION]` proposed 3 per part).
3. Rest interference below `REST_BUDGET_MM3` for all pairs.
4. Build timeout (`[DECISION]` proposed 300 s) and measure timeout (`[DECISION]` 120 s)
   recorded as `reason = timeout`.
5. OOM caught and recorded as `reason = oom` (lichida case).
6. Schema: every param in the row passes `coerce()` with no fallback substitution; any
   substitution is recorded as `reason = schema_default_substituted`.

Expected censoring rate is reported for the whole sweep and per order. `[DECISION]` If any
order has > 40 % invalid rows, that order is reported as unmeasurable in v1 rather than
measured on its survivors.

## 5. Direction predictions (H1)

Sign of ∂(closure)/∂(axis), where closure = `total_deg` for open animals and class rank
(`open < discoidal < spiral < sphaeroidal`) otherwise.

| axis | predicted sign | rationale |
|---|---|---|
| A1 segCount ↑ | + | more joints share the same total curl |
| A2 pleural overlap ↑ | − | earlier pleura–pleura contact |
| A3 cephalon:pygidium → 1 | + (toward sphaeroidal) | margins meet edge to edge |
| A4 vault ↑ | − | pleural tips converge faster on the ventral side |
| A5 taper ↑ | + | posterior pleurae clear anterior ones |
| A6 spine length ↑ | − | spine–spine and spine–pleura contact |

Tested by rank correlation within order, then pooled. A prediction is **confirmed** if the
sign holds in ≥ 7 of 10 orders and the pooled correlation has the predicted sign with
p < 0.01. Magnitudes are reported but no magnitude claim is made.

## 6. Realized taxa (H3)

- Minimum set for v1: the two CC0 enrolled PRI scans (*Flexicalymene meeki* PRI 70754,
  *Eldredgeops crassituberculata* PRI 49388) and their prone partners from the Digital Atlas.
- Extension set: taxa with parameters measured from specimens by the Harvard collaborators.
  Fitting procedure (outline + segment count + ratios → params) is documented in
  `fit_taxon.md` and applied blind to the sweep results.
- Retrodiction: for each enrolled scan, per-joint angles measured from the scan vs
  `theta_joint_deg` from the fitted animal. **Pass** if within `2 × RES_DEG + bevel resolution`
  per joint on the median joint and the class matches.
- Occupation: fraction of fitted taxa whose class is `closed` (any of the three) vs the fraction
  of the sweep that is `closed`. H3 supported if the realized fraction exceeds the sweep
  fraction with a one-sided binomial p < 0.01 given `N ≥ 15` taxa. Below 15 taxa, H3 is
  reported descriptively, not tested.

## 7. Kill conditions

Any one of these, and the corresponding claim is dropped from the paper — not softened.

- **K1 (H1 dead):** fewer than 3 of the 6 axes reach the confirmation criterion in §5.
- **K2 (control fails):** the null axis moves `enroll_class` in > 5 % of the control subset, or
  correlates with `total_deg` at |ρ| > 0.1. This invalidates the instrument, not just H1: stop,
  find the leak, re-run.
- **K3 (H2 dead):** ≥ 90 % of valid animals are `closed`. The feasible region is the buildable
  region and there is no morphospace story; v1 reduces to a methods paper.
- **K4 (retrodiction fails):** either enrolled scan misses the per-joint criterion in §6.
- **K5 (censoring):** > 30 % of all rows invalid or `bound`. The sweep is re-designed
  (ranges or BOUND_DEG), not reported on its survivors.

## 8. Analysis plan (scripts written before the sweep)

`analyze_sweep.py` reads the row CSV and produces, in this order:
1. censoring table (per order, per reason)
2. null-axis control plot and statistics (K2 first — if it fails, nothing else is looked at)
3. per-axis rank correlations and sign table (§5)
4. feasible-region maps: `enroll_class` over each axis pair, per order and pooled
5. realized taxa overlaid on (4), retrodiction table (§6)
6. K1–K5 verdicts printed as a block at the end

No figure or statistic outside this list is added to the paper without an amendment.

## 9. What is locked at the tag

`instrument2.py`, `schema.py`, `fields.py`, the ten presets, this document,
`analyze_sweep.py`, the classifier constants, and `requirements.txt` with pinned versions.

## Amendments

(none)
