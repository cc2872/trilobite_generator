# Trilobite morphospace — state for 9 Sep meeting

Claire Choi · Cornell · 2 pp. Numbers in brackets are from the overnight batch and get filled in before printing.

## 1. What is wrong, first

1. **Instrument 1.0 measured its own stop.** Free curl was (segments + 1) × 18°, so the ceiling scaled with segment count — one of the six dials — and no animal collided before it. Every preset read e_max = 1.0, class "partial". Replaced (§2). Nothing measured before 8 Sep is a measurement.
2. **No negative control yet.** *Harpes* enrolls (Gon p. 32), and the harpetid's old 0.94 was a 1.5 mm³ contact inside tolerance. Non-fulcrate olenellids are the right control; they need articulating facets as geometry, which does not exist yet (weeks 2–3).
3. **Three of ten orders do not measure.** Agnostida interferes at rest (seg1–tail, 10.7 mm³). Odontopleurida's spine booleans drop ~65 % of every segment. Asaphida's tiled plates (blade chord ≥ 1) interfere at the first degree on this hinge — a real geometric finding or a preset fault, not yet decided.
4. **Sculpt is sub-grid.** Furrows and the old heightfield eye are under 1 mm of relief on a ~1.4 mm grid. The eye is fixed (§3); the furrows are not.
5. **Uniform flexion.** The ruler bends every joint the same angle. Esteve et al. (2017) show fossils do not: ~45° at the head joint, 13–23° decreasing rearward. Declared, not corrected.
6. **Hinge geometry varies with the animal.** Hinge height depends on relief and effacement; hinge width on the last ring. The *protocol* is fixed; the joint is not. Stated on the sheet.

## 2. What changed since Saturday: the ruler now measures shape

Instrument 2.0: angle sweep per joint to first collision or closure, on meshes built with a fixed wide bevel so the stop cannot be the limiter; closure measured tail-to-head-body (arms off); collisions baseline-subtracted; reading = θ per joint, total, gap / L, `limited_by ∈ {closed, anatomy, bound}`. Bevel reach is part of the ruler (0.15 pitch at segment joints, 0.5 at the two full-width joints), bracketed on the two failure modes and written down.

External check: three clean closures — corynexochida [22.6°, 203°], lichida [19.1°, 210°], textured 21.4°, 214° — sit on the ~200° total that Esteve et al.'s sphaeroidal model needs. Their model reconstructs an observed pose; this instrument finds the pose the shell cannot pass. Closure total is not universal: the agnostid's curve closes near 100° because head and tail are each 0.4 L. That is an axis, not a class.

Ten orders (Gon's list), on one sheet: [table from `out10/progress.txt` — order · n · bevel · θ · total · limited_by · print].

## 3. The eye, as a primitive with fossil numbers behind it

| | Erbenochile | Phacops sp. | Isotelus | Dipleura |
|---|---|---|---|---|
| band height / R | 2.2 | 1.7 | 1.0 | ~0.3 |
| lean from vertical | 0° | 15° | ~45° | dome |
| brim / R | 0.15 | 0 | 0 | 0 |
| arc | ~180° | ~110° | ~160° | ~160° |
| lattice | schizochroal | schizochroal | holochroal | holochroal |
| source | photos ±15 % | scan, right eye | photos ±20 %, scan pending | photo |

`eye_solid.py`: revolved cap + leaning band over the visual arc, palpebral lobe inward, lens caps on a hex lattice in the band's own coordinates, one boolean. Phacopida: 87 lenses/eye, watertight head in 47 s. Holochroal facets under 0.5 mm at print scale are not built, by rule. On the blueprint: eye plan, section B–B, outward view, unrolled band, section C–C through the eye. FOV ruler v0.1 (arc only); v0.2 (ray-cast occlusion, at rest and enrolled) designed.

## 4. Decisions wanted

1. Type genus per order (Gon's ten; Trinucleida/Olenida excluded).
2. Printed stop: constant across the print run, or per animal at its measured θ. The measurement no longer depends on it.
3. Whether the eye is on this paper's critical path (function: enrollment × vision on one sheet) or a second paper.

## 5. Next four weeks

Facets and fulcrum as geometry (the control) · doublure and ventral concavity · glabella fade and segment cap as parameters · FOV v0.2 · presets to 9–10 / 10 against Gon · first print the day the grid fix lands.

Provenance: dated notebook pages (pre-code plan drawing; 16 May spread) scanned and in the IM statement.
