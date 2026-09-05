# Architecture

## Layers (website/)
- schema.py      every knob as data: name, range, kind, group, doc. UI, validation, hashing derive from it.
- fields.py      quantities along the body as curves: width curve, per-segment spine field, landmarks.
- load_sculpt.py decodes the base sculpt (glTF), merges, orients (X across, Y along, Z up), scales to 130 mm.
- wrap.py        body-coordinate remap: the sculpt's own outline is its identity; knobs move it by ratio,
                 so the reference preset reproduces the sculpt exactly. Also regional spine scaling and
                 segment-count resampling (drop/duplicate segments, re-space, trim overlaps).
- mega.py        anatomy displacement: Phacops-minus-Olenoides fields (glabella, segments, tail), an eye
                 turret, the harpetid brim, applied to the sculpt's vertices along z; effacement = Laplacian
                 smoothing. Then hands the mesh to skin.py. Every generation is therefore a deformed sculpt.
- slicer.py      heightfield of the top surface, midline furrow detection, boundary tracing outward
                 (spline inside the body, straight beyond the margin), cutting with vertical prisms (Manifold).
- skin.py        per joint: ventral stop bevels following the curved cut, clearance channel, interleaving
                 knuckles, side-exiting pin bore, articulating facet (recess under the dorsal flange, longer
                 with depth below the axis). Chain kinematics with per-joint axes; measure() = the instrument
                 on meshes (e_max by bisection over all pairs, closure gap, class).
- instrument.py  the same instrument for the legacy parametric builder (trilobite.py), plus print validity.
- trilobite.py   the synthetic parametric builder (build123d). Kept as fallback; not exposed in the UI.
- trilobite_web.py  HTTP server: /api/config, /api/build, /api/mesh (from memory), /api/download (zip),
                 /api/check, /api/health. Labels each build (describe()).
- index.html     the page: schema-driven sliders, three.js viewer, roll slider, Download STL.

## The fixed ruler
Articulation (stop angle, clearance, wall, knuckle size/count) is held constant across a sweep so every
difference in enrollment is attributable to body shape.

## Coordinates
Body coordinates (s along 0..1, u across -1..1 as a fraction of local half-width, z up) are shared by the
atlases, the part library and the wrap. Every model — sculpt or fossil scan — becomes Z(s,u).

## Research pipelines (research/)
- atlas/atlas.py   register any model into Z(s,u): PCA align, dorsal-up + head-first by the glabella crest,
                   dense dorsal sampling, fossil masking from matrix, outline from vertices, landmarks.
- parts/parts.py   crop head / glabella / eye / segment / pygidium / brim patches; solids cut beyond the margin.
- skin/            the standalone slicer development (now inside website/).
