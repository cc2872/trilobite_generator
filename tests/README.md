# tests/

- `references/<preset>/` — frozen instrument-2.1 targets from the OpenCascade builder (prompt 1). Do not regenerate
  after the mesh builder lands; they are what the rewrite is measured against.
- `make_references.py` — the freezer (BREP builder only).
- `test_regression.py` — current builder vs. references: watertight, surface distance, same reading.
- planned: `sanity_animals.py` (2 segments -> bound; huge lateral spines -> open; tail wider than head -> double),
  `noise_floor.py` (same params x jitter x mesh tol x bevel -> repeatability floor that sets THETA_TOL_DEG).
- `test_mesh.py` — mesh.py: closed/symmetric shells, exact booleans, hinge port, BREP plate comparison (skipped without build123d).
- `test_parts.py` — parts.py: proetida segments/tail vs frozen references (median < 0.3 mm), a/b/c cells fuse to the row
  shell, every frozen order builds closed at its frozen bevel, agnostida's 45-deg tip severing reproduces.
- `test_head.py` — cephalon vs frozen references (six clean orders), eye solid vs the BREP eye (skipped without build123d),
  head cells fuse, prong primitive d = 1/2/3/5, genalPath fallback. phacopida head is an xfail (see CHANGES prompt 4).
- `test_schema.py` — schema 6.0 cell coverage, presets carry no dead keys, migrate(), prongs build, and the whole
  mesh-built animal reads the same on instrument 2.1 as the frozen BREP reading (proetida, corynexochida, harpetida).
- prompt 6: `make_references.py` moved to legacy/ (needs the BREP builder). test_regression.py now builds through
  instrument.build_animal and skips frozen BREP build defects and head surfaces (spline-biased).
- `test_web.py` — Flask API: schema/cells, presets, build + cache + GLB, measure (slow).
