# legacy/ — the OpenCascade builder and the v5 website (kept until prompt 7 replaces the site)
- trilobite.py  the BREP builder (build123d). The mesh builder (mesh.py + parts.py) replaced it; tests/references were frozen from it.
- instrument2.py  instrument 2.1 on the BREP builder; merged into ../instrument.py.
- trilobite_web.py, index.html, Dockerfile  the v5 site (dials). Prompt 7 rewrites it on the 3x3 CELLS contract.
- skins.py  scan-derived skins; becomes fit.py (scan -> parameters) later.
- sheet10.py  order gallery sheet; folds into blueprint.py in prompt 8.
Running the legacy site needs build123d; nothing outside this folder does.
