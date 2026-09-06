# Trilobite Morphospace Generator

Everything in one place: the generator, the enrollment instrument, the website, the skin system,
the extracted reference assets, tests, renders.

## Layout
    schema.py            parameter table as data (52 knobs, 9 groups) — everything derives from it
    fields.py            along-body curves (width curve, spine field, effacement)
    trilobite.py         generator: surface-as-function shells, head/segments/tail, hinges, assembly, export
    instrument.py        enrollment ruler: all-pairs overlap, e_max, closure gap, class, print validity
    skins.py             reference skins as shape keys: register → resample → blend → plate functions
    skins/               registered end-member heads (.npz): olenoides, gltf, proetida, harpetida (+ harpid shield source)
    trilobite_web.py     website server (schema-driven knobs, meshes served from memory)
    index.html           website front end (three.js viewer, roll slider)
    Dockerfile, requirements.txt   deployment (Render etc.)
    tests/               test_invariants.py (generator + instrument), test_skins.py (skin system)
    render.py            offline renders
    reference_assets/    extracted heads, legs, antennae, bodies from the five reference models (whole, no decimation)
                         + manifest.json (provenance, triangle counts, how each head was cut) + extract.py
    v4_gallery.png       reference vs spiny variant, flat and at measured e_max
    skins_check.png      the four registered head fields and a blend
    skin_plates.png      CAD plates built from real skins and a blend

## Run
    pip install -r requirements.txt
    python trilobite.py            # build default animal, export STLs, animate in the OCP viewer
    python instrument.py           # measure it
    python trilobite_web.py        # http://localhost:8765
    pytest -q                      # ~1 min

## Status
- Generator v4 + instrument: reference body e_max 1.0 (126° free curl, partial); spiny variant 0.50.
- Skin system: 4 heads registered and blendable; plates build from any blend in ~1 s. Phacops excluded (bad scan).
- Not yet wired: headFamily selector + blend weights into build_cephalon; segment/tail skins; appendage placement.
- Known: tail occasionally builds as >1 solid (unfused spine); tessellation fallback on some parts.

## Website on GitHub / Render
Push the root files (not reference_assets/ — 440 MB — unless you want them versioned; Git LFS if so).
