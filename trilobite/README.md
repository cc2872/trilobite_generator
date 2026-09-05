# Trilobite — parametric sculpt megafolder

Open **`viewer.html`** in a browser (needs internet for the Three.js CDN). Black-and-white,
minimal. Pick a specimen, then:

- **Body length / Head length** — the real cut sculpt mesh elongates and the cephalon
  re-proportions live.
- **Enrollment** — the pin-hinge chain curls, each segment rotating about its hinge.
- **Legs** — checkbox; box limbs under each segment (curl with their segment).
- **Skeleton** — checkbox; the hinge chain drawn along the base, underneath the animal.

Segment count is fixed per specimen (it's intrinsic to each sculpt). Phacops is a static fossil
(no thorax), included for completeness.

## What's here
```
viewer.html            the black-and-white parametric UI  ← start here
species_data.js        the four sculpts baked to web data (mesh + segment index + joints)
pipeline/              the Python that produces everything
  load_any.py            load / orient / scale each sculpt to the 130 mm frame
  furrow_slicer.py       cut along the interpleural furrows (spines stay whole)
  cutter.py              face-assignment split (no watertight requirement)
  rig.py                 wrap (elongate / head) + curl on the skeleton
  assemble.py            full assembly: pieces + eyes/legs
  real_legs.py           orient the sculpt's real legs into the shell frame
  assemble_final.py      olenoides + real legs → STL
  bake_web.py            export species_data.js for the viewer
  phacops_eyes.py        extract the phacops eyes (fossil route)
  fields.py, schema.py   the parametric skeleton definition (your existing code)
meshes/                printable STLs (colourless)
  olenoides_assembled.stl   shell + real sculpted legs, dorsal-up
  olenoides_real_legs.stl   the legs alone, aligned
  phacops_eyes.stl          extracted schizochroal eyes
renders/               proof images (cut → skeleton → assembled → generator)
generator/             trilobite_generator.html — the fully procedural model (bonus)
docs/                  per-module notes
```

## Two routes, on purpose
- **Sculpt route** (viewer + pipeline + meshes): the real purchased sculpts, cut and rigged.
  Real geometry, fixed segment counts.
- **Procedural route** (generator/): a from-scratch parametric trilobite where *everything* is a
  knob — segment count, procedural eyes by lens count, etc. Use it to explore body plans the
  sculpts don't cover.

## Rebuild the web data
```
cd pipeline && python bake_web.py     # regenerates ../species_data.js from meshes.pkl
```

## Known edges
- The viewer deforms length / head / enrollment / legs. Width and relief scaling exist in `rig.py`
  but aren't exposed in the UI yet.
- Legs in the viewer are box placeholders (the real sculpted legs are in `meshes/` and only exist
  for Olenoides); box limbs keep the UI light and uniform across species.
- The instrument (`e_max`, collision) still needs running on the rigged pieces — that's the next gate.
