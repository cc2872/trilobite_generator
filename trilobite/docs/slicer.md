# Furrow-following slicer

Replaces the straight-cut logic in `website/slicer.py`. The old slicer traced each furrow as a
single-valued top-down curve `y(x)`, stopped when the height-field valley got ambiguous, then
extrapolated **straight** at the margin slope and extruded a **vertical wall**. Because pleural
spines sweep backward and overhang the furrow, that vertical wall sawed straight through them and
split every spine between two pieces — the shredded fringe.

This version traces each seam as the **furrow valley across the width**, constrained to sweep
backward (never forward into the spine ahead), blended toward a straight cut where the surface is
effaced, and continued past the body margin along the furrow's own tangent into the notch between
spine bases. Each pleural spine therefore stays whole with the segment it belongs to.

## What it does NOT rely on
Booleans. The purchased sculpts are open surface shells (tens of thousands of open edges), not
watertight solids, so `trimesh.boolean.intersection` refuses them. Cutting is done by
**face assignment** (`cutter.assign_faces` / `split_faces`): each face's piece index = how many seam
curves its centroid lies behind. Exact, fast, needs no watertightness, and preserves geometry. Feed
the resulting per-segment shells into the existing `skin.py` thicken + hinge stage for the printable /
enrollable route.

## Usage
```python
import furrow_slicer as fs, cutter as C
from load_any import load_species
mesh = load_species("olenoides")          # oriented X-across / Y-along(head -y) / Z-up, 130 mm
info = fs.all_seams(mesh.vertices, n_segments=8, band=(36,102))
idx  = C.assign_faces(info["v"], mesh.faces, info["seams"])   # piece index per face
pieces = C.split_faces(info["v"], mesh.faces, idx)            # head, seg0..segN, tail
```

You specify the body plan (`n_segments`, thorax `band` in mm, `flip_y`); the geometry follows. This
is deliberate — free furrow auto-detection was unreliable across effaced vs spiny forms. Verified
per-species config:

| species   | n_segments | band (mm)  | flip_y | notes                                  |
|-----------|-----------:|------------|:------:|----------------------------------------|
| olenoides | 8          | (36, 102)  | False  | spiny — seams follow pleurae           |
| harpetid  | 10         | (52, 122)  | False  | head = the large low-y brim            |
| proetida  | 9          | (24, 92)   | True   | came in flipped; head was at high y    |
| phacops   | —          | —          | —      | fossil: NOT sliced; eyes extracted     |

## Head stays whole
The head is piece 0 (everything in front of seam 0), so it is one piece by construction. Set the
front of `band` at the cephalon/thorax junction. Long genal prolongations (harpetid) that run the
body length are handled as separate parts in your parts library (`solid_genal_spine.stl`), not carried
by a transverse seam.

## phacops
Fossil scan, effaced, convex-side needs a z-flip. `phacops_eyes.py` extracts the two schizochroal eye
lobes (`phacops_eyes.stl`) rather than slicing the animal.

## Known limitation
A spine that curls *under* the following segment in Z (not just back in plan) is still clipped by the
vertical seam sweep. These sculpts' spines sweep mainly in-plane, so it's a minor correction.
