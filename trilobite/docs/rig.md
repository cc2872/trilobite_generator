# rig.py — assets on the parametric skeleton

Hangs the cut sculpt pieces on the pin-hinge skeleton you already have and makes them
elongate / shorten / articulate when the skeleton parameters change. It uses your real
skeleton functions (`fields.pitch`, `fields.hinge_z`, `fields.width_curve`, `schema`) and the
same body-coordinate deformation law as `wrap.py` — not a parallel invention.

## The binding
1. Each cut piece keeps its body coordinate `s = y / L0` (L0 = the sculpt's own length).
2. **wrap** to the target params P: `s → s'` (piecewise-linear through P's head/thorax/tail
   fractions, thorax linear so segments stay ~uniform), then
   `y' = P.length · s'`, `x' ·= width_curve(P)/width_curve(P0)`, `z' ·= P.relief/relief0`.
   → the asset stretches with **length**, **pitch** and **width**.
3. Hinge axes sit at the wrapped seam positions, height `fields.hinge_z(P)`.
4. **curl**: each piece rotates `−enroll·maxAngle` about its hinge, accumulated head→tail —
   the same chain as `instrument.transforms` / `skin.posed`.

```python
import furrow_slicer as fs, cutter as C, rig, schema
info = fs.all_seams(v, n_segments=8, band=(36,102))
idx  = C.assign_faces(info["v"], faces, info["seams"])
pieces_v = [info["v"]]*(idx.max()+1); pieces_f = [faces[idx==k] for k in range(idx.max()+1)]
P0 = schema.coerce(dict(length=info["v"][:,1].max()))
res = rig.rig(pieces_v, pieces_f, list(info["mins"]), info["v"][:,1].max(),
              P=schema.coerce(dict(P0, length=190)), P0=P0, enroll=0.5)
# res["posed"] : per-piece world vertices;  rig.posed_skeleton(res) : the bone chain
```

`rig_skeleton.png` shows the assets on the posed skeleton at L=130 / 190 / 95 / curled.

## Honest limits
- Binds at the cut's own segment count. Changing `segCount` needs a resample step
  (`wrap.resample_segments` does it for solids via a boolean trim; for these open shells,
  resample by drop/duplicate without the trim). Length / pitch / width / relief / curl are wired;
  segCount-change is not yet wired for the shells.
- The wrap stretches the sculpt's existing furrows proportionally; it does not re-derive furrow
  shape at the new proportions.
- Curl is rigid per-segment about the wrapped hinge axes, so segments can gap/overlap at the
  hinges — that gap is exactly what `instrument.py` is meant to measure. This rig is the poser,
  not the collision check.
