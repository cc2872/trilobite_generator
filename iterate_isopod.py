"""Iterate an isopod-model animal toward a reference picture (9 Nov 2026; generalises iterate_agnostid.py):
build(preset, style, edits) -> (P, pieces, PV, K, plans) with P = preset params + the style's params + edits["P"],
face/outline = the style's + edits, rows = edits.get("rows"). Usage:
    python iterate_isopod.py asaphida isotelus ref.png out.png '{"P": {...}, "outline": {...}, "face": {...}}'"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "isopod_model"))
import numpy as np, trimesh
import schema
import body as IBODY
import head_styles as HS
import render as R
import check as C
from PIL import Image

def build(preset, style, edits):
    P0 = json.load(open(os.path.join(HERE, "presets", f"{preset}.json")))["params"]
    P = schema.coerce(P0, base=schema.table_defaults())
    P.update(HS.style(style)[1]); P.update(edits.get("P", {}))
    P = schema.coerce(P, base=schema.table_defaults())
    face = dict(HS.style(style)[0], **edits.get("face", {}))
    outline = dict(HS.outline(style), **edits.get("outline", {}))
    t0 = time.time()
    parts, plans, PV, K = IBODY.build(P, "shaped", face=face, outline=outline, rows=edits.get("rows"))
    s = K["s"]; pieces = [p.copy().apply_scale(1.0 / s) for p in parts]
    print("built", len(pieces), "pieces in", round(time.time() - t0, 1), "s")
    return P, pieces, PV, K, plans

def report(pieces, PV, K, angles=(0.0, 18.0, 20.0)):
    s = K["s"]; pivots = [(float(y / s), float(z / s)) for (y, z) in PV]
    print("widths:", [round(float(np.ptp(p.vertices[:, 0])), 1) for p in pieces], "| heights:", [round(float(p.vertices[:, 2].max()), 1) for p in pieces])
    for a in angles: print(f"  contacts at {a:g} deg:", C.contacts(pieces, pivots, a) or ("nothing" if a == 0 else "clean"))
    return pivots

def vs_reference(pieces, pivots, ref, out, pose=(10.0,) * 9):
    rot = trimesh.transformations.rotation_matrix(np.pi, (0, 0, 1))
    T = C.pose(pivots, list(pose[: len(pivots)]))
    m = trimesh.util.concatenate([p.copy().apply_transform(T[k]) for k, p in enumerate(pieces)]); m.apply_transform(rot)
    tmp = out.replace(".png", "_r.png"); R.sheet([[(m, 90, -90, "")]], tmp, W=560, H=760)
    a = Image.open(tmp).convert("RGB"); b = Image.open(ref).convert("RGB")
    b = b.resize((int(b.width * a.height / b.height), a.height))
    c = Image.new("RGB", (a.width + b.width, a.height), "black"); c.paste(b, (0, 0)); c.paste(a, (b.width, 0)); c.save(out)

if __name__ == "__main__":
    preset, style, ref, out = sys.argv[1:5]; edits = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    P, pieces, PV, K, plans = build(preset, style, edits); pivots = report(pieces, PV, K); vs_reference(pieces, pivots, ref, out)
