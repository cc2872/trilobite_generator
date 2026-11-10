"""Iterate the isopod-model agnostid toward the reference drawing: build with P/outline/face edits,
render the dorsal view next to the reference. Usage: python iterate_agnostid.py out.png '{"P": {...}, "outline": {...}, "face": {...}}'"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "isopod_model"))
import numpy as np, trimesh
import schema
import body as IBODY
import head_styles as HS
import render as R
from PIL import Image

REF = "/mnt/user-data/uploads/4dab61c3-image.png"

def build(edits):
    P0 = json.load(open(os.path.join(HERE, "presets", "agnostida.json")))["params"]
    P = schema.coerce(P0, base=schema.table_defaults())
    P.update(HS.style("agnostus")[1])                      # the style's own parameter values, as the page applies them
    P.update(edits.get("P", {}))
    P = schema.coerce(P, base=schema.table_defaults())
    face = dict(HS.style("agnostus")[0], **edits.get("face", {}))
    outline = dict(HS.outline("agnostus"), **edits.get("outline", {}))
    t0 = time.time()
    parts, plans, PV, K = IBODY.build(P, "shaped", face=face, outline=outline, rows=edits.get("rows"))
    s = K["s"]
    pieces = [p.copy().apply_scale(1.0 / s) for p in parts]
    print("built", len(pieces), "pieces in", round(time.time() - t0, 1), "s")
    return P, pieces, PV, K, plans

if __name__ == "__main__":
    out = sys.argv[1]; edits = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    P, pieces, PV, K, plans = build(edits)
    whole = trimesh.util.concatenate(pieces)
    whole.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (0, 0, 1)))
    tmp = out.replace(".png", "_r.png")
    R.sheet([[(whole, 90, -90, "TOP")]], tmp, W=560, H=760)
    a = Image.open(tmp).convert("RGB"); b = Image.open(REF).convert("RGB")
    b = b.resize((int(b.width * a.height / b.height), a.height))
    c = Image.new("RGB", (a.width + b.width, a.height), "black")
    c.paste(b, (0, 0)); c.paste(a, (b.width, 0)); c.save(out)
    print("extents", np.round(whole.extents, 1).tolist())
