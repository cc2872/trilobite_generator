"""The agnostid on the isopod model: agnostida preset P, the 'agnostus' shaped head, isopod ball joints.
Writes the isopod sheet, shaded photo views, and the print STL."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "isopod_model"))
import numpy as np, trimesh
import schema, blueprint
import body as IBODY
import head_styles as HS
import render as R

OUT = "/home/claude/outputs"
os.makedirs(OUT, exist_ok=True)

P0 = json.load(open(os.path.join(HERE, "presets", "agnostida.json")))["params"]
P = schema.coerce(P0, base=schema.table_defaults())
P.update({"borderWidth": 0.16, "marginHeight": 0.18, "effacement": 0.45, "widthTail": 0.85})
P = schema.coerce(P, base=schema.table_defaults())

face, _ = HS.style("agnostus")
outline = HS.outline("agnostus")

t0 = time.time()
parts, plans, PV, K = IBODY.build(P, "shaped", face=face, outline=outline, rows=None)
s = K["s"]
pieces = [p.copy().apply_scale(1.0 / s) for p in parts]
names = ["head"] + [f"seg{i}" for i in range(len(pieces) - 2)] + ["tail"]
pivots = [(float(y / s), float(z / s)) for (y, z) in PV]
print("built", len(pieces), "pieces in", round(time.time() - t0, 1), "s; extents",
      np.round(trimesh.util.concatenate(pieces).extents, 1))

whole = trimesh.util.concatenate(pieces)
whole.export(os.path.join(OUT, "agnostid_isopod.stl"))

eye = None
Sh = plans[0][2] if len(plans[0]) > 2 else None
if isinstance(Sh, dict) and Sh.get("eye"):
    Eh = Sh["eye"]; eye = [float(Eh["xe"]) / s, (float(Eh["ye"]) + float(plans[0][3])) / s, float(Eh["eR"]) / s]

info = dict(names=names, pivots=pivots, curl_deg=getattr(IBODY, "CURL", 30.0),
            params=schema.param_hash(P), schema=schema.SCHEMA_VERSION,
            build_seconds=round(time.time() - t0, 1),
            volumes=[float(p.volume) for p in pieces],
            meta=[dict(bodies=p.body_count, watertight=bool(p.is_watertight)) for p in pieces],
            checks={}, head="agnostus", body=None, eye=eye,
            date=time.strftime("%Y-%m-%d"))
blueprint.isopod_sheet(pieces, P, info, os.path.join(OUT, "agnostid_isopod_sheet.png"))
print("sheet done", round(time.time() - t0, 1), "s")

w = whole.copy()
w.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (0, 0, 1)))
R.sheet([[(w, 90, -90, "TOP"), (w, 30, -60, "3/4")],
         [(w, 0, -90, "SIDE"), (w, 10, -25, "FRONT 3/4")]],
        os.path.join(OUT, "agnostid_isopod_3d.png"), W=560, H=420)
print("renders done", round(time.time() - t0, 1), "s")
