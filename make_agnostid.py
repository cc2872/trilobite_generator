"""Build an agnostid-type animal (two subequal shields, 2 segments, blind, broad rims) from the
agnostida preset, then draw the blueprint sheet and shaded 3D views."""
import json, os, time
import numpy as np, trimesh
import schema, assemble, joints, blueprint
from joints import pin as PIN
import sys
sys.path.insert(0, "isopod_model")
import render as R

OUT = "/home/claude/outputs"
os.makedirs(OUT, exist_ok=True)

P0 = json.load(open("presets/agnostida.json"))["params"]
P, notes = schema.coerce_report(P0, base=schema.table_defaults())

# nudges toward the reference image: broad concave border rims on both shields,
# strong effacement with the elongate axial lobe kept, shields a touch rounder
P.update({
    "borderWidth": 0.16,
    "marginHeight": 0.18,
    "effacement": 0.45,
    "widthTail": 0.85,
})
P, notes = schema.coerce_report(P, base=schema.table_defaults())

t0 = time.time()
parts = assemble.animal(P, "pin")
print("built", len(parts), "parts in", round(time.time() - t0, 1), "s")

mats0 = joints.get("pin").transforms_deg(P, 0.0)
flat = trimesh.util.concatenate([m.copy().apply_transform(T) for m, T in zip(parts, mats0)])
flat.export(os.path.join(OUT, "agnostid.stl"))
print("flat extents", np.round(flat.extents, 1))

m = dict(limited_by="not measured", enroll_class="—",
         hinge_z=round(PIN.hinge_z(P), 2), pitch=round(PIN.joint_offsets(P)[1], 2),
         knuckle=round(PIN.hinge_width(P) / int(P["nKnuckles"]) - P["clearance"], 2),
         e_max="—", params=schema.param_hash(P), print_valid=True)
blueprint.sheet(flat, P, m, os.path.join(OUT, "agnostid_blueprint.png"))
print("sheet done", round(time.time() - t0, 1), "s")

w = flat.copy()
w.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (0, 0, 1)))
R.sheet([[(w, 90, -90, "TOP"), (w, 30, -60, "3/4")],
         [(w, 0, -90, "SIDE"), (w, 10, -25, "FRONT 3/4")]],
        os.path.join(OUT, "agnostid_3d.png"), W=560, H=420)
print("renders done", round(time.time() - t0, 1), "s")
