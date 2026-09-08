"""Measure a preset on instrument 2.0 from part STLs already in out10/<name>/ (built by build_parts_only.py)."""
import sys, json, os, trimesh, schema, trilobite as T, instrument2 as I2
name, bound = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
d = json.load(open(f"presets/{name}.json")); P = schema.coerce(d["params"], base=schema.table_defaults()); Pm = I2.measurement_params(P, bound)
names = T.PART_NAMES(P); out = f"out10/{name}"
meshes = [trimesh.load(f"{out}/{n}_bevel{int(bound)}.stl") for n in names]
hb = I2.head_body_mesh(Pm)
r = I2.measure(P, meshes, None, bound_deg=bound, budget_s=200, head_body=hb)
th = r["theta_joint_deg"] or 0.0
r.update(preset=name, order=d.get("order"), blurb=d.get("blurb"), unsane_parts=[n for n, m in zip(names, meshes) if not m.is_watertight],
         closure_ref="head body" if hb is not None else "head", wedge=dict(reach=T.WEDGE_REACH, reach_wide=T.WEDGE_REACH_WIDE),
         gap_curve=I2.sweep_gap(P, meshes, [t for t in (0, 5, 10, 15, 18, 20, 25, 30, 35, 40, 45) if t <= bound], head_body=hb) if not r["measure_timed_out"] else None)
for tag, ang in (("flat", 0.0), ("posed", th)):
    mats = I2.transforms_deg(P, ang); trimesh.util.concatenate([m.copy().apply_transform(M) for m, M in zip(meshes, mats)]).export(f"{out}/{tag}.stl")
for n in names: os.replace(f"{out}/{n}_bevel{int(bound)}.stl", f"{out}/{n}.stl")
json.dump(r, open(f"{out}/measure_v2.json", "w"), indent=1)
msg = f"{name}: bound {bound} | theta {r['theta_joint_deg']} total {r['total_deg']} gap {r['closure_gap_mm']} {r['limited_by']} by {r['stopped_by'][:2]} print {r['print_valid']} {r['violations'][:2]} unsane {r['unsane_parts']}"
print(msg); open("out10/progress.txt", "a").write(msg + "\n")
