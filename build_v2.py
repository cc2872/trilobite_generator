import sys, json, time, os
import schema, trilobite as T, instrument2 as I2
name = sys.argv[1]; out = sys.argv[2]; bound = float(sys.argv[3]) if len(sys.argv) > 3 else I2.BOUND_DEG
os.makedirs(out, exist_ok=True)
if name in schema.PRESETS: P = schema.preset(name)
else:
    d = json.load(open(f"presets/{name}.json")); P = schema.coerce(d.get("params", d), base=schema.table_defaults())
Pm = I2.measurement_params(P, bound)
t0 = time.time(); parts = T.parts_list(Pm); names = T.PART_NAMES(Pm)
meshes = []
for n, p in zip(names, parts):
    m = getattr(p, "_checked_mesh", None) or T.to_trimesh(p, *T.SANE_MESH_TOL)
    m.export(f"{out}/{n}_bevel{int(bound)}.stl"); meshes.append(m)
print(name, "measurement build", round(time.time()-t0), "s  notes", T.BUILD_NOTES, " watertight", [m.is_watertight for m in meshes], flush=True)
hb = I2.head_body_mesh(Pm)
r = I2.measure(P, meshes, parts, bound_deg=bound, budget_s=900, head_body=hb)
r["closure_ref"] = "head body (arms off)" if hb is not None else "head"
r["build_notes"] = list(T.BUILD_NOTES); r["unsane_parts"] = [n[0] for n in T.BUILD_NOTES if n[1] == "UNSANE after retries"]
r["gap_curve"] = I2.sweep_gap(P, meshes, [t for t in (0, 10, 18, 25, 30, 35, 40, 45) if t <= bound], head_body=hb) if not r["measure_timed_out"] else None
json.dump(r, open(f"{out}/measure_v2_bevel{int(bound)}.json", "w"), indent=1)
print(json.dumps({k: r[k] for k in ("theta_joint_deg","total_deg","closure_gap_mm","gap_over_L","limited_by","stopped_by","stop_recommended_deg","v1_equivalent_e_max","print_valid","violations","seconds")}), flush=True)
print("gap curve", r["gap_curve"])
