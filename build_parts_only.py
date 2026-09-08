import sys, json, time, os, pickle
import schema, trilobite as T, instrument2 as I2
name, out, bound, i0, i1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
os.makedirs(out, exist_ok=True)
P = schema.preset(name) if name in schema.PRESETS else schema.coerce(json.load(open(f"presets/{name}.json")).get("params"), base=schema.table_defaults())
Pm = I2.measurement_params(P, bound); names = T.PART_NAMES(Pm)
fns = [lambda: T.build_cephalon(Pm)] + [(lambda i=i: T.build_segment(Pm, i)) for i in range(int(Pm["segCount"]))] + [lambda: T.build_pygidium(Pm)]
for k in range(i0, min(i1, len(names))):
    t0 = time.time(); p = T._build_checked(fns[k], names[k])
    m = getattr(p, "_checked_mesh", None) or T.to_trimesh(p, *T.SANE_MESH_TOL)
    m.export(f"{out}/{names[k]}_bevel{int(bound)}.stl")
    print(names[k], round(time.time()-t0), "s solids", len(p.solids()), [n for n in T.BUILD_NOTES if n[0]==names[k]], flush=True)
