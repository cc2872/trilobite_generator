import sys, json, time, os, glob, traceback
import numpy as np, trimesh
import schema, trilobite as T, instrument2 as I2
BOUNDS = (45.0, 35.0, 25.0)
names = sys.argv[1:] or [os.path.basename(f)[:-5] for f in sorted(glob.glob("presets/*.json"))]
for name in names:
    out = f"out10/{name}"; os.makedirs(out, exist_ok=True)
    log = open(f"{out}/log.txt", "a")
    try:
        d = json.load(open(f"presets/{name}.json")); P = schema.coerce(d["params"], base=schema.table_defaults())
        # probe the bevel on the middle segment
        bound = None; probe = {}
        for b in BOUNDS:
            Pm = I2.measurement_params(P, b); T.BUILD_NOTES.clear(); T.GRID_JITTER = 0
            try:
                s = T.build_segment(Pm, int(P["segCount"]) // 2); n = len(s.solids())
            except Exception as ex:
                n = -1
            probe[b] = n
            if n == 1: bound = b; break
        if bound is None: bound = BOUNDS[-1]
        Pm = I2.measurement_params(P, bound); T.BUILD_NOTES.clear()
        t0 = time.time(); parts = T.parts_list(Pm); pn = T.PART_NAMES(Pm)
        meshes = []
        for n, p in zip(pn, parts):
            m = getattr(p, "_checked_mesh", None) or T.to_trimesh(p, *T.SANE_MESH_TOL)
            m.export(f"{out}/{n}.stl"); meshes.append(m)
        build_s = round(time.time() - t0)
        notes = list(T.BUILD_NOTES); unsane = [n[0] for n in notes if n[1] == "UNSANE after retries"]
        hb = I2.head_body_mesh(Pm)
        r = I2.measure(P, meshes, parts, bound_deg=bound, budget_s=600, head_body=hb)
        th = r["theta_joint_deg"] or 0.0
        r.update(preset=name, order=d.get("order"), blurb=d.get("blurb"), bevel_probe=probe, unsane_parts=unsane,
                 build_notes=notes, build_seconds=build_s, closure_ref="head body" if hb is not None else "head",
                 gap_curve=I2.sweep_gap(P, meshes, [t for t in (0, 5, 10, 15, 18, 20, 25, 30, 35, 40, 45) if t <= bound], head_body=hb)
                           if not r["measure_timed_out"] else None)
        for tag, ang in (("flat", 0.0), ("posed", th)):
            mats = I2.transforms_deg(P, ang)
            trimesh.util.concatenate([m.copy().apply_transform(M) for m, M in zip(meshes, mats)]).export(f"{out}/{tag}.stl")
        json.dump(r, open(f"{out}/measure_v2.json", "w"), indent=1)
        msg = f"{name}: bound {bound} probe {probe} build {build_s}s unsane {unsane} | theta {r['theta_joint_deg']} total {r['total_deg']} gap {r['closure_gap_mm']} {r['limited_by']} by {r['stopped_by'][:2]} print {r['print_valid']} {r['violations']}"
    except Exception as ex:
        msg = f"{name}: FAILED {ex}\n{traceback.format_exc()}"
    print(msg, flush=True); log.write(msg + "\n"); log.close()
    open("out10/progress.txt", "a").write(msg.split("\n")[0] + "\n")
