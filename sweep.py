"""
sweep.py — the v1 morphospace sweep on the mesh builder (PREREG_v1_sweep.md §2, §4), and the ten-preset batch.

    python sweep.py presets [name ...]                 # every presets/*.json -> out/<name>/ (reading, STLs, flat/posed) + out/presets.csv
    python sweep.py design  --n-per-order 150 --n-control 50 --seed 1 --out design.csv
    python sweep.py run     design.csv --out sweep_rows.csv --workers 4 --timeout 600
    python sweep.py one     design.csv <row_index> --out /tmp/x.json        # what a worker runs; for debugging

design: one row per animal; six axes Latin-hypercube sampled inside each order's preset ranges (segCount, overlap,
cephFrac/pygFrac, relief, widthThoraxRear, spine_scale); the control subset varies ONLY eyeSize (the null axis).
run: one child process per design row (OOM isolation), timeouts/kills/exceptions recorded as invalid rows, CSV
flushed per animal, resumable. A whole animal builds in ~2.5 s and measures in ~7 s on the measurement grid.
"""
import sys, os, json, csv, time, argparse, subprocess, traceback, glob
import numpy as np

AXES = {"segCount": ("add", -3, +3), "overlap": ("abs", 0.2, 0.8), "cephFrac": ("mul", 0.7, 1.3), "pygFrac": ("mul", 0.7, 1.3),
        "relief": ("mul", 0.6, 1.4), "widthThoraxRear": ("mul", 0.6, 1.4), "spine_scale": ("abs", 0.0, 1.5)}
SPINE_KEYS = ("spineBase", "genalSpine", "pygSpine", "axialSpine", "pygMarginalLen")
NULL_AXIS = ("eyeSize", 0.0, 0.45)

def lhs(n, k, rng):
    u = np.empty((n, k))
    for j in range(k): u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u

def load_presets():
    out = {}
    for f in sorted(glob.glob("presets/*.json")):
        name = os.path.basename(f)[:-5]
        if not name.startswith("_"): out[name] = json.load(open(f))["params"]
    return out

def design(n_per_order, n_control, seed, out_path):
    import schema
    rng = np.random.default_rng(seed); presets = load_presets(); rows = []; keys = list(AXES)
    for name, base in presets.items():
        base = schema.coerce(base, base=schema.table_defaults()); u = lhs(n_per_order, len(keys), rng)
        for r in range(n_per_order):
            P = dict(base); rec = dict(preset=name, is_control=0, design_seed=seed, design_index=len(rows))
            for j, k in enumerate(keys):
                kind, lo, hi = AXES[k]; x = lo + (hi - lo) * u[r, j]
                if k == "spine_scale":
                    for sk in SPINE_KEYS: P[sk] = base[sk] * x
                    rec["spine_scale"] = round(float(x), 4)
                elif kind == "add": P[k] = base[k] + int(round(x))
                elif kind == "mul": P[k] = base[k] * x
                else: P[k] = x
            Pc, notes = schema.coerce_report(P, base=base)
            rec["params"] = json.dumps(Pc, sort_keys=True); rec["design_notes"] = ";".join(notes); rows.append(rec)
        per = max(2, n_control // len(presets))
        for e in np.linspace(NULL_AXIS[1], NULL_AXIS[2], per):
            rows.append(dict(preset=name, is_control=1, design_seed=seed, design_index=len(rows), spine_scale=None,
                             params=json.dumps(schema.coerce(dict(base, eyeSize=float(e)), base=base), sort_keys=True), design_notes=""))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["design_index", "preset", "is_control", "design_seed", "spine_scale", "design_notes", "params"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} animals -> {out_path}  ({len(presets)} orders x {n_per_order} + control)")

def one(design_path, idx, out_json, bound=None):
    import schema, instrument as I
    rows = list(csv.DictReader(open(design_path))); d = rows[idx]
    P, notes = schema.coerce_report(json.loads(d["params"]), base=schema.table_defaults())
    notes = [n for n in (d.get("design_notes") or "").split(";") if n] + notes
    r, B = I.read(P, bound=bound)
    spine = float(d["spine_scale"]) if d.get("spine_scale") not in (None, "", "None") else None
    rw = I.row(r, P, preset=d["preset"], is_control=int(d["is_control"]), spine_scale=spine, schema_notes=notes, build_seconds=B["build_seconds"])
    rw["design_index"] = idx
    json.dump(dict(row=rw, full=r), open(out_json, "w"), default=str)
    return rw

def invalid_row(d, idx, reason):
    import schema, instrument as I
    P = schema.coerce(json.loads(d["params"]), base=schema.table_defaults())
    rw = I.row(dict(limited_by="invalid", reason=reason, enroll_class="censored", instrument=I.INSTRUMENT_VERSION), P, preset=d["preset"],
               is_control=int(d["is_control"]), spine_scale=(float(d["spine_scale"]) if d.get("spine_scale") not in (None, "", "None") else None))
    rw["valid"] = 0; rw["design_index"] = idx; return rw

def run(design_path, out_path, workers, timeout_s, bound=None):
    import instrument as I, threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = list(csv.DictReader(open(design_path))); cols = ["design_index"] + I.ROW_COLUMNS
    done = {int(r["design_index"]) for r in csv.DictReader(open(out_path))} if os.path.exists(out_path) else set()
    todo = [i for i in range(len(rows)) if i not in done]; print(f"{len(todo)} to run, {len(done)} already in {out_path}", flush=True)
    tmp = out_path + ".children"; os.makedirs(tmp, exist_ok=True); lock = threading.Lock()
    new = not os.path.exists(out_path); fout = open(out_path, "a", newline=""); w = csv.DictWriter(fout, fieldnames=cols)
    if new: w.writeheader(); fout.flush()
    def child(i):
        oj = f"{tmp}/{i}.json"; t0 = time.time()
        cmd = [sys.executable, __file__, "one", design_path, str(i), "--out", oj] + (["--bound", str(bound)] if bound else [])
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if p.returncode == 0 and os.path.exists(oj): rw = json.load(open(oj))["row"]
            elif p.returncode < 0: rw = invalid_row(rows[i], i, f"oom_or_killed:rc{p.returncode}")
            else: rw = invalid_row(rows[i], i, "build_error:" + (p.stderr.strip().splitlines() or ["?"])[-1][:160])
        except subprocess.TimeoutExpired: rw = invalid_row(rows[i], i, f"timeout:{timeout_s}s")
        rw["_wall"] = round(time.time() - t0); return rw
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(child, i): i for i in todo}
        for n, f in enumerate(as_completed(futs), 1):
            rw = f.result()
            with lock: w.writerow({k: rw.get(k) for k in cols}); fout.flush(); os.fsync(fout.fileno())
            print(f"[{n}/{len(todo)}] #{rw['design_index']} {rw['preset']:14s} {rw['limited_by']:9s} {rw['enroll_class']:11s} theta {rw.get('theta_joint_deg')} "
                  f"bevel {rw.get('bevel_built_deg')} {rw.get('reason','')[:60]} ({rw['_wall']}s)", flush=True)
    fout.close()

def presets_batch(names, out_dir="out", grid=None):
    """The ten-preset table: reading + STLs per preset, one CSV. Replaces batch_v2 / build_v2 / measure_from_stl."""
    import instrument as I, schema
    os.makedirs(out_dir, exist_ok=True); csv_path = os.path.join(out_dir, "presets.csv")
    for name in names:
        d = json.load(open(f"presets/{name}.json")); P, notes = schema.coerce_report(d["params"], base=schema.table_defaults())
        od = os.path.join(out_dir, name); os.makedirs(od, exist_ok=True); t0 = time.time()
        try:
            r, B = I.read(P, export_dir=od, **({"grid": grid} if grid else {}))
            th = r["theta_joint_deg"] or 0.0
            I.posed_stl(P, B["meshes"], 0.0, os.path.join(od, "flat.stl")); I.posed_stl(P, B["meshes"], th, os.path.join(od, "posed.stl"))
            if r["theta_joint_deg"] is not None:            # closure curve for the gallery sheet
                b = r["bevel_built_deg"]
                r["gap_curve"] = I.sweep_gap(P, B["meshes"], [t for t in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45) if t <= b], head_body=B["head_body"])
            r.update(preset=name, order=d.get("order"), schema_notes=notes)
            json.dump(r, open(os.path.join(od, "measure.json"), "w"), indent=1, default=str)
            rw = I.row(r, P, preset=name, schema_notes=notes, build_seconds=B["build_seconds"]); I.write_row(rw, csv_path)
            msg = (f"{name}: bevel {r['bevel_built_deg']} probe {r['bevel_probe']} build {B['build_seconds']}s measure {r['seconds']}s | "
                   f"{r['limited_by']} {r['enroll_class']} theta {r['theta_joint_deg']} total {r['total_deg']} s_tail {r['s_tail']} by {r['stopped_by'][:1]} "
                   f"{r['reason']} unsane {r['unsane_parts']} print {r['print_valid']} {r['violations'][:2]}")
        except Exception as ex:
            msg = f"{name}: FAILED {ex}\n{traceback.format_exc()[-600:]}"
        print(msg, flush=True); open(os.path.join(out_dir, "progress.txt"), "a").write(msg.split("\n")[0] + f"  [{round(time.time()-t0)}s]\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("design"); a.add_argument("--n-per-order", type=int, default=150); a.add_argument("--n-control", type=int, default=50)
    a.add_argument("--seed", type=int, default=1); a.add_argument("--out", default="design.csv")
    b = sub.add_parser("run"); b.add_argument("design"); b.add_argument("--out", default="sweep_rows.csv")
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)); b.add_argument("--timeout", type=int, default=600)
    b.add_argument("--bound", type=float, default=None)
    c = sub.add_parser("one"); c.add_argument("design"); c.add_argument("index", type=int); c.add_argument("--out", required=True); c.add_argument("--bound", type=float, default=None)
    p = sub.add_parser("presets"); p.add_argument("names", nargs="*"); p.add_argument("--out", default="out")
    A = ap.parse_args()
    if A.cmd == "design": design(A.n_per_order, A.n_control, A.seed, A.out)
    elif A.cmd == "run": run(A.design, A.out, A.workers, A.timeout, A.bound)
    elif A.cmd == "presets": presets_batch(A.names or [os.path.basename(f)[:-5] for f in sorted(glob.glob("presets/*.json")) if not os.path.basename(f).startswith("_")], A.out)
    else:
        try: one(A.design, A.index, A.out, A.bound); print("ok")
        except Exception: traceback.print_exc(); sys.exit(1)
