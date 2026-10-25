"""
scripts/library.py: build a large library of generated trilobites in batches, one STL per animal, named by its hash.

The animal IS its parameter set. The hash is taken over the parameters (schema version + coerced params, the same JSON
as schema.to_json), not over the STL bytes: Manifold's booleans are multithreaded, so two builds of the same parameters
give the same solid (volume equal to ~1e-10 relative) with a slightly different triangulation (a few hundred faces of
~300k), and the STL bytes differ. The STL is therefore a cache of the recipe; its sha256 is kept only to check the file.

Layout of a library directory:
    lib/LIBRARY.json                      sampler, seed, schema version, git commit, batch size (written by `design`)
    lib/design/batch_0000.csv             hash, index, sampler, preset, params (one row per animal, fixed before any build)
    lib/manifest/batch_0000.csv           one row per built animal (appended as each finishes; resumable)
    lib/stl/3f/3fa9c2d81b0e4c77.stl.gz    the flat (rest-pose) animal, gzip -6, sharded by the first two hex characters

Steps:
    python scripts/library.py design lib --sampler presets --n 20000 --seed 7 --batch-size 1000
    python scripts/library.py build lib --batch 0 --workers 8          # one batch; re-run the same line to resume
    python scripts/library.py build lib --all --workers 8              # every batch not yet finished
    python scripts/library.py status lib
    python scripts/library.py build lib --all --workers 8 --retry-failed   # give failed/timed-out animals another try
    python scripts/library.py landmarks lib --workers 14                  # landmarks only, no mesh (~0.1 s per animal)

Landmarks only: `landmarks` computes the TriloMorph landmarks of every animal in the design straight from its parameters,
without building the mesh, and writes lib/landmarks/batch_0000.csv (hash, index, preset, then x,y of cephalic LM1-16 and
pygidial LM1-7). It is ~500x faster than a build and needs no STL disk, so a million animals is hours, not months. Use it
for counts, accumulation curves and coverage; use `build` on a subset for anything that needs the solid (one-body rate,
renders, prints). A batch file is written whole and renamed when complete, so re-running skips finished batches.

Samplers:
    box      every parameter uniform over its schema range (the space the count was made on; most animals fail the build)
    presets  Latin hypercube around each of the order presets on the sweep's axes (sweep.py AXES), equal per order
             --presets NAME [NAME ...] limits it to named presets, including non-trilobites in presets_other/
             (e.g. --presets pillbug: 10,000 variations of the pill bug on the same axes)
    npz      rows of an .npz with arrays X (n x k) and keys (k) over a table-defaults base (e.g. cell representatives
             from the count), --npz path

Each animal is built in its own process (memory and a timeout per animal), at the widest bevel in (45, 35, 25) at which
every segment is one body, with the pin joint: the same build the instrument measures. --measure also runs instrument 2.1
and adds the reading to the manifest (about 3x the time).
"""
import os, sys, csv, json, gzip, time, glob, shutil, hashlib, argparse, subprocess, tempfile, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
csv.field_size_limit(10 ** 8)

DESIGN_COLS = ["hash", "index", "batch", "sampler", "preset", "seed", "params"]
MANIFEST_COLS = ["hash", "index", "batch", "sampler", "preset", "schema_version", "git_commit", "built", "one_body_each",
                 "unsane_parts", "bevel_deg", "n_parts", "n_faces", "volume_mm3", "length_mm", "width_mm", "stl", "stl_bytes",
                 "stl_sha256", "build_s", "landmarks", "limited_by", "enroll_class", "total_deg", "theta_joint_deg", "reason",
                 "error", "wall_s"]


def recipe_hash(P):
    """sha256 of the canonical recipe (schema version + coerced params), 16 hex = 64 bits (collision odds ~1e-10 at 1e5)."""
    import schema
    return hashlib.sha256(schema.to_json(P).encode()).hexdigest()[:16]


def git_commit():
    try:
        c = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ------------------------------------------------------------------------------------------------ design
def sample_box(n, rng):
    import schema
    params = [p for p in schema.PARAMS if p.kind != "expr" and p.group != "Hinge"]
    base = schema.table_defaults()
    for _ in range(n):
        P = dict(base)
        for p in params:
            P[p.key] = float(rng.uniform(p.lo, p.hi)) if p.kind == "float" else int(rng.integers(int(p.lo), int(p.hi) + 1))
        yield "", schema.coerce(P, quiet=True)


def load_named_presets(names):
    """None -> the ten trilobite orders in presets/ (exactly what sweep.py uses). Otherwise the named presets, looked up in
    presets/ and presets_other/ (non-trilobites such as pillbug live there so the sweep and the studio never see them)."""
    import sweep
    if not names: return sweep.load_presets()
    out = {}
    for nm in names:
        for d in ("presets", "presets_other"):
            f = os.path.join(ROOT, d, f"{nm}.json")
            if os.path.exists(f): out[nm] = json.load(open(f))["params"]; break
        else:
            sys.exit(f"no preset named {nm!r} in presets/ or presets_other/")
    return out


def sample_presets(n, rng, names=None):
    import schema, sweep, numpy as np
    presets = load_named_presets(names); names = list(presets); keys = list(sweep.AXES)
    per = -(-n // len(names))
    for name in names:
        base = schema.coerce(presets[name], base=schema.table_defaults(), quiet=True); u = sweep.lhs(per, len(keys), rng)
        for r in range(per):
            P = dict(base)
            for j, k in enumerate(keys):
                kind, lo, hi = sweep.AXES[k]; x = lo + (hi - lo) * u[r, j]
                if k == "spine_scale":
                    for sk in sweep.SPINE_KEYS: P[sk] = base[sk] * x
                elif kind == "add": P[k] = base[k] + int(round(x))
                elif kind == "mul": P[k] = base[k] * x
                else: P[k] = x
            yield name, schema.coerce(P, base=base, quiet=True)


def sample_npz(path):
    import schema, numpy as np
    d = np.load(path, allow_pickle=True); keys = [str(k) for k in d["keys"]]; base = schema.table_defaults()
    for row in d["X"]:
        P = dict(base); P.update({k: float(v) for k, v in zip(keys, row)})
        yield "", schema.coerce(P, quiet=True)


def design(lib, sampler, n, seed, batch_size, npz=None, presets=None):
    import numpy as np, schema
    if os.path.exists(os.path.join(lib, "LIBRARY.json")):
        sys.exit(f"{lib} already has a design; use a new directory (a design is fixed before anything is built)")
    os.makedirs(os.path.join(lib, "design"), exist_ok=True)
    rng = np.random.default_rng(seed)
    gen = {"box": lambda: sample_box(n, rng), "presets": lambda: sample_presets(n, rng, presets), "npz": lambda: sample_npz(npz)}[sampler]()
    seen, rows, dup = set(), [], 0
    for preset, P in gen:
        if sampler != "npz" and len(rows) >= n: break
        h = recipe_hash(P)
        if h in seen: dup += 1; continue
        seen.add(h); rows.append(dict(hash=h, sampler=sampler, preset=preset, seed=seed, params=json.dumps(P, sort_keys=True)))
    rows = [rows[j] for j in rng.permutation(len(rows))]      # shuffled: every batch is a random sample of the whole design,
    for i, r in enumerate(rows): r.update(index=i, batch=i // batch_size)   # so a library stopped part-way is still unbiased
    nb = 0
    for b in range(0, len(rows), batch_size):
        with open(os.path.join(lib, "design", f"batch_{b // batch_size:04d}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=DESIGN_COLS); w.writeheader(); w.writerows(rows[b:b + batch_size]); nb += 1
    meta = dict(sampler=sampler, n=len(rows), seed=seed, batch_size=batch_size, n_batches=nb, duplicates_dropped=dup, presets=presets,
                schema_version=schema.SCHEMA_VERSION, git_commit=git_commit(), npz=npz, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                hash="sha256(schema.to_json(params))[:16]")
    json.dump(meta, open(os.path.join(lib, "LIBRARY.json"), "w"), indent=1)
    print(f"{len(rows)} animals in {nb} batches of {batch_size} -> {lib}/design  ({dup} duplicate recipes dropped)")


# ------------------------------------------------------------------------------------------------ one animal
def stl_path(lib, h): return os.path.join(lib, "stl", h[:2], f"{h}.stl.gz")


def one(lib, design_csv, row_i, out_json, measure):
    """Build one animal, write its gzip STL by hash, write a result JSON for the parent."""
    warnings.filterwarnings("ignore"); os.chdir(ROOT)
    import numpy as np, trimesh, schema, instrument as I, landmarks
    d = list(csv.DictReader(open(design_csv)))[row_i]
    P = schema.coerce(json.loads(d["params"]), base=schema.table_defaults(), quiet=True)
    h = d["hash"]; assert recipe_hash(P) == h, "design hash does not match its params (schema changed since design?)"
    res = dict(hash=h, index=d["index"], batch=d["batch"], sampler=d["sampler"], preset=d["preset"], schema_version=schema.SCHEMA_VERSION,
               length_mm=P["length"], width_mm=P["width"])
    t0 = time.time()
    bound, _ = I.probe_bound(P)
    B = I.build_animal(P, bound)
    res.update(bevel_deg=bound, unsane_parts=",".join(B["unsane_parts"]), one_body_each=int(not B["unsane_parts"]),
               n_parts=sum(m is not None for m in B["meshes"]), build_s=round(time.time() - t0, 1))
    good = [m for m in B["meshes"] if m is not None]
    if good:
        mats = I.transforms_deg(P, 0.0)
        flat = trimesh.util.concatenate([m.copy().apply_transform(T) for m, T in zip(B["meshes"], mats) if m is not None])
        dst = stl_path(lib, h); os.makedirs(os.path.dirname(dst), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), suffix=".part"); os.close(fd)
        raw = flat.export(file_type="stl")
        with gzip.open(tmp, "wb", compresslevel=6) as gz: gz.write(raw)
        os.replace(tmp, dst)                                   # atomic: a half-written file never carries the hash name
        res.update(built=1, n_faces=len(flat.faces), volume_mm3=round(float(sum(m.volume for m in good)), 3),
                   stl=os.path.relpath(dst, lib), stl_bytes=os.path.getsize(dst), stl_sha256=hashlib.sha256(open(dst, "rb").read()).hexdigest()[:16])
    else:
        res.update(built=0)
    try:
        c = landmarks.cephalon(P); t = landmarks.pygidium(P)
        res["landmarks"] = json.dumps(dict(cephalon={k: [round(float(x), 4) for x in v] for k, v in c["lm"].items()},
                                           pygidium={k: [round(float(x), 4) for x in v] for k, v in t["lm"].items()}))
    except Exception as ex:
        res["landmarks"] = ""; res["error"] = f"landmarks: {str(ex)[:80]}"
    if measure and good:
        r = I.measure(P, B["meshes"], bound_deg=bound, head_body=B["head_body"], bevel_built_deg=B["bevel_built_deg"], unsane_parts=B["unsane_parts"])
        res.update(limited_by=r.get("limited_by"), enroll_class=r.get("enroll_class"), total_deg=r.get("total_deg"),
                   theta_joint_deg=r.get("theta_joint_deg"), reason=r.get("reason") or "")
    json.dump(res, open(out_json, "w"), default=str)


# ------------------------------------------------------------------------------------------------ a batch
def build(lib, batch, workers, timeout, measure, retry_failed=False):
    meta = json.load(open(os.path.join(lib, "LIBRARY.json")))
    import schema
    if meta["schema_version"] != schema.SCHEMA_VERSION:
        sys.exit(f"library designed at schema {meta['schema_version']}, code is {schema.SCHEMA_VERSION}: check out {meta['git_commit']}")
    design_csv = os.path.abspath(os.path.join(lib, "design", f"batch_{batch:04d}.csv"))
    man = os.path.join(lib, "manifest", f"batch_{batch:04d}.csv"); os.makedirs(os.path.dirname(man), exist_ok=True)
    rows = list(csv.DictReader(open(design_csv)))
    if retry_failed and os.path.exists(man):              # keep the built rows, drop the rest so they are tried again
        M = list(csv.DictReader(open(man))); keep = [r for r in M if str(r.get("built")) == "1"]
        if len(keep) < len(M):
            with open(man, "w", newline="") as fh0:
                w0 = csv.DictWriter(fh0, fieldnames=MANIFEST_COLS, extrasaction="ignore"); w0.writeheader(); w0.writerows(keep)
            print(f"batch {batch:04d}: retrying {len(M) - len(keep)} failed animals", flush=True)
    done = {r["hash"] for r in csv.DictReader(open(man))} if os.path.exists(man) else set()
    todo = [i for i, r in enumerate(rows) if r["hash"] not in done]
    print(f"batch {batch:04d}: {len(todo)} to build, {len(done)} already in the manifest", flush=True)
    if not todo: return
    commit = git_commit(); new = not os.path.exists(man)
    fh = open(man, "a", newline=""); w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS, extrasaction="ignore")
    if new: w.writeheader(); fh.flush()
    tmpdir = tempfile.mkdtemp(prefix="lib_")

    def child(i):
        oj = os.path.join(tmpdir, f"{i}.json"); t0 = time.time()
        cmd = [sys.executable, os.path.abspath(__file__), "one", os.path.abspath(lib), design_csv, str(i), oj] + (["--measure"] if measure else [])
        try:
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
            if p.returncode == 0 and os.path.exists(oj) and os.path.getsize(oj): res = json.load(open(oj))
            else:
                err = (p.stderr.strip().splitlines() or [f"rc {p.returncode}"])[-1][:160]
                res = dict(built=0, error=("oom_or_killed:" if p.returncode < 0 else "") + err)
        except subprocess.TimeoutExpired:
            res = dict(built=0, error=f"timeout:{timeout}s")
        d = rows[i]
        for k in ("hash", "index", "batch", "sampler", "preset"): res.setdefault(k, d[k])
        res["git_commit"] = commit; res["wall_s"] = round(time.time() - t0, 1)
        if os.path.exists(oj): os.remove(oj)
        return res

    t_start = time.time(); n_ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(child, i) for i in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result(); n_ok += int(r.get("built") or 0)
            w.writerow(r); fh.flush(); os.fsync(fh.fileno())
            el = time.time() - t_start; eta = el / n * (len(todo) - n)
            print(f"[{n}/{len(todo)}] {r['hash']} {r.get('preset') or r.get('sampler'):14s} "
                  f"{'built' if r.get('built') else 'FAILED'} one-body {r.get('one_body_each', '-')} "
                  f"{(r.get('stl_bytes') or 0) / 1e6:.1f} MB {r['wall_s']}s {r.get('error', '')[:60]}  eta {eta / 60:.0f} min", flush=True)
    fh.close(); shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"batch {batch:04d} done: {n_ok}/{len(todo)} built in {(time.time() - t_start) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------------------------------ landmarks only
LM_COLS = (["hash", "index", "batch", "preset", "head_length_mm", "error"]
           + [f"c{k}_{a}" for k in range(1, 17) for a in "xy"] + [f"p{k}_{a}" for k in range(1, 8) for a in "xy"])


def _landmark_batch(args):
    """One whole design batch, in-process (a worker process of the pool). Returns (batch, n, n_errors, seconds)."""
    lib, f = args
    warnings.filterwarnings("ignore"); os.chdir(ROOT)
    import schema, landmarks
    b = int(os.path.basename(f)[6:10]); out = os.path.join(lib, "landmarks", os.path.basename(f)); tmp = out + ".part"
    t0 = time.time(); n = bad = 0; base = schema.table_defaults()
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LM_COLS, extrasaction="ignore"); w.writeheader()
        for d in csv.DictReader(open(f)):
            r = dict(hash=d["hash"], index=d["index"], batch=d["batch"], preset=d["preset"], error="")
            try:
                P = schema.coerce(json.loads(d["params"]), base=base, quiet=True)
                c = landmarks.cephalon(P); t = landmarks.pygidium(P); r["head_length_mm"] = round(float(c["head_length"]), 4)
                for k in range(1, 17):
                    x, y = c["lm"][f"LM{k}"][:2]; r[f"c{k}_x"], r[f"c{k}_y"] = round(float(x), 4), round(float(y), 4)
                for k in range(1, 8):
                    x, y = t["lm"][f"LM{k}"][:2]; r[f"p{k}_x"], r[f"p{k}_y"] = round(float(x), 4), round(float(y), 4)
            except Exception as ex:
                r["error"] = str(ex)[:120]; bad += 1
            w.writerow(r); n += 1
    os.replace(tmp, out)
    return b, n, bad, round(time.time() - t0, 1)


def landmarks_only(lib, workers):
    from concurrent.futures import ProcessPoolExecutor
    meta = json.load(open(os.path.join(lib, "LIBRARY.json")))
    import schema
    if meta["schema_version"] != schema.SCHEMA_VERSION:
        sys.exit(f"library designed at schema {meta['schema_version']}, code is {schema.SCHEMA_VERSION}: check out {meta['git_commit']}")
    os.makedirs(os.path.join(lib, "landmarks"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(lib, "design", "batch_*.csv")))
    todo = [f for f in files if not os.path.exists(os.path.join(lib, "landmarks", os.path.basename(f)))]
    print(f"{len(todo)} of {len(files)} batches to do", flush=True)
    t0 = time.time(); tot = bad = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (b, n, e, sec) in enumerate(ex.map(_landmark_batch, [(lib, f) for f in todo]), 1):
            tot += n; bad += e; el = time.time() - t0
            print(f"[{k}/{len(todo)}] batch {b:04d}: {n} animals, {e} errors, {sec}s   eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)
    print(f"done: {tot} animals, {bad} errors, {(time.time() - t0) / 60:.1f} min", flush=True)


def status(lib):
    meta = json.load(open(os.path.join(lib, "LIBRARY.json")))
    print(f"{lib}: {meta['n']} animals, sampler {meta['sampler']}, seed {meta['seed']}, schema {meta['schema_version']}, commit {meta['git_commit']}")
    tot = built = one_body = nbytes = 0
    for f in sorted(glob.glob(os.path.join(lib, "design", "batch_*.csv"))):
        b = os.path.basename(f); n = sum(1 for _ in csv.DictReader(open(f))); m = os.path.join(lib, "manifest", b)
        R = list(csv.DictReader(open(m))) if os.path.exists(m) else []
        k = sum(int(r["built"] or 0) for r in R); o = sum(int(r["one_body_each"] or 0) for r in R if r["one_body_each"])
        s = sum(int(r["stl_bytes"] or 0) for r in R if r["stl_bytes"])
        tot += n; built += k; one_body += o; nbytes += s
        print(f"  {b}: {len(R)}/{n} attempted, {k} built, {o} with every part one body, {s / 1e9:.2f} GB")
    print(f"total: {built}/{tot} built, {one_body} with every part one body, {nbytes / 1e9:.2f} GB on disk")
    if built: print(f"at {nbytes / built / 1e6:.1f} MB per animal the full library needs about {nbytes / built * tot / 1e9:.1f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("design"); a.add_argument("lib"); a.add_argument("--sampler", choices=["box", "presets", "npz"], default="presets")
    a.add_argument("--n", type=int, default=10000); a.add_argument("--seed", type=int, default=7)
    a.add_argument("--batch-size", type=int, default=1000); a.add_argument("--npz")
    a.add_argument("--presets", nargs="+", help="with --sampler presets: only these (e.g. pillbug, or phacopida lichida); default the ten orders")
    b = sub.add_parser("build"); b.add_argument("lib"); g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch", type=int); g.add_argument("--all", action="store_true")
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)); b.add_argument("--timeout", type=int, default=900)
    b.add_argument("--measure", action="store_true"); b.add_argument("--retry-failed", action="store_true")
    m = sub.add_parser("landmarks"); m.add_argument("lib"); m.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    o = sub.add_parser("one"); o.add_argument("lib"); o.add_argument("design_csv"); o.add_argument("row", type=int); o.add_argument("out_json")
    o.add_argument("--measure", action="store_true")
    s = sub.add_parser("status"); s.add_argument("lib")
    A = ap.parse_args()
    A.lib = os.path.abspath(A.lib); os.chdir(ROOT)               # presets/ and the builders are read relative to the repo
    if A.cmd == "design": design(A.lib, A.sampler, A.n, A.seed, A.batch_size, A.npz, A.presets)
    elif A.cmd == "one": one(A.lib, A.design_csv, A.row, A.out_json, A.measure)
    elif A.cmd == "status": status(A.lib)
    elif A.cmd == "landmarks": landmarks_only(A.lib, A.workers)
    else:
        if A.all:
            for f in sorted(glob.glob(os.path.join(A.lib, "design", "batch_*.csv"))):
                build(A.lib, int(os.path.basename(f)[6:10]), A.workers, A.timeout, A.measure, A.retry_failed)
        else:
            build(A.lib, A.batch, A.workers, A.timeout, A.measure, A.retry_failed)
