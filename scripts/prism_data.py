"""
scripts/prism_data.py: the data behind /prism (the Raup prism page), written to web/static/prism/prism.json.

    python scripts/prism_data.py                                # the library regenerated: --sampler presets, seed 7, n 10000
    python scripts/prism_data.py --library C:\\trilib            # the real library: reads LIB/design/batch_*.csv
    python scripts/prism_data.py --seed 7 --n 10000 --out web/static/prism/prism.json

What it writes:
    made     every library animal: its preset and the values of the sweep keys it changed (KEYS), so the page can
             rebuild it exactly as /api/preset/<preset> + those values; c, p, n, r for drawing.
    fossils  TriloMorph whole-body specimens measured by eye (data/prism/fossils_trilomorph.csv): genus, order, c, p, n, r.
    genera   the 25 Gon genera's recipe values (data/prism/genera_gon.csv): c, p, n, r and the preset they were built on.
    presets  the ten order presets: c, p, n, r.
    schema   the bounds of the four drawn keys, so the page clips the way schema.coerce does.
c = cephFrac, p = pygFrac, n = segCount, r = widthThoraxRear.
"""
import os, sys, csv, json, glob, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
csv.field_size_limit(10 ** 8)
KEYS = ["segCount", "overlap", "cephFrac", "pygFrac", "relief", "widthThoraxRear",
        "spineBase", "genalSpine", "pygSpine", "axialSpine", "pygMarginalLen"]          # sweep.AXES + sweep.SPINE_KEYS
ORDER_PRESET = {"olenida": "ptychopariida", "trinucleida": "asaphida"}                    # TriloMorph orders without a preset


def made_from_library(lib):
    rows = []
    for f in sorted(glob.glob(os.path.join(lib, "design", "batch_*.csv"))):
        for r in csv.DictReader(open(f, newline="")):
            rows.append((r["preset"], json.loads(r["params"])))
    return rows


def made_from_sampler(n, seed):
    import numpy as np, library
    os.chdir(ROOT)
    return list(library.sample_presets(n, np.random.default_rng(seed)))


def main():
    A = argparse.ArgumentParser(); A.add_argument("--library"); A.add_argument("--n", type=int, default=10000)
    A.add_argument("--seed", type=int, default=7); A.add_argument("--out", default=os.path.join(ROOT, "web", "static", "prism", "prism.json"))
    a = A.parse_args()
    import schema
    rows = made_from_library(a.library) if a.library else made_from_sampler(a.n, a.seed)
    names = sorted({p for p, _ in rows}); idx = {p: i for i, p in enumerate(names)}
    r4 = lambda x: round(float(x), 4)
    made = dict(keys=KEYS, presets=names, preset=[idx[p] for p, _ in rows],
                v=[[r4(P[k]) for k in KEYS] for _, P in rows],
                source=(f"library {os.path.abspath(a.library)}" if a.library else f"library.py sample_presets n={a.n} seed={a.seed}"))
    pre = {}
    os.chdir(ROOT)
    for f in sorted(glob.glob(os.path.join(ROOT, "presets", "*.json"))):
        nm = os.path.basename(f)[:-5]
        if nm.startswith("_"): continue
        P = schema.coerce(json.load(open(f))["params"], base=schema.table_defaults(), quiet=True)
        pre[nm] = dict(c=r4(P["cephFrac"]), p=r4(P["pygFrac"]), n=int(P["segCount"]), r=r4(P["widthThoraxRear"]))
    def nearest(c, p, n):
        return min(pre, key=lambda k: ((pre[k]["c"] - c) / 0.07) ** 2 + ((pre[k]["p"] - p) / 0.1) ** 2 + ((pre[k]["n"] - n) / 3.3) ** 2)
    fos = []
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "prism", "fossils_trilomorph.csv"), newline="")):
        c, p, n = float(r["c"]), float(r["p"]), float(r["n"]); o = (r["order"] or "unknown").strip()
        ol = o.lower(); ps = ORDER_PRESET.get(ol, ol if ol in pre else nearest(c, p, n))
        rr = float(r["r"]) if r.get("r") not in (None, "", "nan") else None
        fos.append(dict(id=r["file"], genus=r["genus"] or "unnamed", order=o, preset=ps, drawn=int(float(r.get("drawn") or 0)),
                        c=r4(c), p=r4(p), n=int(round(n)), r=(r4(rr) if rr is not None and rr == rr else None)))
    gen = []
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "prism", "genera_gon.csv"), newline="")):
        gen.append(dict(genus=r["genus"], preset=r["order"], c=r4(r["c_recipe"]), p=r4(r["p_recipe"]),
                        n=int(float(r["n_recipe"])), r=r4(r["r_recipe"])))
    b = {p.key: [p.lo, p.hi] for p in schema.PARAMS if p.key in ("cephFrac", "pygFrac", "segCount", "widthThoraxRear")}
    out = dict(schema=schema.SCHEMA_VERSION, bounds=b, made=made, fossils=fos, genera=gen, presets=pre,
               credit="Fossils measured from TriloMorph (Serra et al. 2023, CC BY 4.0)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print(f"{len(rows)} made animals, {len(fos)} fossils, {len(gen)} genera, {len(pre)} presets -> {a.out} "
          f"({os.path.getsize(a.out) // 1024} KB)")


if __name__ == "__main__":
    main()
