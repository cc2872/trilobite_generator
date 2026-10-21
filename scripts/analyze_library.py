"""
scripts/analyze_library.py: what a library run says about where the generator breaks.

    python scripts/analyze_library.py C:\\trilib                 # every manifest batch written so far
    python scripts/analyze_library.py C:\\trilib --out C:\\trilib\\analysis

Reads lib/manifest/*.csv (and lib/design/*.csv for the parameters), writes to lib/analysis/:
    summary.txt               counts, one-body rate per order with 95% Wilson intervals, which parts fail, sizes and times
    onebody_by_order.png      one-body rate per order with 95% intervals
    failing_parts.png         per order: share of animals whose head / a thorax segment / tail is not one closed body
    break_map.png             one-body rate against each sweep axis (relative to the order's preset), pooled over orders
    failing_segments.png      which thorax segment fails, by position from the head
    by_order.csv              the per-order table, for plotting elsewhere

"One-body" = every part (head, each segment, tail) came out as one closed, watertight body at the instrument's bevel.
It is a geometric check of the builder, not biology.
"""
import os, sys, csv, json, glob, math, argparse, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
csv.field_size_limit(10 ** 8)


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(lib):
    rows = []
    for f in sorted(glob.glob(os.path.join(lib, "manifest", "batch_*.csv"))):
        rows += list(csv.DictReader(open(f)))
    params = {}
    for f in sorted(glob.glob(os.path.join(lib, "design", "batch_*.csv"))):
        for d in csv.DictReader(open(f)): params[d["hash"]] = json.loads(d["params"])
    return rows, params


def part_kinds(unsane):
    parts = [p for p in (unsane or "").split(",") if p]
    return dict(head=any(p == "head" for p in parts), segment=any(p.startswith("seg") for p in parts), tail=any(p == "tail" for p in parts),
                segs=[int(p[3:]) for p in parts if p.startswith("seg") and p[3:].isdigit()])


def axis_values(P, base):
    """Each sweep axis expressed the way sweep.py varies it: segCount as +/- segments, overlap as its value, the 'mul' axes
    as a ratio to the order's preset, spine_scale as the ratio of spine lengths to the preset's."""
    import sweep
    out = {}
    for k, (kind, lo, hi) in sweep.AXES.items():
        try:
            if k == "spine_scale":
                ks = [s for s in sweep.SPINE_KEYS if abs(base.get(s, 0) or 0) > 1e-9]
                out[k] = (P[ks[0]] / base[ks[0]]) if ks else None
            elif kind == "add": out[k] = P[k] - base[k]
            elif kind == "mul": out[k] = P[k] / base[k] if base[k] else None
            else: out[k] = P[k]
        except (KeyError, ZeroDivisionError, TypeError):
            out[k] = None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("lib"); ap.add_argument("--out"); A = ap.parse_args()
    lib = os.path.abspath(A.lib); out = os.path.abspath(A.out or os.path.join(lib, "analysis")); os.makedirs(out, exist_ok=True)
    os.chdir(ROOT)
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows, params = load(lib)
    if not rows: sys.exit(f"no manifest rows in {lib}\\manifest")
    built = [r for r in rows if str(r.get("built")) == "1"]
    for r in built: r.update(part_kinds(r.get("unsane_parts"))); r["ok"] = str(r.get("one_body_each")) == "1"
    orders = sorted({r["preset"] or r["sampler"] for r in built})
    L = []
    L.append(f"library {lib}\n{len(rows)} attempted, {len(built)} built, {len(rows) - len(built)} not built")
    errs = collections.Counter((r.get("error") or "")[:60] for r in rows if str(r.get("built")) != "1")
    if errs: L.append("not built, by error: " + "; ".join(f"{k or '(no error text)'} x{v}" for k, v in errs.most_common(6)))
    k = sum(r["ok"] for r in built); p, lo, hi = wilson(k, len(built))
    L.append(f"\nevery part one body: {k}/{len(built)} = {p:.1%}  (95% {lo:.1%}-{hi:.1%})")

    # ---- per order
    T = []
    for o in orders:
        R = [r for r in built if (r["preset"] or r["sampler"]) == o]; n = len(R); k = sum(r["ok"] for r in R); p, lo, hi = wilson(k, n)
        T.append(dict(order=o, n=n, one_body=k, rate=round(p, 4), lo=round(lo, 4), hi=round(hi, 4),
                      head_fails=sum(r["head"] for r in R), segment_fails=sum(r["segment"] for r in R), tail_fails=sum(r["tail"] for r in R),
                      mb_mean=round(np.mean([float(r["stl_bytes"] or 0) for r in R]) / 1e6, 2),
                      s_median=round(float(np.median([float(r["wall_s"] or 0) for r in R])), 1)))
    T.sort(key=lambda t: t["rate"])
    with open(os.path.join(out, "by_order.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(T[0])); w.writeheader(); w.writerows(T)
    L.append("\norder            n   one-body  95% interval     head  segment  tail fails   MB   s/animal")
    for t in T:
        L.append(f"{t['order']:15s} {t['n']:4d}   {t['rate']:6.1%}   {t['lo']:5.1%}-{t['hi']:5.1%}   {t['head_fails']:5d} {t['segment_fails']:7d} {t['tail_fails']:6d}"
                 f"   {t['mb_mean']:5.1f}  {t['s_median']:6.1f}")
    bad = [r for r in built if not r["ok"]]
    if bad:
        combo = collections.Counter("+".join(x for x in ("head", "segment", "tail") if r[x]) or "other" for r in bad)
        L.append("\nfailing animals by which parts fail: " + ", ".join(f"{k} {v} ({v / len(bad):.0%})" for k, v in combo.most_common()))

    # ---- figure: one-body by order
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(T) + 1.2))
    y = np.arange(len(T))
    ax.errorbar([t["rate"] for t in T], y, xerr=[[t["rate"] - t["lo"] for t in T], [t["hi"] - t["rate"] for t in T]], fmt="o", color="#2b5d7d", capsize=3)
    ax.set_yticks(y, [f"{t['order']} (n={t['n']})" for t in T]); ax.set_xlim(0, 1); ax.set_xlabel("every part one closed body (95% interval)")
    ax.axvline(sum(r["ok"] for r in built) / len(built), color="#999", lw=1, ls="--"); ax.grid(axis="x", alpha=.3)
    ax.set_title("Builds that come out whole, by order", loc="left", fontsize=11); fig.tight_layout(); fig.savefig(os.path.join(out, "onebody_by_order.png"), dpi=200); plt.close(fig)

    # ---- figure: which part fails
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(T) + 1.2))
    for j, (key, col) in enumerate((("head", "#c0504d"), ("segment", "#e2a23b"), ("tail", "#5b8f5b"))):
        ax.barh(y + (j - 1) * 0.26, [t[f"{key}_fails"] / t["n"] for t in T], height=0.26, color=col, label=key)
    ax.set_yticks(y, [t["order"] for t in T]); ax.set_xlabel("share of animals in which this part is not one body"); ax.legend(frameon=False, loc="lower right")
    ax.set_title("Which part breaks", loc="left", fontsize=11); ax.grid(axis="x", alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "failing_parts.png"), dpi=200); plt.close(fig)

    # ---- figure: which segment fails (position from the head, as a fraction of the thorax)
    pos = [s / max(1, int(params.get(r["hash"], {}).get("segCount", 1)) - 1 or 1) for r in built for s in r["segs"] if r["hash"] in params]
    if pos:
        fig, ax = plt.subplots(figsize=(5.5, 3)); ax.hist(pos, bins=10, range=(0, 1), color="#e2a23b")
        ax.set_xlabel("position of the failing segment (0 = first behind the head, 1 = last before the tail)"); ax.set_ylabel("failures")
        fig.tight_layout(); fig.savefig(os.path.join(out, "failing_segments.png"), dpi=200); plt.close(fig)

    # ---- figure: break map against the sweep axes
    try:
        import sweep, schema
        bases = {o: schema.coerce(p, base=schema.table_defaults(), quiet=True) for o, p in sweep.load_presets().items()}
        other = os.path.join(ROOT, "presets_other")
        for f in glob.glob(os.path.join(other, "*.json")):
            bases[os.path.basename(f)[:-5]] = schema.coerce(json.load(open(f))["params"], base=schema.table_defaults(), quiet=True)
        AX = list(sweep.AXES)
        vals = {a: [] for a in AX}
        for r in built:
            P = params.get(r["hash"]); b = bases.get(r["preset"])
            if P is None or b is None: continue
            v = axis_values(P, b)
            for a in AX:
                if v[a] is not None: vals[a].append((v[a], r["ok"]))
        n_ax = sum(1 for a in AX if len(vals[a]) > 20)
        if n_ax:
            cols = 4; rws = math.ceil(n_ax / cols)
            fig, axs = plt.subplots(rws, cols, figsize=(3.2 * cols, 2.6 * rws), squeeze=False); i = 0
            L.append("\none-body rate across each sweep axis (5 equal-count bins, low -> high):")
            for a in AX:
                if len(vals[a]) <= 20: continue
                xs = np.array([v for v, _ in vals[a]], float); ok = np.array([o for _, o in vals[a]], float)
                edges = np.unique(np.quantile(xs, np.linspace(0, 1, 6)))
                if len(edges) < 3: edges = np.unique(xs)
                mids, rate, lo_, hi_ = [], [], [], []
                for e0, e1 in zip(edges[:-1], edges[1:]):
                    m = (xs >= e0) & (xs <= e1) if e1 == edges[-1] else (xs >= e0) & (xs < e1)
                    if m.sum() == 0: continue
                    p, l, h = wilson(int(ok[m].sum()), int(m.sum())); mids.append(xs[m].mean()); rate.append(p); lo_.append(l); hi_.append(h)
                axx = axs[i // cols][i % cols]; i += 1
                axx.fill_between(mids, lo_, hi_, color="#2b5d7d", alpha=.2); axx.plot(mids, rate, "o-", color="#2b5d7d")
                unit = {"add": "segments added", "mul": "x preset", "abs": "value"}[sweep.AXES[a][0]] if a != "spine_scale" else "x preset spines"
                axx.set_title(a, fontsize=10); axx.set_xlabel(unit, fontsize=8); axx.set_ylim(0, 1); axx.grid(alpha=.3)
                L.append(f"  {a:16s} " + "  ".join(f"{p:.0%}" for p in rate) + f"   (spread {max(rate) - min(rate):.0%})")
            for j in range(i, rws * cols): axs[j // cols][j % cols].axis("off")
            fig.suptitle("Where the builder breaks: whole-build rate along each sweep axis (95% band)", x=0.01, ha="left", fontsize=11)
            fig.tight_layout(); fig.savefig(os.path.join(out, "break_map.png"), dpi=200); plt.close(fig)
    except Exception as ex:
        L.append(f"\n(break map skipped: {ex})")

    sizes = [float(r["stl_bytes"] or 0) for r in built]; walls = [float(r["wall_s"] or 0) for r in rows if r.get("wall_s")]
    L.append(f"\nSTL size: mean {np.mean(sizes) / 1e6:.1f} MB, total {sum(sizes) / 1e9:.1f} GB;  wall time per animal: median {np.median(walls):.0f} s, "
             f"90th percentile {np.percentile(walls, 90):.0f} s, max {max(walls):.0f} s")
    txt = "\n".join(L); open(os.path.join(out, "summary.txt"), "w").write(txt + "\n"); print(txt); print(f"\nfigures and tables in {out}")


if __name__ == "__main__":
    main()
