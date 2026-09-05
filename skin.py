"""
skin.py — the purchased sculpt as a skin over the pin-hinge skeleton, entirely with mesh booleans.

Per joint (between piece i and piece i+1, boundary curve y_i(x) traced from the sculpt's own furrow):
  1. ventral stop bevels on both faces, following the curve, below the hinge axis
  2. clearance channel around the axis in both pieces; interleaving knuckles; pin bore exiting the sides
  3. articulating facet: piece i+1's front is recessed under piece i's dorsal flange, over a length that
     grows with depth below the axis so the pleura can swing under during flexion
Then: chain kinematics with per-joint axes, the enrollment instrument (exact overlap volume), exports.
"""
import numpy as np, trimesh, math, json, time, pickle
from load_sculpt import load
import slicer as S

def box_from_grids(A, B):
    """Closed mesh between two (n,m,3) grids sharing (i,j) parameterisation (A front, B back)."""
    n, m = A.shape[:2]
    V = np.r_[A.reshape(-1, 3), B.reshape(-1, 3)]
    idx = lambda i, j, back: (n * m if back else 0) + i * m + j
    F = []
    for i in range(n - 1):
        for j in range(m - 1):
            a, b, c, d = idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0)
            F += [[a, b, c], [a, c, d]]
            a, b, c, d = idx(i, j, 1), idx(i + 1, j, 1), idx(i + 1, j + 1, 1), idx(i, j + 1, 1)
            F += [[a, c, b], [a, d, c]]
    for i in range(n - 1):                    # j = 0 and j = m-1 edges
        for j0, flip in ((0, 1), (m - 1, 0)):
            a, b, c, d = idx(i, j0, 0), idx(i + 1, j0, 0), idx(i + 1, j0, 1), idx(i, j0, 1)
            F += ([[a, c, b], [a, d, c]] if flip else [[a, b, c], [a, c, d]])
    for j in range(m - 1):                    # i = 0 and i = n-1 edges
        for i0, flip in ((0, 0), (n - 1, 1)):
            a, b, c, d = idx(i0, j, 0), idx(i0, j + 1, 0), idx(i0, j + 1, 1), idx(i0, j, 1)
            F += ([[a, c, b], [a, d, c]] if flip else [[a, b, c], [a, c, d]])
    mesh = trimesh.Trimesh(V, np.array(F), process=True)
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0: mesh.invert()
    return mesh

def bevel_cutter(yfun, zh, phi_deg, rear, xr=90, zlo=-3):
    """Region to remove below the axis: behind (rear=True) or in front of the tilted face."""
    xs = np.linspace(-xr, xr, 40); zs = np.linspace(zlo, zh, 16); t = math.tan(math.radians(phi_deg))
    A = np.array([[(x, yfun(x) - (zh - z) * t if rear else yfun(x) + (zh - z) * t, z) for z in zs] for x in xs])
    B = A.copy(); B[:, :, 1] = A[:, :, 1] + (60 if rear else -60)
    return box_from_grids(A, B)

def strip_box(yfun, L_of_x, zbot_fn, ztop, xr=90, ny=8):
    """Region above zbot(x,y) within the plan strip y ∈ [yfun(x), yfun(x)+L(x)], up to ztop."""
    xs = np.linspace(-xr, xr, 60)
    A = np.array([[(x, yfun(x) + L_of_x(x) * k / (ny - 1), zbot_fn(x, yfun(x) + L_of_x(x) * k / (ny - 1))) for k in range(ny)] for x in xs])
    B = A.copy(); B[:, :, 2] = ztop
    return box_from_grids(A, B)

def cyl_x(xc, y, z, r, length):
    c = trimesh.creation.cylinder(radius=r, height=length, sections=32)
    c.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    c.apply_translation((xc, y, z)); return c

def top_lookup(xs, ys, Z):
    def f(x, y):
        i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2); j = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
        return float(Z[j, i])
    return f

SCULPT_P0 = None   # the sculpt's own parameter vector (its landmarks, outline, count), filled on first build

def sculpt_reference():
    """P0: the knob values that reproduce the purchased sculpt unchanged."""
    global SCULPT_P0
    if SCULPT_P0 is None:
        import wrap
        parts, _ = load(); body = parts["Trilobite_spine"]
        xs, ys, Z, mask = S.heightfield(body, 0.8)
        mins, _ = S.midline_furrows(xs, ys, Z, mask, 38, 124, min_gap=2.5)
        L = body.extents[1]
        SCULPT_P0 = wrap.P0_from(body, mins[0] / L, mins[-1] / L, len(mins) - 1)
    return dict(SCULPT_P0)

def build_skinned(P=None, legs=False, verbose=True, body_override=None):
    import wrap
    P0 = sculpt_reference()
    P = {**dict(wall=2.0, clearance=0.3, boreDia=1.95, barrelR=2.6, nKnuckles=3, maxAngle=18, overlap=3.5, axisFrac=0.24), **P0, **(P or {})}
    parts, _ = load(); body0 = parts["Trilobite_spine"] if body_override is None else body_override; pr = wrap.profiles(parts["Trilobite_spine"])
    body = wrap.wrap_mesh(body0, P, P0, pr)
    if legs: parts["Trilobite_legs"] = wrap.wrap_mesh(parts["Trilobite_legs"], P, P0, pr)
    L = float(body.extents[1]); xw = 0.56 * float(body.extents[0])
    xs, ys, Z, mask = S.heightfield(body, 0.8 * L / 130); ztop = top_lookup(xs, ys, Z)
    mins, crest = S.midline_furrows(xs, ys, Z, mask, 0.29 * L, 0.955 * L, min_gap=2.5 * L / 130)
    traced = [S.trace_boundary(xs, ys, Z, mask, m, x_max=xw) for m in mins]
    # shared sweep beyond the margin so cuts never cross
    slopes = []
    for f, pts in traced:
        if len(pts) > 8: xa = pts[:, 0]; k = xa > xa.max() - 8; slopes.append(np.polyfit(pts[k, 0], pts[k, 1], 1)[0])
    slope = float(np.median(slopes))
    funcs = []
    for (f, pts), m in zip(traced, mins):
        if len(pts) < 6: funcs.append(lambda x, m=m: np.full_like(np.asarray(x, float), m)); continue
        xa = np.sort(np.unique(pts[:, 0])); ya = np.array([pts[pts[:, 0] == x, 1].mean() for x in xa])
        ya = np.convolve(np.pad(ya, 2, mode="edge"), np.array([1, 2, 3, 2, 1]) / 9, mode="valid"); xe, ye = xa[-1], ya[-1]
        funcs.append(lambda x, xa=xa, ya=ya, xe=xe, ye=ye: np.where(np.abs(np.asarray(x, float)) <= xe, np.interp(np.abs(np.asarray(x, float)), xa, ya), ye + slope * (np.abs(np.asarray(x, float)) - xe)))
    pieces = S.slice_body(body, funcs, x_max=xw + 8)
    for i in range(len(pieces) - 1):      # the sculpt's spines overlap each other: give shared volume to the front piece
        d = trimesh.boolean.difference([pieces[i + 1], pieces[i]], engine="manifold")
        if d is not None and len(d.faces) and d.is_volume: pieces[i + 1] = d
    if int(P.get("segCount", len(pieces) - 2)) != len(pieces) - 2:
        pieces, funcs = wrap.resample_segments(pieces, funcs, int(P["segCount"]))
    n = len(pieces); t, c, rB, phi = P["wall"], P["clearance"], P["barrelR"], P["maxAngle"] / 2
    axes = []
    originals = [p.copy() for p in pieces]
    for i in range(n - 1):
        yf = funcs[i]; y0 = float(yf(0.0))
        crest_y = float(np.nanmax(np.where(mask[:, np.abs(xs) < 2], Z[:, np.abs(xs) < 2], np.nan)[np.argmin(np.abs(ys - y0))]))
        zh = crest_y - rB - t - c - 1.0
        wj = float(pieces[i + 1].extents[0]); Wh = max(8.0, P["axisFrac"] * min(wj, pieces[i].extents[0]) - 2)
        nK = int(P["nKnuckles"]); kw = Wh / nK; x0 = -Wh / 2
        axes.append(dict(y=y0, zh=zh, Wh=Wh))
        a, b = pieces[i], pieces[i + 1]
        # 1. bevels below the axis, following the curve
        a = trimesh.boolean.difference([a, bevel_cutter(yf, zh, phi, rear=True)], engine="manifold")
        b = trimesh.boolean.difference([b, bevel_cutter(yf, zh, phi, rear=False)], engine="manifold")
        # 2. hinge: channel in both, interleaving knuckles, through-bore exiting the sides
        chan = cyl_x(0, y0, zh, rB + c, Wh + 4)
        a = trimesh.boolean.difference([a, chan], engine="manifold"); b = trimesh.boolean.difference([b, chan], engine="manifold")
        for k in range(nK):
            xc = x0 + (k + 0.5) * kw; kn = cyl_x(xc, y0, zh, rB, kw - c)
            if k % 2 == 0: a = trimesh.boolean.union([a, kn], engine="manifold")
            else: b = trimesh.boolean.union([b, kn], engine="manifold")
        bore = cyl_x(0, y0, zh, P["boreDia"] / 2, 2 * xr_of(pieces[i]) + 10)
        a = trimesh.boolean.difference([a, bore], engine="manifold"); b = trimesh.boolean.difference([b, bore], engine="manifold")
        # 3. facet + flange
        s_ang = math.sin(math.radians(P["maxAngle"]))
        L = lambda x, yf=yf, zh=zh: P["overlap"] + max(0.0, zh - ztop(x, float(yf(x)))) * s_ang + 1.0
        cstep = c + P["overlap"] * s_ang
        step = strip_box(yf, L, lambda x, y: ztop(x, y) - t - cstep, 90)
        flange_box = strip_box(yf, lambda x: P["overlap"], lambda x, y: ztop(x, y) - t, 90)
        flange = trimesh.boolean.intersection([originals[i + 1], flange_box], engine="manifold")
        b = trimesh.boolean.difference([b, step], engine="manifold")
        if flange is not None and len(flange.faces) and flange.volume > 1: a = trimesh.boolean.union([a, flange], engine="manifold")
        pieces[i], pieces[i + 1] = a, b
        if verbose: print(f"joint {i}: y={y0:.1f} zh={zh:.1f} Wh={Wh:.1f} | {len(a.faces)}/{len(b.faces)} tris, volumes {a.is_volume}/{b.is_volume}", flush=True)
    # rest-pose guarantee: any sliver two neighbours still share (sculpt self-overlaps, sampling mismatch at
    # the flange edge) goes to the front part, so the instrument starts from a clean zero
    for i in range(n - 1):
        d = trimesh.boolean.difference([pieces[i + 1], pieces[i]], engine="manifold")
        if d is not None and len(d.faces) and d.is_volume and d.volume > 0.5 * pieces[i + 1].volume: pieces[i + 1] = d
    if legs:
        comps = parts["Trilobite_legs"].split(only_watertight=False); bounds = [0] + [ax["y"] for ax in axes] + [200]
        for comp in comps:
            ya = comp.vertices[comp.vertices[:, 2].argmax(), 1]; k = sum(1 for bnd in bounds if ya >= bnd) - 1
            k = min(max(k, 0), n - 1); pieces[k] = trimesh.boolean.union([pieces[k], comp], engine="manifold")
    return pieces, axes, P

def xr_of(p): return float(max(abs(p.bounds[0][0]), abs(p.bounds[1][0])))

# ---------------- kinematics + instrument on the skinned animal
def transforms(axes, angles):
    mats = [np.eye(4)]; M = np.eye(4)
    for ax, th in zip(axes, angles):
        y, zh = ax["y"], ax["zh"]; T = np.eye(4)
        R = trimesh.transformations.rotation_matrix(-math.radians(th), [1, 0, 0], point=[0, y, zh])
        M = M @ R; mats.append(M.copy())
    return mats

def posed(pieces, axes, e, maxAngle):
    mats = transforms(axes, [e * maxAngle] * len(axes)); out = []
    for p, M in zip(pieces, mats): q = p.copy(); q.apply_transform(M); out.append(q)
    return out

def collisions(pieces, axes, e, maxAngle, tol=5.0):   # mm^3; below this two parts merely touch (sub-millimetre slivers at full mesh resolution)
    ps = posed(pieces, axes, e, maxAngle); hits = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if not (np.all(ps[i].bounds[0] - 0.5 <= ps[j].bounds[1]) and np.all(ps[j].bounds[0] - 0.5 <= ps[i].bounds[1])): continue
            r = trimesh.boolean.intersection([ps[i], ps[j]], engine="manifold")
            v = abs(r.volume) if r is not None and len(r.faces) else 0.0
            if v > tol: hits.append((i, j, round(v, 1)))
    return hits

def measure(pieces, axes, maxAngle):
    if not collisions(pieces, axes, 1.0, maxAngle): em = 1.0
    elif collisions(pieces, axes, 0.0, maxAngle): em = 0.0
    else:
        lo, hi = 0.0, 1.0
        while hi - lo > 0.004:
            mid = 0.5 * (lo + hi)
            if collisions(pieces, axes, mid, maxAngle): hi = mid
            else: lo = mid
        em = lo
    ps = posed(pieces, axes, em, maxAngle)
    gap = float(trimesh.proximity.closest_point(ps[0], ps[-1].vertices[::20])[1].min())
    return dict(e_max=round(em, 3), free_curl_deg=round(em * len(axes) * maxAngle, 1), total_curl_deg=len(axes) * maxAngle,
                closure_gap_mm=round(gap, 1), enroll_class="none" if em < 0.05 else ("complete" if gap < 3 else "partial"),
                touching_at_zero=len(collisions(pieces, axes, 0.0, maxAngle)) > 0, stopped_by=collisions(pieces, axes, min(1.0, em + 0.01), maxAngle)[:6])

if __name__ == "__main__":
    t0 = time.time(); pieces, axes, P = build_skinned(); print("built %.0fs" % (time.time() - t0), flush=True)
    pickle.dump((pieces, axes, P), open("skinned.pkl", "wb"))
    m = measure(pieces, axes, P["maxAngle"]); print(json.dumps(m))
    for i, p in enumerate(pieces): p.export(f"part{i:02d}.stl")
    trimesh.util.concatenate(posed(pieces, axes, 0.0, P["maxAngle"])).export("skinned_flat.stl")
    trimesh.util.concatenate(posed(pieces, axes, max(m["e_max"], 0.02), P["maxAngle"])).export("skinned_enrolled.stl")
    json.dump(dict(measure=m, axes=axes, P=P), open("skinned_measure.json", "w"), indent=1)
