"""
atlas.py — register a trilobite model into body coordinates and extract its dorsal heightfield.

Body coordinates: s along the body (0 = front of head, 1 = tip of tail), u across (-1..1 as a fraction
of the local half-width), z normalised by the animal's relief.  Output per model: an atlas .npz with
Z(s,u), the outline W(s), landmarks (furrows, widest point), and the physical scale.
Works for closed sculpts and for fossil scans (which get masked from their matrix first).
"""
import numpy as np, trimesh, json

def pca_align(v):
    """Longest axis -> y, thinnest -> z; returns aligned vertices (centred)."""
    c = v - v.mean(0); w, U = np.linalg.eigh(np.cov(c.T)); R = U[:, ::-1]
    a = (c @ R)[:, [1, 0, 2]]
    if np.linalg.det(np.c_[np.eye(3)]) < 0: pass
    return a

def orient(v, head_wider=True):
    """Dorsal up (the side whose top surface varies more), head at low y (the wider end)."""
    def top_var(vv):
        ys = np.linspace(vv[:, 1].min(), vv[:, 1].max(), 40); prof = []
        for a, b in zip(ys[:-1], ys[1:]):
            sel = (vv[:, 1] >= a) & (vv[:, 1] < b) & (np.abs(vv[:, 0]) < 0.05 * (vv[:, 0].max() - vv[:, 0].min()))
            prof.append(vv[sel, 2].max() if sel.any() else np.nan)
        p = np.array(prof); p = p[np.isfinite(p)]; return np.var(np.diff(p)) if len(p) > 3 else 0
    vf = v.copy(); vf[:, 2] *= -1
    if top_var(vf) > top_var(v): v = vf
    y0, y1 = v[:, 1].min(), v[:, 1].max(); L = y1 - y0
    w_front = np.ptp(v[v[:, 1] < y0 + 0.3 * L, 0]); w_back = np.ptp(v[v[:, 1] > y1 - 0.3 * L, 0])
    if (w_back > w_front) == head_wider: v = v.copy(); v[:, 1] *= -1
    v = v - [0, v[:, 1].min(), v[:, 2].min()]
    return v

def heightfield_xy(v, cell, normals=None):
    """Dense dorsal heightfield: linear interpolation over upward-facing vertices, then a max with the raw
    vertex heights so spines and ridges are never lost."""
    from scipy.interpolate import LinearNDInterpolator
    x0, y0 = v[:, 0].min(), v[:, 1].min(); nx = int(np.ptp(v[:, 0]) / cell) + 2; ny = int(np.ptp(v[:, 1]) / cell) + 2
    xs = x0 + cell * np.arange(nx); ys = y0 + cell * np.arange(ny)
    up = v if normals is None else v[normals[:, 2] > 0.15]
    if len(up) > 60000: up = up[np.random.default_rng(0).choice(len(up), 60000, replace=False)]
    f = LinearNDInterpolator(up[:, :2], up[:, 2])
    X, Y = np.meshgrid(xs, ys); Z = f(X, Y)
    Zmax = np.full((ny, nx), np.nan); ix = ((v[:, 0] - x0) / cell).astype(int); iy = ((v[:, 1] - y0) / cell).astype(int)
    o = np.argsort(v[:, 2]); Zmax[iy[o], ix[o]] = v[o, 2]
    Z = np.where(np.isnan(Z), Zmax, np.fmax(Z, np.nan_to_num(Zmax, nan=-1e9)))
    return xs, ys, Z

def footprint_halfwidth(v, ys, cell):
    """Outline from the actual vertices: max |x - midline| per row, smoothed."""
    xmid = np.median(v[:, 0]); W = np.full(len(ys), np.nan)
    iy = np.clip(((v[:, 1] - ys[0]) / cell).astype(int), 0, len(ys) - 1)
    np.fmax.at(W, iy, np.abs(v[:, 0] - xmid))
    W = np.where(np.isnan(W), np.nanmin(W), W); k = np.array([1, 2, 3, 2, 1]) / 9
    return xmid, np.convolve(np.pad(W, 2, mode="edge"), k, mode="valid")

def mask_animal(Z, frac=0.45):
    """Fossil in matrix: keep the largest connected blob of cells above a height threshold."""
    from scipy import ndimage
    zmin, zmax = np.nanmin(Z), np.nanmax(Z); m = np.nan_to_num(Z, nan=zmin) > zmin + frac * (zmax - zmin)
    m = ndimage.binary_closing(m, iterations=3); m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m); sizes = ndimage.sum(m, lab, range(1, n + 1)); m = lab == (1 + int(np.argmax(sizes)))
    return m

def make_atlas(v, name, cell=None, fossil=False, ns=220, nu=101, flip_z=None, normals=None, flip_y=None):
    v = pca_align(v)
    if flip_z is None: v = orient(v)
    else:
        if flip_z: v[:, 2] *= -1
        if flip_y: v[:, 1] *= -1
        v = v - [0, v[:, 1].min(), v[:, 2].min()]
    L = np.ptp(v[:, 1]); cell = cell or L / 260
    xs, ys, Z = heightfield_xy(v, cell, normals)
    if fossil:
        m = mask_animal(Z); Z = np.where(m, Z, np.nan)
        rows = np.where(m.any(1))[0]; ys = ys[rows[0]:rows[-1] + 1]; Z = Z[rows[0]:rows[-1] + 1]; m = m[rows[0]:rows[-1] + 1]
        Z = Z - np.nanpercentile(Z, 2)
        xmid = np.median(xs[np.where(m.any(0))[0]]); W = np.array([np.abs(xs[m[j]] - xmid).max() if m[j].any() else np.nan for j in range(len(ys))])
        k = np.array([1, 2, 3, 2, 1]) / 9; Wn = np.convolve(np.pad(np.where(np.isnan(W), np.nanmin(W), W), 2, mode="edge"), k, mode="valid")
        v = v[(v[:, 1] >= ys[0]) & (v[:, 1] <= ys[-1])]
    else:
        xmid, Wn = footprint_halfwidth(v, ys, cell)
    y0, L = ys[0], ys[-1] - ys[0]
    s = np.linspace(0, 1, ns); u = np.linspace(-1, 1, nu); A = np.full((ns, nu), np.nan)
    yq = y0 + s * L; jj = np.clip(np.searchsorted(ys, yq) - 1, 0, len(ys) - 1)
    for i, j in enumerate(jj):
        xq = xmid + u * Wn[j]; ii = np.clip(np.round((xq - xs[0]) / cell).astype(int), 0, len(xs) - 1); A[i] = Z[j, ii]
    A = np.where(A < 0, np.nan, A); H = np.nanpercentile(A, 99.5); An = np.clip(A / H, 0, 1.2)
    Wn_s = np.interp(s * L + y0, ys, Wn)
    crest = np.nanmax(np.nan_to_num(An[:, nu // 2 - 2:nu // 2 + 3], nan=0), axis=1)
    # head is where the glabella is: the highest crest must lie in the front half; otherwise reverse s
    if int(np.argmax(np.convolve(crest, np.ones(9) / 9, mode="same"))) > ns // 2:
        An = An[::-1]; Wn_s = Wn_s[::-1]; crest = crest[::-1]
    k3 = np.array([1, 2, 1]) / 4; cs = np.convolve(np.pad(crest, 1, mode="edge"), k3, mode="valid")
    fur = []
    for i in range(3, ns - 3):
        if cs[i] < cs[i - 1] and cs[i] <= cs[i + 1] and cs[i] < cs[i - 3] - 0.012 and cs[i] < cs[i + 3] - 0.012 and cs[i] > 0.1:
            if not fur or s[i] - fur[-1] > 0.025: fur.append(float(s[i]))
    meta = dict(name=name, length_mm=float(L), max_halfwidth_mm=float(np.nanmax(Wn)), relief_mm=float(H),
                s_widest=float(s[int(np.argmax(Wn_s))]), furrows_s=fur, fossil=fossil)
    np.savez(f"{name}_atlas.npz", s=s, u=u, Z=An, W=Wn_s / np.nanmax(Wn_s), crest=cs, meta=json.dumps(meta))
    return An, meta

if __name__ == "__main__":
    import sys, re
    sys.path.insert(0, "/home/claude/skin/site")
    out = {}
    # 1. Olenoides (glTF sculpt, already flat)
    from load_sculpt import load
    parts, _ = load(); b = parts["Trilobite_spine"]; out["olenoides"] = make_atlas(b.vertices, "olenoides", normals=b.vertex_normals, flip_z=False, flip_y=False)
    # 2. Proetida (OBJ kit): body groups only, Maya Y-up -> Z-up
    V = []; faces = {}; cur = "d"
    for line in open("/home/claude/ref/src/export.obj"):
        if line.startswith("v "): V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("g "): cur = line[2:].strip()
        elif line.startswith("f "): faces.setdefault(cur, []).append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    V = np.array(V); keep = [n for n in faces if ("Cephalon" in n or "Pygidium" in n or "Segment_" in n)]
    idx = sorted({i for n in keep for f in faces[n] for i in f}); vb = V[idx][:, [0, 2, 1]]
    out["proetida"] = make_atlas(vb, "proetida")
    # 3. Harpetid (closed STL)
    h = trimesh.load("/home/claude/ref/harp/harpetid.stl", process=False); out["harpetid"] = make_atlas(h.vertices, "harpetid", normals=h.vertex_normals)
    # 4. Phacops (fossil scan in matrix)
    p = trimesh.load("/home/claude/ref/phac/src/phacops_aligned.stl", process=False); out["phacops"] = make_atlas(p.vertices, "phacops", fossil=True, flip_z=True, flip_y=False)
    for name, (A, meta) in out.items():
        print(f"{name:10s} L={meta['length_mm']:.1f} W={2*meta['max_halfwidth_mm']:.1f} H={meta['relief_mm']:.1f}  widest s={meta['s_widest']:.2f}  furrows({len(meta['furrows_s'])}) at s={np.round(meta['furrows_s'],2).tolist()}")
