"""
skins.py — reference skins as shape keys.

A skin is a reference mesh (e.g. an extracted cephalon) registered to the generator's canonical frame
and resampled onto the generator's (u, v) grid:
    v ∈ [0, 1]  rear (hinge line, y = 0) → front tip;    u ∈ [-1, 1]  left margin → right margin
    silhouette  w(v)  half-width at row v, as a fraction of the maximum half-width
    field       z(v, u) height above the ground, as a fraction of the skin's length
Every skin on the same grid → blending is a weighted sum. Undercuts are dropped by construction
(top surface only); anything a ray from above cannot see is an appendage, not a skin.
"""
import json, os, hashlib
import numpy as np
import trimesh

NU, NV = 45, 33           # the generator's head grid
RASTER = 0.6              # mm per raster cell (before normalisation); finer than the grid

# ---------------------------------------------------------------- registration
def _principal_axis_xy(v, which="minor"):
    """In-plane principal axis. Heads are wider than long, so the body (length) axis is the MINOR one."""
    c = v[:, :2].mean(0); w, vec = np.linalg.eigh(np.cov((v[:, :2] - c).T))
    return vec[:, np.argmin(w) if which == "minor" else np.argmax(w)]

def register(mesh, rear="auto", axis="minor"):
    """Rigidly align a reference head: up = +z, body axis = y, midline = x=0, ground = z=0, rear edge at y=0,
    head extending toward -y. Returns the aligned mesh and a report (for bug checks)."""
    m = mesh.copy()
    # up axis = thinnest extent (trilobites are flat)
    up = int(np.argmin(m.extents))
    if up != 2:
        R = np.eye(4)
        if up == 1: R[:3, :3] = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])[:3, :3]
        else:       R[:3, :3] = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])[:3, :3]
        m.apply_transform(R)
    # body axis → y
    a = _principal_axis_xy(m.vertices, axis)
    ang = np.arctan2(a[0], a[1])                              # rotate a onto +y
    m.apply_transform(trimesh.transformations.rotation_matrix(-ang, [0, 0, 1]))   # rotate the axis onto +y
    # ground and midline
    m.apply_translation([-m.vertices[:, 0].mean() * 0 - 0.5 * (m.bounds[0][0] + m.bounds[1][0]), 0, -m.bounds[0][2]])
    # which end is the rear? the flat cut / straight margin: compare widths near both ends
    y = m.vertices[:, 1]; L = y.max() - y.min()
    def width_near(yy, frac=0.08):
        sel = np.abs(y - yy) < frac * L
        return np.ptp(m.vertices[sel, 0]) if sel.any() else 0.0
    w_lo, w_hi = width_near(y.min() + 0.04 * L), width_near(y.max() - 0.04 * L)
    if rear == "auto":
        rear_is_high = w_hi > w_lo                            # rear (occipital) edge is the wide, straight end
    else:
        rear_is_high = (rear == "high")
    if not rear_is_high:                                      # flip so the rear is at +y, then shift rear to y=0
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
    m.apply_translation([0, -m.bounds[1][1], 0])              # rear edge → y = 0, head extends toward -y
    m.apply_translation([-0.5 * (m.bounds[0][0] + m.bounds[1][0]), 0, 0])
    rep = dict(length=float(-m.bounds[0][1]), halfwidth=float(m.bounds[1][0]), height=float(m.bounds[1][2]),
               up_axis_was=int(up), rear_was="high" if rear_is_high else "low")
    return m, rep

# ---------------------------------------------------------------- resampling
def rasterize_top(mesh, cell=None, n_samples=250000):
    """Top-surface height map from surface samples + vertices: max z per (x, y) cell.
    Cell size is relative to the mesh (1/140 of its width) so tiny and huge references behave alike."""
    if cell is None: cell = max(mesh.extents[0], mesh.extents[1]) / 140.0
    pts = np.vstack([mesh.vertices, trimesh.sample.sample_surface(mesh, n_samples)[0]])
    x0, x1 = mesh.bounds[0][0], mesh.bounds[1][0]; y0, y1 = mesh.bounds[0][1], mesh.bounds[1][1]
    nx = max(4, int(np.ceil((x1 - x0) / cell)) + 1); ny = max(4, int(np.ceil((y1 - y0) / cell)) + 1)
    ix = np.clip(((pts[:, 0] - x0) / cell).astype(int), 0, nx - 1); iy = np.clip(((pts[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    H = np.full((ny, nx), -np.inf)
    np.maximum.at(H, (iy, ix), pts[:, 2])            # max z per cell
    H = np.where(np.isinf(H), np.nan, H)             # empty cells → NaN
    return H, (x0, y0, cell)

def resample(mesh, nu=NU, nv=NV):
    """Skin arrays on the (u, v) grid: silhouette w[v] (fraction of max half-width) and field z[v, u]
    (fraction of length). Rows run rear (v=0, y=0) → front tip (v=1)."""
    H, (x0, y0, cell) = rasterize_top(mesh)
    ny, nx = H.shape
    L = -mesh.bounds[0][1]; W = mesh.bounds[1][0]
    occupied = ~np.isnan(H)
    # silhouette per raster row: outermost occupied cell; then smooth lightly
    xs = x0 + (np.arange(nx) + 0.5) * cell
    row_w = np.array([np.abs(xs[occupied[j]]).max() if occupied[j].any() else 0.0 for j in range(ny)])
    ys = y0 + (np.arange(ny) + 0.5) * cell            # ascending y: front (most negative) → rear (0)
    # fill holes inside the silhouette by nearest occupied cell along the row
    Hf = H.copy()
    for j in range(ny):
        occ = np.where(occupied[j])[0]
        if len(occ) == 0: continue
        for i in range(nx):
            if np.isnan(Hf[j, i]) and abs(xs[i]) <= row_w[j]:
                Hf[j, i] = H[j, occ[np.argmin(np.abs(occ - i))]]
    Hf = np.where(np.isnan(Hf), 0.0, Hf)
    # sample the grid: v from rear (y=0) to front (y=-L)
    w = np.zeros(nv); z = np.zeros((nv, nu))
    for jv in range(nv):
        v = jv / (nv - 1); yq = -v * L
        j = int(np.clip((yq - y0) / cell, 0, ny - 1))
        wv = max(row_w[j], 0.02 * W); w[jv] = wv / W
        for iu in range(nu):
            u = -1 + 2 * iu / (nu - 1); xq = u * wv
            i = int(np.clip((xq - x0) / cell, 0, nx - 1))
            z[jv, iu] = Hf[j, i] / L
    # symmetry check on the field itself (cheap), then symmetrise: references are bilateral up to scan noise
    asym = float(np.nanpercentile(np.abs(z - z[:, ::-1]) * L, 95))
    z = 0.5 * (z + z[:, ::-1])
    return dict(w=w, z=z, length=L, halfwidth=W, relief=float(mesh.bounds[1][2]), asymmetry_p95_mm=asym)

def make_skin(path, name, out_dir, rear="auto", axis="minor"):
    m = trimesh.load(path)
    if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(list(m.geometry.values()))
    m, rep = register(m, rear=rear, axis=axis)
    s = resample(m)
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(rep); meta.update(name=name, source=os.path.basename(path), length=s["length"], halfwidth=s["halfwidth"],
                relief=s["relief"], asymmetry_p95_mm=s["asymmetry_p95_mm"])
    np.savez(os.path.join(out_dir, name + ".npz"), w=s["w"], z=s["z"], meta=json.dumps(meta))
    return s, rep

def load_skin(path):
    d = np.load(path); return dict(w=d["w"], z=d["z"], meta=json.loads(str(d["meta"])))

def blend(skins, weights):
    """Weighted mix of skins on the common grid. Weights are normalised."""
    wts = np.array(weights, float); wts = wts / wts.sum()
    return dict(w=sum(wt * s["w"] for wt, s in zip(wts, skins)), z=sum(wt * s["z"] for wt, s in zip(wts, skins)))

def skin_functions(skin, length, halfwidth):
    """Turn a (blended) skin into outline(u, v) and zfun(x, y) for trilobite.plate at a target size."""
    w, z = skin["w"], skin["z"]; nv, nu = z.shape
    def outline(u, v):
        wv = float(np.interp(v, np.linspace(0, 1, nv), w)) * halfwidth
        return u * wv, -v * length
    def zfun(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        v = np.clip(-y / length, 0, 1)
        wv = np.interp(v, np.linspace(0, 1, nv), w) * halfwidth
        u = np.clip(x / np.maximum(wv, 1e-6), -1, 1)
        fv = v * (nv - 1); fu = (u + 1) / 2 * (nu - 1)
        j0 = np.clip(np.floor(fv).astype(int), 0, nv - 2); i0 = np.clip(np.floor(fu).astype(int), 0, nu - 2)
        tj = fv - j0; ti = fu - i0
        zz = (1 - tj) * ((1 - ti) * z[j0, i0] + ti * z[j0, i0 + 1]) + tj * ((1 - ti) * z[j0 + 1, i0] + ti * z[j0 + 1, i0 + 1])
        return zz * length
    return outline, zfun
