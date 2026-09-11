"""
mesh.py — geometry core without OpenCascade (prompt 2 of the rewrite, 10 Sep 2026).

Everything the generator builds is a height field z = zfun(x, y) over a plan outline (u, v) -> (x, y), thickened into
a shell, then cut and joined with a handful of solids: hinge barrels, knuckles, stop blocks, the ventral bevel wedge,
spines, eyes. This module does exactly that on triangle meshes:

  * heightfield_shell / under_envelope  — the plate and its clipping envelope, WATERTIGHT AND MIRROR-SYMMETRIC BY
    CONSTRUCTION: the grid is sampled on the right half (u >= 0) and mirrored, so the two sides are the same
    triangles reflected and zfun asymmetry (seeded ornament) cannot leak in. After a Manifold boolean the geometry stays
    symmetric but the triangulation of flat faces may not — test symmetry on the surface (is_symmetric), not the vertices.
  * union / difference / intersection   — exact booleans via Manifold (already the instrument's overlap engine).
  * box / cylinder / frustum / wedge     — the solids the hinge and the spines are made of.
  * hinge                                — a port of trilobite.add_hinge (barrel cut, interleaved knuckles with webs
    clipped to the envelope, pin bore, stop block, ventral bevel clipped to its own side of the hinge line).
    Same geometry, same parameters, no OCC.
  * mirror_x / is_symmetric              — for the a/b/c part split in prompt 3 (build a, mirror to c).

No retries, no grid jitter, no sanity loop: a height-field mesh is closed by construction, and Manifold either returns
a manifold or raises. The only geometric departure from the B-spline builder is that a plate is only as smooth as its
grid; the default grid (NU x NV = 121 x 61) puts vertices ~1 mm apart on a 150 mm animal, under the 0.15 mm
tessellation tolerance the BREP builder exported at once the surface is piecewise-linear anyway.
"""
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh as _MMesh

NU, NV = 121, 61          # default grid: odd NU so the axis (u = 0) is a vertex column shared by both halves
Z_MIN = 0.05              # the BREP builder floored surfaces here
T_MIN = 0.30              # minimum shell thickness (never a zero-thickness region, which is non-manifold)

# ---------------------------------------------------------------- Manifold bridge
def to_manifold(m):
    """trimesh -> Manifold. Raises if the mesh is not a closed manifold (that is the point: fail here, loudly)."""
    if not m.is_watertight:
        raise ValueError(f"to_manifold: mesh is not watertight ({len(m.faces)} faces)")
    man = Manifold(_MMesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    _check(man, "input")
    return man

SLIVER_MM3 = 2.0          # components below this volume are boolean artifacts (coincident faces), not geometry.
                          # 12 Sep 2026: measured bimodal — ghosts 0.001-1 mm3, real severed tips 3-30 mm3; gap at 1.2-6.1.
                          # 1e-3 caught none of the ghosts. 2.0 sits in the gap.

def _drop_slivers(man):
    parts = man.decompose()
    if len(parts) <= 1: return man
    keep = [p for p in parts if abs(p.volume()) > SLIVER_MM3]
    return man if len(keep) == len(parts) else Manifold.compose(keep)

def from_manifold(man):
    man = _drop_slivers(man)
    mm = man.to_mesh()
    v = np.asarray(mm.vert_properties, np.float64)[:, :3]
    f = np.asarray(mm.tri_verts, np.int64)
    return trimesh.Trimesh(v, f, process=False)

def _check(man, what):
    st = man.status()
    if str(st).split(".")[-1] not in ("NoError", "NO_ERROR"):
        raise ValueError(f"Manifold error in {what}: {st}")
    return man

def _as_man(x): return x if isinstance(x, Manifold) else to_manifold(x)

def union(*parts):
    """Boolean union of trimesh or Manifold objects -> trimesh."""
    ms = [_as_man(p) for p in parts if p is not None]
    if not ms: raise ValueError("union of nothing")
    out = ms[0]
    for m in ms[1:]: out = out + m
    return from_manifold(_check(out, "union"))

def difference(a, *tools):
    out = _as_man(a)
    for t in tools:
        if t is None: continue
        out = out - _as_man(t)
    return from_manifold(_check(out, "difference"))

def intersection(a, b):
    return from_manifold(_check(_as_man(a) ^ _as_man(b), "intersection"))

def volume(m): return float(_as_man(m).volume())

def bodies(m, sliver_mm3=SLIVER_MM3):
    """Connected solid bodies of a mesh (a severed pleural tip shows up here, not as a silent extra solid).
    Components below sliver_mm3 are boolean artifacts (coincident/degenerate faces from the Manifold cut),
    not real geometry, and are dropped — otherwise a watertight plate carrying two zero-volume ghost shells
    is miscounted as 'severed' and censored (11 Sep 2026: this false positive was the bulk of the sweep's
    ~70% invalid rate; the real plate volume was intact). Pass sliver_mm3=0 to count every split component."""
    comps = m.split(only_watertight=True)
    if sliver_mm3 <= 0:
        return comps
    return [c for c in comps if c.volume >= sliver_mm3]

# ---------------------------------------------------------------- primitives (all centred conventions stated)
def box(sx, sy, sz, at=(0, 0, 0), align=("c", "c", "c")):
    """Axis-aligned box. align per axis: 'c' centred on `at`, 'min' box starts at `at`, 'max' box ends at `at`."""
    m = trimesh.creation.box(extents=(sx, sy, sz))
    off = []
    for a, s in zip(align, (sx, sy, sz)):
        off.append(0.0 if a == "c" else (s / 2 if a == "min" else -s / 2))
    m.apply_translation(np.asarray(at, float) + np.asarray(off))
    return m

def cylinder(r, length, axis="z", at=(0, 0, 0), sections=48):
    """Cylinder centred on `at`, along x, y or z."""
    m = trimesh.creation.cylinder(radius=r, height=length, sections=sections)
    if axis == "x": m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    elif axis == "y": m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    m.apply_translation(at)
    return m

def frustum(r0, r1, length, sections=32):
    """Tapered rod along +z from z = 0 (radius r0) to z = length (radius r1). Spines are made of these."""
    r1 = max(r1, 0.05)
    man = Manifold.cylinder(length, r0, r1, sections)
    return from_manifold(man)

def wedge(band, L, phi_deg, rear, y=0.0, z=0.0):
    """The ventral bevel tool: a box of half-size L tilted phi about the x-axis through the hinge line (y, z),
    on the rear (+y) or front (-y) side, matching trilobite.add_hinge's Box(...).rotate(Axis.X, ±phi)."""
    if rear: m = box(band, L, L, align=("c", "min", "max"))
    else:    m = box(band, L, L, align=("c", "max", "max"))
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-phi_deg if rear else phi_deg), (1, 0, 0)))
    m.apply_translation((0, y, z))
    return m

def mirror_x(m):
    """Reflect across the sagittal plane x = 0, winding fixed."""
    v = m.vertices.copy(); v[:, 0] *= -1
    return trimesh.Trimesh(v, m.faces[:, ::-1], process=False)

def is_symmetric(m, tol=0.05, n=6000, seed=0):
    """True if the SURFACE is invariant under x -> -x: points sampled on the mesh, mirrored, lie on the mesh within tol.
    (A vertex-set test is wrong after a boolean: Manifold may triangulate a flat face with an extra vertex on one side
    without changing the shape. The raw height field IS vertex-symmetric; booleans keep the geometry symmetric to Manifold's
    tolerance — measured 0.03 mm on a hinge at a 20 deg bevel — so the default tol is 0.05 mm, under any nozzle.)"""
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    pts = pts.copy(); pts[:, 0] *= -1
    _, d, _ = trimesh.proximity.closest_point(m, pts)
    return bool(np.max(d) < tol)

def mirror_asymmetry_mm(m, n=6000, seed=0):
    """Largest distance a mirrored surface sample is from the surface — 0 for a symmetric solid."""
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    pts = pts.copy(); pts[:, 0] *= -1
    _, d, _ = trimesh.proximity.closest_point(m, pts)
    return float(np.max(d))

# ---------------------------------------------------------------- height fields
REF_NU = 53               # the BREP builder's segment grid: its 2-pass blur defines the smoothing the presets were tuned on

def sample_grid(outline, zfun, nu=NU, nv=NV, passes=None, symmetric=True):
    """Sample (x, y, z) on the plan grid. u in [-1, 1] across, v in [0, 1] along. With symmetric=True the right half
    (u >= 0) is sampled and mirrored — outline(u, v) is only ever called for u >= 0 and zfun only on x >= 0.
    The 3-tap blur is the BREP builder's; passes=None scales the pass count with grid density so the smoothing LENGTH
    (not the cell count) matches the 53-wide reference grid — a finer grid then adds resolution without changing shape."""
    if nu % 2 == 0: nu += 1
    if passes is None: passes = max(1, int(round(2 * ((nu - 1) / (REF_NU - 1)) ** 2)))
    mid = nu // 2
    xs = np.zeros((nv, nu)); ys = np.zeros((nv, nu))
    cols = range(mid, nu) if symmetric else range(nu)
    for j in range(nv):
        v = j / (nv - 1)
        for i in cols:
            u = -1 + 2 * i / (nu - 1)
            x, y = outline(u, v); xs[j, i] = x; ys[j, i] = y
    if symmetric:
        xs[:, :mid] = -xs[:, nu - 1:mid:-1]; ys[:, :mid] = ys[:, nu - 1:mid:-1]
        xs[:, mid] = 0.0                                        # the axis column is exactly on the sagittal plane
    zs = zfun(np.abs(xs.ravel()), ys.ravel()).reshape(nv, nu) if symmetric else zfun(xs.ravel(), ys.ravel()).reshape(nv, nu)
    for _ in range(passes):
        zp = np.pad(zs, 1, mode="edge"); zs = 0.25 * zp[1:-1, :-2] + 0.5 * zp[1:-1, 1:-1] + 0.25 * zp[1:-1, 2:]
        zp = np.pad(zs, 1, mode="edge"); zs = 0.25 * zp[:-2, 1:-1] + 0.5 * zp[1:-1, 1:-1] + 0.25 * zp[2:, 1:-1]
    if symmetric: zs[:, :mid] = zs[:, nu - 1:mid:-1]
    # a pinched row/column (all plan points coincident) must carry ONE z, or the merge leaves a fan of vertical edges
    for j in (0, nv - 1):
        if np.ptp(xs[j]) < 1e-9 and np.ptp(ys[j]) < 1e-9: zs[j, :] = zs[j].mean()
    for i in (0, nu - 1):
        if np.ptp(xs[:, i]) < 1e-9 and np.ptp(ys[:, i]) < 1e-9: zs[:, i] = zs[:, i].mean()
    return xs, ys, zs

def _grid_faces(nv, nu, offset=0, flip=False):
    """Two triangles per quad, diagonals mirrored about the middle column so the triangulation itself is symmetric."""
    mid = nu // 2; F = []
    idx = lambda j, i: offset + j * nu + i
    for j in range(nv - 1):
        for i in range(nu - 1):
            a, b, c, d = idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i)
            if i >= mid: F += [(a, b, c), (a, c, d)]
            else:        F += [(a, b, d), (b, c, d)]
    F = np.asarray(F, np.int64)
    return F[:, ::-1] if flip else F

def _closed(top_xyz, bot_xyz, nv, nu):
    """Close a top and a bottom grid surface with side strips around the (u, v) boundary; merge coincident vertices
    (pinched outlines collapse a whole row to one point) and drop the degenerate triangles that leaves behind."""
    V = np.vstack([top_xyz, bot_xyz]); n = len(top_xyz)
    F = [_grid_faces(nv, nu, 0, flip=False), _grid_faces(nv, nu, n, flip=True)]
    def strip(loop_top, loop_bot, flip):
        S = []
        for k in range(len(loop_top) - 1):
            a, b = loop_top[k], loop_top[k + 1]; c, d = loop_bot[k + 1], loop_bot[k]
            S += [(a, b, c), (a, c, d)]
        S = np.asarray(S, np.int64)
        return S[:, ::-1] if flip else S
    idx = lambda j, i: j * nu + i
    row0 = [idx(0, i) for i in range(nu)]; row1 = [idx(nv - 1, i) for i in range(nu)]
    colL = [idx(j, 0) for j in range(nv)]; colR = [idx(j, nu - 1) for j in range(nv)]
    # boundary traversed so that outward normals are consistent with the top (counter-clockwise from above)
    F.append(strip(row0, [n + k for k in row0], flip=True))       # front edge (v = 0)
    F.append(strip(row1, [n + k for k in row1], flip=False))      # rear edge (v = 1)
    F.append(strip(colL, [n + k for k in colL], flip=False))      # left edge (u = -1)
    F.append(strip(colR, [n + k for k in colR], flip=True))       # right edge (u = +1)
    m = trimesh.Trimesh(V, np.vstack(F), process=True)            # merges coincident vertices
    m.update_faces(m.nondegenerate_faces()); m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    if not m.is_watertight:
        raise ValueError("height-field solid is not watertight (degenerate outline?)")
    if m.volume < 0: m.invert()
    return m

def heightfield_shell(outline, zfun, t, nu=NU, nv=NV, z_min=Z_MIN, symmetric=True):
    """Shell of vertical thickness t under z = zfun(x, y) on the plan outline(u, v). Same floors as trilobite.plate:
    top >= z_min, bottom = max(top - t, z_min), and never thinner than T_MIN."""
    xs, ys, zs = sample_grid(outline, zfun, nu, nv, symmetric=symmetric)
    nv_, nu_ = zs.shape
    top = np.maximum(zs, z_min + T_MIN)
    bot = np.minimum(np.maximum(zs - t, z_min), top - T_MIN)
    T = np.column_stack([xs.ravel(), ys.ravel(), top.ravel()]); B = np.column_stack([xs.ravel(), ys.ravel(), bot.ravel()])
    return _closed(T, B, nv_, nu_)

def under_envelope(outline, zfun, nu=NU, nv=NV, floor=-1.0, z_min=Z_MIN, symmetric=True):
    """Everything under the surface down to z = floor: the clipping envelope for hinge webs and spine roots."""
    xs, ys, zs = sample_grid(outline, zfun, nu, nv, symmetric=symmetric)
    nv_, nu_ = zs.shape
    top = np.maximum(zs, z_min + T_MIN)
    T = np.column_stack([xs.ravel(), ys.ravel(), top.ravel()]); B = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, floor)])
    return _closed(T, B, nv_, nu_)

# ---------------------------------------------------------------- the hinge (port of trilobite.add_hinge)
def hinge(part, envelope, y_axis, rear, zh, Wh, barrel_r, clearance, n_knuckles, ring_top, wall, bore_d,
          band, reach, bevel_deg, wide=False):
    """Interleaving-knuckle pin hinge on a plate edge, plus the ventral bevel that sets the flexion stop.
    Geometry identical to trilobite.add_hinge; all quantities are passed in (mm, degrees), nothing read from P.
      y_axis   hinge line y;  rear = True for the plate's rear edge (knuckles on the odd slots), False for its front
      zh, Wh   hinge line height and knuckle span;  band  width of the bevel cut;  reach  how far past the hinge line
      bevel_deg  full included bevel angle (the wedge is tilted by half of it)."""
    rB, c, nK = barrel_r, clearance, int(n_knuckles)
    Wh_min = nK * c + 1.0
    if Wh < Wh_min: Wh = Wh_min
    kw, x0 = Wh / nK, -Wh / 2
    L = 80.0
    def barrel(xc, length, r): return cylinder(r, length, axis="x", at=(xc, y_axis, zh))
    P = to_manifold(part); E = to_manifold(envelope)
    P = P - to_manifold(barrel(0, Wh + 4, rB + c))
    webs = None
    for i in range(nK):
        if (i % 2 == 0) != rear: continue
        xc = x0 + (i + 0.5) * kw
        P = P + to_manifold(barrel(xc, kw - c, rB))
        z_lo = zh - rB - 0.5 * c - 1.5
        web = to_manifold(box(kw - c, rB + c + 1.0, ring_top + 1.0 - z_lo, at=(xc, y_axis, z_lo),
                              align=("c", "max" if rear else "min", "min")))
        webs = web if webs is None else webs + web
    if webs is not None: P = P + (webs ^ E)
    P = P - to_manifold(barrel(0, Wh + 10, bore_d / 2))
    blk_h = max(zh - rB - c - 0.5, 0.3)
    P = P + to_manifold(box(Wh + 2, 2 * wall, blk_h, at=(0, y_axis, 0), align=("c", "max" if rear else "min", "min")))
    phi = bevel_deg / 2
    W = to_manifold(wedge(band, L, phi, rear, y=y_axis, z=zh))
    if rear: own = box(band + 2, L + reach, 2 * L, at=(0, y_axis - reach, zh), align=("c", "min", "c"))
    else:    own = box(band + 2, L + reach, 2 * L, at=(0, y_axis + reach, zh), align=("c", "max", "c"))
    keep = to_manifold(barrel(0, Wh + 4, rB + c))
    for clip in (own, box(Wh + 2, 4 * L, 4 * L, at=(0, y_axis, zh))):
        cut = (W ^ to_manifold(clip)) - keep
        P = P - cut
    return from_manifold(_check(P, "hinge"))

# ---------------------------------------------------------------- convenience for the tests and prompt 3
def superellipse_outline(L, W, n_front=2.2, n_rear=2.2, pinch_front=True, pinch_rear=True):
    """A symmetric plan: x = u * halfwidth(v), y = v * L, with the half-width following a superellipse that pinches
    to a point at either end. Enough to stand in for a cephalon or a pygidium in tests."""
    def hw(v):
        s = 2 * v - 1
        n = n_rear if s > 0 else n_front
        return 0.5 * W * max(1 - abs(s) ** n, 0.0) ** (1 / n)
    def outline(u, v):
        h = hw(v)
        if (v <= 0 and pinch_front) or (v >= 1 and pinch_rear): h = 0.0
        return u * h, v * L
    return outline
