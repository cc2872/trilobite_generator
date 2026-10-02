"""
isopod_model/crescent_head.py (was scripts/isopod_head.py): the crescent head (anatomy/head_crescent.py) layered onto
the isopod asset's head piece. This is the model's head; isopod_model/build.py takes it from here.

    python isopod_model/crescent_head.py [out_dir]   -> <out_dir>/isopod_crescent_head.stl (all 8 isopod pieces, head
                                                        piece replaced) + .json (the joints' free ranges)

Nothing of the isopod is removed or moved: the layer is only ever added ABOVE the head piece's top surface, so the
ball socket, the leg sockets, the midline pocket and the underside are untouched, and pieces 1-7 are the originals.

  * fit: the crescent at uniform scale (its shape is not stretched), its round front inside the head piece's
    outline with a MARGIN, its widest line where that allows the largest head.
  * over the head piece: the crescent's relief (head_crescent's surface x scale) is added on top of the head piece's
    own top surface, the layer's underside buried a little in it (never more than half the plate's thickness).
  * behind the head piece: the horns (and the cheeks behind its rear edge) ride over segments 1-3 on an air gap,
    like the isopod's own plates overlap without touching. Their underside is GAP above the highest point any of
    segments 1-3 reaches anywhere in the joints' free range measured on the original model (dorsal/ventral pitch,
    yaw, roll at each ball, combined), so the head cannot touch the body in any pose the original allows.
Checks: the original head piece is contained in the new one; the new head piece is one body; symmetric; and at
every pose of a grid over the free range, the new head piece meets segments 1-3 no more than the original did.
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, trimesh
from manifold3d import Manifold, Mesh as MMesh
import mesh as M
from anatomy import head_crescent as HC

ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "isopod", "giant_isopod_main_itbefred.stl")
MARGIN = 1.0      # the crescent's front stays this far inside the head piece's outline
GAP = 1.5         # horns' clearance over the swept body (the isopod's plates: 1.4-5 mm)
EMBED = 0.8       # the layer's underside sinks this far into the head piece (at most half its local thickness)
STEP = 0.5        # envelope / surface sampling, mm
GRID = (241, 121)
SKIN = 1.2        # the details sit on a skin this thick over the head piece (furrows cut into the skin, not the piece)
FADE = 3.0        # ... which fades to nothing over this distance from the edge, so the edge is flush
DRAPE = 8.0       # horns fall from the head piece's rear edge to the body's line over this distance
HORN_T = 1.6      # minimum horn thickness where it rides over the body
CHECK = True
RIM_H = 1.5       # band height at its edges: the head's rim level
BAND_PEAK = 4.0   # band ridge height at the horn root
HORN_W0 = 7.0     # band width at the widest line
HORN_W1 = 3.0     # horn width at the tip
HORN_REACH = 0.85 # horn length behind the widest line / the head's half-width (head3d: 20 / 24)
NOSE_LIP = 2.0    # band width at the nose      # run the pose clearance check (needs the pose grid even when the envelope is cached)


def man(p): return Manifold(MMesh(vert_properties=np.asarray(p.vertices, np.float32), tri_verts=np.asarray(p.faces, np.uint32)))
def rot(axis, deg, c): return trimesh.transformations.rotation_matrix(np.radians(deg), axis, c)
AX = dict(pitch=(1, 0, 0), yaw=(0, 0, 1), roll=(0, 1, 0))


def load():
    m = trimesh.load(ASSET); z0 = m.bounds[0][2]; xc = 0.5 * (m.bounds[0][0] + m.bounds[1][0])
    m.apply_translation((-xc, 0, -z0))                                    # work frame: midline x = 0, bed z = 0
    return sorted(m.split(only_watertight=False), key=lambda p: p.bounds[0][1]), (xc, z0)


def ball_centre(p):
    """Centre of a piece's front ball: a sphere on the midline, measured from its sections 3.5-5 mm off axis."""
    y_hi = p.bounds[0][1] + 13; o = []
    for dx in (3.5, 4.0, 4.5, 5.0):
        for d in p.section(plane_origin=(dx, 0, 0), plane_normal=(1, 0, 0)).discrete:
            c = d.mean(0)
            if d[:, 1].max() < y_hi and 2 < c[2] < 13: o.append(c[1:])
    return np.array([0.0, *np.mean(o, 0)])


def free_range(A, B, c, step=1.0, lim=45.0):
    """Degrees each way B turns about c (pitch/yaw/roll) before it meets A (overlap > 0.5 mm3), from rest."""
    A, B = man(A), man(B); out = {}
    for name, ax in AX.items():
        for sgn in (+1, -1):
            a = lim
            for deg in np.arange(step, lim + step, step):
                if (A ^ B.transform(rot(ax, sgn * deg, c)[:3, :])).volume() > 0.5: a = deg - step; break
            out[(name, sgn)] = a
    return out


def poses(rng, n_pitch_down=3):
    """A grid over one joint's free range: dorsal pitch limit and rest, ventral pitch in steps, yaw and roll limits."""
    P = [rng[("pitch", 1)], 0.0] + [-rng[("pitch", -1)] * k / n_pitch_down for k in range(1, n_pitch_down + 1)]
    Y = [-rng[("yaw", -1)], 0.0, rng[("yaw", 1)]]; R = [-rng[("roll", -1)], 0.0, rng[("roll", 1)]]
    return list(itertools.product(P, Y, R))


def T_joint(c, pitch, yaw, roll):
    return rot(AX["pitch"], pitch, c) @ rot(AX["yaw"], yaw, c) @ rot(AX["roll"], roll, c)


def chain_poses(parts, C, ranges):
    """[(T1, T2, T3)]: segments 1-3 posed over joints 0-1, 1-2 (grid) and 2-3 (rest and its dorsal limit), keeping
    only poses the ORIGINAL model allows (no neighbouring pair overlapping by more than 0.5 mm3): each joint's
    limits were measured from rest, and combining extremes of two joints can ask for poses the body cannot reach."""
    A = [man(p) for p in parts[:4]]; out = []
    g23 = [(0.0, 0.0, 0.0), (ranges[3][("pitch", 1)], 0.0, 0.0)]
    for a in poses(ranges[1]):
        T1 = T_joint(C[1], *a)
        for b in poses(ranges[2], n_pitch_down=1):
            T2 = T1 @ T_joint(C[2], *b)
            for g in g23:
                Ts = (T1, T2, T2 @ T_joint(C[3], *g))
                P = [A[0]] + [A[k + 1].transform(Ts[k][:3, :]) for k in range(3)]
                if all((P[k] ^ P[k + 1]).volume() <= 0.5 for k in range(3)): out.append(Ts)
    return out


def envelope(parts, TT, xs, ys):
    """H[y, x]: the highest z any of segments 1-3 reaches over each cell, over every pose in TT."""
    H = np.full((len(ys), len(xs)), -np.inf)
    clouds = []
    for k, p in enumerate(parts[1:4]):                         # upward-facing surface only (the envelope is a max)
        pts, fi = p.sample(int(p.area * 4), seed=k, return_index=True)
        pts = pts[p.face_normals[fi, 2] > -0.6]; clouds.append(np.c_[pts, np.ones(len(pts))])
    for Ts in TT:
        for T, C in zip(Ts, clouds):
            w = C @ T.T
            i = np.round((w[:, 0] - xs[0]) / STEP).astype(int); j = np.round((w[:, 1] - ys[0]) / STEP).astype(int)
            ok = (i >= 0) & (i < len(xs)) & (j >= 0) & (j < len(ys))
            np.maximum.at(H, (j[ok], i[ok]), w[ok, 2])
    # fill the sampling holes and dilate by a cell: an upper envelope, never below any sampled point
    from scipy.ndimage import maximum_filter
    return maximum_filter(H, size=3, mode="nearest")


def head_surface(p0, xs, ys):
    """Top of the head piece and the bottom of its top plate, per cell (NaN where it is absent)."""
    X, Y = np.meshgrid(xs, ys); O = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 200.0)]
    loc, idx, _ = p0.ray.intersects_location(O, np.tile([0, 0, -1.0], (len(O), 1)), multiple_hits=True)
    top = np.full(len(O), np.nan); second = np.full(len(O), np.nan)
    order = np.lexsort((-loc[:, 2], idx)); loc, idx = loc[order], idx[order]
    first = np.r_[True, idx[1:] != idx[:-1]]
    top[idx[first]] = loc[first, 2]
    nxt = np.r_[False, first[:-1]] & ~first                                  # the second hit of each ray
    second[idx[nxt]] = loc[nxt, 2]
    return top.reshape(X.shape), second.reshape(X.shape)


def cover(top, xs, ys):
    """Smallest uniform scale k and widest line y_w such that the crescent's round front covers the head piece's
    whole outline plus MARGIN."""
    present = ~np.isnan(top); Y, X = np.nonzero(present); px, py = np.abs(xs[X]), ys[Y]
    y_nose = py.min(); r = px.max() + MARGIN
    while True:                                   # front of the crescent at the head's nose, widest line behind it
        y_w = y_nose - MARGIN + r; front = py <= y_w
        if np.hypot(px[front], py[front] - y_w).max() <= r: return r / HC.RO, y_w      # MARGIN is already in r and y_w
        r += 0.25


def fit(top, xs, ys):
    """Largest uniform scale k and widest line y_w such that the crescent's round front (radius 24 k about (0, y_w))
    lies inside the head piece's outline minus MARGIN."""
    present = ~np.isnan(top); W0 = np.array([np.abs(xs[r]).max() if r.any() else 0.0 for r in present])
    best = (0, 0)
    for y_w in np.arange(30, 55, 0.25):
        lo, hi = 0.2, 3.0
        for _ in range(40):
            k = 0.5 * (lo + hi); r = HC.RO * k
            yy = ys[(ys >= y_w - r) & (ys <= y_w)]
            need = np.sqrt(np.clip(r ** 2 - (yy - y_w) ** 2, 0, None)) + MARGIN
            ok = (y_w - r - MARGIN >= ys[present.any(1)].min()) and np.all(need <= np.interp(yy, ys, W0))
            lo, hi = (k, hi) if ok else (lo, k)
        if lo > best[0]: best = (lo, y_w)
    return best


def build_layer(top0, bot0, H, xs, ys, k, y_w):
    """The crescent on the isopod's own dome. The head keeps the head piece's smooth top and its front outline (no
    brim); the crescent brings its horns and notch and, as a thin skin that fades out at the edge, its details
    (head3d's surface minus head3d's dome: glabella, furrows, border, eyes, occipital ring, horn ridge). Nothing is
    cut from the head piece. Behind it the horns drape smoothly down to ride GAP over the swept body."""
    from scipy.ndimage import maximum_filter, gaussian_filter
    from scipy.interpolate import RegularGridInterpolator as RGI
    # smooth upper envelope of the body, and gap-filled head surfaces for bilinear reads
    Hf = np.where(np.isfinite(H), H, -50.0)
    Hd = maximum_filter(Hf, size=5); Hs = np.maximum(gaussian_filter(Hd, 3.0), maximum_filter(Hf, size=3))
    fill = lambda A: np.where(np.isfinite(A), A, np.nan_to_num(maximum_filter(np.nan_to_num(A, nan=-50.0), size=5), nan=-50.0))
    rd = lambda A: RGI((ys, xs), A, bounds_error=False, fill_value=None)
    Ht, T0, B0 = rd(Hs), rd(fill(top0)), rd(fill(bot0))
    present = np.isfinite(top0)
    y_front0 = np.array([ys[present[:, c]].min() if present[:, c].any() else np.inf for c in range(len(xs))])
    y_rear0 = np.array([ys[present[:, c]].max() if present[:, c].any() else -np.inf for c in range(len(xs))])

    nu, nv = GRID; U, V = np.meshgrid(np.linspace(-1, 1, nu), np.linspace(0, 1, nv))
    Xh = U * HC.RO; x = k * Xh
    Yr = HC._y_rear(np.abs(Xh))
    yf_c = y_w + k * (HC._y_front(np.abs(Xh)) - HC.C)
    yf_p = np.interp(np.abs(x), xs, y_front0) + 0.3                       # never in front of the head piece
    Yf = (np.maximum(yf_c, yf_p) - y_w) / k + HC.C
    Yh = Yr - V * (Yr - Yf); y = y_w + k * (Yh - HC.C)
    q = np.c_[y.ravel(), np.abs(x).ravel()]
    t0, b0, hs = T0(q).reshape(x.shape), B0(q).reshape(x.shape), Ht(q).reshape(x.shape)
    inside = (y <= np.interp(np.abs(x), xs, y_rear0)) & (y >= np.interp(np.abs(x), xs, y_front0))
    floor_ = hs + GAP
    on_head = inside & (t0 >= floor_)

    # the crescent's details: head3d's surface minus its dome
    yc = HC.Y0 - 0.19 * HC.LC; aD = HC.DOME_FILL * HC.RO; bD = 0.88 * HC.DOME_FILL * HC.LC
    rD = (np.abs(Xh) / aD) ** HC.DOME_EXP + (np.abs(Yh - yc) / bD) ** HC.DOME_EXP
    dome = HC.RIM + (HC.RELIEF - HC.RIM) * np.clip(1 - rD, 0, 1) ** (1 / HC.DOME_EXP)
    detail = k * (HC._z(Xh, Yh) - dome)
    skin = np.maximum(SKIN + detail, 0.2)                                 # never below the head piece's own top
    fade = np.clip(np.minimum(y - np.maximum(yf_c, yf_p), k * HC.RO - np.abs(x)) / FADE, 0, 1)
    fade = fade * fade * (3 - 2 * fade)

    # behind the head piece: drape from its rear edge down onto the body's line
    t_edge = T0(np.c_[np.minimum(y, np.interp(np.abs(x), xs, y_rear0)).ravel(), np.abs(x).ravel()]).reshape(x.shape)
    d_e = np.clip(y - np.interp(np.abs(x), xs, y_rear0), 0, None)
    drape = floor_ + np.clip(t_edge - floor_, 0, None) * (1 - np.clip(d_e / DRAPE, 0, 1) ** 2) ** 2
    base = np.where(on_head, t0, np.maximum(floor_, np.where(inside, np.maximum(t0, floor_), drape)))
    thick = np.where(on_head, skin * fade, np.maximum(skin, HORN_T))
    topz = base + np.maximum(thick, 0.05)
    embed = np.where(np.isfinite(b0), np.minimum(EMBED, 0.5 * np.clip(t0 - b0, 0, None)), EMBED)
    bottom = np.where(on_head, t0 - np.maximum(embed, 0.05), base)
    T = np.c_[x.ravel(), y.ravel(), topz.ravel()]; B = np.c_[x.ravel(), y.ravel(), bottom.ravel()]
    layer = M._closed(T, B, nv, nu)
    print(f"layer: {len(layer.faces)} faces, watertight {layer.is_watertight}, {layer.volume:.0f} mm3")
    return layer


def build_skin(top0, bot0, xs, ys, k, y_w, face=None):
    """The crescent's details as a thin skin on the head piece's own dome, over the head piece's footprint and
    inside the crescent's outline, fading to nothing at both edges. Never below the head piece's top."""
    from scipy.interpolate import RegularGridInterpolator as RGI
    from scipy.ndimage import maximum_filter
    present = np.isfinite(top0)
    fill = lambda A: np.where(np.isfinite(A), A, maximum_filter(np.nan_to_num(A, nan=-50.0), size=5))
    T0, B0 = (RGI((ys, xs), fill(A), bounds_error=False, fill_value=None) for A in (top0, bot0))
    yf0 = np.array([ys[present[:, c]].min() if present[:, c].any() else np.nan for c in range(len(xs))])
    yr0 = np.array([ys[present[:, c]].max() if present[:, c].any() else np.nan for c in range(len(xs))])
    ok = np.isfinite(yf0) & (xs >= 0); xw = xs[ok].max() - 0.5
    nu, nv = GRID; U, V = np.meshgrid(np.linspace(-1, 1, nu), np.linspace(0, 1, nv))
    x = U * xw; ax = np.abs(x)
    yf = np.interp(ax, xs[ok], yf0[ok]) + 0.3; yr = np.interp(ax, xs[ok], yr0[ok]) - 0.3
    y = yr - V * (yr - yf)
    Xh, Yh = x / k, (y - y_w) / k + HC.C
    yc = HC.Y0 - 0.19 * HC.LC; aD = HC.DOME_FILL * HC.RO; bD = 0.88 * HC.DOME_FILL * HC.LC
    rD = (np.abs(Xh) / aD) ** HC.DOME_EXP + (np.abs(Yh - yc) / bD) ** HC.DOME_EXP
    dome = HC.RIM + (HC.RELIEF - HC.RIM) * np.clip(1 - rD, 0, 1) ** (1 / HC.DOME_EXP)
    detail = np.maximum(SKIN + k * (HC._z(Xh, Yh, face) - dome), 0.2)
    sm = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
    edge = np.minimum(np.minimum(y - yf, yr - y), xw - ax)                    # to the head piece's edge
    notch = (HC._y_rear(np.abs(Xh)) - Yh) * k                                  # to the crescent's notch (behind it: none)
    fade = sm(edge / FADE) * sm(notch / FADE)
    q = np.c_[y.ravel(), ax.ravel()]; t0 = T0(q).reshape(x.shape); b0 = B0(q).reshape(x.shape)
    embed = np.minimum(EMBED, 0.5 * np.clip(t0 - b0, 0, None)); embed = np.maximum(embed, 0.05)
    top = t0 + np.maximum(detail * fade, 0.05); bot = t0 - embed
    return M._closed(np.c_[x.ravel(), y.ravel(), top.ravel()], np.c_[x.ravel(), y.ravel(), bot.ravel()], nv, nu)


def body_sides(parts, TT, ys, z_top):
    """E(y): the farthest |x| any of segments 1-3 reaches at heights 0 .. z_top, over every pose in TT."""
    E = np.zeros(len(ys))
    clouds = []
    for kk, p in enumerate(parts[1:4]):
        pts = p.sample(int(p.area * 4), seed=kk); clouds.append(np.c_[pts, np.ones(len(pts))])
    for Ts in TT:
        for T, Cc in zip(Ts, clouds):
            w = Cc @ np.asarray(T).T; sel = (w[:, 2] >= -0.5) & (w[:, 2] <= z_top)
            j = np.round((w[sel, 1] - ys[0]) / STEP).astype(int); okj = (j >= 0) & (j < len(ys))
            np.maximum.at(E, j[okj], np.abs(w[sel, 0][okj]))
    from scipy.ndimage import maximum_filter1d, gaussian_filter1d
    Ed = maximum_filter1d(E, size=int(round(3.0 / STEP)) | 1)
    return np.maximum(gaussian_filter1d(Ed, 1.5 / STEP), Ed)


def curl_poses(C, ranges, n=4):
    """Rest and the ventral curl only (every joint pitched down together in n steps): the poses the horns must clear.
    Side swing and twist are left to the horns to stop, as the isopod's own plates do."""
    out = []
    for f in np.linspace(0, 1, n + 1):
        T1 = T_joint(C[1], -f * ranges[1][("pitch", -1)], 0, 0)
        T2 = T1 @ T_joint(C[2], -f * ranges[2][("pitch", -1)], 0, 0)
        out.append((T1, T2, T2 @ T_joint(C[3], -f * ranges[3][("pitch", -1)], 0, 0)))
    return out


def build_band(parts, TT, top0, xs, ys, y_w):
    """The crescent on the bed, drawn with smooth curves (never the body's bumps):
      outer edge: an ellipse round the head's front (NOSE_LIP ahead of the nose, HORN_W0 beyond the head's widest
                  point), carried on behind the widest line as one gentle arc that tapers into the horn tips;
      inner edge: fused to the head piece in front; beside the body, a quadratic fitted to the body's sides and
                  pushed out until it clears them by GAP everywhere, at rest and through the curl.
    head3d's ridge on top, RIM_H at both edges."""
    from scipy.ndimage import distance_transform_edt, binary_erosion, gaussian_filter, gaussian_filter1d
    p0 = np.isfinite(top0)
    W0 = gaussian_filter1d(np.array([np.abs(xs[r]).max() if r.any() else 0.0 for r in p0]), 2.0)
    E = body_sides(parts, TT, ys, BAND_PEAK + GAP)
    has = E > 0; y_b = ys[has].min(); y_nose = ys[p0.any(1)].min()
    A = W0.max() + HORN_W0; B = y_w - (y_nose - NOSE_LIP)
    y_tip = y_w + HORN_REACH * A
    sel = has & (ys <= y_tip)
    from scipy.ndimage import maximum_filter1d
    need = np.where(has, E + GAP, 0.0)
    broad = gaussian_filter1d(maximum_filter1d(need, size=int(round(15.0 / STEP)) | 1), 4.0 / STEP)
    I_s = broad + max(np.max((need - broad)[sel]), 0.0)          # smooth, and clear of the body everywhere
    # outer edge behind the widest line: from A down to the tip width, eased
    t = np.clip((ys - y_w) / (y_tip - y_w), 0, 1); ease = t * t * (3 - 2 * t)
    O_back = (1 - ease) * A + ease * (I_s + HORN_W1)
    O = np.where(ys <= y_w, A * np.sqrt(np.clip(1 - ((y_w - ys) / B) ** 2, 0, 1)), np.maximum(O_back, I_s + HORN_W1))
    rt = 0.5 * HORN_W1; yct = y_tip - rt; xct = float(np.interp(yct, ys, I_s)) + rt
    head_er = binary_erosion(p0, iterations=1)
    Hm = lambda qy, qx: head_er[np.clip(np.round((qy - ys[0]) / STEP).astype(int), 0, len(ys) - 1),
                                np.clip(np.round((qx - xs[0]) / STEP).astype(int), 0, len(xs) - 1)]
    def inside(qx, qy):
        ax = np.abs(qx); o = np.interp(qy, ys, O); ii = np.where(qy >= y_b, np.interp(qy, ys, I_s), 0.0)
        m = (ax <= o) & (ax >= ii) & (qy >= y_nose - NOSE_LIP) & (qy <= y_tip) & ~Hm(qy, ax)
        return m & ~((qy > yct) & (np.hypot(ax - xct, qy - yct) > rt))
    # the ridge: rim at the edges, BAND_PEAK in the middle, tapering toward the tips; smooth
    fx = np.arange(0, A + 2, 0.25); fy = np.arange(y_nose - NOSE_LIP - 1, y_tip + 1, 0.25)
    FX, FY = np.meshgrid(fx, fy); band = inside(FX, FY)
    d = distance_transform_edt(band) * 0.25
    half = np.maximum(0.5 * (np.interp(fy, ys, O) - np.where(fy >= y_b, np.interp(fy, ys, I_s), np.interp(fy, ys, W0))), 0.6)[:, None]
    taper = 1 - 0.4 * np.clip((FY - y_w) / (y_tip - y_w), 0, 1)
    z = RIM_H + (BAND_PEAK - RIM_H) * taper * np.clip(d / half, 0, 1) ** 0.6
    z = np.where(band, gaussian_filter(np.where(band, z, RIM_H), 3.0), RIM_H)
    from scipy.interpolate import RegularGridInterpolator as RGI
    Z = RGI((fy, fx), z, bounds_error=False, fill_value=RIM_H)
    print(f"band: {A:.1f} mm each side at y {y_w:.1f}, inner edge {I_s[sel & (ys > y_b + 5)].min():.1f}-{I_s[sel].max():.1f} mm beside the body, tips at y {y_tip:.1f}")
    return _band_mesh(inside, Z, y_w)


def _band_mesh(inside, Z, y_w):
    """Sweep the band in one piece along rays from (0, y_w), from one horn tip round the front to the other: each ray
    takes the first stretch of band it meets, found at 0.05 mm on the exact outline. Bottom on the bed."""
    rr = np.arange(0.0, 110.0, 0.05)
    def hit(phi):
        m = inside(rr * np.sin(phi), y_w + rr * np.cos(phi))
        if not m.any(): return None
        i0 = int(np.argmax(m)); i1 = i0 + int(np.argmin(m[i0:])) - 1 if not m[i0:].all() else len(m) - 1
        return rr[i0], rr[i1]
    lo_phi = next(ph for ph in np.radians(np.arange(0.5, 180, 0.1)) if hit(ph) is not None)
    phis = np.linspace(lo_phi, 2 * np.pi - lo_phi, 2001)
    hr = [hit(ph) for ph in phis]; ok = np.array([h is not None for h in hr])
    phis = phis[ok]; rin = np.array([h[0] for h in hr if h]); rout = np.array([h[1] for h in hr if h])
    rout = np.maximum(rout, rin + 0.6)
    n_u, n_v = len(phis), 13
    V = np.linspace(0, 1, n_v)[:, None]; R = rin[None, :] + V * (rout - rin)[None, :]
    x = R * np.sin(phis)[None, :]; y = y_w + R * np.cos(phis)[None, :]
    top = Z(np.c_[y.ravel(), np.abs(x).ravel()]).reshape(x.shape)
    return M._closed(np.c_[x.ravel(), y.ravel(), top.ravel()], np.c_[x.ravel(), y.ravel(), np.zeros(x.size)], n_v, n_u)


def main(out):
    parts, (xc, z0) = load(); p0 = parts[0]
    C = {j: ball_centre(parts[j]) for j in (1, 2, 3)}
    ranges = {j: free_range(parts[j - 1], parts[j], C[j]) for j in (1, 2, 3)}
    for j in (1, 2, 3):
        print(f"joint {j-1}-{j}: ball y {C[j][1]:.2f} z {C[j][2]:.2f}; free range " +
              ", ".join(f"{n}{'+' if s > 0 else '-'} {a:g}" for (n, s), a in ranges[j].items()))
    xs = np.arange(-45, 45 + STEP, STEP); ys = np.arange(5, 100 + STEP, STEP)
    top0, bot0 = head_surface(p0, xs, ys)
    present = np.isfinite(top0); W0 = np.array([np.abs(xs[r]).max() if r.any() else 0.0 for r in present])
    k = (W0.max() - MARGIN) / HC.RO; y_w = float(ys[np.argmax(W0)])          # as wide as the head, widest lines together
    print(f"crescent scale {k:.3f} (half-width {HC.RO * k:.1f} mm), widest line y = {y_w:.2f}")
    cache = os.path.join(out, "isopod_poses_cache.npz")
    if os.path.exists(cache): TT = [tuple(t) for t in np.load(cache)["TT"]]
    else:
        TT = chain_poses(parts, C, ranges); os.makedirs(out, exist_ok=True); np.savez(cache, TT=np.array(TT))
    print(f"{len(TT)} poses the original allows")
    skin = build_skin(top0, bot0, xs, ys, k, y_w)
    band = build_band(parts, curl_poses(C, ranges), top0, xs, ys, y_w)
    layer = M.union(skin, band)
    A0 = man(p0); new0 = A0 + man(layer)
    lost = (A0 - new0).volume()
    new0_mesh = M.from_manifold(new0)
    n_bodies = len(new0.decompose())
    print(f"new head piece: {n_bodies} body, original lost {lost:.4f} mm3 (must be 0), added {new0.volume() - A0.volume():.0f} mm3, "
          f"symmetric {M.is_symmetric(new0_mesh, tol=0.1)}")
    assert n_bodies == 1 and lost < 1e-3

    # the check: the new head piece meets segments 1-3 no more than the original, at every pose of the grid
    B1, B2, B3 = man(parts[1]), man(parts[2]), man(parts[3]); worst = 0.0
    check = curl_poses(C, ranges, n=8) if CHECK else []
    for Ts in check:
        for T, Bk in zip(Ts, (B1, B2, B3)):
            P = Bk.transform(T[:3, :])
            d = (new0 ^ P).volume() - (A0 ^ P).volume(); worst = max(worst, d)
    print(f"clearance check over {len(check)} poses x 3 segments: extra overlap vs the original, worst {worst:.4f} mm3")
    assert worst < 0.05
    # the side swing the horns now allow at the first joint (was ranges[1] yaw)
    yaw_new = free_range(M.from_manifold(new0), parts[1], C[1])[("yaw", 1)]
    print(f"side swing at joint 0-1: {ranges[1][('yaw', 1)]:g} deg in the original, {yaw_new:g} deg with the horns")

    allp = [new0_mesh] + parts[1:]
    whole = trimesh.util.concatenate(allp); whole.apply_translation((xc, 0, z0))      # back to the asset's frame
    os.makedirs(out, exist_ok=True); path = os.path.join(out, "isopod_crescent_head.stl"); whole.export(path)
    json.dump(dict(scale=k, widest_y=y_w, gap=GAP, margin=MARGIN, ranges={f"{j}:{n}{s:+d}": a for j in ranges for (n, s), a in ranges[j].items()},
                   poses=len(TT), checked=len(check), worst_extra_overlap_mm3=worst), open(os.path.join(out, "isopod_crescent_head.json"), "w"), indent=1)
    print(path)
    return whole


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")


# ---------------------------------------------------------------- the head piece for any face (28 Sep 2026)
SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


def _solidify_underside(p0, step=0.5):
    """Fill the giant-isopod head piece's naturally hollow underside flat to the bed, so the printed head has no
    ventral cavities (the pockets under the glabella and eyes). Raycast its top surface over a grid, fill each
    footprint column bed->top, clip to the (convex) footprint so there is no off-head brim. The source carapace is
    watertight, so this is a clean one-time boolean; the result is cached on disk (source/head_piece_solid.stl)."""
    import scipy.spatial, shapely.geometry
    (x0, y0, _), (x1, y1, z1) = p0.bounds
    FX, FY = np.meshgrid(np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step))
    o = np.c_[FX.ravel(), FY.ravel(), np.full(FX.size, z1 + 10)]
    locs, ri, _ = p0.ray.intersects_location(o, np.tile((0, 0, -1.0), (len(o), 1)), multiple_hits=True)
    top = np.zeros(FX.size); np.maximum.at(top, ri, np.asarray(locs)[:, 2]); top = np.clip(top, 0.3, None)
    fill = M._closed(np.c_[FX.ravel(), FY.ravel(), top], np.c_[FX.ravel(), FY.ravel(), np.zeros(FX.size)], FX.shape[0], FX.shape[1])
    hull = scipy.spatial.ConvexHull(p0.vertices[:, :2]); poly = shapely.geometry.Polygon(p0.vertices[hull.vertices][:, :2])
    col = trimesh.creation.extrude_polygon(poly, 400.0); col.apply_translation((0, 0, -200.0))
    fill = M.from_manifold(M.to_manifold(fill) ^ M.to_manifold(col))
    return max(M.from_manifold(M.to_manifold(p0) + M.to_manifold(fill)).split(only_watertight=False), key=lambda b: b.volume)


def head_piece(face=None, fit="face"):
    """The approved head piece with the crescent's face set by `face` (head_crescent.FACE keys: eye size, place and
    height, glabella inflation and rise, ...): the isopod's head piece + the rim-level band (cached in
    source/crescent_band.stl, it does not depend on the face) + the detail skin for this face. Work frame (midline
    x = 0, bed z = 0), isopod size. Also returns where the face's eye sits: (x, y, radius) on the right side.
    fit: where the crescent's face sits along the head. "face" (default): nose to notch, as the face and eye tests
    placed it (scripts/isopod_face.py, isopod_eyes.py), so the eyes sit on the cheeks; "widest": the widest lines
    together, as the first approved head (27 Sep 2026: face=None with fit="widest" is that head exactly)."""
    parts, _ = load(); p0 = parts[0]
    solid_f = os.path.join(SOURCE, "head_piece_solid.stl")                  # solid-underside head piece: no ventral cavities (cached, like the band)
    if os.path.exists(solid_f): p0 = trimesh.load(solid_f)
    else: p0 = _solidify_underside(p0); p0.export(solid_f)
    xs = np.arange(-45, 45 + STEP, STEP); ys = np.arange(5, 100 + STEP, STEP)
    top0, bot0 = head_surface(p0, xs, ys)
    present = np.isfinite(top0); W0 = np.array([np.abs(xs[r]).max() if r.any() else 0.0 for r in present])
    k = (W0.max() - MARGIN) / HC.RO; y_w = float(ys[np.argmax(W0)])
    band_f = os.path.join(SOURCE, "crescent_band.stl")
    if os.path.exists(band_f): band = trimesh.load(band_f)
    else:
        info = json.load(open(os.path.join(SOURCE, "isopod_crescent_head.json")))
        C = {j: ball_centre(parts[j]) for j in (1, 2, 3)}
        ranges = {j: {("pitch", -1): info["ranges"][f"{j}:pitch-1"]} for j in (1, 2, 3)}
        band = build_band(parts, curl_poses(C, ranges), top0, xs, ys, y_w); band.export(band_f)
    y_f = y_w if fit == "widest" else float(ys[present.any(1)].min()) + k * HC.RO
    skin = build_skin(top0, bot0, xs, ys, k, y_f, face)
    head = M.from_manifold(man(p0) + man(M.union(skin, band)))
    E = HC._eye(face)
    return head, dict(k=k, y_w=y_w, y_face=y_f, eye=(k * E["xe"], y_f + k * (E["ye"] - HC.C), k * E["eR"]))
