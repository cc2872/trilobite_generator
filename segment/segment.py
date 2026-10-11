"""
segment.py -- ONE thoracic segment of a trilobite, as a printable solid. Nothing else: no chain, no joints, no copying.

    python segment.py                              -> out/segment.stl + out/segment.png (+ segment.json: the numbers used)
    python segment.py --width 70 --pitch 12 --tip 25 --sweep 35 --spine-w 0.5 --out out/spiny
    python segment.py --copies 5 --taper 0.95      -> five copies one pitch apart, each 5 % narrower, half-rings tucked under
    python segment.py --list                       -> every parameter with its meaning and default

The piece is PARTIALLY FILLED: solid from the bed up to `fill` mm, and above that a shell `wall` thick (measured
along the surface normal, so slopes are not thinner). The hollow is one pocket under the ring and the inner pleurae,
its ceiling drafted at `draft` deg from the pocket's rim so the rim prints without support; the pocket stays open at
the rear under the ring (the mouth the next segment's half-ring tucks into). fill 0 = a plain shell on its doublure;
wall 0 = solid. Every build reports the bed contact, the widest unsupported span of the pocket ceiling (a bridge:
FDM needs support under it past ~30 mm, resin does not care), the overhang, and whether copies intersect.

Frame: x across the animal (0 on the axis, mirror-symmetric), y along it (0 = the front of the pleurae, + rearward),
z up from the print bed. Millimetres. Everything is a height field z(x, y) over a plan outline, closed down to an
underside b(x, y) (the bed, or the pocket ceiling) to make a solid.

What a segment has (Whittington's terms), each with its own number:
  axial ring        the convex median lobe, arched across and along; `axis` is its half-width / the body half-width
  half-ring         the articulating half-ring: a lower lobe in FRONT of the ring (half-elliptical in plan and across)
                    that tucks under the ring ahead with `clearance` to spare; the ring's front face rises from it over `step`
  axial furrows     the grooves either side of the ring
  pleurae           inner run (horizontal to the fulcrum), outer blade (slopes down and sweeps back), both with the
                    pleural furrow running along them from the axis toward the tip
  facet             the articulating facet: the anterolateral corner of the blade bevelled down and cut back, so the
                    pleura ahead can slide over it when the animal enrols
  tip               rounded (tip = 0) or drawn into a spine of length `tip`
"""
import argparse, json, math, os, sys
import numpy as np
import trimesh

PARAMS = [  # (name, default, unit, meaning)
    ("width",      60.0, "mm",   "half-width of the segment at the pleural edge (spine not included)"),
    ("pitch",      10.0, "mm",   "length of the segment along the axis, front of the ring to its rear"),
    ("axis",       0.30, "frac", "axial ring half-width / width"),
    ("relief",     14.0, "mm",   "height of the ring's crest over the bed"),
    ("ring_arch",  0.25, "frac", "how much the ring arches along its length (0 flat, 1 a full barrel)"),
    ("halfring",   0.45, "frac", "articulating half-ring length / pitch (0 none)"),
    ("halfring_drop", 0.30, "frac", "how far the half-ring sits below the ring crest / relief (it is also kept `clearance` under the ring ahead)"),
    ("ax_furrow",  1.2,  "mm",   "axial furrow depth"),
    ("ax_furrow_w", 1.6, "mm",   "axial furrow width (sigma)"),
    ("pleura_h",   0.72, "frac", "pleural height at the axial furrow / relief"),
    ("fulcrum",    0.45, "frac", "where the pleura bends down, / (width - axis half-width)"),
    ("slope",      0.70, "frac", "drop from the fulcrum to the pleural edge / pleural height"),
    ("sweep",      18.0, "deg",  "the outer blade sweeps back by this angle"),
    ("curve",       8.0, "deg",  "extra backward bend over the blade's length"),
    ("pl_furrow",  0.9,  "mm",   "pleural furrow depth"),
    ("pl_furrow_w", 1.4, "mm",   "pleural furrow width (sigma)"),
    ("pl_furrow_pos", 0.42, "frac", "where the pleural furrow runs across the pleura, 0 front edge .. 1 rear edge"),
    ("facet",      0.35, "frac", "articulating facet: how far in from the tip it reaches / blade length (0 none)"),
    ("facet_cut",  0.35, "frac", "how far back the facet cuts the front edge at the tip / pitch"),
    ("facet_drop", 0.5,  "frac", "how far the facet bevels down at the tip / local pleural height"),
    ("tip",        0.0,  "mm",   "pleural spine length past the edge (0 = rounded tip)"),
    ("spine_w",    0.6,  "frac", "spine width at its root / pitch"),
    ("spine_h",    0.5,  "frac", "spine height at its root / pleural edge height"),
    ("edge_h",     3.0,  "mm",   "height of the pleural edge over the bed (the rim the piece stands on)"),
    ("step",       1.2,  "mm",   "the ring's front face: the rise from the half-ring to the ring happens over this length"),
    ("wall",       2.0,  "mm",   "shell thickness, normal to the surface (0 = solid to the bed)"),
    ("fill",       8.5,  "mm",   "solid from the bed up to this height; hollow only above it (0 = plain shell). Past pleura_h*relief - wall the pleurae are solid and only the ring is hollow"),
    ("draft",      45.0, "deg",  "the pocket ceiling rises from the fill height at this angle off its rim (printable without support)"),
    ("doublure",   4.0,  "mm",   "the margin turned down to the bed, this wide, along the pleural edge (0 none)"),
    ("copies",     1,    "n",    "how many copies, placed one pitch (+ gap) apart along the axis"),
    ("taper",      0.96, "frac", "each copy's width x this (1 = identical copies)"),
    ("gap",        0.6,  "mm",   "axial clearance between copies at the pleurae"),
    ("clearance",  0.6,  "mm",   "least vertical clearance wanted between a copy's half-ring and the ring ahead"),
    ("grid",       161,  "n",    "grid columns across (odd); rows along scale with the pitch"),
]
DEFAULTS = {n: d for n, d, _, _ in PARAMS}


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)


def plan(p):
    """outline(u, v) -> (x, y) for u in [-1, 1] across and v in [0, 1] front->rear, and the scalars the surface needs."""
    w, L, a = p["width"], p["pitch"], p["axis"] * p["width"]
    f = min(max(p["fulcrum"], 0.05), 0.95); sw, cv = math.radians(p["sweep"]), math.radians(p["curve"])
    tip = max(p["tip"], 0.0); X_tip = w + tip * math.cos(sw + cv); tw = p["spine_w"] * L
    hr = p["halfring"] * L

    def q_of(ax): return np.clip((ax - a) / max(w - a, 1e-6), 0, 1)           # 0 at the axial furrow, 1 at the pleural edge
    def back(ax):                                                           # how far back the pleura has swept at |x|
        q = q_of(ax); qb = np.clip((q - f) / (1 - f), 0, 1)                   # the inner run is straight; the blade sweeps
        return np.tan(sw) * qb * (w - a) * (1 - f) + np.tan(cv) * qb * qb * (w - a) * (1 - f) * 0.5
    def edges(ax):
        ax = np.asarray(ax, float); b = back(ax); q = q_of(ax)
        yf = b.copy(); yr = L + b
        # the articulating half-ring: a half-ellipse in plan ahead of the ring, hr long on the axis, confined to the
        # axial lobe (half-width a - 0.5) so nothing of it sits under the ring ahead's axial furrow
        yf = yf - hr * np.sqrt(np.clip(1 - (ax / max(a - 0.5, 1.0)) ** 2, 0, 1))
        # the articulating facet: the front edge cut back toward the tip over the outer `facet` of the blade
        if p["facet"] > 0.01:
            fq = np.clip((q - (1 - p["facet"])) / max(p["facet"], 1e-6), 0, 1)
            yf = yf + p["facet_cut"] * L * fq * fq
        if tip > 0.05:                                                      # the spine: a tapering chord past the edge
            pp = np.clip((ax - w) / max(X_tip - w, 1e-6), 0, 1); bt = np.tan(sw + cv) * np.maximum(ax - w, 0)
            yc = 0.5 * (yf + yr) + bt; half = 0.5 * tw * (1 - pp) ** 0.8 + 0.4 * pp
            yf = np.where(ax > w, yc - half, yf); yr = np.where(ax > w, yc + half, yr)
        else:                                                               # a rounded tip: the last 12 % of the blade closes in a half-ellipse
            r = np.clip((q - 0.88) / 0.12, 0, 1); shrink = 0.5 * (L) * (1 - np.sqrt(np.clip(1 - r * r, 0, 1)))
            yf = yf + shrink; yr = yr - shrink
        return yf, np.maximum(yr, yf + 0.3)

    def outline(u, v):
        x = u * X_tip; yf, yr = edges(np.array([abs(x)]))
        return x, float(yf[0] + v * (yr[0] - yf[0]))
    return dict(outline=outline, edges=edges, back=back, q_of=q_of, w=w, L=L, a=a, f=f, X_tip=X_tip, hr=hr, tip=tip)


def surface(p, S):
    """z(x, y): the segment's top."""
    w, L, a, f, hr = S["w"], S["L"], S["a"], S["f"], S["hr"]; h = p["relief"]; hp = p["pleura_h"] * h

    def rear(ax):
        """the top at the ring's rear (y = L), ring + pleura + axial furrow only"""
        q = S["q_of"](ax); across = np.sqrt(np.clip(1 - (ax / max(a, 1e-6)) ** 2, 0, 1))
        ring = hp + (h - hp) * across * max(1 - p["ring_arch"], 0.0)
        qb = np.clip((q - f) / (1 - f), 0, 1); pleura = np.maximum(hp * (1 - p["slope"] * qb ** 1.3), p["edge_h"])
        inring = smoothstep(a + 1.2, a - 1.2, ax)
        return inring * ring + (1 - inring) * pleura - p["ax_furrow"] * np.exp(-((ax - a) / p["ax_furrow_w"]) ** 2)

    def ceiling(ax):
        """the underside of the ring's rear: `wall` below `rear`, measured along the normal (slope factor capped at 3)"""
        d = 0.2; g = (rear(ax + d) - rear(np.maximum(ax - d, 0))) / (2 * d)
        return rear(ax) - p["wall"] * np.minimum(np.sqrt(1 + g * g), 3.0)

    def z(x, y):
        ax = np.abs(np.asarray(x, float)); y = np.asarray(y, float); q = S["q_of"](ax); b = S["back"](ax)
        # ---- axial ring: arched across (an ellipse over the ring's width) and along (ring_arch)
        across = np.sqrt(np.clip(1 - (ax / max(a, 1e-6)) ** 2, 0, 1))
        along = 1 - p["ring_arch"] * ((y - 0.5 * L) / (0.5 * L)) ** 2
        ring = hp + (h - hp) * across * np.clip(along, 0, None)
        # ---- pleura: the inner run level, the blade dropping from the fulcrum, the edge at edge_h
        qb = np.clip((q - f) / (1 - f), 0, 1)
        pleura = hp * (1 - p["slope"] * qb ** 1.3)
        pleura = np.maximum(pleura, p["edge_h"])
        inring = smoothstep(a + 1.2, a - 1.2, ax)                            # 1 on the ring, 0 on the pleura (blend over 2.4 mm)
        zz = inring * ring + (1 - inring) * pleura
        # ---- axial furrows: a groove either side of the ring
        zz = zz - p["ax_furrow"] * np.exp(-((ax - a) / p["ax_furrow_w"]) ** 2)
        # ---- pleural furrow: along the pleura, following its sweep, fading out near the tip
        yc = b + p["pl_furrow_pos"] * L
        onpl = smoothstep(a + 0.5, a + 3.0, ax) * (1 - smoothstep(0.90, 1.0, q))
        zz = zz - p["pl_furrow"] * onpl * np.exp(-((y - yc) / p["pl_furrow_w"]) ** 2)
        # ---- articulating half-ring: a lower, flat lobe ahead of the ring (y < 0 on the axis)
        if hr > 0.05:
            # the lobe: elliptical across (half-width a - 0.5), `halfring_drop` below the crest, and never higher than
            # the ring ahead's underside at its rear minus `clearance` (the ring ahead has this ring's profile)
            E = np.sqrt(np.clip(1 - (ax / max(a - 0.5, 1.0)) ** 2, 0, 1))
            hz = (h - p["halfring_drop"] * h) * E
            if p["wall"] > 0.01:
                hz = np.minimum(hz, ceiling(ax) - p["clearance"])
            lobe = np.minimum(zz, hz)
            sy = smoothstep(0.0, max(p["step"], 0.1), y)                     # the ring's front face: a ramp over `step` mm
            zz = np.where(ax < a, lobe * (1 - sy) + zz * sy, zz)
        # ---- articulating facet: bevel down toward the front corner of the blade
        if p["facet"] > 0.01:
            fq = np.clip((q - (1 - p["facet"])) / max(p["facet"], 1e-6), 0, 1)
            yf0 = b + (p["facet_cut"] * L * fq * fq); fy = np.clip(1 - (y - yf0) / max(0.45 * L, 1e-6), 0, 1)
            zz = zz - p["facet_drop"] * zz * fq * fy * (ax <= w)
        # ---- the spine: height tapering from spine_h x edge height to a point
        if S["tip"] > 0.05:
            pp = np.clip((ax - w) / max(S["X_tip"] - w, 1e-6), 0, 1)
            zz = np.where(ax > w, np.maximum(p["spine_h"] * p["edge_h"] * (1 - 0.6 * pp), 1.4), zz)
        return np.maximum(zz, 1.0)
    return z


def underside(p, S, xs, ys, zs):
    """The underside b(x, y). wall 0: the bed everywhere (solid). Otherwise the shell ceiling is `wall` below the top
    measured along the normal (vertical offset wall * sqrt(1 + |grad z|^2), capped), never thinner than 0.8 mm; the
    doublure strip is solid to the bed; then the fill: wherever that ceiling is below `fill` the piece is solid to the
    bed, and the remaining pocket's ceiling is drafted at `draft` deg up from the fill height off the pocket rim. The
    mouth -- the rear `hr` + 1 mm of the ring, |x| < a + 0.5 -- keeps the plain shell ceiling so the next segment's
    half-ring tucks under it."""
    if p["wall"] <= 0.01: return np.zeros_like(zs)
    nv, nu = zs.shape
    dx = float(np.abs(np.diff(xs, axis=1)).mean()); dy = float(np.abs(np.diff(ys, axis=0)).mean())
    gy, gx = np.gradient(zs, dy, dx)
    k = np.minimum(np.sqrt(1 + gx ** 2 + gy ** 2), 3.0)                    # normal offset -> vertical offset, capped at 3x
    bot = np.maximum(zs - p["wall"] * k, 0.0)
    ax = np.abs(xs); on_doublure = (S["w"] - ax) < p["doublure"]
    bot = np.where(on_doublure, 0.0, bot)
    bot = np.minimum(bot, np.maximum(zs - 0.8, 0.0))                       # never thinner than 0.8 mm
    if p["fill"] > 0.01:
        mouth = (ys > S["L"] - S["hr"] - 1.0) & (ax < S["a"] + 0.5) if S["hr"] > 0.05 else np.zeros_like(bot, bool)
        pocket = (bot > p["fill"]) | mouth
        from scipy import ndimage
        d = ndimage.distance_transform_edt(pocket, sampling=(dy, dx))      # mm to the nearest solid cell
        roof = p["fill"] + math.tan(math.radians(min(max(p["draft"], 5.0), 89.0))) * d
        bot = np.where(pocket, np.where(mouth, bot, np.minimum(bot, roof)), 0.0)
    return bot


def closed_solid(S, zfun, nu=161, nv=None, p=None):
    """Sample the plan grid, close top to the underside, return a watertight trimesh. Symmetric by construction."""
    if nu % 2 == 0: nu += 1
    nv = nv or max(41, int(S["L"] / 0.35) | 1)
    mid = nu // 2; xs = np.zeros((nv, nu)); ys = np.zeros((nv, nu))
    for j in range(nv):
        v = j / (nv - 1)
        for i in range(mid, nu):
            u = -1 + 2 * i / (nu - 1); x, y = S["outline"](u, v); xs[j, i] = x; ys[j, i] = y
    xs[:, :mid] = -xs[:, nu - 1:mid:-1]; ys[:, :mid] = ys[:, nu - 1:mid:-1]; xs[:, mid] = 0.0
    zs = zfun(np.abs(xs), ys)                                               # no blur: the grid is not uniform in y, a
    bs = underside(p or DEFAULTS, S, xs, ys, zs)                            # stencil blur smears the ring into the half-ring
    T = np.c_[xs.ravel(), ys.ravel(), zs.ravel()]; B = np.c_[xs.ravel(), ys.ravel(), bs.ravel()]
    n = xs.size; V = np.vstack([T, B]); F = []
    idx = lambda j, i: j * nu + i
    for j in range(nv - 1):
        for i in range(nu - 1):
            A_, B_, C_, D_ = idx(j, i), idx(j, i + 1), idx(j + 1, i), idx(j + 1, i + 1)
            F += [[A_, B_, D_], [A_, D_, C_], [n + A_, n + D_, n + B_], [n + A_, n + C_, n + D_]]
    # the side strips round the boundary (the outline), top edge to bottom edge
    loop = [idx(0, i) for i in range(nu)] + [idx(j, nu - 1) for j in range(1, nv)] + [idx(nv - 1, i) for i in range(nu - 2, -1, -1)] + [idx(j, 0) for j in range(nv - 2, 0, -1)]
    for k in range(len(loop)):
        a_, b_ = loop[k], loop[(k + 1) % len(loop)]
        F += [[a_, n + a_, b_], [b_, n + a_, n + b_]]
    m = trimesh.Trimesh(V, np.array(F), process=True)
    m.merge_vertices(); m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    m.metadata["grid"] = dict(xs=xs, ys=ys, zs=zs, bs=bs)
    return m


def stand(m):
    """From the build grid: the bed contact as a fraction of the footprint, and the widest run of hollow across any
    row (the longest bridge the pocket ceiling has to make, mm)."""
    g = m.metadata.get("grid")
    if g is None: return None, None
    hollow = g["bs"] > 0.05; contact = 1 - hollow.mean(); span = 0.0
    for j in range(hollow.shape[0]):
        row = hollow[j]; i = 0
        while i < len(row):
            if row[i]:
                k = i
                while k < len(row) and row[k]: k += 1
                span = max(span, abs(g["xs"][j, k - 1] - g["xs"][j, i])); i = k
            else: i += 1
    return float(contact), float(span)


def thin(m, wall):
    """Share of the top surface where the shell, measured normal to the surface, is under 0.6 * wall (from the grid)."""
    g = m.metadata.get("grid")
    if g is None or wall <= 0.01: return 0.0
    xs, ys, zs, bs = g["xs"], g["ys"], g["zs"], g["bs"]
    dx = float(np.abs(np.diff(xs, axis=1)).mean()); dy = float(np.abs(np.diff(ys, axis=0)).mean())
    gy, gx = np.gradient(zs, dy, dx); t = (zs - bs) / np.sqrt(1 + gx ** 2 + gy ** 2)
    return float(((t < 0.6 * wall) & (bs > 0.05)).mean())


def overhang(m):
    """Fraction of the surface that is a downward face steeper than 45 deg and off the bed: what needs support."""
    n = m.face_normals; a = m.area_faces; zc = m.triangles_center[:, 2]
    bad = (n[:, 2] < -0.707) & (zc > 0.6); return float(a[bad].sum() / a.sum())


def build_one(p):
    S = plan(p); z = surface(p, S); m = closed_solid(S, z, nu=int(p["grid"]), p=p)
    return m, S, z


def halfring_clearance(p, S, z, S2, z2, dy):
    """The next copy's half-ring (plan S2, surface z2, placed dy behind) under this copy's ring: least (my underside -
    its top) over the overlap, on the axis. Negative = they intersect."""
    if S2["hr"] <= 0.05: return None
    ys = np.linspace(S["L"] - S2["hr"] + 0.2, S["L"] - 0.2, 40); xs = np.linspace(0, S2["a"] + 2.5, 24)
    X, Y = np.meshgrid(xs, ys); under = z(X, Y) - (p["wall"] if p["wall"] > 0.01 else z(X, Y))   # solid: no room at all
    top2 = z2(X, Y - dy); yf2, _ = S2["edges"](X.ravel()); there = (Y - dy) >= yf2.reshape(X.shape)
    yr1 = S["edges"](X.ravel())[1].reshape(X.shape); there &= Y <= yr1                      # and under this copy's material
    if not there.any(): return None
    return float((under - top2)[there].min())


def build(p):
    """The segment, or `copies` of it one pitch (+ gap) apart, each `taper` narrower. Returns [(mesh, plan, zfun, dy)],
    plus notes: the overhang fraction and the half-ring clearance between neighbours."""
    out = []; notes = []; y = 0.0; q = dict(p)
    for k in range(max(int(p["copies"]), 1)):
        m, S, z = build_one(q); m.apply_translation((0, y, 0)); out.append((m, S, z, y)); y += q["pitch"] + p["gap"]; q = dict(q, width=q["width"] * p["taper"])
    m0 = out[0][0]; contact, span = stand(m0)
    if contact is not None:
        g = m0.metadata["grid"]; hx = float(np.abs(g["xs"][g["bs"] > 0.05]).max()) if (g["bs"] > 0.05).any() else 0.0
        notes.append(f"stands on {contact * 100:.0f} % of its footprint; hollow out to |x| = {hx:.0f} mm ({'the ring only' if hx < out[0][1]['a'] + 3 else 'ring and inner pleurae'}); widest bridge in the pocket ceiling {span:.0f} mm")
    notes.append(f"overhang (needs support): {overhang(m0) * 100:.1f} % of the surface; shell thinner than {0.6 * p['wall']:.1f} mm over {thin(m0, p['wall']) * 100:.1f} % of the top")
    cs = [halfring_clearance(p, S1, z1, S2, z2, y2 - y1) for (m1, S1, z1, y1), (m2, S2, z2, y2) in zip(out, out[1:])]
    cs = [c for c in cs if c is not None]
    if cs: notes.append(f"half-ring under the ring ahead: {min(cs):+.2f} mm least clearance over {len(cs)} joint(s)" + ("  <-- below the clearance wanted" if min(cs) < p["clearance"] else ""))
    if len(out) > 1:                                                        # the authoritative collision check: a boolean
        vol = 0.0
        for (m1, *_), (m2, *_) in zip(out, out[1:]):
            try:
                I = trimesh.boolean.intersection([m1, m2], engine="manifold"); vol += abs(I.volume) if len(I.faces) else 0.0
            except BaseException as e: notes.append(f"intersection check failed: {e}"); break
        notes.append(f"copies intersect: {vol:.2f} mm3" + ("  <-- they touch; raise gap, clearance or halfring_drop" if vol > 0.01 else " (none)"))
    return out, notes


def sheet(pieces, p, path, notes=()):
    """plan + side + a 3/4 render on one black sheet (matplotlib; no GPU). pieces: [(mesh, plan, zfun, dy)]."""
    m, S, z, _ = pieces[0]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(16, 9), facecolor="k")
    ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 3); ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    for a in (ax1, ax2): a.set_facecolor("k"); a.tick_params(colors="#777", labelsize=7); [s.set_color("#333") for s in a.spines.values()]
    for (mk, Sk, zk, dy) in pieces:
        xs = np.linspace(0, Sk["X_tip"], 200); yf, yr = Sk["edges"](xs); yf = yf + dy; yr = yr + dy
        for sgn in (1, -1):
            ax1.plot(sgn * xs, -yf, "w", lw=0.9); ax1.plot(sgn * xs, -yr, "w", lw=0.9)
            ax1.plot([sgn * Sk["a"]] * 2, [-yf[0], -yr[0]], color="#777", lw=0.5, ls=":")
        ax1.plot([xs[-1], xs[-1]], [-yf[-1], -yr[-1]], "w", lw=0.9); ax1.plot([-xs[-1], -xs[-1]], [-yf[-1], -yr[-1]], "w", lw=0.9)
    xs = np.linspace(0, S["X_tip"], 200); yf, yr = S["edges"](xs)
    ax1.set_aspect("equal"); ax1.set_title("plan (front up) · dotted: axial furrows" + (f" · {len(pieces)} copies" if len(pieces) > 1 else ""), color="w", fontsize=9)
    for (mk, Sk, zk, dy) in pieces:
        xs_ = np.linspace(0, Sk["X_tip"], 200); yf_, yr_ = Sk["edges"](xs_); yy = np.linspace(float(yf_.min()) - 0.5, float(yr_.max()) + 0.5, 300)
        for xq, lab, c in ((0.0, "axis", "w"), (0.5 * (Sk["a"] + Sk["w"]), "mid-pleura", "#999"), (Sk["w"] * 0.97, "near the edge", "#666")):
            f_, r_ = Sk["edges"](np.array([xq])); sel = (yy >= f_[0]) & (yy <= r_[0]); top = zk(np.full(sel.sum(), xq), yy[sel])
            ax2.plot(yy[sel] + dy, top, color=c, lw=0.9, label=lab if dy == 0 else None)
            g = mk.metadata.get("grid")
            if p["wall"] > 0.01 and g is not None:
                col = np.abs(g["xs"][0] - xq).argmin(); ax2.plot(g["ys"][:, col] + dy, g["bs"][:, col], color=c, lw=0.5, ls="--")
    ax2.set_aspect("equal"); ax2.set_ylim(0, p["relief"] * 1.3); ax2.legend(fontsize=7, facecolor="k", labelcolor="w", edgecolor="#333"); ax2.set_title("side (sections along y at three x; dashed = the underside: bed, fill, pocket ceiling)", color="w", fontsize=9)
    ax3.set_facecolor("k"); ax3.set_axis_off()
    allm = trimesh.util.concatenate([pc_[0] for pc_ in pieces])
    tri = allm.vertices[allm.faces]; n = allm.face_normals; shade = 0.35 + 0.65 * np.clip(n @ np.array([0.3, -0.5, 0.8]), 0, 1)
    pc = Poly3DCollection(tri, facecolors=np.c_[shade, shade, shade * 0.97, np.ones_like(shade)], edgecolors="none"); ax3.add_collection3d(pc)
    b = allm.bounds; c = b.mean(0); e = b[1] - b[0]; r = e.max() * 0.5
    ax3.set_xlim(c[0] - r, c[0] + r); ax3.set_ylim(c[1] - r, c[1] + r); ax3.set_zlim(0, 2 * r); ax3.view_init(elev=38, azim=-62); ax3.set_box_aspect((1, 1, 1), zoom=1.9)
    ax3.set_title(f"{allm.extents[0]:.0f} × {allm.extents[1]:.0f} × {allm.extents[2]:.0f} mm · {len(allm.faces):,} faces · watertight {all(pc_[0].is_watertight for pc_ in pieces)}\n" + " · ".join(notes), color="w", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=85, facecolor="k"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for n, d, u, doc in PARAMS: ap.add_argument("--" + n.replace("_", "-"), type=float, default=None, help=f"{doc} [{d} {u}]")
    ap.add_argument("--out", default="out/segment", help="output stem: <stem>.stl, <stem>.png, <stem>.json")
    ap.add_argument("--params", default=None, help="a JSON file of parameters (the command line overrides it)")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--no-sheet", action="store_true")
    A = ap.parse_args()
    if A.list:
        for n, d, u, doc in PARAMS: print(f"{n:14s} {d:>7} {u:5s} {doc}")
        return
    p = dict(DEFAULTS)
    if A.params: p.update(json.load(open(A.params)))
    for n, _, _, _ in PARAMS:
        v = getattr(A, n); p[n] = v if v is not None else p[n]
    pieces, notes = build(p)
    os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
    whole = trimesh.util.concatenate([pc[0] for pc in pieces]); whole.export(A.out + ".stl"); json.dump(p, open(A.out + ".json", "w"), indent=1)
    if len(pieces) > 1:
        for k, (m, S, z, dy) in enumerate(pieces): m.export(f"{A.out}_{k:02d}.stl")
    m = pieces[0][0]
    print(f"{A.out}.stl  {whole.extents[0]:.1f} x {whole.extents[1]:.1f} x {whole.extents[2]:.1f} mm  {len(pieces)} piece(s)  one copy: volume {m.volume / 1000:.1f} cm3  faces {len(m.faces):,}  watertight {m.is_watertight}  one body {len(m.split(only_watertight=False)) == 1}")
    for n_ in notes: print("  " + n_)
    if not A.no_sheet: sheet(pieces, p, A.out + ".png", notes); print(f"{A.out}.png")


if __name__ == "__main__":
    main()
