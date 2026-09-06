"""
trilobite.py v4 — anatomically sculpted, physically enrollable, parametric trilobite.

THE IDEA: the dorsal exoskeleton is a FUNCTION z(x, y). Vault + axial lobe + glabella + eyes
+ border are positive terms; every furrow (axial, ring, pleural, glabellar, occipital, border)
is a smooth trough. The surface is sampled on a grid that follows each part's plan outline,
turned into a spline surface and thickened into a shell. Booleans are used only for what a
surface cannot express: hinges, stops, spines, doublure.

Coordinates: X across, Y along the body (head at -Y), Z up. Each part's front hinge (the
head's rear hinge) is at its local y = 0.  Parameters: schema.py.  Fields: fields.py.
"""
import math
import numpy as np
from build123d import *
import schema
from fields import (width_curve, seg_halfwidth, head_halfwidth, tail_halfwidth,
                    pleural_spine_field, furrow_amp, landmarks)

VIEWER_PORT = 3939
P = schema.defaults()

# =====================================================================
#  derived scalars
# =====================================================================
def pitch(P):
    return P["length"] * (1 - P["cephFrac"] - P["pygFrac"]) / P["segCount"]

def ring_top(P):
    return P["relief"] * (1 + P["axisRise"])

def hinge_z(P):
    """Hinge axis height. The clearance groove (radius barrelR + clearance) must sit clear below the
    shell's inner surface at the ring furrow, and the knuckle top must clear the flap above it."""
    return ring_top(P) - P["barrelR"] - P["wall"] - P["clearance"] - 0.4 - 0.7 * furrow_amp(P)

def hinge_width(P):
    """Constant for every joint, sized to the narrowest axial ring (last segment)."""
    return 2 * P["axisFrac"] * seg_halfwidth(P, P["segCount"] - 1) - 2

def joint_offsets(P):
    d = pitch(P)
    return [0] + [d] * P["segCount"]

# =====================================================================
#  surface vocabulary (all numpy, all smooth)
# =====================================================================
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def plateau(v, half, edge=0.7):
    """1 inside |v| < half, 0 outside, smooth over `edge`."""
    return 1 - smoothstep(half - edge, half + edge, np.abs(v))

def trough(dist, sigma):
    return np.exp(-(dist / sigma) ** 2)

def vault(u, P):
    """Cross-section height factor vs normalized lateral position u ∈ [-1, 1].
    tent=0: the v4 fulcrum vault.  tent=1: axial Gaussian dome + straight pleural slope, soft-max blended —
    the form fitted from the reference sculpt (5 parameters, 0.12 mm rms on seg2)."""
    u = np.abs(u); f = P["fulcrum"]
    inner = 1 - 0.35 * (u / f) ** 2
    outer = 0.65 * (1 - (np.maximum(u - f, 0) / (1 - f)) ** 1.6)
    v4 = np.where(u <= f, inner, outer)
    k = P["tent"]
    if k <= 0.001: return v4
    m = P["marginHeight"]; k1 = P["pleuralSlope"]
    zb = 1.0 - 0.04                                      # pleural line starts just under the crest
    k2 = zb - k1 - m                                     # quadratic term fixed so the slope lands on the margin at the tip
    def raw(uu):
        dome = np.exp(-(uu / P["axisSigma"]) ** 2)
        pleural = zb - k1 * uu - k2 * uu ** 2
        return np.log(np.exp(4 * dome) + np.exp(4 * pleural)) / 4      # soft max of the two
    t0, t1 = raw(np.float64(0.0)), raw(np.float64(1.0))
    tent = (raw(u) - t1) / max(t0 - t1, 1e-6)             # normalise so vault(0)=1, vault(1)=0 like v4
    return (1 - k) * v4 + k * tent

# =====================================================================
#  plate builder: grid → spline surface → shell solid, plus an "under" envelope
# =====================================================================
def _grid(outline, zfun, nu, nv, dz=0.0, floor=None, passes=2):
    """Sample z on the plan grid, then lightly blur it (a few 3-tap passes) so no feature is sharper
    than the grid can carry — that is what stops the spline from overshooting."""
    xs = np.zeros((nv, nu)); ys = np.zeros((nv, nu))
    for j in range(nv):
        v = j / (nv - 1)
        for i in range(nu):
            u = -1 + 2 * i / (nu - 1)
            x, y = outline(u, v); xs[j, i] = x; ys[j, i] = y
    zs = zfun(xs.ravel(), ys.ravel()).reshape(nv, nu)
    for _ in range(passes):
        zp = np.pad(zs, 1, mode="edge")
        zs = 0.25 * zp[1:-1, :-2] + 0.5 * zp[1:-1, 1:-1] + 0.25 * zp[1:-1, 2:]
        zp = np.pad(zs, 1, mode="edge")
        zs = 0.25 * zp[:-2, 1:-1] + 0.5 * zp[1:-1, 1:-1] + 0.25 * zp[2:, 1:-1]
    zs = np.maximum(zs + dz, 0.05)
    if floor is not None: zs[:] = floor
    return [[(float(xs[j, i]), float(ys[j, i]), float(zs[j, i])) for i in range(nu)] for j in range(nv)]

def _surface(rows):
    """Least-squares B-spline approximation to a small tolerance. (Variational smoothing washes the
    sculpture out; pure interpolation overshoots on sharp features.)"""
    return Face.make_surface_from_array_of_points(rows, tol=0.08, max_deg=3)

def safe_expr(expr, s):
    """Evaluate a user formula in s with numpy math only. Bad input → 's**2' and a BUILD_NOTES entry."""
    ns = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "arctan", "minimum", "maximum", "clip", "where")}
    ns.update(pi=np.pi, e=np.e, s=s)
    try:
        v = eval(compile(str(expr), "<genalPath>", "eval"), {"__builtins__": {}}, ns)
        v = np.broadcast_to(np.asarray(v, float), s.shape).copy()
        if not np.all(np.isfinite(v)): raise ValueError("non-finite")
        return v
    except Exception as ex:
        BUILD_NOTES.append(("head", "genalPath rejected", f"{expr!r}: {str(ex)[:40]}")); return s ** 2

GRID_JITTER = 0          # added to nu/nv by plate()/under() when a build is retried (see parts_list)

def plate(outline, zfun, t, nu=45, nv=27):
    nu += GRID_JITTER; nv += GRID_JITTER
    """Shell of thickness t under the surface z = zfun(x, y), on the plan given by outline(u, v)."""
    top = _surface(_grid(outline, zfun, nu, nv))
    bot = _surface(_grid(outline, zfun, nu, nv, dz=-t))
    sides = []
    for e1 in top.edges():
        m = e1.position_at(0.5)
        e2 = min(bot.edges(), key=lambda e: (e.position_at(0.5) - m).length)
        sides.append(Face.make_surface_from_curves(e1, e2))
    return Solid(Shell([top, bot] + sides))

def under(outline, zfun, nu=45, nv=27):
    nu += GRID_JITTER; nv += GRID_JITTER
    """Solid filling everything under the surface down to z = -1 (envelope for clipping)."""
    top = _surface(_grid(outline, zfun, nu, nv))
    bot = _surface(_grid(outline, zfun, nu, nv, floor=-1.0))
    sides = []
    for e1 in top.edges():
        m = e1.position_at(0.5)
        e2 = min(bot.edges(), key=lambda e: (e.position_at(0.5) - m).length)
        sides.append(Face.make_surface_from_curves(e1, e2))
    return Solid(Shell([top, bot] + sides))

def tubercles(x, y, P, region, seed, count_scale):
    """Seeded granules: sum of small Gaussian bumps inside `region(xk, yk) -> bool`."""
    if P["tubercles"] <= 0.001: return 0.0
    rng = np.random.default_rng(int(P["seed"]) * 1000 + seed)
    n = int(P["tubercles"] * count_scale)
    xmin, xmax, ymin, ymax = region["box"]
    if xmax <= xmin or ymax <= ymin: return 0.0     # degenerate plan (very short pygidium/head at slider extremes): skip, don't crash
    z = np.zeros_like(x, dtype=float); r = P["tubercleSize"]; k = 0; tries = 0
    while k < n and tries < 20 * n + 20:
        tries += 1
        xk = rng.uniform(xmin, xmax); yk = rng.uniform(ymin, ymax)
        if not region["ok"](xk, yk): continue
        z += 0.45 * r * np.exp(-((x - xk) ** 2 + (y - yk) ** 2) / (0.55 * r) ** 2); k += 1
    return z

def spine(base_r, tip_r, length, at, yaw_deg, pitch_deg=0):
    """Tapered spine from `at`, pointing +Y (rear); yaw about Z (+ toward +X), pitch up."""
    s = Cone(base_r, tip_r, length, align=(Align.CENTER, Align.CENTER, Align.MIN)).rotate(Axis.X, -90)
    return s.rotate(Axis.X, pitch_deg).rotate(Axis.Z, -yaw_deg).moved(Location(at))

# =====================================================================
#  hinge (unchanged mechanism: interleaving knuckles, pin bore, stop block, ventral bevel)
# =====================================================================
def add_hinge(part, envelope, P, y_axis, rear):
    rB, c, nK = P["barrelR"], P["clearance"], int(P["nKnuckles"])
    zh, Wh = hinge_z(P), hinge_width(P)
    kw, x0 = Wh / nK, -Wh / 2
    W = P["width"]

    def barrel(xc, length, r):
        return Cylinder(r, length).rotate(Axis.Y, 90).moved(Location((xc, y_axis, zh)))

    part -= barrel(0, Wh + 4, rB + c)
    webs = None
    for i in range(nK):
        if (i % 2 == 0) != rear: continue
        xc = x0 + (i + 0.5) * kw
        part += barrel(xc, kw - c, rB)
        z_lo = zh - rB - 0.5 * c - 1.5
        web = Box(kw - c, rB + c + 1.0, ring_top(P) + 1.0 - z_lo,               # web reaches the shell whatever the top does
                  align=(Align.CENTER, Align.MAX if rear else Align.MIN, Align.MIN)).moved(Location((xc, y_axis, z_lo)))
        webs = web if webs is None else webs + web
    part += webs & envelope
    part -= barrel(0, Wh + 10, P["boreDia"] / 2)
    blk = Box(Wh + 2, 2 * P["wall"], zh - rB - c - 0.5,
              align=(Align.CENTER, Align.MAX if rear else Align.MIN, Align.MIN)).moved(Location((0, y_axis, 0)))
    part += blk
    phi, L = P["maxAngle"] / 2, 80
    if P["bladeChord"] < 1.0:                                   # separate ribs: only the ring and the rib roots overlap
        band = 2 * (P["axisFrac"] * (W / 2)) + 0.12 * W
    else:                                                        # v4 plates tile: bevel the inner pleurae too
        band = 2 * (P["axisFrac"] * (W / 2)) + 0.45 * W
    if rear:
        wedge = Box(band, L, L, align=(Align.CENTER, Align.MIN, Align.MAX)).rotate(Axis.X, -phi)
    else:
        wedge = Box(band, L, L, align=(Align.CENTER, Align.MAX, Align.MAX)).rotate(Axis.X, phi)
    part -= wedge.moved(Location((0, y_axis, zh))) - barrel(0, Wh + 4, rB + c)
    return part

# =====================================================================
#  THORACIC SEGMENT i
# =====================================================================

BUILD_NOTES = []          # (part label, n fragments dropped, total mm^3) — read by the instrument's readout

def prune_slivers(part, min_vol=1.0, frac=0.02, label=""):
    """Drop degenerate zero-volume solids left behind by tangent booleans, and detached fragments smaller than
    `frac` of the main solid (a blade tip severed by the ventral bevel would fall off the print anyway).
    Anything larger is kept so the single-solid test still catches a genuinely unfused spine. Dropped
    fragments are recorded in BUILD_NOTES — weakness-first, never silent."""
    sols = part.solids()
    if len(sols) <= 1: return part
    vmax = max(x.volume for x in sols)
    keep = [x for x in sols if x.volume >= max(min_vol, frac * vmax)]
    dropped = [x for x in sols if x not in keep]
    if dropped:
        BUILD_NOTES.append((label, len(dropped), round(sum(x.volume for x in dropped), 1)))
    return part if len(keep) == len(sols) else Part(children=keep)

def build_segment(P, i=0):
    t, c, h = P["wall"], P["clearance"], P["relief"]
    d = pitch(P); ovl = P["overlap"] * d; flap = max(ovl - 2.0, 1.0)
    w = seg_halfwidth(P, i)
    a = P["axisFrac"] * w
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    sweep = P["tipSweep"] * d

    L0 = d + flap
    Ls = pleural_spine_field(P)[i] * w                                   # spine length: the rib's arc simply continues
    X_tip = w + Ls * math.cos(math.radians(P["spineSweep"]))            # lateral reach of the whole arc
    S_sp = Ls * math.sin(math.radians(P["spineSweep"]))                  # how far back the spine trails
    R_TIP = 0.7                                                          # half-chord at the point (≥ nozzle)
    def q_of(x):
        return np.clip((np.abs(x) - a) / (w - a), 0, 1)
    def p_of(x):
        return np.clip((np.abs(x) - w) / max(X_tip - w, 1e-6), 0, 1) if Ls > 0.5 else np.zeros_like(np.asarray(x, float))
    def edges(x):
        """Front and rear edge of the pleura at lateral position x. One arc: the blade sweeps back and narrows,
        and — if this segment carries a spine — the same arc keeps going, its curvature increasing and its
        chord closing to a point. There is no separate spine object."""
        q = q_of(x); p = p_of(x)
        # spine arc constants measured on Olenoides serratus (8 spines, 6 Sep 2026): sweep ∝ q^1.52 ± 0.39,
        # chord ∝ (1−q)^0.47 ± 0.30 — spines stay wide and close fast at the tip, blunter than a needle
        yc = 0.5 * L0 + sweep * q ** 1.6 + S_sp * p ** 1.5
        root = smoothstep(0.0, 0.3, q)
        half = 0.5 * L0 * (1 - root) + 0.5 * min(P["bladeChord"] * d, L0) * root
        hb = half * (1 - P["tipTaper"] * q ** 2.5)
        hb = hb * (1 - p) ** 0.5 + R_TIP * p                             # spine: chord closes to the point
        return yc - hb, yc + hb
    def outline(u, v):
        x = u * X_tip
        yf, yr = edges(np.array([abs(x)]))
        return x, float(yf[0] + v * (yr[0] - yf[0]))

    def zfun(x, y):
        ax = np.abs(x)
        z = margin + (h - margin) * vault(x / w, P)
        z += rise * plateau(x, a)                                            # axial ring
        z -= F * trough(ax - (a + 0.6), 0.9)                                 # axial furrows
        z -= 0.7 * F * trough(y - d, 0.8) * plateau(x, a + 1.5, 1.0)         # ring furrow at the flap
        yf0, _ = edges(x); px = q_of(x); ly = yf0 + 0.30 * d + px * 0.25 * d
        z -= 0.8 * F * trough(y - ly, 0.9) * (ax > a + 1.0)                  # pleural furrow, following the blade
        z += tubercles(x, y, P, dict(box=(-w + 2, w - 2, ovl + 1, d - 1), ok=lambda xk, yk: abs(xk) > a + 1.5 or abs(xk) < a - 1.5),
                       seed=i, count_scale=0.9 * w)
        # convexity along the body (reference sculpt: ring arch R≈1.9 mm on a 9 mm chord, blade R≈2.5 mm on a
        # 5.3 mm chord at 35 mm scale). Both are additive humps that vanish at the two hinge lines (y=0, y=d),
        # so ring_top, hinge_z and the flap are untouched — the fixed ruler stays fixed.
        # The chord of each hump runs from the end of the stepped front band (which sits under the previous
        # segment's flap) to the rear hinge line, so the hump never rises into the flap above it.
        y0r = ovl - 1.0
        yc_r = 0.5 * (y0r + d); hr = max(0.5 * (d - y0r), 0.5)
        ring_hump = P["ringArch"] * h * np.clip(1 - ((y - yc_r) / hr) ** 2, 0, 1)
        yf_b, yr_b = edges(x); y0b = yf_b + ovl - 1.0
        yc_b = 0.5 * (y0b + yr_b); hb_b = np.maximum(0.5 * (yr_b - y0b), 0.5)
        blade_hump = P["bladeCamber"] * h * np.clip(1 - ((y - yc_b) / hb_b) ** 2, 0, 1)
        ax_w = plateau(x, a + 1.0, 1.5)
        window = smoothstep(ovl - 1.5, ovl + 0.5, y - yf_b)                  # belt and braces: nothing under the flap
        z += window * (ax_w * ring_hump + (1 - ax_w) * blade_hump)
        if Ls > 0.5:                                                          # the spine droops toward its point
            p = p_of(x)
            z = np.where(p > 0, np.maximum(margin * (1 - 0.55 * p), t + 0.6), z)
        yf, _ = edges(x)
        t_prev = t * (P["headWall"] if i == 0 else 1.0)                            # the part shingled over this front
        drop = np.minimum(t_prev + c + 0.7 * F + 0.3, np.maximum(z - (t + 0.6), 0))    # never thinner than the wall
        z -= drop * (1 - smoothstep(ovl - 1.5, ovl, y - yf))                          # stepped front band, following the blade
        return z

    seg = plate(outline, zfun, t, nu=53, nv=29)
    env = under(outline, zfun, nu=53, nv=29)
    # doublure: vertical rim + inward lip along the blade's margin (the blade is swept, so follow it)
    yf_m, yr_m = (float(v[0]) for v in edges(np.array([w])))
    rim_len = (yr_m - yf_m) - 1.0
    if rim_len > 2.0 and P["tipTaper"] < 0.3:                 # doublure lips only on broad, square-tipped pleurae
        for s in (1, -1):
            seg += Box(t, rim_len, max(margin - t, 1.0), align=(Align.MAX if s > 0 else Align.MIN, Align.MIN, Align.MIN)).moved(Location((s * w, yf_m + 0.5, 0)))
            seg += Box(0.06 * P["width"], rim_len, t, align=(Align.MAX if s > 0 else Align.MIN, Align.MIN, Align.MIN)).moved(Location((s * w, yf_m + 0.5, 0)))
    seg = add_hinge(seg, env, P, d, rear=True)
    seg = add_hinge(seg, env, P, 0, rear=False)
    # pleural spines are part of the plate (edges() above): body, not ornament — the bevel and the instrument see them
    r = 0.45 * margin
    if P["axialSpine"] > 0.02:
        seg += spine(0.6 * r + 0.6, 0.5, P["axialSpine"] * h, (0, ovl + 0.45 * (d - ovl), h + rise - 1.0), 0, pitch_deg=60)
    return prune_slivers(seg, label=f"seg{i}")

# =====================================================================
#  CEPHALON
# =====================================================================
_SKIN_CACHE = {}
_SKIN_NAMES = ("olenoides", "gltf", "harpetida", "proetida")           # matches skins/*.npz and the schema's skin<Name> weights

def _load_skin_cached(name):
    if name not in _SKIN_CACHE:
        import os
        import skins
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skins", f"{name}.npz")
        _SKIN_CACHE[name] = skins.load_skin(path)
    return _SKIN_CACHE[name]

def head_skin_zfun(P, Lc, wh):
    """Blend of the registered reference head skins (skins.py), resampled to this head's own
    footprint (mm). Falls back to pure Olenoides if every weight is ~0."""
    import skins
    weights = [P[f"skin{n.capitalize()}"] for n in _SKIN_NAMES]
    if sum(weights) < 1e-6: weights = [1.0, 0.0, 0.0, 0.0]
    blended = skins.blend([_load_skin_cached(n) for n in _SKIN_NAMES], weights)
    _, zfun_skin = skins.skin_functions(blended, Lc, wh)
    return zfun_skin

def build_cephalon(P):
    t, c, h = P["wall"] * P["headWall"], P["clearance"], P["relief"] * P["headRelief"]
    Lc = P["cephFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P); flap = max(ovl - 2.0, 1.0)
    wh = head_halfwidth(P)
    par = P["cephParallel"] * Lc
    a = P["axisFrac"] * wh
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    Le = Lc - par                                                       # elliptical front length
    skin_zfun = head_skin_zfun(P, Lc, wh) if P["headSkin"] > 0.001 else None

    nO = P["headOutlineExp"]
    def xmax(y):
        yy = np.asarray(y, float)
        front = np.clip((-yy - par) / Le, 0, 1)
        return np.maximum(wh * np.clip(1 - front ** nO, 0, 1) ** (1 / nO), 1.5)     # superellipse front (reference n=2.15)

    gs = P["genalSweep"] * pitch(P)
    Lg = P["genalSpine"] * Lc                                            # genal horn length (0 = plain genal angle)
    gc = math.radians(P["genalCurve"])
    H0 = 0.72                                                            # the crescent's horn begins at 0.72 of half-width
    arc = P["headRearArc"] * Lc; nR = P["headRearExp"]
    def cheek(u):
        q = np.clip((np.abs(u) * wh - a) / (wh - a), 0, 1)
        # rear edge: the genal sweep (v4, ≤ 0.6 pitch) plus the crescent's concave rear arc — the head's
        # top-down band thins from the axis to the genal angles as the rear edge bows back by headRearArc·Lc
        return flap + gs * q ** 2.2 + arc * q ** nR
    Y_tip = float(cheek(H0)) + (Lg * math.cos(gc) if Lg > 0.5 else 0.0)  # ... measured from the rear edge where they start
    def y_front(ax):
        """front edge of the head at |x| ≤ wh (inverse of the superellipse xmax)."""
        r = np.clip(ax / wh, 0, 1)
        return -par - Le * np.clip(1 - r ** nO, 0, 1) ** (1 / nO)
    W_arm = P["genalWidthMM"] if P["genalWidthMM"] > 0.05 else P["genalWidth"] * wh
    W_arm = max(W_arm, 1.5)                                               # crescent arm thickness across, mm
    kT = P["genalTaper"]                                                  # arm end: 0.5 blunt/rounded … 3 pointed
    x_in = wh - W_arm
    S_ = np.linspace(0, 1, 241)
    path = safe_expr(P["genalPath"], S_)                                  # user formula f(s): sideways offset shape
    path = path - path[0]
    if np.max(np.abs(path)) > 1e-9: path = path / np.max(np.abs(path))    # normalised so genalCurve sets the amount
    x_off = math.tan(gc) * Lg * path                                      # sideways (mm) along the arm
    xc_s = (wh - 0.5 * W_arm) + x_off                                     # centreline x(s); y(s) = yr_root + Lg·s
    W_s = W_arm * np.clip(1 - S_ ** (2.0 / max(kT, 0.2)), 0, 1) ** (0.5 * kT) + 0.7   # width along the arm: end taper
    X_tip = float(np.max(xc_s + 0.5 * W_s))
    def head_edges(x):
        """Front and rear edge of the head BODY at lateral |x|: superellipse front + concave rear.
        The crescent's arms are separate swept bands (build_arm) fused on afterwards, so a path that turns
        inward leaves open water between the arm and the thorax instead of filling it."""
        ax = np.abs(np.asarray(x, float))
        return y_front(np.minimum(ax, wh)), cheek(np.minimum(ax, wh) / wh)
    def outline(u, v):
        x = u * X_tip
        yf, yr = head_edges(np.array([abs(x)]))
        return x, float(yr[0] - v * (yr[0] - yf[0]))

    def glab_half(y):
        yy = np.asarray(y, float)
        f = np.clip(-yy / Lc, 0, 1)
        return a * (1 + (P["glabInflate"] - 1) * f)

    def zfun(x, y):
        ax = np.abs(x)
        xm = xmax(y)
        front = np.clip((-y - par) / Le, 0, 1)
        tz = np.sqrt(np.clip(1 - front ** 2, 0, 1)) * 0.9 + 0.1
        z_v4 = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
        # superellipsoid dome (reference head: m=1.49, semi-axes 0.82 wh × 0.72 Lc, centred 0.19 Lc in from the rear,
        # flat border outside it). Blended with the v4 separable vault by `tent`.
        mD, fill = P["headDomeExp"], P["headDomeFill"]
        rim = margin * 0.6
        yc = -0.19 * Lc; aD = fill * wh; bD = 0.88 * fill * Lc
        rD = (ax / aD) ** mD + (np.abs(y - yc) / bD) ** mD
        z_dome = rim + (h - rim) * np.clip(1 - rD, 0, 1) ** (1 / mD)
        z = (1 - P["tent"]) * z_v4 + P["tent"] * z_dome
        g = glab_half(y)
        gl = plateau(x, g) * (1 - smoothstep(0.80 * Lc, 0.92 * Lc, -y))    # glabella, fading at the front
        z += (rise + P["glabRise"] * h * np.clip(-y / Lc, 0, 1)) * gl
        z -= F * trough(ax - (g + 0.7), 0.9) * (-y < 0.9 * Lc)             # axial furrows round the glabella
        z -= 0.8 * F * trough(y + 0.13 * Lc, 0.8) * plateau(x, g + 1.5, 1.0)   # occipital furrow
        for k in range(int(P["glabLobes"])):                                # lateral glabellar furrows
            yk = -Lc * (0.28 + 0.16 * k)
            z -= 0.7 * F * trough(y - yk, 0.9) * trough(ax - (g - 1.2), 2.2) * (ax > 0.3 * g)
        # eyes: palpebral lobe + crescent visual surface facing outward
        if P["eyeSize"] > 0.01:
            eR = P["eyeSize"] * wh; ye = -P["eyePos"] * Lc
            xe = float(glab_half(ye)) + eR + 1.0
            r = np.hypot(ax - xe, y - ye)
            ang = np.degrees(np.arctan2(y - ye, ax - xe))
            mask = plateau(ang, P["eyeArc"] / 2, 12)
            eH = P["eyeHeight"] * eR
            z += eH * np.exp(-(r / (0.95 * eR)) ** 4)                        # domed eye (super-Gaussian, flat top)
            z += 0.35 * eH * trough(r - eR, 1.0) * mask                       # crescent visual surface rim
        z += tubercles(x, y, P, dict(box=(-wh + 3, wh - 3, -Lc + 4, -2), ok=lambda xk, yk: True), seed=99, count_scale=2.2 * wh)
        # border: furrow and raised rim along the outline
        if P["borderWidth"] > 0.005:
            bw = P["borderWidth"] * wh
            dist = np.minimum(xm - ax, np.where(-y > par, (1 - front) * Le, 1e9))
            z -= 0.8 * F * trough(dist - bw, 0.8 + 0.2 * bw)
            z += 0.35 * F * plateau(dist, 0.45 * bw, 0.4 * bw)
        if skin_zfun is not None:                                           # blend toward the real registered skin(s)
            z = (1 - P["headSkin"]) * z + P["headSkin"] * skin_zfun(x, y)
        w0_ = seg_halfwidth(P, 0)
        def z_th_arm(axx):
            return margin + (h - margin) * vault(np.clip(axx / w0_, 0, 1), P) + P["bladeCamber"] * h + t + c + 0.3
        # rear band: the head's shingle over segment 0 must follow the thoracic vault, whatever the dome does,
        # so the stepped front of segment 0 slides under it exactly as it does under another segment
        w0 = seg_halfwidth(P, 0)
        z_th = margin + (h - margin) * vault(np.clip(ax / w0, 0, 1), P)
        rear_band = 1 - smoothstep(flap + 1.0, flap + 4.0, -y)
        z = np.maximum(z, z_th * rear_band)
        if arc > 0.5:                                                          # crescent arms behind the hinge line: a hood
            hood = smoothstep(flap - 0.5, flap + 2.0, y)                       # over the ribs, never into them
            z = np.maximum(z, hood * (z_th + P["bladeCamber"] * h + t + c + 0.3))
        # occipital ring: the axis must stand at ring_top over the head hinge whatever the dome does
        occ = ring_top(P) * plateau(x, a, 1.0) * (1 - smoothstep(0.10 * Lc, 0.16 * Lc, -y))
        return np.maximum(z, occ)

    head = plate(outline, zfun, t, nu=45, nv=33)                          # t = wall·headWall
    env = under(outline, zfun, nu=45, nv=33)
    head = add_hinge(head, env, P, 0, rear=True)
    # the crescent's arms: swept bands of width W_s along the user's path, starting inside the head so the fuse is solid
    if Lg > 0.5:
        yr_root = float(cheek(np.array([1.0]))[0])
        S0 = -0.12                                                        # start 12 % of the arm inside the head
        def arm_outline(side):
            def outline(u, v):
                s = S0 + (1 - S0) * 0.5 * (u + 1)
                sc = np.clip(s, 0, 1)
                xc = float(np.interp(sc, S_, xc_s)); w = float(np.interp(sc, S_, W_s))
                # tangent of the centreline for the across direction
                ds = 1e-3; xa = float(np.interp(np.clip(sc + ds, 0, 1), S_, xc_s)); xb = float(np.interp(np.clip(sc - ds, 0, 1), S_, xc_s))
                tx, ty = (xa - xb), Lg * 2 * ds; nrm = math.hypot(tx, ty); tx, ty = tx / nrm, ty / nrm
                nx, ny = ty, -tx                                          # normal (across the arm)
                off = (v - 0.5) * w
                return side * (xc + nx * off), yr_root + Lg * s + ny * off
            return outline
        def arm_z(x, y):
            ax = np.abs(x)
            z_h = max(P["marginHeight"] * h, 0.0) + t + 0.6 + c
            z_in = margin + (h - margin) * vault(np.clip(ax / seg_halfwidth(P, 0), 0, 1), P) + P["bladeCamber"] * h + t + c + 0.3
            inboard = ax < seg_halfwidth(P, 0) + 1.0
            z = np.where(inboard & (y > yr_root - 0.5), np.maximum(z_h, z_in), z_h)
            body = y < yr_root + 0.5                                     # inside the head: follow the head's own surface
            zb = zfun(np.asarray(x, float), np.asarray(y, float)) - 0.3      # (slightly inside it, so the fuse is solid)
            blend = smoothstep(yr_root - 0.5, yr_root + 2.5, y)
            return np.where(body, zb, (1 - blend) * zb + blend * z)
        for side in (1, -1):
            arm = plate(arm_outline(side), arm_z, t, nu=41, nv=7)
            if arm is not None and len(arm.solids()) == 1 and arm.volume > 0:
                head += arm
            else:
                BUILD_NOTES.append(("head", "arm failed", f"side {side}"))
    if P["occipitalSpine"] > 0.02:
        head += spine(0.6 * margin, 0.5, P["occipitalSpine"] * Lc, (0, -0.07 * Lc, ring_top(P) - 1.0), 0, pitch_deg=55)
    return prune_slivers(head, label="head")

# =====================================================================
#  PYGIDIUM
# =====================================================================
def build_pygidium(P):
    t, c, h = P["wall"] * P["tailWall"], P["clearance"], P["relief"] * P["tailRelief"]
    Lp = P["pygFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P)
    wp = tail_halfwidth(P)
    a = P["axisFrac"] * wp
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    n = int(P["pygRings"])

    def xmax(y):
        yy = np.clip(np.asarray(y, float) / Lp, 0, 1)
        return np.maximum(wp * np.sqrt(np.clip(1 - yy ** 2, 0, 1)), 1.5)
    # ---- the shield plan is parametrised by angle round the margin, so anything the margin grows —
    #      the two forks, the marginal spines — is a radial continuation of it. No cones anywhere.
    Lf = P["pygSpine"] * Lp
    n_m = int(P["pygMarginal"]); Lm = P["pygMarginalLen"] * Lp
    sp = math.radians(P["pygSplay"])
    phi_f = math.atan(wp / Lp * math.tan(math.pi / 2 - sp)) if Lf > 0.5 else None    # fork leaves the margin at this angle
    hphi_f = math.radians(13)
    phi_m = [math.radians(50 + 80 * (k + 0.5) / n_m) for k in range(n_m)]          # marginal spines round the rear arc (clear of the last ribs)
    hphi_m = math.radians(80 / max(n_m, 1)) * 0.5 * 1.3      # roots overlap a little: a scalloped margin tessellates, needle roots do not
    def bump(phi, phi_k, h, e):
        return np.clip(1 - np.abs(phi - phi_k) / h, 0, 1) ** e
    def radial_extra(phi):
        """Extra radial length of the margin at angle phi: forks + marginal spines, each a smooth point."""
        ex = np.zeros_like(phi)
        if Lf > 0.5:
            for pf in (phi_f, math.pi - phi_f): ex = np.maximum(ex, Lf * bump(phi, pf, hphi_f, 1.5))
        for pk in phi_m:
            for pm in (pk, math.pi - pk): ex = np.maximum(ex, Lm * bump(phi, pm, hphi_m, 1.2))
        return ex
    def margin_pt(phi):
        x0, y0 = wp * np.cos(phi), Lp * np.sin(phi)
        r0 = np.hypot(x0, y0); f = 1 + radial_extra(phi) / np.maximum(r0, 1e-6)
        return x0 * f, y0 * f
    def outline(u, v):
        phi = 0.5 * math.pi * (1 - u)
        mx, my = margin_pt(np.array([phi])); fx = u * (wp - 1.0)             # front hinge line, slightly inset at the corners
        return float((1 - v) * fx + v * mx[0]), float(v * my[0])
    def outside(x, y):
        """How far outside the basic ellipse a point is, as a fraction of the longest growth (0 inside)."""
        r = np.sqrt((x / wp) ** 2 + (y / Lp) ** 2)
        Lmax = max(Lf if Lf > 0.5 else 0.0, Lm if n_m else 0.0, 1e-6)
        return np.clip((r - 1) * np.hypot(x, y) / np.maximum(r, 1e-6) / Lmax, 0, 1)

    def zfun(x, y):
        ax = np.abs(x); xm = xmax(y); yy = np.clip(y / Lp, 0, 1)
        nT = P["tailDomeExp"]                                                   # 2 = elliptical fall-off (v4); higher = flatter, then a steeper rear
        tz = np.clip(1 - yy ** nT, 0, 1) ** (1 / nT) * 0.9 + 0.1
        z = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
        ay = a * (1 - 0.85 * yy)                                            # tapering pygidial axis
        z += rise * plateau(x, ay) * (1 - smoothstep(0.75 * Lp, 0.95 * Lp, y))
        z -= F * trough(ax - (ay + 0.6), 0.9) * (y < 0.9 * Lp)
        for k in range(n):                                                  # axial rings + pleural ribs
            yk = Lp * (k + 1) / (n + 1.5)
            fade = 1 - 0.6 * k / max(n, 1)
            z -= 0.7 * F * fade * trough(y - yk, 0.8) * plateau(x, ay + 1.5, 1.0)
            px = np.clip((ax - ay) / np.maximum(xm - ay, 1), 0, 1); ly = yk + px * 0.5 * Lp / (n + 1.5)
            z -= 0.6 * F * fade * trough(y - ly, 0.9) * (ax > ay + 1.0)
        if P["borderWidth"] > 0.005:
            bw = P["borderWidth"] * wp
            dist = np.minimum(xm - ax, (1 - yy) * Lp)
            z -= 0.8 * F * trough(dist - bw, 0.8 + 0.2 * bw)
            z += 0.35 * F * plateau(dist, 0.45 * bw, 0.4 * bw)
        z += tubercles(x, y, P, dict(box=(-wp + 3, wp - 3, ovl + 1, Lp - 3), ok=lambda xk, yk: abs(xk) < wp * math.sqrt(max(0.0, 1 - (yk / Lp) ** 2)) - 2.5),
                       seed=77, count_scale=1.6 * wp)
        ov = outside(ax, y)                                                            # forks and marginal spines droop
        z = np.where(ov > 0, np.maximum(margin * 0.6 * (1 - 0.5 * ov), t + 0.6), z)     # toward their points
        if P["tailRelief"] < 0.999:                                                    # only when the tail is lowered: axis holds
            z = np.maximum(z, ring_top(P) * plateau(x, a, 1.0) * (1 - smoothstep(0.5 * ovl, ovl + 1.0, y)))   # ring_top over the hinge
        drop = np.minimum(t + c + 0.7 * F + 0.3, np.maximum(z - (t + 0.6), 0))
        z -= drop * (1 - smoothstep(ovl - 1.5, ovl, y))                               # stepped front under the last flap (last word)
        return z

    nu_t = 53 if (n_m or Lf > 0.5) else 45                                     # points need columns (73 stalls the fit)
    tail = plate(outline, zfun, t, nu=nu_t, nv=27)
    env = under(outline, zfun, nu=nu_t, nv=27)
    tail = add_hinge(tail, env, P, 0, rear=False)
    # the paired tail spines are the forks in tail_edges() above — the margin running on, not cones
    if P["termSpine"] > 0.02:
        tail += spine(0.5 * margin, 0.6, P["termSpine"] * Lp, (0, 0.85 * Lp, 0.5 * margin), 0)
    # marginal spines are scallops of the margin (tail_edges above) — the last cone family is gone
    return prune_slivers(tail, label="tail")

# =====================================================================
#  ASSEMBLY, CHECK, ANIMATION, EXPORT
# =====================================================================
PART_NAMES = lambda P: ["head"] + [f"seg{i}" for i in range(int(P["segCount"]))] + ["tail"]

def _sane(part):
    """A part is sane if it is one solid of positive volume that fits in its own bounding box."""
    try:
        bb = part.bounding_box().size
        floor = 0.25 * bb.X * bb.Y * 2.0                     # at least a 2 mm plate over a quarter of the footprint
        return len(part.solids()) == 1 and floor < part.volume <= bb.X * bb.Y * bb.Z * 1.05
    except Exception:
        return False

def _build_checked(fn, label):
    """Build a part; if OpenCascade hands back garbage (failed fuse → tool only, or an inside-out body),
    retry with a slightly different surface grid. Every retry is recorded in BUILD_NOTES."""
    global GRID_JITTER
    for k, jit in enumerate((0, 2, -2, 4)):
        GRID_JITTER = jit
        n0 = len(BUILD_NOTES)
        try:
            part = fn()
        except Exception as e:
            BUILD_NOTES.append((label, "build error", str(e)[:60])); part = None
        # a pruned negative-volume solid means the boolean failed even if what is left looks sane
        garbage = any(isinstance(n[2], (int, float)) and n[2] < 0 for n in BUILD_NOTES[n0:])
        if part is not None and _sane(part) and not garbage:
            if k: BUILD_NOTES.append((label, "rebuilt", f"grid jitter {jit:+d}"))
            GRID_JITTER = 0; return part
    GRID_JITTER = 0
    BUILD_NOTES.append((label, "UNSANE after retries", ""))
    return part

def parts_list(P):
    P = schema.coerce(P)
    return ([_build_checked(lambda: build_cephalon(P), "head")]
            + [_build_checked(lambda i=i: build_segment(P, i), f"seg{i}") for i in range(int(P["segCount"]))]
            + [_build_checked(lambda: build_pygidium(P), "tail")])

def chain_transforms(P, enroll):
    """Location of each part for a uniform enrollment (0..1)."""
    zh = hinge_z(P); theta = enroll * P["maxAngle"]
    rot = Location((0, 0, zh)) * Location((0, 0, 0), (-theta, 0, 0)) * Location((0, 0, -zh))
    offs = joint_offsets(P); locs = [Location()]
    loc = Location()
    for i in range(1, len(offs) + 1):
        loc = loc * (Location((0, offs[i - 1], 0)) * rot); locs.append(loc)
    return locs

def assemble(P, enroll, parts=None):
    parts = parts or parts_list(P)
    return [p.moved(l) for p, l in zip(parts, chain_transforms(P, enroll))]

def check_enrollment(P, samples=(0.0, 0.5, 0.95, 1.05, 1.3), parts=None):
    """Neighbour-pair overlap by CAD boolean (slow, exact). The instrument module does all pairs on meshes."""
    parts = parts or parts_list(P)
    print("enrollment check (overlap mm^3 per joint, head->tail):")
    for e in samples:
        posed = assemble(P, e, parts)
        vols = [((posed[i] & posed[i + 1]).volume if (posed[i] & posed[i + 1]) else 0.0) for i in range(len(posed) - 1)]
        print(f"  enroll={e:4.2f} ({e*P['maxAngle']:5.1f} deg): " + " ".join(f"{v:6.1f}" for v in vols)
              + f"  {'OK' if max(vols) < 1 else 'COLLIDES'}")

def build_chain(P, parts=None):
    parts = parts or parts_list(P)
    zh, offs = hinge_z(P), joint_offsets(P); names = PART_NAMES(P)
    node = None
    for i in reversed(range(len(parts))):
        body = parts[i].moved(Location((0, 0, -zh))); body.label = "shell"
        kids = [body] + ([node] if node is not None else [])
        node = Compound(children=kids, label=names[i])
        if i > 0: node.location = Location((0, offs[i - 1], 0))
    return Compound(children=[node], label="trilobite").moved(Location((0, 0, zh))), names

def animate_enrollment(P, parts=None, seconds=2.5):
    from ocp_vscode import show, Animation
    chain, names = build_chain(P, parts)
    show(chain, render_joints=False)
    anim = Animation(); times = [0, seconds / 2, seconds]
    for i in range(1, len(names)):
        anim.add_track("/trilobite/" + "/".join(names[:i + 1]), "rx", times, [0, -P["maxAngle"], 0])
    anim.animate(speed=1)

# =====================================================================
#  meshing / export (build123d's export_stl uses a tolerance far too fine for spline shells)
# =====================================================================
def to_trimesh(part, tol=0.15, ang=0.3):
    """Robust tessellation: OpenCascade occasionally skips a face at a given tolerance, so retry with
    nearby tolerances, then fall back to meshing face by face and stitching."""
    import trimesh
    best = None
    for k in (1.0, 0.8, 1.25, 0.6, 1.6):
        try:
            v, tris = part.tessellate(tol * k, ang)
        except AttributeError:
            continue
        m = trimesh.Trimesh(np.array([(p.X, p.Y, p.Z) for p in v]), np.array(tris), process=True)
        if m.is_watertight: return m
        trimesh.repair.fill_holes(m)
        if m.is_watertight: return m
        if best is None or len(m.faces) > len(best.faces): best = m
    if best is not None: return best
    pieces = []
    for f in part.faces():
        try:
            v, tris = f.tessellate(tol, ang)
            pieces.append(trimesh.Trimesh(np.array([(p.X, p.Y, p.Z) for p in v]), np.array(tris), process=False))
        except AttributeError:
            pass
    m = trimesh.util.concatenate(pieces); m.merge_vertices(); trimesh.repair.fix_normals(m); trimesh.repair.fill_holes(m)
    return m

def save_mesh(part, path, tol=0.15, ang=0.3):
    """STL/GLB/OBJ export by extension. Returns the trimesh."""
    m = to_trimesh(part, tol, ang)
    m.export(path)
    return m

def save_posed(P, enroll, parts, path, tol=0.2):
    import trimesh
    ms = [to_trimesh(p, tol, 0.4) for p in assemble(P, enroll, parts)]
    trimesh.util.concatenate(ms).export(path)

if __name__ == "__main__":
    import sys, time
    P = schema.defaults()
    t0 = time.time(); parts = parts_list(P)
    for n, p in zip(PART_NAMES(P), parts):
        print(f"{n:5s} solids: {len(p.solids())} valid: {p.is_valid} size: {tuple(round(v,1) for v in p.bounding_box().size)}")
    print(f"built in {time.time()-t0:.0f}s  | params {schema.param_hash(P)}")
    for n, p in zip(PART_NAMES(P), parts): save_mesh(p, f"{n}.stl")
    save_posed(P, 0.0, parts, "animal_flat.stl"); save_posed(P, 1.0, parts, "animal_enrolled.stl")
    if "--check" in sys.argv: check_enrollment(P, parts=parts)
    try:
        from ocp_vscode import set_port; set_port(VIEWER_PORT); animate_enrollment(P, parts)
    except Exception as e:
        print("viewer not available:", e)
