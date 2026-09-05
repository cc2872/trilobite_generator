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
    """Cross-section height factor vs normalized lateral position u ∈ [-1, 1], with a fulcrum."""
    u = np.abs(u); f = P["fulcrum"]
    inner = 1 - 0.35 * (u / f) ** 2
    outer = 0.65 * (1 - (np.maximum(u - f, 0) / (1 - f)) ** 1.6)
    return np.where(u <= f, inner, outer)

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

def plate(outline, zfun, t, nu=45, nv=27):
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
        web = Box(kw - c, rB + c + 1.0, 2 * rB + c + 3,
                  align=(Align.CENTER, Align.MAX if rear else Align.MIN, Align.CENTER)).moved(Location((xc, y_axis, zh)))
        webs = web if webs is None else webs + web
    part += webs & envelope
    part -= barrel(0, Wh + 10, P["boreDia"] / 2)
    blk = Box(Wh + 2, 2 * P["wall"], zh - rB - c - 0.5,
              align=(Align.CENTER, Align.MAX if rear else Align.MIN, Align.MIN)).moved(Location((0, y_axis, 0)))
    part += blk
    phi, L = P["maxAngle"] / 2, 80
    band = 2 * (P["axisFrac"] * (W / 2)) + 0.45 * W          # bevel only the axis and inner pleurae; blades sweep freely
    if rear:
        wedge = Box(band, L, L, align=(Align.CENTER, Align.MIN, Align.MAX)).rotate(Axis.X, -phi)
    else:
        wedge = Box(band, L, L, align=(Align.CENTER, Align.MAX, Align.MAX)).rotate(Axis.X, phi)
    part -= wedge.moved(Location((0, y_axis, zh))) - barrel(0, Wh + 4, rB + c)
    return part

# =====================================================================
#  THORACIC SEGMENT i
# =====================================================================
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
    def q_of(x):
        return np.clip((np.abs(x) - a) / (w - a), 0, 1)
    def edges(x):
        """Front and rear edge of the pleural blade at lateral position x: a sickle that sweeps back
        and narrows toward its tip."""
        q = q_of(x)
        yc = 0.5 * L0 + sweep * q ** 1.6
        hb = 0.5 * L0 * (1 - P["tipTaper"] * q ** 2.5)
        return yc - hb, yc + hb
    def outline(u, v):
        q = np.clip((abs(u) * w - a) / (w - a), 0, 1)
        x = u * w * (1 + 0.06 * q ** 2)                                  # tips reach slightly outward
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
        yf, _ = edges(x)
        drop = np.minimum(t + c + 0.7 * F + 0.3, np.maximum(z - (t + 0.6), 0))    # never thinner than the wall
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
    # pleural spines from the field — added AFTER the hinge so the stop bevel never trims them;
    # whether they clear the next segment when curling is the instrument's job to find out
    Ls = pleural_spine_field(P)[i] * w
    r = 0.45 * margin
    if Ls > 0.5:
        _, yr_tip = edges(np.array([w]))
        for s in (1, -1):   # needle spine continuing the blade tip
            seg += spine(r, 0.5, Ls, (s * (w - 0.3 * r), float(yr_tip[0]) - 1.2 * r, r), s * P["spineSweep"])
    if P["axialSpine"] > 0.02:
        seg += spine(0.6 * r + 0.6, 0.5, P["axialSpine"] * h, (0, ovl + 0.45 * (d - ovl), h + rise - 1.0), 0, pitch_deg=60)
    return seg

# =====================================================================
#  CEPHALON
# =====================================================================
def build_cephalon(P):
    t, c, h = P["wall"], P["clearance"], P["relief"]
    Lc = P["cephFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P); flap = max(ovl - 2.0, 1.0)
    wh = head_halfwidth(P)
    par = P["cephParallel"] * Lc
    a = P["axisFrac"] * wh
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    Le = Lc - par                                                       # elliptical front length

    def xmax(y):
        yy = np.asarray(y, float)
        front = np.clip((-yy - par) / Le, 0, 1)
        return np.maximum(wh * np.sqrt(np.clip(1 - front ** 2, 0, 1)), 1.5)

    gs = P["genalSweep"] * pitch(P)
    def cheek(u):
        q = np.clip((abs(u) * wh - a) / (wh - a), 0, 1)
        return flap + gs * q ** 2.2                              # rear edge sweeps back toward the genal angle
    def outline(u, v):
        yr = cheek(u)
        y = yr - v * (Lc + yr)
        return u * float(xmax(y)), y

    def glab_half(y):
        yy = np.asarray(y, float)
        f = np.clip(-yy / Lc, 0, 1)
        return a * (1 + (P["glabInflate"] - 1) * f)

    def zfun(x, y):
        ax = np.abs(x)
        xm = xmax(y)
        front = np.clip((-y - par) / Le, 0, 1)
        tz = np.sqrt(np.clip(1 - front ** 2, 0, 1)) * 0.9 + 0.1
        z = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
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
        return z

    head = plate(outline, zfun, t, nu=45, nv=33)
    env = under(outline, zfun, nu=45, nv=33)
    head = add_hinge(head, env, P, 0, rear=True)
    if P["genalSpine"] > 0.02:                      # after the hinge: the bevel must not trim them
        Lg = P["genalSpine"] * Lc
        for s in (1, -1):
            head += spine(0.55 * margin, 0.6, Lg, (s * (wh + 0.3 * margin), -0.14 * Lc, 0.5 * margin), s * (18 + 0.5 * P["genalCurve"]))
    if P["occipitalSpine"] > 0.02:
        head += spine(0.6 * margin, 0.5, P["occipitalSpine"] * Lc, (0, -0.07 * Lc, ring_top(P) - 1.0), 0, pitch_deg=55)
    return head

# =====================================================================
#  PYGIDIUM
# =====================================================================
def build_pygidium(P):
    t, c, h = P["wall"], P["clearance"], P["relief"]
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

    def outline(u, v):
        y = v * Lp
        return u * float(xmax(y)), y

    def zfun(x, y):
        ax = np.abs(x); xm = xmax(y); yy = np.clip(y / Lp, 0, 1)
        tz = np.sqrt(np.clip(1 - yy ** 2, 0, 1)) * 0.9 + 0.1
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
        drop = np.minimum(t + c + 0.7 * F + 0.3, np.maximum(z - (t + 0.6), 0))
        z -= drop * (1 - smoothstep(ovl - 1.5, ovl, y))                               # stepped front under the last flap
        return z

    tail = plate(outline, zfun, t, nu=45, nv=27)
    env = under(outline, zfun, nu=45, nv=27)
    tail = add_hinge(tail, env, P, 0, rear=False)
    if P["pygSpine"] > 0.02:
        Ls = P["pygSpine"] * Lp
        for s in (1, -1):
            tail += spine(0.5 * margin, 0.6, Ls, (s * 0.5 * wp, 0.6 * Lp, 0.5 * margin), s * P["pygSplay"])
    if P["termSpine"] > 0.02:
        tail += spine(0.5 * margin, 0.6, P["termSpine"] * Lp, (0, 0.85 * Lp, 0.5 * margin), 0)
    n_m = int(P["pygMarginal"])
    if n_m > 0:
        Lm = P["pygMarginalLen"] * Lp; r = 0.4 * margin
        for k in range(n_m):
            phi = math.radians(15 + (150 * (k + 0.5) / n_m))            # around the elliptical margin, one side
            for s in (1, -1):
                bx, by = s * 0.92 * wp * math.cos(phi), 0.92 * Lp * math.sin(phi)
                nx, ny = s * math.cos(phi) / wp, math.sin(phi) / Lp        # outward normal of the ellipse
                dx, dy = nx, ny + 0.6 / Lp                                  # swept back
                yaw = math.degrees(math.atan2(dx, dy))
                tail += spine(r, 0.5, Lm, (bx, by, 0.5 * margin), yaw)
    return tail

# =====================================================================
#  ASSEMBLY, CHECK, ANIMATION, EXPORT
# =====================================================================
PART_NAMES = lambda P: ["head"] + [f"seg{i}" for i in range(int(P["segCount"]))] + ["tail"]

def parts_list(P):
    P = schema.coerce(P)
    return [build_cephalon(P)] + [build_segment(P, i) for i in range(int(P["segCount"]))] + [build_pygidium(P)]

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
