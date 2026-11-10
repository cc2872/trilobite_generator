"""
isopod_model/shaped_head.py (2 Oct 2026): the head's OUTLINE as parameters.

The crescent head (anatomy/head_crescent.py) has one silhouette. The guide's orders differ first of all in the
silhouette: a harpetid horseshoe, a phacopid's rounded shield with no spines, an agnostid's parabola, a trinucleid's
needles three heads long. This module draws any of them from twelve numbers and lays the crescent head's whole
surface (dome, glabella, furrows, eyes, border, brim, tubercles ... every FACE key) on it.

    OUTLINE                      the parameters and their defaults (the default is the crescent's own silhouette)
    Shape(outline)               the outline at scale 1 (half-width 24, as head_crescent): y_front / y_rear per column,
                                 and the fields the surface needs (distance to the outer and rear margins, the horn)
    plan(P, plans, K, outline, face)   a head plan in model mm for body.build (same keys as anatomy/head_crescent.plan)

Frame at scale 1: x across (midline 0), y along (0 = the rear margin on the axis, the head toward -y), as every head.
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from anatomy import head_crescent as HC
from anatomy import genal as GN
from anatomy.common import smoothstep

RO = HC.RO
# 20 Oct 2026: the defaults are an ANATOMICAL head (a pointed shield, slim tapered genal spines leaving a near-straight rear
# margin), not the print kit's. Until today OUTLINE held the crescent piece's own silhouette (head3d: a round dome widest at
# the rear, two parallel round-ended bands 1/6 of the half-width wide, and a U notch as deep as the bands are long), which
# is why the "plain" animal in the studio drew a bell with two sticks and the thorax rows inside the notch. Every named
# style already overrode those keys; the kit silhouette is kept below as CRESCENT_OUTLINE (= the "crescent" style).
OUTLINE = dict(
    width=1.15,        # head half-width / the first thorax segment's half-width
    length=1.05,       # axial length / half-width (semicircle 1.0; long parabola 1.5; short and transverse 0.7)
    front_exp=1.6,     # the front's curve = POINTINESS: 1.2 sharp, 1.6 ogival / pointed, 2 round, 3+ blunt and square-shouldered
    front_point=0.0,   # an anterior median point, / length (Dalmanites, Huntonia)
    widest=0.03,       # where the head is widest, from the rear margin, / length (agnostid: 0.45)
    side=0.0,          # straight side behind the widest line, / length (subparallel-sided heads)
    rear_taper=0.0,    # the side narrows back to the genal angle by this fraction of the half-width
    corner=0.12,       # radius of a rounded genal angle, / half-width (when there is no genal spine)
    genal=0.6,         # genal spine / prolongation length, / axial length (0 = none; TriloMorph median 0.23, 95th pct 1.4)
    genal_w=0.10,      # its width at the root, / half-width (harpetid prolongation 0.3; trinucleid needle 0.07)
    genal_tip=0.12,    # POINTED TO ROUND: tip width / root width (0 = drawn to a point; 1 = a parallel band with a round end)
    genal_angle=4.0,   # 1 Nov 2026: where it points from its root, deg: 0 straight back, +90 out, 180 forward, -90 in (the full circle)
    genal_kink=0.0,    # where it bends, / its length (0 = straight, no kink)
    genal_kink_angle=4.0,  # where the part after the kink points (the same circle)
    genal_web=0.0,     # an edge from the rear margin to the tip, filling the corner: its anchor, from the genal angle (0 = none) in
                       # toward the axis (1)
    genal_web_round=0.5,   # the web's edge: 0 straight, +1 hugging the corner, -1 bulging out
    notch=0.048,       # the rear margin bows back between the genal angles: depth / axial length (0 straight across)
    notch_exp=2.0,     # 2 = an elliptical U; higher = straight across, then a late bend
    dome_fill=0.82,    # the vaulted part / the whole (0.6 leaves a broad flat brim: Harpetida)
    horn_h=0.32,       # the spine's ridge height / relief (a flat prolongation ~0.18)
    horn_aspect=1.0,   # 9 Nov 2026: the spine's height / its local width, at most (THE genal-spine bug: the crest was horn_h x the head's
                       # relief whatever the spine's width, so a thin spine was a tall standing blade); 1 = as tall as wide; 9 = as built
    genal_sharp=0.0,   # 9 Nov 2026: 0 = the ridge as built (quarter-ellipse section, crest tapering to 45 % at the tip); 1 = a sharp spine
                       # (triangular section, crest tapering to a point at the tip)
    height=1.0,        # crown height, x the model's usual head height
    fringe_n=0,        # 2 Nov 2026: a fringe of small spines along the side margin (Miraspis, Kettneraspis): how many a side (0 = none)
    fringe_len=0.25,   # ... each one's length / the head's axial length
    fringe_w=0.035,    # ... its root radius / the head's half-width
    fringe_from=0.35,  # ... where the row starts and ends along the front-and-side margin, from the axis (0) to the
    fringe_to=0.9,     #     genal angle (1)
    fringe_rise=8.0,   # ... how far they lift as they leave, deg
    dot_r=0.08,        # 7 Nov 2026: how far along the margin each of the user's dots (`dots`) pulls it, / the half-outline's length
    # ---- 3 Oct 2026: four more shape groups. Every default is "off".
    genal_at=0.0,      # where the spine leaves the margin: 0 the genal angle; +1 moved inward along the rear margin
                       # (intergenal spine); -1 moved forward along the side (a lateral cephalic spine)
    front_flat=0.0,    # a straight, truncate front: this fraction of the half-width is cut square across
    front_dent=0.0,    # ... and may be indented on the axis, / length
    point_w=0.10,      # the anterior process (front_point is its length): width / half-width
    point_fork=0.0,    # ... forked: each tine this far off the axis, / half-width (0 = one median point)
    brim_w=0.0,        # a brim as its own surface round the front and sides: width / half-width (Harpetida 0.35)
    brim_rise=0.12,    # height of its inner edge / relief: 0 = flat on the bed, 0.3 = steeply inclined
    brim_curve=0.0,    # its cross-section: 0 a straight slope, +1 convex (bulging), -1 concave (dished)
    brim_roll=0.0,     # a raised roll along the brim's inner edge, / relief (the harpetid genal roll)
    brim_pits=0,       # rows of pits on it
)
# the crescent print piece's own silhouette (head3d: R 20, arm 4 x 20, the U notch), as it was the default until 20 Oct 2026
CRESCENT_OUTLINE = dict(OUTLINE, length=1.08, front_exp=2.0, widest=0.08, genal=0.77, genal_w=0.167, genal_tip=1.0, genal_angle=0.0, genal_kink_angle=0.0, notch=0.693)   # notch: 0.9 x the band length 0.77 (24 Oct: / axial length)
TRACE_MAX = 400     # points a traced outline may carry

OUTLINE_RANGE = dict(   # what the silhouette is drawn for (11 Oct 2026: a 0.2-long head crashed in interp, a 3 x 3 x genal-3 one printed 2.7 m)
    width=(0.5, 4.0), length=(0.4, 2.5), front_exp=(1.0, 8.0), front_point=(0.0, 1.5), widest=(0.0, 0.8), side=(0.0, 1.0), rear_taper=(0.0, 0.9),
    corner=(0.0, 0.5), genal=(0.0, 3.5), genal_w=(0.02, 1.0), genal_tip=(0.0, 1.0), genal_angle=(-180.0, 180.0), genal_kink=(0.0, 0.9),
    genal_kink_angle=(-180.0, 180.0), genal_web=(0.0, 1.0), genal_web_round=(-1.0, 1.0),
    notch=(0.0, 1.0), notch_exp=(1.0, 8.0), dome_fill=(0.3, 1.0), horn_h=(0.0, 1.0), genal_sharp=(0.0, 1.0), horn_aspect=(0.3, 9.0), height=(0.2, 3.0), genal_at=(-1.0, 1.0),
    front_flat=(0.0, 0.9), front_dent=(0.0, 0.5), point_w=(0.02, 1.0), point_fork=(0.0, 0.9), brim_w=(0.0, 1.0), brim_rise=(-0.5, 1.0),
    brim_curve=(-1.0, 1.0), brim_roll=(0.0, 0.5), brim_pits=(0, 10), fringe_n=(0, 24), fringe_len=(0.02, 1.0), fringe_w=(0.005, 0.15),
    fringe_from=(0.0, 1.0), fringe_to=(0.0, 1.0), fringe_rise=(-30.0, 60.0), dot_r=(0.02, 0.3))
DOTS_MAX = 40       # dots a head may carry

def dots_of(d):
    """The user's dots on the head's margin (7 Nov 2026): [[u, dx, dy], ...]. u is where the dot sits along the head's
    right half-outline (0 the front on the axis, round the side, 1 the rear margin on the axis), measured on the outline
    the sliders draw; (dx, dy) is how far the dot has been dragged from there, in half-widths (x out, y toward the rear).
    The margin passes through every dot and eases back to the drawn outline over dot_r on either side; dots closer than
    that make one smooth curve through them all. None or [] = no dots. The client's order is kept (the studio indexes
    them). Raises ValueError on anything else."""
    if d is None or (isinstance(d, (list, tuple)) and len(d) == 0): return None
    if not isinstance(d, (list, tuple)) or len(d) > DOTS_MAX: raise ValueError(f"dots: a list of at most {DOTS_MAX} [u, dx, dy]")
    out = []
    for p in d:
        try: u, dx, dy = float(p[0]), float(p[1]), float(p[2])
        except (TypeError, ValueError, IndexError): raise ValueError("dots: each dot is [u, dx, dy]")
        if not all(math.isfinite(v) for v in (u, dx, dy)): raise ValueError("dots: not finite")
        if not 0.0 <= u <= 1.0: raise ValueError("dots: u (where along the margin) is 0 .. 1")
        if abs(dx) > 1.5 or abs(dy) > 1.5: raise ValueError("dots: a dot moves at most 1.5 half-widths")
        out.append([round(u, 5), round(dx, 5), round(dy, 5)])
    return out


def dot_field(dots, r, us, length=None, slope=2.5):
    """How far the margin is pushed out along its normal (n,) at positions us (0 .. 1 along the half-outline), for dots
    [[u, d]] (d > 0 out, < 0 in): a monotone cubic through every dot, back to 0 over its reach either side of a lone dot,
    straight through neighbours closer than that. A dot's reach is r (a fraction of the half-outline), or, when `length`
    (the half-outline's length, in d's unit) is given, at least slope x |d| along it -- a dot dragged far pulls a wide
    bump, never an overhang. The rear margin on the axis (u = 1, the joint) never moves."""
    from scipy.interpolate import PchipInterpolator
    us = np.asarray(us, float)
    if not dots: return np.zeros(len(us))
    d = sorted(([float(a), float(b)] for a, b in dots), key=lambda p: p[0])
    m = []                                                         # dots at one u: the last one wins
    for p in d:
        if m and p[0] - m[-1][0] < 1e-4: m[-1] = p
        else: m.append(p)
    m = [p for p in m if p[0] < 1 - 1e-4] or [[1 - 2e-4, 0.0]]
    rr = [max(r, slope * abs(p[1]) / length) if length else r for p in m]
    ku, kv = [], []
    u0 = m[0][0]
    if u0 > 1e-4:                                                  # the front on the axis
        if u0 > rr[0] + 1e-4: ku += [0.0, u0 - rr[0]]; kv += [0.0, 0.0]
        else: ku.append(0.0); kv.append(m[0][1])
    for i, p in enumerate(m):
        ku.append(p[0]); kv.append(p[1])
        if i + 1 < len(m) and m[i + 1][0] - p[0] > rr[i] + rr[i + 1] + 1e-4: ku += [p[0] + rr[i], m[i + 1][0] - rr[i + 1]]; kv += [0.0, 0.0]
    u1 = m[-1][0]
    if 1 - u1 > rr[-1] + 1e-4: ku += [u1 + rr[-1], 1.0]; kv += [0.0, 0.0]
    else: ku.append(1.0); kv.append(0.0)                           # the rear margin on the axis stays put
    return PchipInterpolator(np.array(ku), np.array(kv))(np.clip(us, 0, 1))


def _normals(ring):
    """Outward unit normals of the right half-outline (front on the axis -> side -> rear on the axis), smoothed so a
    corner (the genal angle) turns them gradually instead of fanning them out."""
    from scipy.ndimage import gaussian_filter1d
    t = np.gradient(ring, axis=0); t = gaussian_filter1d(t, 6, axis=0, mode="nearest")
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-12)[:, None]
    return np.c_[t[:, 1], -t[:, 0]]                                # the tangent turned -90 deg: out of the head


def dot_targets(ring0, u_ring, N, dots, window=0.25):
    """Each dot [u, dx, dy] (scale 1) as [u*, d]: dragged to T = the outline at u + (dx, dy), it sits at u*, the point of
    the outline nearest T within `window` of u, and pushes the margin out (d > 0) or in by its distance there. A dot
    dragged along the margin slides; dragged across it, it pushes."""
    out = []
    for u, dx, dy in dots:
        i0 = int(np.argmin(np.abs(u_ring - u))); T = ring0[i0] + (dx, dy)
        lo, hi = np.searchsorted(u_ring, [u - window, u + window]); lo = max(lo, 0); hi = max(min(hi, len(u_ring)), lo + 1)
        j = lo + int(np.argmin(np.hypot(*(ring0[lo:hi] - T).T)))
        out.append([float(u_ring[j]), float(np.dot(T - ring0[j], N[j]))])
    return out


def _ring(axb, yfb, yrb, n=900):
    """The head proper's right half-outline from its columns: the front edge out to the widest column, down its side,
    the rear edge back to the axis -- resampled to n points evenly along its length. (points (n, 2), u (n,))."""
    P = np.r_[np.c_[axb, yfb], np.c_[axb[::-1], yrb[::-1]]]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    keep = np.r_[True, np.diff(s) > 1e-9]; P, s = P[keep], s[keep]
    u = np.linspace(0, 1, n); t = u * s[-1]
    return np.c_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], u


def _columns_of(poly, axb):
    """Front (min y) and rear (max y) edge of a closed polygon at each column x = axb (NaN where it has none)."""
    poly = np.vstack([poly, poly[:1]]); yf = np.full(len(axb), np.nan); yr = np.full(len(axb), np.nan)
    for (x0_, y0_), (x1_, y1_) in zip(poly[:-1], poly[1:]):
        lo_, hi_ = min(x0_, x1_), max(x0_, x1_)
        if hi_ - lo_ < 1e-9: continue
        m = (axb >= lo_ - 1e-9) & (axb <= hi_ + 1e-9)
        if not m.any(): continue
        yy = y0_ + (axb[m] - x0_) * (y1_ - y0_) / (x1_ - x0_)
        yf[m] = np.fmin(yf[m], yy); yr[m] = np.fmax(yr[m], yy)
    return yf, yr

def trace_of(t):
    """A traced head margin (20 Oct 2026): the RIGHT half, from the front on the axis back to the genal angle, as [[x, y], ...]
    in half-widths (x >= 0 across, y along: 0 = the rear margin on the axis, the head toward -y). Validated and folded
    (|x|), ends snapped: the first point to the axis. None or [] = no trace. Raises ValueError on anything else."""
    if t is None or (isinstance(t, (list, tuple)) and len(t) == 0): return None
    if not isinstance(t, (list, tuple)) or not 3 <= len(t) <= TRACE_MAX: raise ValueError(f"trace: 3 to {TRACE_MAX} points")
    out = []
    for p in t:
        try: x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError): raise ValueError("trace: each point is [x, y]")
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 1.5 or not -4.0 <= y <= 1.0: raise ValueError("trace: points lie within |x| <= 1.5 half-widths, -4 <= y <= 1")
        out.append([abs(x), y])
    out[0][0] = 0.0; ymin = min(p[1] for p in out)
    if max(p[0] for p in out) < 0.2: raise ValueError("trace: no width")
    if out[0][1] > ymin + 0.08: raise ValueError("trace: the first point is the front of the head on the axis")
    out[0][1] = ymin                                               # the front, exactly (a hand's or a resampler's jitter snapped)
    return out

def migrate_outline(o):
    """A head saved before 1 Nov 2026: its spine left straight back, turned out to genal_spread over its first 0.3 and on by
    genal_curve along its length (and genal_tip above 1 widened it). Now: two straight pieces through the old spine's root,
    middle and tip (genal_angle, genal_kink 0.5, genal_kink_angle; one piece when it was straight). New keys win."""
    if not o or not any(k in o for k in ("genal_spread", "genal_curve")): return o
    o = dict(o); sp = float(o.pop("genal_spread", OLD_SPREAD)); cu = float(o.pop("genal_curve", 0.0))
    if not any(k in o for k in ("genal_angle", "genal_kink", "genal_kink_angle")):
        S = np.linspace(0, 1, 241); e = np.clip(S / 0.3, 0, 1); th = np.radians(sp * (3 * e ** 2 - 2 * e ** 3) + cu * S)
        x = np.r_[0, np.cumsum(np.sin(th)[1:])]; y = np.r_[0, np.cumsum(np.cos(th)[1:])]
        o["genal_angle"], o["genal_kink"], o["genal_kink_angle"] = GN.from_old_path(x, y)
    if "genal_tip" in o:
        try: o["genal_tip"] = min(float(o["genal_tip"]), 1.0)
        except (TypeError, ValueError): pass
    return o
OLD_SPREAD = 4.0        # the old default genal_spread

def outline_of(o=None):
    o = migrate_outline(o)
    f = dict(OUTLINE, **(o or {})); tr = f.pop("trace", None); dots = f.pop("dots", None); bad = set(f) - set(OUTLINE)
    if bad: raise KeyError(f"unknown outline keys {sorted(bad)} (have {sorted(OUTLINE)}, 'trace' and 'dots')")
    for k, v in f.items():
        try: v = float(v)
        except (TypeError, ValueError): raise ValueError(f"outline {k} is not a number")
        if not math.isfinite(v): raise ValueError(f"outline {k} is not finite")
        lo, hi = OUTLINE_RANGE.get(k, (-10.0 * max(1.0, abs(OUTLINE[k])), 10.0 * max(1.0, abs(OUTLINE[k]))))
        if not lo <= v <= hi: raise ValueError(f"outline {k} = {v:g} is outside {lo:g}..{hi:g}")
        f[k] = v
    tr = trace_of(tr)
    if tr is not None: f["trace"] = tr
    dots = dots_of(dots)
    if dots is not None: f["dots"] = dots
    return f

def _dense(pts, step=0.15):
    """Points along a polyline at <= step spacing (jumps between columns become their vertical edges)."""
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step)); t = np.linspace(0, 1, n + 1)[1:, None]; out.append(a + t * (b - a))
    return np.vstack(out)

def _convex_floor(x, y):
    """The lower convex envelope of the points (x, y) (x increasing), sampled back at x: y with every concave dip filled."""
    pts = np.c_[x, y]; hull = []
    for p_ in pts:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (b[0] - a[0]) * (p_[1] - a[1]) - (b[1] - a[1]) * (p_[0] - a[0]) <= 1e-12: hull.pop()   # b lies above the chord a-p: drop it
            else: break
        hull.append(p_)
    hull = np.array(hull); return np.interp(x, hull[:, 0], hull[:, 1])


def _convex_columns(x, yf, yr):
    """A convex silhouette's columns from the 2-D convex hull of its (mirrored) outline: the front and rear edges meet at
    the widest column with one tangent, which two separate 1-D envelopes do not give (a corner at the widest point)."""
    from scipy.spatial import ConvexHull
    P = np.r_[np.c_[x, yf], np.c_[x, yr], np.c_[-x, yf], np.c_[-x, yr]]
    try: hv = P[ConvexHull(P).vertices]
    except Exception: return yf, yr
    hv = np.vstack([hv, hv[:1]]); lo = np.full(len(x), np.inf); hi = np.full(len(x), -np.inf)
    for (x0_, y0_), (x1_, y1_) in zip(hv[:-1], hv[1:]):
        a_, b_ = min(x0_, x1_), max(x0_, x1_); m = (x >= a_ - 1e-9) & (x <= b_ + 1e-9)
        if b_ - a_ < 1e-9: yy = np.where(m, np.minimum(y0_, y1_), np.inf); lo = np.minimum(lo, yy); hi = np.maximum(hi, np.where(m, max(y0_, y1_), -np.inf)); continue
        yy = y0_ + (x - x0_) * (y1_ - y0_) / (x1_ - x0_)
        lo = np.where(m, np.minimum(lo, yy), lo); hi = np.where(m, np.maximum(hi, yy), hi)
    ok = np.isfinite(lo) & np.isfinite(hi)
    return np.where(ok, lo, yf), np.where(ok, hi, yr)


class Shape:
    def __init__(self, outline=None, n=1601, organic=None, body_reach=None):
        o = self.o = outline_of(outline); tr = o.get("trace"); self.notes = []   # what the silhouette had to refuse (the studio shows them)
        if tr is not None:                                         # a traced margin: its own length, widest line and genal angle
            T = np.array(tr, float) * RO; L1 = self.L1 = float(-T[:, 1].min())
            iw = int(np.argmax(T[:, 0])); yw = float(T[iw, 1])
            rearish = T[T[:, 0] > 0.5 * RO]; ig = int(np.argmax(rearish[:, 1])); xg, yg = float(rearish[ig, 0]), float(rearish[ig, 1])
            Lf = L1 + yw; nF = o["front_exp"]
        else:
            L1 = self.L1 = o["length"] * RO; yw = -o["widest"] * L1; Lf = L1 + yw; yg = yw + o["side"] * L1
            xg = RO * (1 - o["rear_taper"]); nF = o["front_exp"]
        Lh = o["genal"] * L1; w0 = o["genal_w"] * RO; horn = Lh > 0.5
        at = float(np.clip(o["genal_at"], -1, 1)); moved = horn and abs(at) > 0.01
        corner_horn = horn and not moved                           # the spine IS the genal angle (the crescent's case)
        x_in = xg - w0 if corner_horn else xg
        # ---- the head proper, column by column (0 .. RO)
        nb = 1201; axb = np.linspace(0, RO, nb)
        fl = float(np.clip(o["front_flat"], 0, 0.85)); a2 = np.clip((axb - fl * RO) / (RO * (1 - fl)), 0, 1)
        if tr is not None:                                         # columns of the traced polygon: front edge = its min y, rear edge = its max y
            poly = np.vstack([T, [[0.0, max(0.0, yg)]]]) if T[-1, 0] > 1e-6 else T   # close along the rear margin to the axis
            poly = np.vstack([poly, poly[:1]]); yfb = np.full(nb, np.nan); yrb = np.full(nb, np.nan)
            for (x0_, y0_), (x1_, y1_) in zip(poly[:-1], poly[1:]):
                lo_, hi_ = min(x0_, x1_), max(x0_, x1_); m = (axb >= lo_) & (axb <= hi_)
                if hi_ - lo_ < 1e-9: continue
                yy = y0_ + (axb[m] - x0_) * (y1_ - y0_) / (x1_ - x0_)
                yfb[m] = np.fmin(yfb[m], yy); yrb[m] = np.fmax(yrb[m], yy)
            okc = np.isfinite(yfb) & np.isfinite(yrb)
            yfb = np.interp(axb, axb[okc], yfb[okc]); yrb = np.interp(axb, axb[okc], yrb[okc])
            from scipy.ndimage import gaussian_filter1d
            yfb = gaussian_filter1d(yfb, 3, mode="nearest"); yrb = gaussian_filter1d(yrb, 3, mode="nearest")   # the hand's jitter
            # 25 Oct 2026: a cephalon's front margin is convex everywhere (the frontal point and dent go on below), and the
            # hand's trace, even smoothed, still turned in and out by up to 0.2 mm with ~10 curvature reversals along it
            # (head_check). The front edge takes its own lower convex envelope -- the same curve with the wobble gone.
            front_cols = axb <= xg - 0.5 * o["genal_w"] * RO
            yfb[front_cols] = _convex_floor(axb[front_cols], yfb[front_cols])
            yrb = np.maximum(yrb, yfb + 0.5)
        else:
            yfb = yw - Lf * np.clip(1 - a2 ** nF, 0, 1) ** (1 / nF)
            # 24 Oct 2026: the notch is the HEAD's rear margin bowing back (depth / axial length), no longer a fraction of the
            # spine's length (which made every lengthening of the spine deepen the notch), and it runs out to the genal
            # angle: the columns between the spine's inner edge and the angle sat at yg, a wall of notch x Lh wherever the
            # spine did not run straight back over them (the "stub" at the genal corner, studio, 23 Oct).
            # The U is deepest at the spine's inner edge (x_in) and RETURNS to the genal angle over the spine's root width,
            # so its wall tops out exactly where the spine leaves (hidden under a band that runs straight back; the
            # margin's own return where the band leaves at an angle). Without a corner spine the margin runs to the angle as before.
            y_n = yg + o["notch"] * L1; q = o["notch_exp"]
            w_ret = w0 if corner_horn else 0.0; x_n = max(xg - w_ret, 1e-6)
            yrb = np.full(nb, yg); inb = axb <= x_n; ret = (axb > x_n) & (axb <= xg)
            yrb[inb] = y_n * (1 - np.clip(1 - (axb[inb] / x_n) ** q, 0, 1) ** (1 / q))
            if ret.any(): yrb[ret] = yg + (y_n - yg) * np.clip(1 - ((axb[ret] - x_n) / w_ret) ** q, 0, 1) ** (1 / q)
            sidec = axb > xg                                       # a side that tapers back from the widest line
            if sidec.any(): yrb[sidec] = yw + (yg - yw) * np.sqrt(np.clip((RO - axb[sidec]) / max(RO - xg, 1e-6), 0, 1))
        if tr is None and not corner_horn and o["corner"] > 0.01:  # round the genal angle: open the silhouette with a disc
            from scipy.ndimage import distance_transform_edt as edt, gaussian_filter1d
            rc = o["corner"] * RO; st = 0.1; pad = int(rc / st) + 4
            gy = np.arange(yfb.min() - 1, yrb.max() + 1, st); gx = np.arange(-RO - 1, RO + 1 + st / 2, st)
            GX, GY = np.meshgrid(gx, gy); a_ = np.abs(GX)
            m = (a_ <= RO) & (GY >= np.interp(a_, axb, yfb)) & (GY <= np.interp(a_, axb, yrb))
            m = np.pad(m, pad); er = edt(m) * st > rc; op = edt(~er) * st <= rc; op = op[pad:-pad, pad:-pad]
            good = np.ones(nb, bool)
            for i in range(nb):
                col = op[:, int(round((axb[i] + RO + 1) / st))]
                if col.any(): yfb[i] = gy[col.argmax()]; yrb[i] = gy[len(col) - 1 - col[::-1].argmax()]
                else: good[i] = False
            yfb[good] = gaussian_filter1d(yfb[good], 4, mode="nearest"); yrb[good] = gaussian_filter1d(yrb[good], 4, mode="nearest")
            mid_b = np.interp(axb, axb[good], 0.5 * (yfb + yrb)[good]); yfb = np.where(good, yfb, mid_b); yrb = np.where(good, yrb, mid_b)
            # 25 Oct 2026: the opening pulls the widest point in by a little, and the columns past it were filled with the
            # mid-line -- the front and rear edges pinched to a POINT at the widest column (a cusp on every rounded head:
            # neoasaphus, triarthrus in head_check). Stretch the opened silhouette back to its full half-width instead.
            if good.any() and not good[-1]:
                xm = float(axb[good].max())
                if xm > 0.5 * RO:
                    yfb = np.interp(axb, axb[good] * (RO / xm), yfb[good]); yrb = np.interp(axb, axb[good] * (RO / xm), yrb[good])
            # 25 Oct 2026: the raster (0.1 mm) leaves a ripple of up to 0.2 mm with hundreds of curvature reversals along the
            # margin (head_check: agnostus, neoasaphus, phacops ...). The rounded outline is convex, so each edge takes its
            # convex envelope: the front edge its floor; the rear edge its ceiling when the rear margin does not bow
            # forward (notch 0) -- a bowed rear margin is concave by design and is left as drawn.
            if o["notch"] <= 1e-9 and not horn: yfb, yrb = _convex_columns(axb, yfb, yrb)   # the whole rounded silhouette is convex: its 2-D hull
            else: yfb = _convex_floor(axb, yfb)
        # ---- 7 Nov 2026: the user's dots. The outline drawn so far (sliders, rounding, or a trace) is the base; each dot sits at
        # its place u along it and pulls the margin to where it was dragged, easing back over dot_r either side. The pulled
        # outline is read back into columns (its front and rear edge at each x), so everything below -- the surface, the
        # brim, the fields, the genal piece -- follows it.
        ring0, u_ring = _ring(axb, yfb, yrb); dots = o.get("dots") or []; Nr = _normals(ring0)
        self.dot_at = dot_targets(ring0, u_ring, Nr, [[u, dx * RO, dy * RO] for u, dx, dy in dots]) if dots else []
        Lring = float(np.hypot(*np.diff(ring0, axis=0).T).sum())
        dd = dot_field(self.dot_at, o["dot_r"], u_ring, length=Lring)
        if dots:                                                   # the margin may not fold over itself: |d'| <= 0.7 along it
            dd = dd * (1 - smoothstep(0.93, 1.0, u_ring))          # ... and meets the axis at the joint flat (no cusp in the mirrored margin)
            ds = 0.7 * Lring / (len(u_ring) - 1)
            for k in range(1, len(dd)): dd[k] = np.clip(dd[k], dd[k - 1] - ds, dd[k - 1] + ds)
            for k in range(len(dd) - 2, -1, -1): dd[k] = np.clip(dd[k], dd[k + 1] - ds, dd[k + 1] + ds)
        Dr = dd[:, None] * Nr
        ring = ring0 + Dr
        self.ring0, self.u_ring, self.ring_D = ring0, u_ring, Dr; self._T_ring0 = cKDTree(ring0)
        if dots:
            ring[:, 0] = np.maximum(ring[:, 0], 0.0)
            reach = float(ring[:, 0].max())
            if abs(reach - RO) > 0.002 * RO:                       # the head's half-width is the width handle's: the dots set its shape
                ring[:, 0] *= RO / reach
                self.notes.append(f"the dots made the head {'wider' if reach > RO else 'narrower'} at its widest; it is drawn at the width the width handle sets")
            poly = np.r_[[[0.0, ring[0, 1]]], ring, [[0.0, ring[-1, 1]]]]
            yf_d, yr_d = _columns_of(poly, axb); ok_d = np.isfinite(yf_d) & np.isfinite(yr_d)
            if ok_d.sum() > 10:
                from scipy.ndimage import gaussian_filter1d
                yfb = gaussian_filter1d(np.interp(axb, axb[ok_d], yf_d[ok_d]), 1.5, mode="nearest")
                yrb = gaussian_filter1d(np.interp(axb, axb[ok_d], yr_d[ok_d]), 1.5, mode="nearest")
                yrb = np.maximum(yrb, yfb + 0.12)
        self.ring = ring
        # the front's median features go on after the rounding (it would erase anything this narrow)
        if tr is None and o["front_dent"] > 0: yfb = yfb + o["front_dent"] * L1 * np.exp(-(axb / (0.30 * RO)) ** 2)
        if o["front_point"] > 0:
            pw = max(o["point_w"], 0.03) * RO
            yfb = yfb - o["front_point"] * L1 * np.exp(-((axb - o["point_fork"] * RO) / pw) ** 2)
        # ---- the genal piece (1 Nov 2026, anatomy/genal.py): no longer drawn into the columns (which let it run only outward
        # of its root), but a solid of its own: a capsule from the root aimed anywhere, an optional kink, a tip from pointed to
        # round, and a web from the rear margin to the tip. Its root is where the spine left before: the genal angle, or moved
        # along the margin by genal_at.
        self.piece = None
        if horn:
            if not moved: x0, y0 = xg - 0.5 * w0, yg; inward = (-0.6, -0.8)
            elif at > 0:                                           # inward along the rear margin
                x0 = xg * (1 - 0.85 * at); y0 = float(np.interp(x0, axb, yrb)) - 0.3 * w0; inward = (0.0, -1.0)
            else:                                                  # forward along the side, then round the front
                y0 = yg + at * (yg - (yw - 0.8 * Lf))
                if y0 >= yw: x0 = RO - (RO - xg) * ((y0 - yw) / max(yg - yw, 1e-6)) ** 2
                else: x0 = RO * (1 - fl) * np.clip(1 - ((yw - y0) / Lf) ** nF, 0, 1) ** (1 / nF) + fl * RO
                x0 = x0 - 0.3 * w0; inward = (-1.0, 0.0)
            if at <= 0.01:                                         # a dot near the genal angle carries the spine's root with it
                dxy = self.dot_shift(x0, y0); x0, y0 = x0 + dxy[0], y0 + dxy[1]
            anchor = None; web = float(o["genal_web"])
            if web > 0.01:
                xa = x0 - web * (x0 - 0.12 * RO)
                if body_reach is not None:                             # 9 Nov 2026: the web stays BESIDE the thorax -- an anchor inside
                    xa = max(xa, min(float(body_reach) + 1.0, x0 - 0.2))   # the segments' reach put the web over the body, where the kit
                anchor = (xa, float(np.interp(xa, axb, yrb)))          # lifts it onto the segments (a flat shelf at their height)
            self.piece = GN.Piece((x0, y0), Lh, w0, o["genal_angle"], kink=o["genal_kink"], kink_angle=o["genal_kink_angle"],
                                  tip=o["genal_tip"], web=anchor, web_round=o["genal_web_round"], hw_min=0.35, inward=inward, organic=organic)
            self.notes += self.piece.notes
        self.horn, self.x_in, self.yg, self.xg, self.yw = horn, x_in, yg, xg, yw
        gd = self.dot_shift(xg, yg); self.xg_d, self.yg_d = xg + gd[0], yg + gd[1]   # the genal angle where the dots put it (its handle)
        Xmax = self.Xmax = float(RO)                               # the columns hold the head proper; the genal piece is its own solid
        ax = self.ax = np.linspace(0, Xmax, n); body = ax <= RO
        yf = np.where(body, np.interp(ax, axb, yfb), np.inf); yr = np.where(body, np.interp(ax, axb, yrb), -np.inf)
        self.body_yf, self.body_yr = yf.copy(), yr.copy()         # the head without its spine (fields: what is "spine")
        rear_is_outer = body & (ax > xg)
        ok_c = np.isfinite(yf) & np.isfinite(yr) & (yr > yf)      # empty columns (past the tip) close to a point
        with np.errstate(invalid="ignore"): mid_all = 0.5 * (yf + yr)
        mid_c = np.interp(ax, ax[ok_c], mid_all[ok_c])
        yf = np.where(ok_c, yf, mid_c); yr = np.where(ok_c, yr, mid_c)
        yr = np.maximum(yr, yf + 0.12)                             # never a zero-length column (the solid would not close)
        self.yf, self.yr = yf, yr
        # ---- margins as point sets: outer (front, side, the spine's outer edge) and rear (notch, the spine's inner edge)
        front = _dense(np.c_[ax, yf]); edge = _dense(np.array([[Xmax, yf[-1]], [Xmax, yr[-1]]]))
        rear = np.c_[ax, yr]; ro = rear_is_outer
        segs_o, segs_i = [front, edge], []
        k = 0
        while k < n:                                               # runs of the rear curve by which margin they belong to
            e = k
            while e + 1 < n and ro[e + 1] == ro[k]: e += 1
            run = _dense(rear[k:min(e + 2, n)]) if e > k or e + 1 < n else rear[k:k + 1]
            (segs_o if ro[k] else segs_i).append(run); k = e + 1
        out = np.vstack(segs_o); self.T_out = cKDTree(out)
        self.arc = np.r_[0, np.cumsum(np.hypot(*np.diff(out, axis=0).T))]
        inn = np.vstack(segs_i) if segs_i else np.array([[0.0, 1e4]]); self.T_in = cKDTree(inn)
        self.Lh = Lh
        # 9 Nov 2026 -- ONE SURFACE for a brimmed head: the brim is measured from ONE outer margin, the head's own running on
        # down the genal piece's outer edge to its tip, so the plate and the piece compute the same height where they meet
        # (they are separate solids, unioned: two surfaces that disagree there leave creases, ledges and steps). The piece's
        # inner edge is kept too: its section rounds off there.
        self.T_oc = self.T_pi = None
        if o["brim_w"] > 0 and self.piece is not None:
            outer, inner = [], []
            for nm_, ofn, _ in self.piece.bands:
                if nm_ in ("root", "web"): continue
                U_ = np.linspace(-1, 1, 241)
                e1 = np.array([ofn(u_, 1.0) for u_ in U_]); e0 = np.array([ofn(u_, 0.0) for u_ in U_])
                if np.abs(e0[:, 0]).mean() > np.abs(e1[:, 0]).mean(): e0, e1 = e1, e0       # the outer edge is the one farther out
                outer.append(e1); inner.append(e0)
            if outer:
                po = _dense(np.vstack(outer)); pi = _dense(np.vstack(inner))
                po = np.c_[np.abs(po[:, 0]), po[:, 1]]; pi = np.c_[np.abs(pi[:, 0]), pi[:, 1]]
                self.T_oc = cKDTree(np.vstack([out, po])); self.T_pi = cKDTree(pi)

    def dot_shift(self, x, y):
        """How far the dots moved the margin nearest (x, y) (scale 1): (0, 0) without dots."""
        if not (self.o.get("dots")): return np.zeros(2)
        return self.ring_D[self._T_ring0.query([float(x), float(y)])[1]]

    def edges(self, x):
        a = np.abs(np.asarray(x, float)); return np.interp(a, self.ax, self.yf), np.interp(a, self.ax, self.yr)

    def fields(self, ax, y):
        """What head_crescent._z_base needs to know about this outline at (|x|, y), scale 1."""
        q = np.column_stack([np.ravel(ax), np.ravel(y)]); sh = np.shape(ax)
        do, io = self.T_out.query(q); dn, _ = self.T_in.query(q)
        fill = min(self.o["dome_fill"], 1 - 0.9 * self.o["brim_w"]) if self.o["brim_w"] > 0 else self.o["dome_fill"]
        g = dict(do=do.reshape(sh), dn=dn.reshape(sh), arc=self.arc[io].reshape(sh), dome_fill=fill, horn_h=self.o["horn_h"])
        g.update(behind=np.zeros(sh), taper=np.ones(sh), hw=np.ones(sh))      # 1 Nov 2026: the genal piece has its own surface (piece_z)
        g["front"] = 1 - g["behind"]
        return g

    def z(self, x, y, face=None):
        """The surface at scale 1 (relief 10): the crescent head's, on this outline. Features sit at the same
        FRACTIONS of the head's length as on the crescent (the glabella, the eyes, the dome's centre)."""
        ax = np.abs(np.asarray(x, float)); y = np.broadcast_to(np.asarray(y, float), ax.shape)
        Yh = HC.Y0 + y * (HC.LC / self.L1)
        G = self.fields(ax, y); o = self.o
        z = HC._z_base(ax, Yh, face, geom=G)
        if o["brim_w"] > 0:                                        # the brim: its own surface between the margin and the vault
            bw = o["brim_w"] * RO; do = G["do"]; fr = G["front"]; c = o["brim_curve"]
            corner = np.ones_like(do)
            if self.T_oc is not None:                              # 9 Nov 2026: one margin, head and genal piece (see __init__)
                do = self.T_oc.query(np.column_stack([np.ravel(ax), np.ravel(y)]))[0].reshape(np.shape(ax))
                pd0 = 0.88 * bw / max(int(o["brim_pits"]), 1)
                # 1 on the head, 0 from the genal corner back: the whole rear lobe a crescent margin (notch) sweeps back into the
                # piece's inner edge carries the piece's section, not the vault's rim and the posterior furrow (they crossed the
                # piece's surface there in an X-shaped crease)
                corner = 1 - smoothstep(self.yg - 2.5 * pd0, self.yg + 0.5 * pd0, y) * smoothstep(0.42 * RO, 0.6 * RO, ax)
            t = np.clip(do / bw, 0, 1)
            prof = t ** (1.0 / (1.0 + c)) if c >= 0 else t ** (1.0 - c)
            hb = o["brim_rise"] * HC.RELIEF; edge = 0.55 * HC.RIM
            zb = edge + hb * prof
            if int(o["brim_pits"]) > 0:                            # the pits stop short of the genal corner (the piece carries none)
                rows = int(o["brim_pits"]); pd = 0.88 * bw / rows; uu = G["do"] / pd; jr = np.clip(np.floor(uu), 0, rows - 1)   # rows and arc from the SAME margin
                s_ = G["arc"] / pd + 0.5 * np.mod(jr, 2)
                zb = zb - 0.5 * corner * np.exp(-(((uu - (jr + 0.5)) ** 2 + (s_ - np.round(s_)) ** 2) * pd ** 2) / (0.30 * pd) ** 2) * (G["do"] > 0.06 * bw) * (G["do"] < 0.94 * bw)
            w = (1 - smoothstep(0.92 * bw, 1.08 * bw, do)) * fr
            z = (1 - w) * (z + hb * fr) + w * zb                   # the vault stands on the brim's inner edge
            z = z + corner * o["brim_roll"] * HC.RELIEF * np.exp(-((do - bw) / (0.28 * bw)) ** 2) * fr
            if self.T_oc is not None:                              # at the genal corner the plate takes the piece's own section, so
                zp = self._piece_section(ax, y)                    # the posterior border's furrow and rim do not cross the brim there
                z = corner * z + (1 - corner) * zp                 # (they made an X-shaped crease and a seam line at the root)
        return z + HC._detail(ax, Yh, face) if HC.has_detail(face) else z

    def _piece_section(self, x, y):
        """A brimmed head's genal piece (9 Nov 2026): the brim's profile from the combined outer margin (head + piece; see
        __init__), with no pits and no girder roll, rounded off toward the piece's inner edge by a quarter round over 0.4 bw
        (less where the blade is narrower). The plate takes this same section at the genal corner (z), so they are one."""
        o = self.o; bw = o["brim_w"] * RO; c = o["brim_curve"]
        q = np.column_stack([np.abs(np.ravel(x)), np.ravel(y)])
        d_o = self.T_oc.query(q)[0].reshape(np.shape(x)); d_i = self.T_pi.query(q)[0].reshape(np.shape(x))
        t = np.clip(d_o / bw, 0, 1)
        prof = t ** (1.0 / (1.0 + c)) if c >= 0 else t ** (1.0 - c)
        hb = o["brim_rise"] * HC.RELIEF; edge = 0.55 * HC.RIM
        zb = edge + hb * prof
        r_in = np.maximum(np.minimum(0.4 * bw, 0.45 * (d_o + d_i)), 1e-3)
        f = np.sqrt(np.clip(1 - (1 - np.clip(d_i / r_in, 0, 1)) ** 2, 0, 1))
        return edge + (zb - edge) * f

    def piece_z(self, name, s, v, x, y, face=None):
        """The genal piece's surface at scale 1 (relief 10), on band `name` at position s along it (0 root .. 1 tip) and v
        across (0 .. 1): a rounded ridge, as the column spine was (a quarter-ellipse each side from the shield's edge floor,
        0.6 rim, to a crest horn_h x relief tapering along the length), rising out of the head's own surface at the root.
        The web is a low plate at the edge floor; the root stub follows the head's surface just under it."""
        s = np.asarray(s, float); v = np.asarray(v, float)
        e_ = 0.6 * float(HC.face_of(face).get("rim", HC.RIM))                # 9 Nov 2026: the shield's edge floor follows the face's
        if name in ("root", "web") and not self.o["brim_w"] > 0: return np.full(np.shape(s), e_ + 0.15)   # rim (it was HC.RIM: a 0.9-unit plinth under every spine,
        gs = float(self.o.get("genal_sharp", 0.0))                           # the vertical walls that made the spines read as slabs)
        crest = self.o["horn_h"] * HC.RELIEF * (1 - (0.55 + 0.42 * gs) * np.clip(s, 0, 1))
        p = self.piece; hw_s = p.hw(np.clip(s, 0, 1) * p.Ls)                 # the band's half-width here (plan units)
        cap = e_ + float(self.o.get("horn_aspect", 1.0)) * 2.0 * hw_s / max(getattr(self, "zk", 1.0), 1e-6)   # 9 Nov 2026: no taller
        crest = np.minimum(crest, cap)                                        # than horn_aspect x its width, in real mm
        t_ = np.clip(1 - np.abs(2 * v - 1), 0, 1)
        sect = (1 - gs) * np.sqrt(np.clip(1 - (1 - t_) ** 2, 0, 1)) + gs * t_ ** 0.75   # round ridge .. sharp (triangular) spine
        if self.o["brim_w"] > 0 and self.T_oc is not None:                    # 9 Nov 2026: a brimmed head's genal piece IS the brim running
            return np.maximum(self._piece_section(x, y), e_)                  # on, on ONE surface with the plate (_piece_section)
        ridge = e_ + (crest - e_) * sect
        w = smoothstep(0.0, min(1.0, 1.5 * p.W / max(p.L, 1e-6)), s)   # the ridge rises out of the margin
        return e_ + w * (ridge - e_)


def plan(P, plans, K, outline=None, face=None, z_thorax=None):
    """A head plan for body.build, model mm, head frame (the rear joint plane at y = 0, the rear margin on the axis one
    shingle flap behind it, as the classic head lies over segment 0)."""
    from anatomy.common import pitch
    import mesh as M
    from anatomy import organic as OR
    o_w = float(outline_of(outline)["width"])
    sh = Shape(outline, organic=OR.k_of(P), body_reach=RO / max(o_w, 1e-6))   # the thorax's half-width in the head's plan units (= w_seg / k)
    o = sh.o
    S1 = plans[1][2]; w_seg = max(abs(S1["outline"](u, v)[0]) for u in np.linspace(-1, 1, 201) for v in (0.0, 0.5, 1.0))
    k = o["width"] * w_seg / RO
    if z_thorax is None:
        z_thorax = max(float(M.sample_grid(Sk["outline"], Sk["zfun"], 61, 31)[2].max()) for _, _, Sk, _ in (plans[1:-1] or plans[1:]))
    gx, gy = np.meshgrid(np.linspace(0, 0.6 * RO, 40), np.linspace(-sh.L1, 0, 60))
    base = {kk: vv for kk, vv in (face or {}).items() if kk in ("eye_size", "eye_pos", "eye_height", "eye_lat", "glab_inflate", "glab_rise", "glab_front", "glab_lobes")}
    z_ref = float(sh.z(gx, gy, dict(base, eye_size=0.0)).max())             # the unsculpted crown: a boss or a tubercle does not set the scale
    kz = 0.85 * z_thorax * float(P.get("headRelief", 1.0)) * o["height"] / z_ref          # body.HEAD_HEIGHT
    sh.zk = kz / max(k, 1e-9)                                  # 9 Nov 2026: z units per plan unit, for the spine section (piece_z)
    flap = max(P["overlap"] * pitch(P) - 2.0, 1.0)
    def outline_fn(u, v):
        x = u * sh.Xmax; yf, yr = sh.edges(abs(x)); return k * x, flap + k * float(yr - v * (yr - yf))
    def zfun(x, y): return kz * sh.z(np.asarray(x, float) / k, (np.asarray(y, float) - flap) / k, face)
    F_ = HC.face_of(face); E = HC._eye(face); f_eye = (HC.Y0 - E["ye"]) / HC.LC
    eye = dict(xe=k * E["xe"], ye=flap - k * f_eye * sh.L1, eR=k * E["eR"], eH=k * E["eH"], arc_deg=float(P["eyeArc"]),
               exponent=float(P.get("eyeProfile", 4.0)), blind=F_["eye_size"] <= 0.01)
    def xmax(y):
        yy = (np.asarray(y, float) - flap) / k
        return k * np.where(yy < sh.yw, RO * np.clip(1 - (np.clip(sh.yw - yy, 0, None) / (sh.L1 + sh.yw)) ** o["front_exp"], 0, 1) ** (1 / o["front_exp"]), RO)
    return dict(outline=outline_fn, zfun=zfun, xmax=xmax, t=P["wall"] * P["headWall"], c=P["clearance"], h=kz * HC.RELIEF, kz=kz,
                Lc=k * sh.L1, wh=k * RO, a=k * F_["axis_frac"] * RO, margin=kz * HC.RIM, Lg=0.0, u_a=F_["axis_frac"] * RO / sh.Xmax,
                eye=eye, scale=k, shape=sh, flap=flap)


# ---------------------------------------------------------------- the diagram (3 Oct 2026)
# The site's sketch (plan + elevation, white hairlines on black) draws the classic head from its own formula in
# web/index.html. For a shaped head the sketch must be the SAME curves the solid is built from, so this returns
# them: every line below is read off Shape / head_crescent, nothing is redrawn by hand. Units: fractions of the
# head's half-width (x) and the same unit along y and z, y = 0 at the rear margin on the axis, head toward -y.
HANDLES = {            # diagram handle -> the outline / face keys it drags (what the site would bind)
    "front":   ("length", "front_exp", "front_flat", "front_dent", "rim"),          # rim: the margin's height (9 Nov 2026)
    "process": ("front_point", "point_w", "point_fork"),
    "widest":  ("width", "widest", "side", "rear_taper"),
    "angle":   ("corner",),
    "spine":   ("genal", "genal_angle", "genal_w", "genal_tip", "genal_sharp", "horn_aspect"),
    "kink":    ("genal_kink", "genal_kink_angle"),
    "web":     ("genal_web", "genal_web_round"),
    "root":    ("genal_at",),
    "notch":   ("notch", "notch_exp"),
    "glabella": ("glab_len", "glabInflate", "glab_bulge", "glab_boss"),
    "cap":     ("front_cap", "front_furrow"),                  # 9 Nov 2026: the closed capsule's front cap (shoulder grip)
    "eye":     ("eyePos", "eyeLat", "eyeSize", "eye_len"),
    "brim":    ("brim_w", "brim_rise", "brim_curve", "brim_roll", "dome_fill"),
    "crown":   ("height", "glabRise", "glab_round"),
}

def plan_outline(sh):
    """The head's plan outline at scale 1, right half: from the front on the axis round to the rear margin on the axis --
    the head proper (its columns) with the genal piece unioned on (1 Nov 2026). A piece aimed in across the axis is cut at it."""
    body = np.r_[np.c_[sh.ax, sh.yf], np.c_[sh.ax, sh.yr][::-1]]
    if sh.piece is None: return body
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    B = Polygon(np.r_[[[0.0, sh.yf[0]]], body, [[0.0, sh.yr[0]]]]).buffer(0)
    U = unary_union([B, sh.piece.polygon()]).intersection(box(0.0, -1e4, 1e4, 1e4)).simplify(0.01, preserve_topology=True)
    if U.geom_type != "Polygon": U = max(getattr(U, "geoms", [U]), key=lambda g: g.area)
    ring = np.asarray(U.exterior.coords)[:-1]
    on = ring[:, 0] < 1e-6
    if not on.any(): return np.vstack([ring, ring[:1]])
    # rotate so the walk starts on the axis at the front and ends on it at the rear, dropping the axis itself
    n_ = len(ring); idx = np.where(on)[0]
    front_i = idx[np.argmin(ring[idx, 1])]; rear_i = idx[np.argmax(ring[idx, 1])]
    def walk(a, b, step):
        out = [a]; k = a
        while k != b: k = (k + step) % n_; out.append(k)
        return out
    w1 = walk(front_i, rear_i, 1); w2 = walk(front_i, rear_i, -1)
    score = lambda w: (sum(1 for k in w[1:-1] if on[k]), -len(w))   # round the outside: off the axis, the long way
    w = min((w1, w2), key=score)
    return ring[w]

def diagram(outline=None, face=None, n=260):
    """Polylines of one head for the sketch: plan (right half; mirror it) and elevation (the section on the axis and
    the highest profile). Returns dict(outline, glabella, eye, brim, boss, section, profile, handles, size)."""
    sh = Shape(outline); o = sh.o; F_ = HC.face_of(face); u = 1.0 / RO
    ax = sh.ax; plan_ = plan_outline(sh) * u
    gl = float(F_["glab_len"]); f = np.linspace(0, min(0.92 * gl, 1.0), n)
    gw = HC._glab_half(f / gl, face) * (1 - smoothstep(0.80, 0.92, f / gl) * 0.0)
    nose = np.clip(1 - np.clip((f / gl - 0.70) / 0.30, 0, 1) ** F_["glab_front"], 0, 1) ** (1 / F_["glab_front"])
    glab = np.c_[gw, -f * sh.L1] * u
    cap = None                                                     # 9 Nov 2026: with front_furrow the glabella is a CLOSED capsule --
    if float(F_.get("front_furrow", 0.0)) > 0.005:                 # flanks to f0, then an elliptical cap (half-width g(f0), length
        kc = float(F_.get("front_cap", 1.0)); f0 = 0.85            # front_cap x g(f0)) ending at the glabella's front, as in _z
        for _ in range(3): f0 = max(1.0 - kc * float(HC._glab_half(f0, face)) / (gl * HC.LC), 0.5)
        g0 = float(HC._glab_half(f0, face)); cl = kc * g0 * (sh.L1 / HC.LC)      # the cap's length in the sketch's y (mm)
        ff = np.linspace(0, f0 * gl, n // 2); fw = HC._glab_half(ff / gl, face)
        t = np.linspace(0, np.pi / 2, n // 2); yc = f0 * gl * sh.L1
        glab = np.r_[np.c_[fw, -ff * sh.L1], np.c_[g0 * np.cos(t), -(yc + cl * np.sin(t))]] * u
        cap = (g0 * u, -yc * u)
    E = HC._eye(face); eye = None
    if F_["eye_size"] > 0.01:
        t = np.linspace(0, 2 * np.pi, 49); fe = (HC.Y0 - E["ye"]) / HC.LC
        eye = np.c_[E["xe"] + E["eR"] * np.cos(t), -fe * sh.L1 + E["eR"] * F_["eye_len"] * (sh.L1 / HC.LC) * np.sin(t)] * u
    brim = None
    b = max(F_["brim"], o["brim_w"], (1 - o["dome_fill"]) if o["dome_fill"] < 0.75 else 0.0)
    if b > 0.02:                                                   # the brim's inner edge: the outer margin moved in
        keep = ax <= RO * (1 - b) ; xs = ax[keep]
        brim = np.c_[xs, np.interp(xs / (1 - b), ax, sh.yf) * (1 - b) + sh.yw * b] * u
    boss = None
    if F_["glab_boss"] > 0:
        t = np.linspace(0, 2 * np.pi, 49); Rb = F_["glab_boss"] * RO; fc = 0.90 * gl - Rb / HC.LC
        boss = np.c_[Rb * np.cos(t), -fc * sh.L1 + Rb * (sh.L1 / HC.LC) * np.sin(t)] * u
    ys = np.linspace(sh.yf.min(), sh.yr.max(), n)
    sec = sh.z(np.zeros(n), ys, face); sec = np.where((ys >= np.interp(0, ax, sh.yf)) & (ys <= np.interp(0, ax, sh.yr)), sec, 0.0)
    X, Y = np.meshgrid(np.linspace(0, sh.Xmax, 90), ys); yf_, yr_ = sh.edges(X)
    prof = np.where((Y >= yf_) & (Y <= yr_), sh.z(X, Y, face), 0.0).max(1)
    kz = 0.85 * o["height"] * 0.42 / max(sec.max(), 1e-6)         # drawn at the model's usual head height / half-width
    H = dict(front=(0.0, float(sh.yf[0]) * u), widest=(1.0, sh.yw * u), angle=(sh.xg_d * u, sh.yg_d * u),
             notch=(0.5 * sh.x_in * u, float(np.interp(0.5 * sh.x_in, ax, sh.yr)) * u),
             glabella=(0.0, -min(0.92 * gl, 1.0) * sh.L1 * u) if cap is None else (0.0, float(glab[-1, 1])))
    if cap is not None: H["cap"] = cap                              # the cap's shoulder: drag fore-aft for front_cap
    if o.get("trace") is not None:                                 # a traced margin: the front, side and rear handles have nothing to drag
        for k in ("front", "widest", "angle", "notch"): H.pop(k, None)
    if sh.piece is not None:
        pc = sh.piece; H["spine"] = tuple(float(c) * u for c in pc.tip); H["root"] = tuple(float(c) * u for c in pc.root)
        H["kink"] = tuple(float(c) * u for c in pc.kink)                               # the kink, or the middle of a straight spine
        A = pc.anchor if pc.anchor is not None else (pc.root[0] - max(0.9 * pc.W, 1.5), float(np.interp(pc.root[0] - max(0.9 * pc.W, 1.5), sh.ax, sh.yr)))
        H["web"] = (float(A[0]) * u, float(A[1]) * u)                                    # the web's anchor on the rear margin (beside the root: none yet)
    if o["front_point"] > 0: H["process"] = (o["point_fork"], float(sh.yf[int(np.argmin(sh.yf))]) * u)
    if b > 0.02: H["brim"] = (0.7 * (1 - b), float(np.interp(0.7 * RO, ax, sh.yf)) * (1 - b) * u + sh.yw * b * u)
    if eye is not None: H["eye"] = (E["xe"] * u, float(eye[:, 1].mean()))
    reach = float(np.abs(plan_[:, 0]).max()) if len(plan_) else sh.Xmax * u
    # 7 Nov 2026: the margin the user's dots sit on. rim: [x, y, u, dx, dy] every few points along the pulled outline (u where
    # along it, (dx, dy) how far the dots have moved it there) -- a click on it adds a dot that keeps the outline where it is;
    # dots: each dot's position, in the order given.
    rim = np.c_[sh.ring * u, sh.u_ring, (sh.ring - sh.ring0) * u][::4]
    dpos = [sh.ring[int(np.argmin(np.abs(sh.u_ring - ua)))] * u for ua, _ in sh.dot_at]   # where each dot landed, on the pulled outline
    return dict(outline=plan_, glabella=glab, eye=eye, brim=brim, boss=boss, section=np.c_[ys * u, sec * kz], profile=np.c_[ys * u, prof * kz],
                handles=H, size=dict(length=sh.L1 * u, halfwidth=max(sh.Xmax * u, reach), width_vs_thorax=o["width"]), notes=list(sh.notes),
                trace=(np.array(o["trace"], float) if o.get("trace") is not None else None),
                rim=rim, dots=(np.array(dpos) if dpos else None))
