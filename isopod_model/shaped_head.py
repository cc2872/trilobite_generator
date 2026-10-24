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
    genal_tip=0.12,    # tip width / root width (1 = a parallel band with a round end; 0.1 = drawn to a point)
    genal_spread=4.0,  # it leaves at this angle outward from straight back, deg
    genal_curve=0.0,   # and turns by this much along its length, deg (+ outward, - back inward)
    notch=0.048,       # the rear margin bows back between the genal angles: depth / axial length (0 straight across)
    notch_exp=2.0,     # 2 = an elliptical U; higher = straight across, then a late bend
    dome_fill=0.82,    # the vaulted part / the whole (0.6 leaves a broad flat brim: Harpetida)
    horn_h=0.32,       # the spine's ridge height / relief (a flat prolongation ~0.18)
    height=1.0,        # crown height, x the model's usual head height
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
CRESCENT_OUTLINE = dict(OUTLINE, length=1.08, front_exp=2.0, widest=0.08, genal=0.77, genal_w=0.167, genal_tip=1.0, genal_spread=0.0, notch=0.693)   # notch: 0.9 x the band length 0.77 (24 Oct: / axial length)
TRACE_MAX = 400     # points a traced outline may carry

OUTLINE_RANGE = dict(   # what the silhouette is drawn for (11 Oct 2026: a 0.2-long head crashed in interp, a 3 x 3 x genal-3 one printed 2.7 m)
    width=(0.5, 2.5), length=(0.4, 2.5), front_exp=(1.0, 8.0), front_point=(0.0, 1.5), widest=(0.0, 0.8), side=(0.0, 1.0), rear_taper=(0.0, 0.9),
    corner=(0.0, 0.5), genal=(0.0, 3.5), genal_w=(0.02, 1.0), genal_tip=(0.02, 1.5), genal_spread=(-60.0, 80.0), genal_curve=(-90.0, 90.0),
    notch=(0.0, 1.0), notch_exp=(1.0, 8.0), dome_fill=(0.3, 1.0), horn_h=(0.0, 1.0), height=(0.2, 3.0), genal_at=(-1.0, 1.0),
    front_flat=(0.0, 0.9), front_dent=(0.0, 0.5), point_w=(0.02, 1.0), point_fork=(0.0, 0.9), brim_w=(0.0, 1.0), brim_rise=(-0.5, 1.0),
    brim_curve=(-1.0, 1.0), brim_roll=(0.0, 0.5), brim_pits=(0, 10))

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

def outline_of(o=None):
    f = dict(OUTLINE, **(o or {})); tr = f.pop("trace", None); bad = set(f) - set(OUTLINE)
    if bad: raise KeyError(f"unknown outline keys {sorted(bad)} (have {sorted(OUTLINE)} and 'trace')")
    for k, v in f.items():
        try: v = float(v)
        except (TypeError, ValueError): raise ValueError(f"outline {k} is not a number")
        if not math.isfinite(v): raise ValueError(f"outline {k} is not finite")
        lo, hi = OUTLINE_RANGE.get(k, (-10.0 * max(1.0, abs(OUTLINE[k])), 10.0 * max(1.0, abs(OUTLINE[k]))))
        if not lo <= v <= hi: raise ValueError(f"outline {k} = {v:g} is outside {lo:g}..{hi:g}")
        f[k] = v
    tr = trace_of(tr)
    if tr is not None: f["trace"] = tr
    return f

def _dense(pts, step=0.15):
    """Points along a polyline at <= step spacing (jumps between columns become their vertical edges)."""
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step)); t = np.linspace(0, 1, n + 1)[1:, None]; out.append(a + t * (b - a))
    return np.vstack(out)

class Shape:
    def __init__(self, outline=None, n=1601):
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
        # the front's median features go on after the rounding (it would erase anything this narrow)
        if tr is None and o["front_dent"] > 0: yfb = yfb + o["front_dent"] * L1 * np.exp(-(axb / (0.30 * RO)) ** 2)
        if o["front_point"] > 0:
            pw = max(o["point_w"], 0.03) * RO
            yfb = yfb - o["front_point"] * L1 * np.exp(-((axb - o["point_fork"] * RO) / pw) ** 2)
        # ---- the spine: discs along a centreline that leaves the margin and may spread and curve
        if horn:
            if not moved: x0, y0 = xg - 0.5 * w0, yg
            elif at > 0:                                           # inward along the rear margin
                x0 = xg * (1 - 0.85 * at); y0 = float(np.interp(x0, axb, yrb)) - 0.3 * w0
            else:                                                  # forward along the side, then round the front
                y0 = yg + at * (yg - (yw - 0.8 * Lf))
                if y0 >= yw: x0 = RO - (RO - xg) * ((y0 - yw) / max(yg - yw, 1e-6)) ** 2
                else: x0 = RO * (1 - fl) * np.clip(1 - ((yw - y0) / Lf) ** nF, 0, 1) ** (1 / nF) + fl * RO
                x0 = x0 - 0.3 * w0
            S = np.linspace(0, 1, 241); th = np.radians(o["genal_spread"] + o["genal_curve"] * S)
            dsx, dsy = np.sin(th) * Lh / 240, np.cos(th) * Lh / 240
            xc = x0 + np.r_[0, np.cumsum(dsx[1:])]; yc = y0 + np.r_[0, np.cumsum(dsy[1:])]
            r = 0.5 * w0 * (1 - (1 - o["genal_tip"]) * S); r = np.maximum(r, 0.35)
            # 24 Oct 2026: the band lives in the silhouette's columns (one front and one rear edge per x), so it can only
            # run OUTWARD of its own root: a centreline that turns in over the head (negative spread, a curve that hooks
            # back) made the head's rear margin the band's upper edge and the solid filled down to the band (the studio's
            # spines through the pygidium, 23 Oct). The band is cut where its CENTRELINE would cross its root's inner
            # edge (a drift of up to the root half-width is tolerated: harpes' -6 deg curl always had that sliver);
            # a spine that turns under or over the head is the genal HORN (P.genalHorn*, a solid of its own).
            bad = np.where(xc < x0 - r[0])[0]
            if len(bad) and bad[0] < len(S) - 1:
                cut = max(int(bad[0]), 2)
                self.notes.append(f"genal band cut at {S[cut]:.0%} of its length: it would turn in over the head (use the genal horn for a curved or inward spine)")
                S, xc, yc, r = S[:cut + 1] / S[cut], xc[:cut + 1], yc[:cut + 1], r[:cut + 1]; Lh = float(np.hypot(xc[-1] - x0, yc[-1] - y0))
        else:
            S = xc = yc = r = None
        self.horn, self.x_in, self.yg, self.xg, self.yw = horn, x_in, yg, xg, yw
        Xmax = self.Xmax = float(max(RO, (xc + r).max() if horn else RO))
        ax = self.ax = np.linspace(0, Xmax, n); body = ax <= RO
        yf = np.where(body, np.interp(ax, axb, yfb), np.inf); yr = np.where(body, np.interp(ax, axb, yrb), -np.inf)
        self.body_yf, self.body_yr = yf.copy(), yr.copy()         # the head without its spine (fields: what is "spine")
        rear_is_outer = body & (ax > xg)
        if horn:
            dx = ax[:, None] - xc[None, :]; ok = np.abs(dx) < r[None, :]
            hgt = np.sqrt(np.clip(r[None, :] ** 2 - dx ** 2, 0, None))
            top = np.where(ok, yc[None, :] + hgt, -np.inf); bot = np.where(ok, yc[None, :] - hgt, np.inf)
            j = top.argmax(1); hmax = top.max(1); hmin = bot.min(1); has = ok.any(1)
            use = has & (hmax > yr)
            rear_is_outer = np.where(use, ax >= xc[j], rear_is_outer)
            yr = np.where(use, hmax, yr); yf = np.where(has, np.minimum(yf, hmin), yf)
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
        if horn: self.T_c = cKDTree(np.c_[xc, yc]); self.S, self.r, self.Lh, self.xc, self.yc = S, r, Lh, xc, yc

    def edges(self, x):
        a = np.abs(np.asarray(x, float)); return np.interp(a, self.ax, self.yf), np.interp(a, self.ax, self.yr)

    def fields(self, ax, y):
        """What head_crescent._z_base needs to know about this outline at (|x|, y), scale 1."""
        q = np.column_stack([np.ravel(ax), np.ravel(y)]); sh = np.shape(ax)
        do, io = self.T_out.query(q); dn, _ = self.T_in.query(q)
        fill = min(self.o["dome_fill"], 1 - 0.9 * self.o["brim_w"]) if self.o["brim_w"] > 0 else self.o["dome_fill"]
        g = dict(do=do.reshape(sh), dn=dn.reshape(sh), arc=self.arc[io].reshape(sh), dome_fill=fill, horn_h=self.o["horn_h"])
        if self.horn:
            _, ic = self.T_c.query(q); s = self.S[ic].reshape(sh)
            byf = np.interp(ax, self.ax, self.body_yf); byr = np.interp(ax, self.ax, self.body_yr)   # +-inf past the head: spine
            with np.errstate(invalid="ignore"): d_out = np.nan_to_num(np.maximum(byf - y, y - byr), nan=1e3, posinf=1e3)
            beh = smoothstep(-1.5, 1.5, d_out) * smoothstep(self.x_in - 3.0, self.x_in, ax) if abs(self.o["genal_at"]) <= 0.01 else smoothstep(-0.5, 1.5, d_out)
            g.update(behind=beh, taper=1 - 0.55 * s, hw=self.r[ic].reshape(sh))
        else:
            g.update(behind=np.zeros(sh), taper=np.ones(sh), hw=np.ones(sh))
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
            bw = o["brim_w"] * RO; do = G["do"]; fr = G["front"]; t = np.clip(do / bw, 0, 1); c = o["brim_curve"]
            prof = t ** (1.0 / (1.0 + c)) if c >= 0 else t ** (1.0 - c)
            hb = o["brim_rise"] * HC.RELIEF; edge = 0.55 * HC.RIM
            zb = edge + hb * prof
            if int(o["brim_pits"]) > 0:
                rows = int(o["brim_pits"]); pd = 0.88 * bw / rows; uu = do / pd; jr = np.clip(np.floor(uu), 0, rows - 1)
                s_ = G["arc"] / pd + 0.5 * np.mod(jr, 2)
                zb = zb - 0.5 * np.exp(-(((uu - (jr + 0.5)) ** 2 + (s_ - np.round(s_)) ** 2) * pd ** 2) / (0.30 * pd) ** 2) * (do > 0.06 * bw) * (do < 0.94 * bw)
            w = (1 - smoothstep(0.92 * bw, 1.08 * bw, do)) * fr
            z = (1 - w) * (z + hb * fr) + w * zb                   # the vault stands on the brim's inner edge
            z = z + o["brim_roll"] * HC.RELIEF * np.exp(-((do - bw) / (0.28 * bw)) ** 2) * fr
        return z + HC._detail(ax, Yh, face) if HC.has_detail(face) else z


def plan(P, plans, K, outline=None, face=None, z_thorax=None):
    """A head plan for body.build, model mm, head frame (the rear joint plane at y = 0, the rear margin on the axis one
    shingle flap behind it, as the classic head lies over segment 0)."""
    from anatomy.common import pitch
    import mesh as M
    sh = Shape(outline); o = sh.o
    S1 = plans[1][2]; w_seg = max(abs(S1["outline"](u, v)[0]) for u in np.linspace(-1, 1, 201) for v in (0.0, 0.5, 1.0))
    k = o["width"] * w_seg / RO
    if z_thorax is None:
        z_thorax = max(float(M.sample_grid(Sk["outline"], Sk["zfun"], 61, 31)[2].max()) for _, _, Sk, _ in (plans[1:-1] or plans[1:]))
    gx, gy = np.meshgrid(np.linspace(0, 0.6 * RO, 40), np.linspace(-sh.L1, 0, 60))
    base = {kk: vv for kk, vv in (face or {}).items() if kk in ("eye_size", "eye_pos", "eye_height", "eye_lat", "glab_inflate", "glab_rise", "glab_front", "glab_lobes")}
    z_ref = float(sh.z(gx, gy, dict(base, eye_size=0.0)).max())             # the unsculpted crown: a boss or a tubercle does not set the scale
    kz = 0.85 * z_thorax * float(P.get("headRelief", 1.0)) * o["height"] / z_ref          # body.HEAD_HEIGHT
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
    return dict(outline=outline_fn, zfun=zfun, xmax=xmax, t=P["wall"] * P["headWall"], c=P["clearance"], h=kz * HC.RELIEF,
                Lc=k * sh.L1, wh=k * RO, a=k * F_["axis_frac"] * RO, margin=kz * HC.RIM, Lg=0.0, u_a=F_["axis_frac"] * RO / sh.Xmax,
                eye=eye, scale=k, shape=sh, flap=flap)


# ---------------------------------------------------------------- the diagram (3 Oct 2026)
# The site's sketch (plan + elevation, white hairlines on black) draws the classic head from its own formula in
# web/index.html. For a shaped head the sketch must be the SAME curves the solid is built from, so this returns
# them: every line below is read off Shape / head_crescent, nothing is redrawn by hand. Units: fractions of the
# head's half-width (x) and the same unit along y and z, y = 0 at the rear margin on the axis, head toward -y.
HANDLES = {            # diagram handle -> the outline / face keys it drags (what the site would bind)
    "front":   ("length", "front_exp", "front_flat", "front_dent"),
    "process": ("front_point", "point_w", "point_fork"),
    "widest":  ("width", "widest", "side", "rear_taper"),
    "angle":   ("corner",),
    "spine":   ("genal", "genal_spread", "genal_curve", "genal_w", "genal_tip"),
    "root":    ("genal_at",),
    "notch":   ("notch", "notch_exp"),
    "glabella": ("glab_len", "glabInflate", "glab_bulge", "glab_boss"),
    "eye":     ("eyePos", "eyeLat", "eyeSize", "eye_len"),
    "brim":    ("brim_w", "brim_rise", "brim_curve", "brim_roll", "dome_fill"),
    "crown":   ("height", "glabRise", "glab_round"),
}

def diagram(outline=None, face=None, n=260):
    """Polylines of one head for the sketch: plan (right half; mirror it) and elevation (the section on the axis and
    the highest profile). Returns dict(outline, glabella, eye, brim, boss, section, profile, handles, size)."""
    sh = Shape(outline); o = sh.o; F_ = HC.face_of(face); u = 1.0 / RO
    ax = sh.ax; plan_ = np.r_[np.c_[ax, sh.yf], np.c_[ax, sh.yr][::-1]] * u
    gl = float(F_["glab_len"]); f = np.linspace(0, min(0.92 * gl, 1.0), n)
    gw = HC._glab_half(f / gl, face) * (1 - smoothstep(0.80, 0.92, f / gl) * 0.0)
    nose = np.clip(1 - np.clip((f / gl - 0.70) / 0.30, 0, 1) ** F_["glab_front"], 0, 1) ** (1 / F_["glab_front"])
    glab = np.c_[gw, -f * sh.L1] * u
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
    H = dict(front=(0.0, float(sh.yf[0]) * u), widest=(1.0, sh.yw * u), angle=(sh.xg * u, sh.yg * u),
             notch=(0.5 * sh.x_in * u, float(np.interp(0.5 * sh.x_in, ax, sh.yr)) * u),
             glabella=(0.0, -min(0.92 * gl, 1.0) * sh.L1 * u))
    if o.get("trace") is not None:                                 # a traced margin: the front, side and rear handles have nothing to drag
        for k in ("front", "widest", "angle", "notch"): H.pop(k, None)
    if sh.horn:
        H["spine"] = (float(sh.xc[-1]) * u, float(sh.yc[-1]) * u); H["root"] = (float(sh.xc[0]) * u, float(sh.yc[0]) * u)
    if o["front_point"] > 0: H["process"] = (o["point_fork"], float(sh.yf[int(np.argmin(sh.yf))]) * u)
    if b > 0.02: H["brim"] = (0.7 * (1 - b), float(np.interp(0.7 * RO, ax, sh.yf)) * (1 - b) * u + sh.yw * b * u)
    if eye is not None: H["eye"] = (E["xe"] * u, float(eye[:, 1].mean()))
    return dict(outline=plan_, glabella=glab, eye=eye, brim=brim, boss=boss, section=np.c_[ys * u, sec * kz], profile=np.c_[ys * u, prof * kz],
                handles=H, size=dict(length=sh.L1 * u, halfwidth=sh.Xmax * u, width_vs_thorax=o["width"]), notes=list(sh.notes),
                trace=(np.array(o["trace"], float) if o.get("trace") is not None else None))
