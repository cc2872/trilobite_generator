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
OUTLINE = dict(
    width=1.15,        # head half-width / the first thorax segment's half-width
    length=1.08,       # axial length / half-width (semicircle 1.0; long parabola 1.5; short and transverse 0.7)
    front_exp=2.0,     # the front's curve: 2 round, 1.5 ogival / pointed, 3+ blunt and square-shouldered
    front_point=0.0,   # an anterior median point, / length (Dalmanites, Huntonia)
    widest=0.08,       # where the head is widest, from the rear margin, / length (agnostid: 0.45)
    side=0.0,          # straight side behind the widest line, / length (subparallel-sided heads)
    rear_taper=0.0,    # the side narrows back to the genal angle by this fraction of the half-width
    corner=0.12,       # radius of a rounded genal angle, / half-width (when there is no genal spine)
    genal=0.77,        # genal spine / prolongation length, / axial length (0 = none)
    genal_w=0.167,     # its width at the root, / half-width (harpetid prolongation 0.3; trinucleid needle 0.07)
    genal_tip=1.0,     # tip width / root width (1 = a parallel band with a round end; 0.1 = drawn to a point)
    genal_spread=0.0,  # it leaves at this angle outward from straight back, deg
    genal_curve=0.0,   # and turns by this much along its length, deg (+ outward, - back inward)
    notch=0.9,         # the rear margin between the spines: 0 straight across, 1 a U as deep as the spine is long
    notch_exp=2.0,     # 2 = an elliptical U; higher = straight across, then a late bend
    dome_fill=0.82,    # the vaulted part / the whole (0.6 leaves a broad flat brim: Harpetida)
    horn_h=0.32,       # the spine's ridge height / relief (a flat prolongation ~0.18)
    height=1.0,        # crown height, x the model's usual head height
)

def outline_of(o=None):
    f = dict(OUTLINE, **(o or {})); bad = set(f) - set(OUTLINE)
    if bad: raise KeyError(f"unknown outline keys {sorted(bad)} (have {sorted(OUTLINE)})")
    return f

def _dense(pts, step=0.15):
    """Points along a polyline at <= step spacing (jumps between columns become their vertical edges)."""
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step)); t = np.linspace(0, 1, n + 1)[1:, None]; out.append(a + t * (b - a))
    return np.vstack(out)

class Shape:
    def __init__(self, outline=None, n=1601):
        o = self.o = outline_of(outline)
        L1 = self.L1 = o["length"] * RO; yw = -o["widest"] * L1; Lf = L1 + yw; yg = yw + o["side"] * L1
        xg = RO * (1 - o["rear_taper"]); nF = o["front_exp"]
        Lh = o["genal"] * L1; w0 = o["genal_w"] * RO; horn = Lh > 0.5
        # ---- the genal spine: discs along a centreline that leaves the genal angle and may spread and curve
        if horn:
            S = np.linspace(0, 1, 241); th = np.radians(o["genal_spread"] + o["genal_curve"] * S)
            dsx, dsy = np.sin(th) * Lh / 240, np.cos(th) * Lh / 240
            xc = (xg - 0.5 * w0) + np.r_[0, np.cumsum(dsx[1:])]; yc = yg + np.r_[0, np.cumsum(dsy[1:])]
            r = 0.5 * w0 * (1 - (1 - o["genal_tip"]) * S); r = np.maximum(r, 0.35)
            x_in = xg - w0
        else:
            S = xc = yc = r = None; x_in = xg
        self.horn, self.x_in, self.yg, self.xg, self.yw = horn, x_in, yg, xg, yw
        Xmax = self.Xmax = float(max(RO, (xc + r).max() if horn else RO))
        ax = self.ax = np.linspace(0, Xmax, n)
        # ---- the head proper, column by column
        body = ax <= RO
        yf = np.full(n, np.inf); yr = np.full(n, -np.inf)
        yf[body] = yw - Lf * np.clip(1 - (ax[body] / RO) ** nF, 0, 1) ** (1 / nF) - o["front_point"] * L1 * np.exp(-(ax[body] / (0.10 * RO)) ** 2)
        y_n = yg + (o["notch"] * Lh if horn else 0.0); q = o["notch_exp"]
        inb = ax <= x_in
        yr[inb] = y_n * (1 - np.clip(1 - (ax[inb] / max(x_in, 1e-6)) ** q, 0, 1) ** (1 / q))
        mid = (~inb) & (ax <= xg); yr[mid] = yg
        sidec = (ax > xg) & body                                   # a side that tapers back from the widest line
        if sidec.any(): yr[sidec] = yw + (yg - yw) * np.sqrt(np.clip((RO - ax[sidec]) / max(RO - xg, 1e-6), 0, 1))
        rear_is_outer = sidec.copy()
        if not horn and o["corner"] > 0.01:                        # round the genal angle: open the silhouette with a disc
            from scipy.ndimage import distance_transform_edt as edt
            rc = o["corner"] * RO; st = 0.1; pad = int(rc / st) + 4
            gy = np.arange(np.nanmin(yf[body]) - 1, np.nanmax(yr[body]) + 1, st); gx = np.arange(-RO - 1, RO + 1 + st / 2, st)
            GX, GY = np.meshgrid(gx, gy); a_ = np.abs(GX)
            m = (a_ <= RO) & (GY >= np.interp(a_, ax[body], yf[body])) & (GY <= np.interp(a_, ax[body], yr[body]))
            m = np.pad(m, pad); er = edt(m) * st > rc; op = edt(~er) * st <= rc; op = op[pad:-pad, pad:-pad]
            for i in np.nonzero(body)[0]:
                col = op[:, int(round((ax[i] + RO + 1) / st))]
                if col.any(): yf[i] = gy[col.argmax()]; yr[i] = gy[len(col) - 1 - col[::-1].argmax()]
                else: yf[i] = np.inf; yr[i] = -np.inf
            from scipy.ndimage import gaussian_filter1d
            okb = np.isfinite(yf) & body
            yf[okb] = gaussian_filter1d(yf[okb], 4, mode="nearest"); yr[okb] = gaussian_filter1d(yr[okb], 4, mode="nearest")
        # ---- the spine's extent in each column
        if horn:
            dx = ax[:, None] - xc[None, :]; ok = np.abs(dx) < r[None, :]
            hgt = np.sqrt(np.clip(r[None, :] ** 2 - dx ** 2, 0, None))
            top = np.where(ok, yc[None, :] + hgt, -np.inf); bot = np.where(ok, yc[None, :] - hgt, np.inf)
            j = top.argmax(1); hmax = top.max(1); hmin = bot.min(1); has = ok.any(1)
            use = has & (hmax > yr)
            rear_is_outer = np.where(use, ax >= xc[j], rear_is_outer)
            yr = np.where(use, hmax, yr); yf = np.where(has, np.minimum(yf, hmin), yf)
        ok_c = np.isfinite(yf) & np.isfinite(yr) & (yr > yf)      # empty columns (past the tip) close to a point
        mid_c = np.interp(ax, ax[ok_c], 0.5 * (yf + yr)[ok_c])
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
        if horn: self.T_c = cKDTree(np.c_[xc, yc]); self.S, self.r, self.Lh = S, r, Lh

    def edges(self, x):
        a = np.abs(np.asarray(x, float)); return np.interp(a, self.ax, self.yf), np.interp(a, self.ax, self.yr)

    def fields(self, ax, y):
        """What head_crescent._z_base needs to know about this outline at (|x|, y), scale 1."""
        q = np.column_stack([np.ravel(ax), np.ravel(y)]); sh = np.shape(ax)
        do, io = self.T_out.query(q); dn, _ = self.T_in.query(q)
        g = dict(do=do.reshape(sh), dn=dn.reshape(sh), arc=self.arc[io].reshape(sh), dome_fill=self.o["dome_fill"], horn_h=self.o["horn_h"])
        if self.horn:
            _, ic = self.T_c.query(q); s = self.S[ic].reshape(sh)
            beh = smoothstep(self.yg - 3.0, self.yg + 2.0, y) * smoothstep(self.x_in - 3.0, self.x_in, ax)
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
        z = HC._z_base(ax, Yh, face, geom=self.fields(ax, y))
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
        z_thorax = max(float(M.sample_grid(Sk["outline"], Sk["zfun"], 61, 31)[2].max()) for _, _, Sk, _ in plans[1:-1])
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
