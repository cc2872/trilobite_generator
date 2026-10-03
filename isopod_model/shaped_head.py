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
        at = float(np.clip(o["genal_at"], -1, 1)); moved = horn and abs(at) > 0.01
        corner_horn = horn and not moved                           # the spine IS the genal angle (the crescent's case)
        x_in = xg - w0 if corner_horn else xg
        # ---- the head proper, column by column (0 .. RO)
        nb = 1201; axb = np.linspace(0, RO, nb)
        fl = float(np.clip(o["front_flat"], 0, 0.85)); a2 = np.clip((axb - fl * RO) / (RO * (1 - fl)), 0, 1)
        yfb = yw - Lf * np.clip(1 - a2 ** nF, 0, 1) ** (1 / nF)
        y_n = yg + (o["notch"] * Lh if corner_horn else 0.0); q = o["notch_exp"]
        yrb = np.full(nb, yg); inb = axb <= x_in
        yrb[inb] = y_n * (1 - np.clip(1 - (axb[inb] / max(x_in, 1e-6)) ** q, 0, 1) ** (1 / q))
        sidec = axb > xg                                           # a side that tapers back from the widest line
        if sidec.any(): yrb[sidec] = yw + (yg - yw) * np.sqrt(np.clip((RO - axb[sidec]) / max(RO - xg, 1e-6), 0, 1))
        if not corner_horn and o["corner"] > 0.01:                 # round the genal angle: open the silhouette with a disc
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
        if o["front_dent"] > 0: yfb = yfb + o["front_dent"] * L1 * np.exp(-(axb / (0.30 * RO)) ** 2)
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
    if sh.horn:
        H["spine"] = (float(sh.xc[-1]) * u, float(sh.yc[-1]) * u); H["root"] = (float(sh.xc[0]) * u, float(sh.yc[0]) * u)
    if o["front_point"] > 0: H["process"] = (o["point_fork"], float(sh.yf[int(np.argmin(sh.yf))]) * u)
    if b > 0.02: H["brim"] = (0.7 * (1 - b), float(np.interp(0.7 * RO, ax, sh.yf)) * (1 - b) * u + sh.yw * b * u)
    if eye is not None: H["eye"] = (E["xe"] * u, float(eye[:, 1].mean()))
    return dict(outline=plan_, glabella=glab, eye=eye, brim=brim, boss=boss, section=np.c_[ys * u, sec * kz], profile=np.c_[ys * u, prof * kz],
                handles=H, size=dict(length=sh.L1 * u, halfwidth=sh.Xmax * u, width_vs_thorax=o["width"]))
