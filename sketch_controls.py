import math, numpy as np, matplotlib
import matplotlib.pyplot as plt

try:
    import schema
    DEFAULTS = {p.key: p.default for p in schema.PARAMS}
except Exception:
    DEFAULTS = dict(length=130, width=65, relief=16, segCount=6, cephFrac=0.33, pygFrac=0.14, marginHeight=0.34,
                    fulcrum=0.55, widthThoraxFront=0.92, widthThoraxRear=0.62, headOutlineExp=2.15, glabInflate=1.25,
                    axisFrac=0.33, eyeSize=0.14, eyePos=0.45, eyeLat=0.0, eyeArc=150, eyeHeight=0.8, eyeSlope=15,
                    headRelief=1.0, tailRelief=1.0, ringArch=0.22, axisRise=0.15, headProngs=0, headProngLen=0.6,
                    headProngCurl=0.0, headProngSplay=30, headProngStem=0.0, headProngWidth=0.45, tailProngs=0, tailProngLen=0.5,
                    tailProngSplay=30, tailProngStem=0.0, genalSpine=0.35, eyeStalk=0.0, lensD=0.16, lensGap=0.3, lensRise=0.35,
                    lensStyle="raised", lensLattice="hex")

PACKING = {"holochroal": dict(lensD=0.06, lensGap=0.05, lensRise=0.3, lensStyle="flush", lensLattice="hex"),
           "schizochroal": dict(lensD=0.11, lensGap=0.5, lensRise=0.5, lensStyle="sunken", lensLattice="hex"),
           "abathochroal": dict(lensD=0.09, lensGap=0.35, lensRise=0.4, lensStyle="raised", lensLattice="grid")}
BG = "#000000"; LINE = "#9a9a9a"; DIM = "#5a5a5a"; TXT = "#bbbbbb"; HANDLE_EC = "#ffffff"; TOL_PX = 9


# ----------------------------------------------------------------------------- geometry (simplified, mm)
class Geo:
    """Plan and elevation curves for the sketch. y runs head (+) → tail (−) in mm, hinge at y = 0."""
    def __init__(self, P):
        self.P = P
        L, W = P["length"], P["width"]
        self.Lc = P["cephFrac"] * L; self.Lp = P["pygFrac"] * L
        self.Lt = L - self.Lc - self.Lp
        self.n = int(P["segCount"]); self.pitch = self.Lt / max(self.n, 1)
        self.wh = 0.5 * W                       # head half-width (shoulder)
        self.relief = P["relief"]

    # plan --------------------------------------------------------------------------------------
    def head_outline(self, m=60):
        e = self.P["headOutlineExp"]; t = np.linspace(0, 1, m)
        x = self.wh * (1 - t ** e) ** (1 / e); y = self.Lc * t
        return np.r_[x, -x[::-1]], np.r_[y, y[::-1]]

    def seg_quad(self, i):
        f0, f1 = self.P["widthThoraxFront"], self.P["widthThoraxRear"]
        wa = self.wh * (f0 + (f1 - f0) * i / max(self.n - 1, 1))
        wb = self.wh * (f0 + (f1 - f0) * (i + 1) / max(self.n - 1, 1)) if i + 1 < self.n else wa
        y0, y1 = -i * self.pitch, -(i + 1) * self.pitch
        return [(-wa, y0), (wa, y0), (wb, y1), (-wb, y1)]

    def tail_outline(self, m=40):
        w = self.wh * self.P["widthThoraxRear"]; t = np.linspace(0, math.pi, m)
        return w * np.cos(t), -self.Lt - self.Lp * np.sin(t)

    def eye(self):
        eR = self.P["eyeSize"] * self.wh; ye = self.P["eyePos"] * self.Lc
        lat = self.P["eyeLat"]; glab = self.P["axisFrac"] * self.wh * self.P["glabInflate"]
        xe = lat * self.wh if lat > 0.01 else glab + eR + 1.0
        return xe, ye, eR

    def prong(self, head=True):
        """Fork at the head tip (or tail tip): root y, split y, tine tips [(x, y)], root half-width (mm)."""
        P = self.P; k = "head" if head else "tail"
        n = int(P[k + "Prongs"]); Lref = self.Lc if head else self.Lp
        L = P[k + "ProngLen"] * Lref; sgn = 1 if head else -1
        y0 = self.Lc if head else -self.Lt - self.Lp
        stem = P[k + "ProngStem"] * L; ysplit = y0 + sgn * stem
        splay = math.radians(P[k + "ProngSplay"]); tips = []
        for i in range(n):
            a = (-0.5 + (i + 0.5) / n) * splay if n > 1 else 0.0
            tips.append((math.sin(a) * (L - stem), ysplit + sgn * math.cos(a) * (L - stem)))
        return dict(n=n, y0=y0, ysplit=ysplit, tips=tips, sgn=sgn, L=L, rw=P[k + "ProngWidth"] * self.relief * self.P["marginHeight"] * 0.5)

    # elevation ---------------------------------------------------------------------------------
    def axial_profile(self, m=80):
        """(z, y) of the dorsal axial section: head dome, rings, tail."""
        R = self.relief; P = self.P
        t = np.linspace(0, 1, m); yh = self.Lc * t
        zh = R * P["headRelief"] * np.sqrt(np.clip(1 - t ** 2, 0, 1)) * 0.85 + R * 0.15 * (1 - t)
        ys, zs = [], []
        for i in range(self.n):
            y0, y1 = -i * self.pitch, -(i + 1) * self.pitch
            arch = P["ringArch"] * R; base = R * (1 + P["axisRise"])
            ys += [y0, 0.5 * (y0 + y1), y1]; zs += [base - arch, base, base - arch]
        tt = np.linspace(0, 1, m); yt = -self.Lt - self.Lp * tt
        zt = R * P["tailRelief"] * np.sqrt(np.clip(1 - tt ** 2.2, 0, 1))
        return (np.r_[zh[::-1], zs, zt], np.r_[yh[::-1], ys, yt])

    def margin_profile(self):
        z = self.relief * self.P["marginHeight"]
        return np.array([z * 0.6, z, z, z * 0.5]), np.array([self.Lc * 0.9, 0, -self.Lt, -self.Lt - self.Lp * 0.9])


# ----------------------------------------------------------------------------- the control panel
class SketchControls:
    def __init__(self, P=None, on_change=None, figsize=(8.4, 9.6)):
        self.P = dict(DEFAULTS); self.P.update(P or {})
        self.on_change = on_change
        self.fig = plt.figure(figsize=figsize); self.fig.patch.set_facecolor(BG)
        gs = self.fig.add_gridspec(2, 3, width_ratios=[0.9, 4, 2.4], height_ratios=[8, 1.3], wspace=0.04, hspace=0.05)
        self.axr, self.axp, self.axe = (self.fig.add_subplot(gs[0, i]) for i in range(3))
        self.axs = self.fig.add_subplot(gs[1, :])
        for ax in (self.axr, self.axp, self.axe, self.axs):
            ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
        self.sel_pos = {}
        self.drag = None; self.hover = None
        self.fig.canvas.mpl_connect("button_press_event", self._press)
        self.fig.canvas.mpl_connect("button_release_event", self._release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._motion)
        self.redraw()

    # ---- handle table: name -> dict(view, pos(g), apply(g, x, y), twin, word)
    def handles(self, g):
        P = self.P; xe, ye, eR = g.eye(); R = g.relief
        def setp(k, v, lo, hi): P[k] = float(min(max(v, lo), hi))
        H = {
            # rail (left): length and the two tagma stations
            "rail_len":   dict(view="r", pos=(6, -g.Lt - g.Lp), apply=lambda x, y: setp("length", g.Lc - y, 60, 250), twin="length_bot", word="length"),
            "rail_head":  dict(view="r", pos=(2, 0), apply=lambda x, y: setp("cephFrac", (g.Lc - y) / P["length"], 0.18, 0.45), twin="width", word="head"),
            "rail_tail":  dict(view="r", pos=(2, -g.Lt), apply=lambda x, y: setp("pygFrac", (y + g.Lt + g.Lp) / P["length"], 0.05, 0.4), twin="tail_h", word="tail"),
            "length_top": dict(view="p", pos=(0, g.Lc + 4), apply=lambda x, y: setp("length", (y - 4) / P["cephFrac"], 60, 250), twin="curl", word="length"),
            "length_bot": dict(view="p", pos=(0, -g.Lt - g.Lp - 4),
                               apply=lambda x, y: setp("length", P["length"] - (y + 4 + g.Lt + g.Lp), 60, 250), twin="tail_h", word="length"),
            "width":      dict(view="p", pos=(g.wh, 0), apply=lambda x, y: setp("width", 2 * abs(x), 30, 140), twin="hinge", word="width"),
            "w_rear":     dict(view="p", pos=(g.wh * P["widthThoraxRear"], -g.Lt * 0.5),
                               apply=lambda x, y: setp("widthThoraxRear", abs(x) / g.wh, 0.3, 1.0), twin="margin", word="w"),
            "eye_c":      dict(view="p", pos=(xe, ye),
                               apply=lambda x, y: (setp("eyePos", y / g.Lc, 0.1, 0.9), setp("eyeLat", abs(x) / g.wh, 0.0, 0.9)), twin="eye_h", word="eye"),
            "eye_r":      dict(view="p", pos=(xe + eR, ye), apply=lambda x, y: setp("eyeSize", abs(x - xe) / g.wh, 0.02, 0.45), twin="eye_h", word="size"),
            "eye_arc":    dict(view="p", pos=(xe + eR * math.cos(math.radians(P["eyeArc"] / 2)), ye + eR * math.sin(math.radians(P["eyeArc"] / 2))),
                               apply=lambda x, y: setp("eyeArc", 2 * math.degrees(math.atan2(y - ye, x - xe)), 60, 300), twin="eye_h", word="arc"),
        }
        # forks: split point (y → stem, x → splay), outer tine tip (→ length), root width
        for k, head in (("head", True), ("tail", False)):
            f = g.prong(head)
            if f["n"] == 0: continue
            H.pop("length_top" if head else "length_bot", None)      # the rail still owns length; the fork owns the tip
            Lref = g.Lc if head else g.Lp; sgn = f["sgn"]; y0 = f["y0"]
            def _split(x, y, k=k, f=f, Lref=Lref, sgn=sgn, y0=y0):
                setp(k + "ProngStem", sgn * (y - y0) / max(f["L"], 1e-6), 0.0, 0.8)
                if f["n"] > 1: setp(k + "ProngSplay", 2 * math.degrees(math.atan2(abs(x), max(f["L"] - sgn * (y - y0), 1e-6))) * f["n"] / (f["n"] - 1), 0, 90)
            def _tip(x, y, k=k, f=f, Lref=Lref, sgn=sgn, y0=y0):
                setp(k + "ProngLen", math.hypot(x, y - f["ysplit"]) / Lref + P[k + "ProngStem"] * P[k + "ProngLen"] * 0, 0.1, 2.0)
            def _root(x, y, k=k, g=g): setp(k + "ProngWidth", 2 * abs(x) / (g.relief * P["marginHeight"]), 0.2, 1.2)
            H[k + "_split"] = dict(view="p", pos=(f["tips"][-1][0] * 0.5 if f["n"] > 1 else 0, f["ysplit"]), apply=_split, twin="curl" if head else "tail_h", word="split")
            H[k + "_tip"] = dict(view="p", pos=f["tips"][-1], apply=_tip, twin="curl" if head else "tail_h", word="tip")
            H[k + "_root"] = dict(view="p", pos=(f["rw"], y0), apply=_root, twin="curl" if head else "tail_h", word="root")
        # packing swatch (bottom): lens diameter and gap, in mm at the current eye radius
        D = P["lensD"] * eR; pitch = D * (1 + P["lensGap"])
        H.update({
            "lens_d":   dict(view="s", pos=(D, 0), apply=lambda x, y: setp("lensD", abs(x) / eR, 0.02, 0.4), twin="eye_r", word="lens"),
            "lens_gap": dict(view="s", pos=(pitch, 0), apply=lambda x, y: setp("lensGap", abs(x) / max(D, 1e-6) - 1, 0.0, 0.6), twin="eye_r", word="gap"),
        })
        H.update({
            # elevation: horizontal drags only (x = z in mm)
            "relief":     dict(view="e", pos=(R * (1 + P["axisRise"]), -g.Lt * 0.5), apply=lambda x, y: setp("relief", x / (1 + P["axisRise"]), 6, 40), twin="w_rear", word="relief"),
            "margin":     dict(view="e", pos=(R * P["marginHeight"], -g.Lt * 0.5), apply=lambda x, y: setp("marginHeight", x / R, 0.05, 0.6), twin="w_rear", word="margin"),
            "head_h":     dict(view="e", pos=(R * P["headRelief"] * 0.85 + R * 0.15 * 0.5, g.Lc * 0.5), apply=lambda x, y: setp("headRelief", (x - R * 0.075) / (R * 0.85), 0.6, 1.6), twin="width", word="head"),
            "tail_h":     dict(view="e", pos=(R * P["tailRelief"] * 0.9, -g.Lt - g.Lp * 0.45), apply=lambda x, y: setp("tailRelief", x / (0.9 * R), 0.5, 1.4), twin="length_bot", word="tail"),
            "eye_h":      dict(view="e", pos=(R * 0.9 + P["eyeHeight"] * eR, ye), apply=lambda x, y: setp("eyeHeight", (x - R * 0.9) / eR, 0.2, 1.6), twin="eye_c", word="eye"),
            "stalk":      dict(view="e", pos=(R * 0.9 - P.get("eyeStalk", 0.0) * eR, ye), apply=lambda x, y: setp("eyeStalk", (R * 0.9 - x) / eR, 0.0, 4.0), twin="eye_c", word="stalk"),
            "curl":       dict(view="e", pos=(R * 0.3 + 0.4 * P["headProngCurl"], g.Lc + 4), apply=lambda x, y: setp("headProngCurl", (x - R * 0.3) / 0.4, -40, 40), twin="length_top", word="curl"),
        })
        return H

    def _ax(self, view): return {"r": self.axr, "p": self.axp, "e": self.axe, "s": self.axs}[view]

    # ---- drawing
    def redraw(self):
        g = Geo(self.P); P = self.P; axp, axe, axr = self.axp, self.axe, self.axr
        axp.cla(); axe.cla(); axr.cla(); self.axs.cla()
        for a in list(self.fig.artists): a.remove()
        for ax in (axr, axp, axe, self.axs): ax.set_aspect("equal"); ax.axis("off")
        # rail: frame, 50 mm scale, length bar, station bar
        ytop, ybot = g.Lc, -g.Lt - g.Lp
        axr.add_patch(plt.Rectangle((-2, ybot - 4), 10, ytop - ybot + 8, fill=False, ec=LINE, lw=1.0))
        axr.plot([6, 6], [ybot, ytop], color=LINE, lw=0.8); axr.plot([5, 7], [ytop, ytop], color=LINE, lw=0.8); axr.plot([5, 7], [ybot, ybot], color=LINE, lw=0.8)
        axr.plot([2, 2], [ybot, ytop], color=LINE, lw=0.8); axr.plot([1, 3], [ytop, ytop], color=LINE, lw=0.8)
        axr.text(-2, ytop + 8, "50 mm", color=TXT, fontsize=8, family="monospace", va="bottom")
        axr.plot([-2, -2], [ytop + 5, ytop + 5], color=LINE)   # anchor for the label only
        # plan
        x, y = g.head_outline(); axp.plot(x, y, color=LINE, lw=1.1)
        gw = P["axisFrac"] * g.wh; axp.plot([-gw, -gw, -gw * P["glabInflate"], gw * P["glabInflate"], gw, gw],
                                            [0, g.Lc * 0.55, g.Lc * 0.82, g.Lc * 0.82, g.Lc * 0.55, 0], color=LINE, lw=0.9)
        xe, ye, eR = g.eye(); th = np.linspace(0, 2 * math.pi, 60)
        for s in (1, -1): axp.plot(s * xe + eR * np.cos(th), ye + eR * np.sin(th), color=LINE, lw=0.8)
        for i in range(g.n):
            q = g.seg_quad(i); q.append(q[0]); axp.plot(*zip(*q), color=LINE, lw=0.8)
        x, y = g.tail_outline(); axp.plot(x, y, color=LINE, lw=1.1)
        gs = P["genalSpine"] * g.Lc
        for s in (1, -1): axp.plot([s * g.wh, s * (g.wh + 0.25 * gs)], [0, -0.6 * gs], color=LINE, lw=1.0)
        for head in (True, False):
            f = g.prong(head)
            if f["n"] == 0:
                y = g.Lc if head else -g.Lt - g.Lp; axp.plot([0, 0], [y, y + (3 if head else -3)], color=LINE, lw=1.0); continue
            axp.plot([-f["rw"], -f["rw"], f["rw"], f["rw"]], [f["y0"], f["ysplit"], f["ysplit"], f["y0"]], color=LINE, lw=1.0)
            for tx, ty in f["tips"]: axp.plot([0, tx], [f["ysplit"], ty], color=LINE, lw=1.0)
        # elevation strip (z → right)
        axe.axvline(0, color=DIM, lw=0.6)
        z, yy = g.axial_profile(); axe.plot(z, yy, color=LINE, lw=1.1)
        zm, ym = g.margin_profile(); axe.plot(zm, ym, color=LINE, lw=0.7, ls=(0, (2, 2)))
        axe.plot([0, g.relief * 0.55], [0, 0], color=DIM, lw=2)                       # hinge tick
        axe.plot([0, 0], [g.Lc, g.Lc + 3], color=LINE, lw=1.0); axe.plot([0, 0], [-g.Lt - g.Lp, -g.Lt - g.Lp - 3], color=LINE, lw=1.0)
        eb = g.relief * 0.9; eh = P["eyeHeight"] * eR
        axe.add_patch(plt.Rectangle((eb, ye - 0.35 * eR), eh, 0.7 * eR, fill=False, ec=LINE, lw=1.0))
        st = P.get("eyeStalk", 0.0) * eR
        if st > 0: axe.plot([eb - st, eb], [ye, ye], color=LINE, lw=1.0)
        # packing swatch: a patch of the visual band at true mm, plus the three-way selector
        axs = self.axs; D = P["lensD"] * eR; pitch = D * (1 + P["lensGap"]); rows = 3; cols = 6
        hexlat = P.get("lensLattice", "hex") == "hex"; style = P.get("lensStyle", "raised")
        for i in range(rows):
            for k in range(cols):
                cx = (k + (0.5 * (i % 2) if hexlat else 0)) * pitch; cy = -i * (0.87 if hexlat else 1.0) * pitch
                if style == "raised": axs.add_patch(plt.Circle((cx, cy), D / 2, fc=DIM, ec=LINE, lw=0.5))
                elif style == "sunken": axs.add_patch(plt.Circle((cx, cy), D / 2, fc=BG, ec=LINE, lw=1.0)); axs.add_patch(plt.Circle((cx, cy), D / 2 * 0.6, fc=DIM, ec="none"))
                else: axs.add_patch(plt.Circle((cx, cy), D / 2, fc=BG, ec=DIM, lw=0.5))
        self.sel_pos = {}
        for name, fx in zip(PACKING, (0.50, 0.66, 0.83)):
            on = all(abs(float(P.get(k, 0)) - float(v)) < 1e-6 if not isinstance(v, str) else P.get(k) == v for k, v in PACKING[name].items())
            axs.text(fx, 0.5, name, color=(HANDLE_EC if on else DIM), fontsize=8, family="monospace", va="center", transform=axs.transAxes)
            self.sel_pos[name] = (fx, 0.5)
        span = max(6 * pitch, 3.0); axs.set_xlim(-D, span * 3.2); axs.set_ylim(-rows * pitch * 0.9 - D, D)
        # handles
        self.H = self.handles(g)
        for name, h in self.H.items():
            self._ax(h["view"]).plot(*h["pos"], "o", ms=7, mfc=BG, mec=HANDLE_EC, mew=1.0, zorder=5)
        # hover feedback
        if self.hover in self.H:
            h = self.H[self.hover]; ax = self._ax(h["view"])
            ax.annotate(h["word"], h["pos"], xytext=(8, 4), textcoords="offset points", fontsize=9, color=TXT, family="monospace")
            t = self.H.get(h["twin"])
            if t and t["view"] != h["view"]:
                con = matplotlib.patches.ConnectionPatch(h["pos"], t["pos"], "data", "data", axesA=ax, axesB=self._ax(t["view"]), color=LINE, lw=0.6, ls=(0, (3, 3)), zorder=4, clip_on=False)
                self.fig.add_artist(con)
        # same px/mm in both views: identical ylim, xlim span fixed by each axes' aspect ratio in the figure
        pad = 12; y0, y1 = -g.Lt - g.Lp - pad, g.Lc + pad
        fw, fh = self.fig.get_size_inches()
        for ax, x0 in ((axr, -6), (axp, None), (axe, -6)):
            bb = ax.get_position(original=True); xspan = (y1 - y0) * (bb.width * fw) / (bb.height * fh)
            ax.set_ylim(y0, y1)
            ax.set_xlim(-xspan / 2, xspan / 2) if x0 is None else ax.set_xlim(x0, x0 + xspan)
            ax.set_aspect("equal", adjustable="box")
        self.fig.canvas.draw_idle()

    # ---- interaction
    def _nearest(self, ev):
        if ev.inaxes not in (self.axr, self.axp, self.axe, self.axs): return None
        best, bd = None, TOL_PX
        for name, h in self.H.items():
            ax = self._ax(h["view"])
            if ax is not ev.inaxes: continue
            px, py = ax.transData.transform(h["pos"]); d = math.hypot(px - ev.x, py - ev.y)
            if d < bd: best, bd = name, d
        return best

    def _press(self, ev):
        if ev.inaxes is self.axs and ev.xdata is not None:
            for name, (x, y) in self.sel_pos.items():
                px, py = self.axs.transAxes.transform((x, y))
                if abs(ev.y - py) < 10 and 0 <= ev.x - px < 90:
                    self.P.update(PACKING[name]); self.redraw()
                    if self.on_change: self.on_change(dict(self.P))
                    return
        self.drag = self._nearest(ev)

    def _release(self, ev):
        if self.drag and self.on_change: self.on_change(dict(self.P))
        self.drag = None

    def _motion(self, ev):
        if self.drag:
            h = self.H[self.drag]; ax = self._ax(h["view"])
            if ev.inaxes is not ax and ev.x is not None:
                ev.xdata, ev.ydata = ax.transData.inverted().transform((ev.x, ev.y))
            if ev.xdata is None: return
            if h["view"] == "e": h["apply"](ev.xdata, h["pos"][1])           # elevation: horizontal only
            elif h["view"] in ("r",): h["apply"](h["pos"][0], ev.ydata)      # rail: vertical only
            elif h["view"] == "s": h["apply"](ev.xdata, 0)                    # swatch: horizontal only
            else: h["apply"](ev.xdata, ev.ydata)
            self.hover = self.drag; self.redraw()
        else:
            n = self._nearest(ev)
            if n != self.hover: self.hover = n; self.redraw()


if __name__ == "__main__":
    def show(P): print({k: round(v, 3) for k, v in P.items() if k in ("length", "width", "relief", "eyePos", "eyeLat", "eyeSize", "eyeArc", "eyeHeight", "eyeStalk", "marginHeight", "headRelief", "tailRelief", "headProngCurl", "headProngStem", "headProngSplay", "headProngLen", "lensD", "lensGap", "lensStyle")})
    SketchControls(on_change=show); plt.show()
