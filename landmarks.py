"""
landmarks.py — the TriloMorph landmark protocol (Serra et al. 2023, Tables 1–2) read off the generator's own geometry,
and a TPS writer their R functions (shapRead / shapFix) and geomorph::readland.tps both accept.

20 Oct 2026: replaces the twelve formula-points of the 11 Sep version (kept in legacy/landmarks_v1.py). The points are
taken from anatomy.head.plan(P) and anatomy.tail.plan(P) — the very functions the meshes are sampled from — so a
generated animal is landmarked the way a photographed one is, and the two can share one Procrustes superimposition.

    cephalon(P)   -> dict(lm={ "LM1": (x, y), ... "LM16": ... }, curves={ "glabella": (N,2), "suture": ..., "anterior": ..., "posterior": ... })
    pygidium(P)   -> dict(lm={ "LM1" .. "LM7" }, curves={ "axis", "border", "margin" })
    tps_string(specimens, part="C"|"P", semilandmarks=...)   one TPS block per (id, P); "_C" cephala, "_P" pygidia
    write_tps(specimens, path, part)

Frame (theirs): the RIGHT half of the dorsal plan, x lateral (+ right, 0 the sagittal axis), y anterior POSITIVE,
millimetres. Our head plan runs the other way (rear hinge at y = 0, front toward -y), so y is negated here.

What is honestly not on the mesh and how it is placed (say so when you publish):
  * LM2, the glabella's front: where the glabellar relief and the axial furrow end, (1 - preglabField) x Lc from the rear
    (anatomy/head.py, gf). The plan's glab_half() keeps tapering to the margin beyond it, but nothing is raised there.
  * LM3 / LM6, the occipital furrow: the trough at 0.13 Lc from the rear in head.plan.zfun_base (the SO furrow).
  * LM10–LM12, the facial suture: head.suture_path — a drawn groove, placed by the sutureEnd parameter, not anatomy
    grown from the mesh. LM11 (suture × border furrow) sits borderWidth inside LM12; with no border, LM11 = LM12.
  * LM15: "width at the level of LM4" degenerates on our swept rear edge (it would sit on the axial furrow), so the
    table's alternative is used: the genal angle, as in their Fig. 3 (Ellipsocephalus).
  * Blind forms (eyeSize <= 0.01): LM8, LM9, LM14 collapse onto LM10, as their files do; shapFix then drops the eye.
  * Pygidium: LM1 is the axis at the anterior margin (y = 0 in the tail frame, which is where the plate meets the last
    segment); LM2 where the axial rise fades (0.85 Lp); the border furrow from borderWidth, as on the head.
"""
import math
import numpy as np

NAMES_C = [f"LM{i}" for i in range(1, 17)]
NAMES_P = [f"LM{i}" for i in range(1, 8)]
CURVES_C = ["glabella", "suture", "anterior", "posterior"]       # LM2–LM7, LM10–LM12, LM13–LM15, LM4–LM15 (their curve.txt)
CURVES_P = ["axis", "border", "margin"]                          # LM1–LM2, LM3–LM6, LM4–LM5
SEMI_C = (12, 20, 20, 20)                                        # their default template (analyze_data.md)
SEMI_P = (12, 20, 30)
OCC_FURROW = 0.13                                                # head.plan: the SO trough at -y = 0.13 Lc
AXIS_END = 0.85                                                  # tail.plan: the axial rise fades over 0.75–0.95 Lp


def _resample(poly, n):
    """n points evenly spaced along the polyline (N, 2), ends included."""
    poly = np.asarray(poly, float)
    if len(poly) < 2: return np.repeat(poly[:1], n, axis=0)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    if d[-1] <= 0: return np.repeat(poly[:1], n, axis=0)
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, poly[:, 0]), np.interp(t, d, poly[:, 1])]


def cephalon(P):
    from anatomy import head as H
    S = H.plan(P)
    Lc, wh, a = S["Lc"], S["wh"], S["a"]
    yf = lambda x: float(_yfront(S, x))                                                        # front edge at |x|
    yr = lambda x: float(S["outline"](x / _xtip(S), 0.0)[1])                                  # rear edge at |x| (v = 0)
    gh = lambda y: float(_glab_half(S, P, y))
    # --- sagittal points (head frame, then y negated)
    y1 = yf(0.0)                                                 # front margin on the axis
    y4 = yr(0.0)                                                 # rear margin on the axis (the flap)
    y3 = -OCC_FURROW * Lc                                        # occipital furrow
    ys = np.linspace(y1 + 1e-3, y4 - 1e-3, 600)
    g = np.array([gh(y) for y in ys])
    y2 = -(1.0 - float(P.get("preglabField", 0.10))) * Lc        # where the glabella's relief and the axial furrow end (head.plan: gf)
    k5 = int(np.argmax(g)); y5, x5 = float(ys[k5]), float(g[k5])  # widest glabella
    x6 = gh(y3)                                                  # occipital × axial furrow
    x7 = gh(y4 - 1e-3); y7 = yr(x7)                               # posterior margin × axial furrow
    # --- eye
    G = S["eye"]
    blind = G is None
    # --- suture
    su = np.asarray(S["suture"], float)                          # (N, 2), front margin -> margin end
    x10, y10 = 0.0, y1                                           # marginal suture at the axis (ours starts beside the glabella)
    x12, y12 = float(su[-1, 0]), float(su[-1, 1])
    bw = float(P["borderWidth"]) * wh
    if bw > 0.05 and len(su) > 2:                                # LM11: the point on the posterior branch bw before its end
        d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(su[::-1], axis=0).T))]
        x11, y11 = float(np.interp(bw, d, su[::-1][:, 0])), float(np.interp(bw, d, su[::-1][:, 1]))
    else: x11, y11 = x12, y12
    # --- margins and genal
    x15, y15 = wh, yr(wh)                                        # genal angle (their alternative LM15)
    Lg = S["Lg"]
    if Lg > 0.05:
        xt, yt = S["arm_outline"](1.0, 0.5); x16, y16 = float(xt), float(yt)
    else: x16, y16 = x15, y15
    x13, y13 = 0.0, y1                                           # no anterior spine on the classic head (prongs are separate ornaments)
    lm = {"LM1": (0.0, y1), "LM2": (0.0, y2), "LM3": (0.0, y3), "LM4": (0.0, y4), "LM5": (x5, y5), "LM6": (x6, y3), "LM7": (x7, y7),
          "LM10": (x10, y10), "LM11": (x11, y11), "LM12": (x12, y12), "LM13": (x13, y13), "LM15": (x15, y15), "LM16": (x16, y16)}
    if blind: lm.update(LM8=lm["LM10"], LM9=lm["LM10"], LM14=lm["LM10"])
    else:
        ax = float(P.get("eyeAxial", 1.0))
        lm.update(LM8=(G["xe"], G["ye"] - G["eR"] * ax), LM9=(G["xe"], G["ye"] + G["eR"] * ax), LM14=(G["xe"] + G["eR"], G["ye"]))
    # --- curves (their endpoints)
    yg = np.linspace(y2, y7, 80); glab = np.c_[[gh(y) for y in yg], yg]
    xs = np.linspace(0.0, wh, 80)
    anterior = np.c_[xs, [yf(x) for x in xs]]                    # LM13 -> LM15 along the front edge
    posterior = np.c_[xs, [yr(x) for x in xs]]                   # LM4 -> LM15 along the rear edge
    curves = dict(glabella=glab, suture=su, anterior=anterior, posterior=posterior)
    flip = lambda pts: np.c_[np.asarray(pts, float)[:, 0], -np.asarray(pts, float)[:, 1]]
    return dict(lm={k: (float(x), float(-y)) for k, (x, y) in lm.items()}, curves={k: flip(v) for k, v in curves.items()},
                blind=blind, head_length=float(y4 - y1), head_halfwidth=float(wh))


def pygidium(P):
    from anatomy import tail as T
    S = T.plan(P)
    Lp, wp, a = S["Lp"], S["wp"], S["a"]
    tip = Lp * (1.0 + (float(P["termSpine"]) if float(P["termSpine"]) > 0.05 else 0.0))   # LM4 at the spine tip if spiny
    bw = float(S.get("border_frac", P["borderWidth"])) * wp      # 6.4: the tail's own border (pygBorderWidth), else the head's
    ax_end = float(S.get("axis_len", AXIS_END)); ka = ax_end / AXIS_END
    lm = {"LM1": (0.0, 0.0), "LM2": (0.0, ax_end * Lp), "LM4": (0.0, tip), "LM5": (a, 0.0), "LM7": (wp, 0.0)}
    if bw > 0.05: lm.update(LM3=(0.0, Lp - bw), LM6=(wp - bw, 0.0))
    else: lm.update(LM3=lm["LM4"], LM6=lm["LM7"])
    # curves: axis = the axial furrow ay(y); margin = the ellipse (plus spines) from the tip round to the front; border = inset
    yy = np.linspace(0.0, ax_end * Lp, 60); axis = np.c_[a * (1 - 0.85 * np.clip(yy / Lp / ka, 0, 1)), yy]
    phi = np.linspace(math.pi / 2, math.radians(1.5), 120)
    mx = np.array([S["outline"](float(1 - 2 * (p - math.radians(1.5)) / (math.pi - 2 * math.radians(1.5))), 1.0) for p in phi])
    margin = np.r_[mx, [[a, 0.0]]]                                # LM4 -> round the margin -> LM5 on the anterior edge
    if bw > 0.05:
        s = 1 - bw / max(wp, 1e-6)
        border = np.c_[mx[:, 0] * s, mx[:, 1] * (1 - bw / max(Lp, 1e-6))][::1]
    else: border = np.array([lm["LM3"], lm["LM6"]], float)
    # tail frame: y runs from the front (0) to the tip, i.e. anterior is -y in TriloMorph's sense -> flip so anterior is +y
    flip = lambda pts: np.c_[np.asarray(pts, float)[:, 0], -np.asarray(pts, float)[:, 1]]
    return dict(lm={k: (float(x), float(-y)) for k, (x, y) in lm.items()}, curves=dict(axis=flip(axis), border=flip(border), margin=flip(margin)),
                pyg_length=float(tip), pyg_halfwidth=float(wp))


# ---- helpers that read the plan's own functions -------------------------------------------------------------------
def _yfront(S, x):
    """front edge y at |x| (head frame) via the plan's outline(u, 1)."""
    return S["outline"](abs(x) / _xtip(S), 1.0)[1]

def _xtip(S):
    """the plan's u = 1 abscissa (X_tip): recover it from outline(1, ·)."""
    return S["outline"](1.0, 0.0)[0]

def _glab_half(S, P, y):
    """the plan's glab_half(y) is a closure; head.plan exposes it through the suture plan dict in >= 27 Sep builds."""
    f = S.get("glab_half")
    if f is not None: return f(np.array([y]))[0]
    # fallback: recompute as head.plan does (kept identical to anatomy/head.py)
    from anatomy.common import smoothstep
    Lc, a = S["Lc"], S["a"]
    ff = np.clip(-y / Lc, 0, 1)
    w = a * (1 + (P["glabInflate"] - 1) * ff) * (1 + 0.10 * np.sin(np.pi * ff))
    nF = P.get("glabFront", 2.5); fn = np.clip((ff - 0.70) / 0.30, 0, 1)
    w = w * np.clip(1 - fn ** nF, 0, 1) ** (1 / nF) * (0.75 + 0.25 * smoothstep(0.0, 0.10, ff))
    return max(w, 0.35 * a)


# ---- TPS -------------------------------------------------------------------------------------------------------
def tps_block(sid, part, semilandmarks=None):
    """One tpsDig-style block: fixed landmarks, CURVES= with POINTS= per curve, IMAGE/ID/SCALE/COMMENT."""
    names, order, semi = (NAMES_C, CURVES_C, SEMI_C) if part["kind"] == "C" else (NAMES_P, CURVES_P, SEMI_P)
    semi = semilandmarks or semi
    out = ["LM=%d" % len(names)] + ["%.5f %.5f" % part["lm"][n] for n in names]
    out.append("CURVES=%d" % len(order))
    for cname, n in zip(order, semi):
        pts = _resample(part["curves"][cname], n)
        out.append("POINTS=%d" % n); out += ["%.5f %.5f" % (x, y) for x, y in pts]
    out += ["IMAGE=%s_%s.png" % (sid, part["kind"]), "ID=%s" % sid, "SCALE=1.0",
            "COMMENT=trilobite_generator; protocol Serra et al. 2023 (TriloMorph); mm; right half, anterior +y; curves=%s" % ",".join(order), ""]
    return "\n".join(out)


def tps_string(specimens, part="C", semilandmarks=None):
    """specimens: list of (id, P). part "C" (cephala) or "P" (pygidia). One block per specimen."""
    blocks = []
    for sid, P in specimens:
        d = cephalon(P) if part == "C" else pygidium(P); d["kind"] = part
        blocks.append(tps_block(sid, d, semilandmarks))
    return "\n".join(blocks)


def write_tps(specimens, path, part="C", semilandmarks=None):
    with open(path, "w") as f: f.write(tps_string(specimens, part, semilandmarks))
    return path


if __name__ == "__main__":                                       # demo: every preset -> cephala TPS on stdout
    import sys, os, json, glob
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import schema
    specs = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets", "*.json"))):
        n = os.path.basename(f)[:-5]
        if not n.startswith("_"): specs.append((n, schema.coerce(json.load(open(f))["params"], base=schema.table_defaults())))
    print(tps_string(specs, "C"))
