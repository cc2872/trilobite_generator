"""
parts.py — the animal's parts on the mesh builder (prompt 3 of the rewrite, 11 Sep 2026).

Thorax rows (cells b2 / a2 / c2 = mirror a2) and the pygidium (b3 / a3 / c3). The head (b1 / a1) lands in prompt 4.

The plan functions (outline, zfun) are the BREP builder's, transcribed from trilobite.build_segment and
trilobite.build_pygidium with ONE deliberate omission — the tubercles (ornament is out of the core, 10 Sep verdict).
Everything a fossil shows in the shell is here: vault, axial ring and furrows, ring furrow, pleural furrow that follows
the blade, ring arch and blade camber, the stepped front band that shingles under the previous part, the flap, the
pleural spine as the blade's own arc running on, forks and marginal spines as the tail margin growing radially.

Cells. A row is three physical regions: the axial ring (b) and two pleurae (a, c). They are built as three
height-field shells on the same plan — the ring on |u| <= u_a, the right pleura on u in [u_a - lap, 1], the left
pleura as the mirror of the right — and fused into one solid for the hinge and the instrument. `segment_cells()` hands
the three back separately for the Flexi row (print-in-place hooks come with that design, v1.1); `segment()` is the
fused row the instrument measures, identical in shape to the BREP one.

Two-length pleura. The cross-section already bends at the fulcrum: inner run = fulcrum * (w - a), outer blade =
(1 - fulcrum) * (w - a), blade slope = pleuralSlope (the "tent" vault). `pleura_lengths()` reports them in mm; the
schema rewrite (prompt 5) exposes the a2 cell as those two lengths and the bend instead of fulcrum/width fractions.

No OpenCascade, no retries, no jitter. A part is a closed manifold or the build raises.
"""
import math
import numpy as np
import mesh as M
from fields import seg_halfwidth, tail_halfwidth, pleural_spine_field, furrow_amp

WEDGE_REACH = 0.15        # seg-seg joints: bevel reach past the hinge line, fraction of the pitch (trilobite.py, 8 Sep)
WEDGE_REACH_WIDE = 0.5    # head-seg0 and last-seg-tail joints (full-width plates, no tips to sever)
GRID_SEG = (121, 61)      # (nu, nv) — 10 Sep verdict: 120x60 per part; nu odd so the axis is a vertex column
GRID_TAIL = (121, 61)
TIP_KEEP_MM = 3.0         # the bevel band stops this far short of a segment's pleural tip (11 Sep: harpetida seg7)

# ---------------------------------------------------------------- derived scalars (trilobite.py, unchanged)
def pitch(P): return P["length"] * (1 - P["cephFrac"] - P["pygFrac"]) / P["segCount"]
def ring_top(P): return P["relief"] * (1 + P["axisRise"])
def hinge_z(P): return ring_top(P) - P["barrelR"] - P["wall"] - P["clearance"] - 0.4 - 0.7 * furrow_amp(P)
def hinge_width(P): return 2 * P["axisFrac"] * seg_halfwidth(P, P["segCount"] - 1) - 2
def joint_offsets(P): return [0] + [pitch(P)] * int(P["segCount"])

# ---------------------------------------------------------------- surface vocabulary (numpy)
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)
def plateau(v, half, edge=0.7): return 1 - smoothstep(half - edge, half + edge, np.abs(v))
def trough(dist, sigma): return np.exp(-(dist / sigma) ** 2)

def vault(u, P):
    """Cross-section height factor vs normalized lateral position u in [-1, 1] (trilobite.vault, verbatim)."""
    u = np.abs(u); f = P["fulcrum"]
    inner = 1 - 0.35 * (u / f) ** 2
    outer = 0.65 * (1 - (np.maximum(u - f, 0) / (1 - f)) ** 1.6)
    v4 = np.where(u <= f, inner, outer)
    k = P["tent"]
    if k <= 0.001: return v4
    m = P["marginHeight"]; k1 = P["pleuralSlope"]
    zb = 1.0 - 0.04; k2 = zb - k1 - m
    def raw(uu):
        dome = np.exp(-(uu / P["axisSigma"]) ** 2)
        pleural = zb - k1 * uu - k2 * uu ** 2
        return np.log(np.exp(4 * dome) + np.exp(4 * pleural)) / 4
    t0, t1 = raw(np.float64(0.0)), raw(np.float64(1.0))
    tent = (raw(u) - t1) / max(t0 - t1, 1e-6)
    return (1 - k) * v4 + k * tent

def pleura_lengths(P, i):
    """(inner run, outer blade, bend angle deg) of segment i's pleura in mm — the two-length profile in the sketch."""
    w = seg_halfwidth(P, i); a = P["axisFrac"] * w
    inner = P["fulcrum"] * (w - a); blade = (1 - P["fulcrum"]) * (w - a)
    bend = math.degrees(math.atan(P["pleuralSlope"] * P["relief"] / max(w - a, 1e-6)))
    return inner, blade, bend

# ---------------------------------------------------------------- solids other than the shell
def spine_solid(base_r, tip_r, length, at, yaw_deg, pitch_deg=0):
    """Tapered spine from `at`, pointing +y (rear); yaw about z (+ toward +x), pitch up. trilobite.spine's frames."""
    s = M.frustum(base_r, tip_r, length)
    R = __import__("trimesh").transformations.rotation_matrix
    s.apply_transform(R(math.radians(-90), (1, 0, 0)))
    s.apply_transform(R(math.radians(pitch_deg), (1, 0, 0)))
    s.apply_transform(R(math.radians(-yaw_deg), (0, 0, 1)))
    s.apply_translation(at)
    return s

def _hinge_geometry(P, wide, halfwidth=None):
    """Bevel band width and reach. 11 Sep 2026: the band follows the LOCAL half-width of the plate (the BREP builder
    sized it from the animal's maximum width, so on a narrow rear segment the 'axial band + 0.12 W' covered nearly the
    whole pleura and the wedge severed its tips — harpetida seg7; OCC's grid-jitter retries had been hiding it).
    Fix: seg-seg bands are capped 3 mm short of the local pleural tip, so the tips stay for the instrument to find.
    Wide joints (head-seg0, last seg-tail) keep the global-width rule the readings were validated on."""
    W = P["width"]
    if P["bladeChord"] < 1.0 and not wide: band = 2 * (P["axisFrac"] * (W / 2)) + 0.12 * W
    else:                                    band = 2 * (P["axisFrac"] * (W / 2)) + 0.45 * W
    if not wide and halfwidth is not None:                      # seg-seg joints: never within 3 mm of the local tip
        band = min(band, 2 * (halfwidth - TIP_KEEP_MM))         # (wide joints keep the validated global-width rule)
    reach = (WEDGE_REACH_WIDE if wide else WEDGE_REACH) * pitch(P)
    return band, reach

def add_hinge(part, env, P, y_axis, rear, wide=False, bevel_deg=None, halfwidth=None):
    """trilobite.add_hinge on the mesh builder. bevel_deg = None uses the printed stop P['maxAngle']; the instrument
    passes its own fixed bevel here — no global, no clamp in the way. halfwidth = the plate's local half-width."""
    band, reach = _hinge_geometry(P, wide, halfwidth)
    return M.hinge(part, env, y_axis=y_axis, rear=rear, zh=hinge_z(P), Wh=hinge_width(P), barrel_r=P["barrelR"],
                   clearance=P["clearance"], n_knuckles=int(P["nKnuckles"]), ring_top=ring_top(P), wall=P["wall"],
                   bore_d=P["boreDia"], band=band, reach=reach,
                   bevel_deg=(P["maxAngle"] if bevel_deg is None else float(bevel_deg)), wide=wide)

# ---------------------------------------------------------------- THORACIC SEGMENT i
def segment_plan(P, i):
    """outline(u, v), zfun(x, y) and the scalars for segment i. Transcribed from trilobite.build_segment, minus tubercles."""
    t, c, h = P["wall"], P["clearance"], P["relief"]
    d = pitch(P); ovl = P["overlap"] * d; flap = max(ovl - 2.0, 1.0)
    w = seg_halfwidth(P, i)
    a = P["axisFrac"] * w
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    sweep = P["tipSweep"] * d
    L0 = d + flap
    Ls = pleural_spine_field(P)[i] * w
    X_tip = w + Ls * math.cos(math.radians(P["spineSweep"]))
    S_sp = Ls * math.sin(math.radians(P["spineSweep"]))
    R_TIP = 0.7
    def q_of(x): return np.clip((np.abs(x) - a) / (w - a), 0, 1)
    def p_of(x): return np.clip((np.abs(x) - w) / max(X_tip - w, 1e-6), 0, 1) if Ls > 0.5 else np.zeros_like(np.asarray(x, float))
    def edges(x):
        q = q_of(x); p = p_of(x)
        yc = 0.5 * L0 + sweep * q ** 1.6 + S_sp * p ** 1.5
        root = smoothstep(0.0, 0.3, q)
        half = 0.5 * L0 * (1 - root) + 0.5 * min(P["bladeChord"] * d, L0) * root
        hb = half * (1 - P["tipTaper"] * q ** 2.5)
        hb = hb * (1 - p) ** 0.5 + R_TIP * p
        return yc - hb, yc + hb
    def outline(u, v):
        x = u * X_tip
        yf, yr = edges(np.array([abs(x)]))
        return x, float(yf[0] + v * (yr[0] - yf[0]))
    def zfun(x, y):
        ax = np.abs(x)
        z = margin + (h - margin) * vault(x / w, P)
        z += rise * plateau(x, a)
        z -= F * trough(ax - (a + 0.6), 0.9)
        z -= 0.7 * F * trough(y - d, 0.8) * plateau(x, a + 1.5, 1.0)
        yf0, _ = edges(x); px = q_of(x); ly = yf0 + 0.30 * d + px * 0.25 * d
        z -= 0.8 * F * trough(y - ly, 0.9) * (ax > a + 1.0)
        y0r = ovl - 1.0
        yc_r = 0.5 * (y0r + d); hr = max(0.5 * (d - y0r), 0.5)
        ring_hump = P["ringArch"] * h * np.clip(1 - ((y - yc_r) / hr) ** 2, 0, 1)
        yf_b, yr_b = edges(x); y0b = yf_b + ovl - 1.0
        yc_b = 0.5 * (y0b + yr_b); hb_b = np.maximum(0.5 * (yr_b - y0b), 0.5)
        blade_hump = P["bladeCamber"] * h * np.clip(1 - ((y - yc_b) / hb_b) ** 2, 0, 1)
        ax_w = plateau(x, a + 1.0, 1.5)
        window = smoothstep(ovl - 1.5, ovl + 0.5, y - yf_b)
        z += window * (ax_w * ring_hump + (1 - ax_w) * blade_hump)
        if Ls > 0.5:
            p = p_of(x)
            z = np.where(p > 0, np.maximum(margin * (1 - 0.55 * p), t + 0.6), z)
        yf, _ = edges(x)
        t_prev = t * (P["headWall"] if i == 0 else 1.0)
        drop = np.minimum(t_prev + c + 0.7 * F + 0.3, np.maximum(z - (t + 0.6), 0))
        z -= drop * (1 - smoothstep(ovl - 1.5, ovl, y - yf))
        return z
    return dict(outline=outline, zfun=zfun, edges=edges, t=t, w=w, a=a, d=d, ovl=ovl, margin=margin, rise=rise, h=h,
                X_tip=X_tip, u_a=a / X_tip, last=(i == int(P["segCount"]) - 1))

def _doublure(seg, S, P):
    """Vertical rim + inward lip along the blade margin, only on broad square-tipped pleurae (as in the BREP builder)."""
    yf_m, yr_m = (float(v[0]) for v in S["edges"](np.array([S["w"]])))
    rim_len = (yr_m - yf_m) - 1.0
    if rim_len > 2.0 and P["tipTaper"] < 0.3:
        t, w, margin = S["t"], S["w"], S["margin"]
        tools = []
        for s in (1, -1):
            tools.append(M.box(t, rim_len, max(margin - t, 1.0), at=(s * w, yf_m + 0.5, 0), align=("max" if s > 0 else "min", "min", "min")))
            tools.append(M.box(0.06 * P["width"], rim_len, t, at=(s * w, yf_m + 0.5, 0), align=("max" if s > 0 else "min", "min", "min")))
        seg = M.union(seg, *tools)
    return seg

def segment(P, i, bevel_deg=None, grid=GRID_SEG):
    """Segment i as ONE solid: shell, doublure, rear hinge (wide on the last segment), front hinge (wide on the first),
    axial spine. Same construction order as trilobite.build_segment."""
    S = segment_plan(P, i)
    seg = M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=grid[0], nv=grid[1])
    env = M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1])
    seg = _doublure(seg, S, P)
    seg = add_hinge(seg, env, P, S["d"], rear=True, wide=S["last"], bevel_deg=bevel_deg, halfwidth=S["w"])
    seg = add_hinge(seg, env, P, 0.0, rear=False, wide=(i == 0), bevel_deg=bevel_deg, halfwidth=S["w"])
    if P["axialSpine"] > 0.02:
        r = 0.45 * S["margin"]
        seg = M.union(seg, spine_solid(0.6 * r + 0.6, 0.5, P["axialSpine"] * S["h"],
                                       (0, S["ovl"] + 0.45 * (S["d"] - S["ovl"]), S["h"] + S["rise"] - 1.0), 0, pitch_deg=60))
    return seg

def segment_cells(P, i, lap=0.06, grid=GRID_SEG):
    """The row as its three cells — ring (b2), right pleura (a2), left pleura (c2 = mirror a2) — each a closed shell
    on the shared plan. Roots overlap by `lap` of the plan width so a union of the three is the fused row's shell.
    This is the Flexi row's part decomposition; hooks between them are the v1.1 joint design."""
    S = segment_plan(P, i); ua = S["u_a"]
    ring_out = lambda u, v: S["outline"](u * ua, v)
    pl_out = lambda u, v: S["outline"]((ua - lap) + 0.5 * (u + 1) * (1 - (ua - lap)), v)
    ring = M.heightfield_shell(ring_out, S["zfun"], S["t"], nu=grid[0] // 3 | 1, nv=grid[1])
    pleura = M.heightfield_shell(pl_out, S["zfun"], S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False)
    return dict(ring=ring, pleura_right=pleura, pleura_left=M.mirror_x(pleura))

# ---------------------------------------------------------------- PYGIDIUM
def pygidium_plan(P):
    """outline(u, v), zfun(x, y) and scalars for the tail. Transcribed from trilobite.build_pygidium, minus tubercles."""
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
    Lf = P["pygSpine"] * Lp
    n_m = int(P["pygMarginal"]); Lm = P["pygMarginalLen"] * Lp
    sp = math.radians(P["pygSplay"])
    phi_f = math.atan(wp / Lp * math.tan(math.pi / 2 - sp)) if Lf > 0.5 else None
    hphi_f = math.radians(13)
    phi_m = [math.radians(50 + 80 * (k + 0.5) / n_m) for k in range(n_m)]
    hphi_m = math.radians(80 / max(n_m, 1)) * 0.5 * 1.3
    def bump(phi, phi_k, hh, e): return np.clip(1 - np.abs(phi - phi_k) / hh, 0, 1) ** e
    def radial_extra(phi):
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
    PHI_MIN = math.radians(1.5)   # the margin never reaches the front line: at phi = 0 the outer grid column would lie
    def outline(u, v):            # along y = 0 and its corner triangles would be collinear (the spline fit used to hide it)
        phi = PHI_MIN + (math.pi - 2 * PHI_MIN) * 0.5 * (1 - u)
        mx, my = margin_pt(np.array([phi])); fx = u * (wp - 1.0)
        return float((1 - v) * fx + v * mx[0]), float(v * my[0])
    def outside(x, y):
        r = np.sqrt((x / wp) ** 2 + (y / Lp) ** 2)
        Lmax = max(Lf if Lf > 0.5 else 0.0, Lm if n_m else 0.0, 1e-6)
        return np.clip((r - 1) * np.hypot(x, y) / np.maximum(r, 1e-6) / Lmax, 0, 1)
    def zfun(x, y):
        ax = np.abs(x); xm = xmax(y); yy = np.clip(y / Lp, 0, 1)
        nT = P["tailDomeExp"]
        tz = np.clip(1 - yy ** nT, 0, 1) ** (1 / nT) * 0.9 + 0.1
        z = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
        ay = a * (1 - 0.85 * yy)
        z += rise * plateau(x, ay) * (1 - smoothstep(0.75 * Lp, 0.95 * Lp, y))
        z -= F * trough(ax - (ay + 0.6), 0.9) * (y < 0.9 * Lp)
        for k in range(n):
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
        ov = np.where(y > 0.35 * Lp, outside(ax, y), 0.0)
        z = np.where(ov > 0, np.maximum(margin * 0.6 * (1 - 0.5 * ov), t + 0.6), z)
        if P["tailRelief"] < 0.999:
            z = np.maximum(z, ring_top(P) * plateau(x, a, 1.0) * (1 - smoothstep(0.5 * ovl, ovl + 1.0, y)))
        drop = np.minimum(t + c + 0.7 * F + 0.3, np.maximum(z - 1.2, 0))
        z -= drop * (1 - smoothstep(ovl - 1.5, ovl, y))
        return z
    return dict(outline=outline, zfun=zfun, t=t, Lp=Lp, wp=wp, a=a, margin=margin, u_a=a / max(wp, 1e-6))

def pygidium(P, bevel_deg=None, grid=GRID_TAIL):
    """The tail as ONE solid: shell, front hinge (wide), terminal spine. Forks and marginal spines are the margin itself."""
    S = pygidium_plan(P)
    tail = M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=grid[0], nv=grid[1])
    env = M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1])
    tail = add_hinge(tail, env, P, 0.0, rear=False, wide=True, bevel_deg=bevel_deg, halfwidth=S["wp"])
    if P["termSpine"] > 0.02:
        tail = M.union(tail, spine_solid(0.5 * S["margin"], 0.6, P["termSpine"] * S["Lp"], (0, 0.85 * S["Lp"], 0.5 * S["margin"]), 0))
    if int(P.get("tailProngs", 0)) > 0:                       # posterior prong: d tines fanning rearward from the tail tip
        tail = M.union(tail, prong(int(P["tailProngs"]), P["tailProngLen"] * S["Lp"], P["tailProngSplay"],
                                   P.get("tailProngWidth", 0.45) * S["margin"] + 0.6, 0.45, (0, 0.90 * S["Lp"], 0.5 * S["margin"]),
                                   yaw_deg=0.0, pitch_deg=5.0, stem_frac=P.get("tailProngStem", 0.0), center_bias=P.get("tailProngCenter", 1.0),
                                   curl_deg=P.get("tailProngCurl", 0.0)))
    return tail

def pygidium_cells(P, lap=0.06, grid=GRID_TAIL):
    """b3 (axis) / a3 (right pleural field + margin) / c3 = mirror a3, on the shared plan."""
    S = pygidium_plan(P); ua = S["u_a"]
    ring_out = lambda u, v: S["outline"](u * ua, v)
    pl_out = lambda u, v: S["outline"]((ua - lap) + 0.5 * (u + 1) * (1 - (ua - lap)), v)
    axis = M.heightfield_shell(ring_out, S["zfun"], S["t"], nu=grid[0] // 3 | 1, nv=grid[1])
    pleura = M.heightfield_shell(pl_out, S["zfun"], S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False)
    return dict(axis=axis, pleura_right=pleura, pleura_left=M.mirror_x(pleura))

# ---------------------------------------------------------------- CEPHALON (prompt 4)
from fields import head_halfwidth

def safe_expr(expr, s, notes=None):
    """Evaluate a user formula in s with numpy math only. Bad input -> s**2 (trilobite.safe_expr)."""
    ns = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "arctan", "minimum", "maximum", "clip", "where")}
    ns.update(pi=np.pi, e=np.e, s=s)
    try:
        v = eval(compile(str(expr), "<genalPath>", "eval"), {"__builtins__": {}}, ns)
        v = np.broadcast_to(np.asarray(v, float), s.shape).copy()
        if not np.all(np.isfinite(v)): raise ValueError("non-finite")
        return v
    except Exception as ex:
        if notes is not None: notes.append(("head", "genalPath rejected", f"{expr!r}: {str(ex)[:40]}"))
        return s ** 2

def eye_geometry(P):
    """Where the eye sits and how big it is, head frame (rear hinge y = 0, head toward -y). eyes.eye_geometry, verbatim."""
    Lc = P["cephFrac"] * P["length"]
    wh = head_halfwidth(P); a = P["axisFrac"] * wh
    eR = P["eyeSize"] * wh; ye = -P["eyePos"] * Lc
    f = min(max(-ye / Lc, 0.0), 1.0)
    glab = a * (1 + (P["glabInflate"] - 1) * f)
    lat = P.get("eyeLat", 0.0)
    xe = lat * wh if lat > 0.01 else glab + eR + 1.0
    return dict(xe=float(xe), ye=float(ye), eR=float(eR), eH=float(P["eyeHeight"] * eR), arc_deg=float(P["eyeArc"]),
                exponent=float(P.get("eyeProfile", 4.0)), glab_half=float(glab), head_halfwidth=float(wh), head_length=float(Lc),
                blind=bool(P["eyeSize"] <= 0.01))

def cephalon_plan(P, notes=None):
    """outline(u, v), zfun(x, y), the crescent-arm path and scalars. Transcribed from trilobite.build_cephalon, minus
    tubercles and the skin blend (ornament and scan-fit are out of the core; no preset used either)."""
    t, c, h = P["wall"] * P["headWall"], P["clearance"], P["relief"] * P["headRelief"]
    Lc = P["cephFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P); flap = max(ovl - 2.0, 1.0)
    wh = head_halfwidth(P)
    par = P["cephParallel"] * Lc
    a = P["axisFrac"] * wh
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    Le = Lc - par
    nO = P["headOutlineExp"]
    def xmax(y):
        yy = np.asarray(y, float)
        front = np.clip((-yy - par) / Le, 0, 1)
        return np.maximum(wh * np.clip(1 - front ** nO, 0, 1) ** (1 / nO), 1.5)
    gs = P["genalSweep"] * pitch(P)
    Lg = P["genalSpine"] * Lc
    gc = math.radians(P["genalCurve"])
    arc = P["headRearArc"] * Lc; nR = P["headRearExp"]
    def cheek(u):
        q = np.clip((np.abs(u) * wh - a) / (wh - a), 0, 1)
        return flap + gs * q ** 2.2 + arc * q ** nR
    def y_front(ax):
        r = np.clip(ax / wh, 0, 1)
        return -par - Le * np.clip(1 - r ** nO, 0, 1) ** (1 / nO)
    W_arm = P["genalWidthMM"] if P["genalWidthMM"] > 0.05 else P["genalWidth"] * wh
    W_arm = max(W_arm, 1.5)
    kT = P["genalTaper"]
    S_ = np.linspace(0, 1, 241)
    path = safe_expr(P["genalPath"], S_, notes)
    path = path - path[0]
    if np.max(np.abs(path)) > 1e-9: path = path / np.max(np.abs(path))
    x_off = math.tan(gc) * Lg * path
    xc_s = (wh - 0.5 * W_arm) + x_off
    W_s = W_arm * np.clip(1 - S_ ** (2.0 / max(kT, 0.2)), 0, 1) ** (0.5 * kT) + 0.7
    X_tip = float(np.max(xc_s + 0.5 * W_s))
    def head_edges(x):
        ax = np.abs(np.asarray(x, float))
        return y_front(np.minimum(ax, wh)), cheek(np.minimum(ax, wh) / wh)
    def outline(u, v):
        x = u * X_tip
        yf, yr = head_edges(np.array([abs(x)]))
        return x, float(yr[0] - v * (yr[0] - yf[0]))
    def glab_half(y):
        """Half-width of the glabella along the head. 11 Sep 2026: ovoid, not a bar — the sides bulge, the widest point
        moves forward with inflate, and the front closes as a rounded nose (superellipse over the last 30 %, exponent
        glabFront; 2.5 = ovoid, 6 = blunt). Rear corners soften into the occipital ring."""
        f = np.clip(-np.asarray(y, float) / Lc, 0, 1)
        w = a * (1 + (P["glabInflate"] - 1) * f)
        w = w * (1 + 0.10 * np.sin(np.pi * f))                                   # bulged sides
        nF = P.get("glabFront", 2.5); fn = np.clip((f - 0.70) / 0.30, 0, 1)
        w = w * np.clip(1 - fn ** nF, 0, 1) ** (1 / nF)                          # rounded nose
        w = w * (0.75 + 0.25 * smoothstep(0.0, 0.10, f))                          # soft rear corners
        return np.maximum(w, 0.35 * a)
    w0 = seg_halfwidth(P, 0)
    G = eye_geometry(P) if P["eyeSize"] > 0.01 else None
    def zfun(x, y):
        ax = np.abs(x)
        xm = xmax(y)
        front = np.clip((-y - par) / Le, 0, 1)
        tz = np.sqrt(np.clip(1 - front ** 2, 0, 1)) * 0.9 + 0.1
        z_v4 = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
        mD, fill = P["headDomeExp"], P["headDomeFill"]
        rim = margin * 0.6
        yc = -0.19 * Lc; aD = fill * wh; bD = 0.88 * fill * Lc
        rD = (ax / aD) ** mD + (np.abs(y - yc) / bD) ** mD
        z_dome = rim + (h - rim) * np.clip(1 - rD, 0, 1) ** (1 / mD)
        z = (1 - P["tent"]) * z_v4 + P["tent"] * z_dome
        g = glab_half(y)
        gl = plateau(x, g) * (1 - smoothstep(0.80 * Lc, 0.92 * Lc, -y))
        z += (rise + P["glabRise"] * h * np.clip(-y / Lc, 0, 1)) * gl
        z -= F * trough(ax - (g + 0.7), 0.9) * (-y < 0.9 * Lc)
        z -= 0.8 * F * trough(y + 0.13 * Lc, 0.8) * plateau(x, g + 1.5, 1.0)
        for k in range(int(P["glabLobes"])):
            yk = -Lc * (0.28 + 0.16 * k)
            z -= 0.7 * F * trough(y - yk, 0.9) * trough(ax - (g - 1.2), 2.2) * (ax > 0.3 * g)
        if G is not None:
            eR, ye, xe, eH = G["eR"], G["ye"], G["xe"], G["eH"]
            r = np.hypot(ax - xe, y - ye)
            ang = np.degrees(np.arctan2(y - ye, ax - xe))
            mask = plateau(ang, P["eyeArc"] / 2, 12)
            if P.get("eyeSolid", 0) < 0.5: z += eH * np.exp(-(r / (0.95 * eR)) ** G["exponent"])
            else:                           z += 0.15 * eH * np.exp(-(r / (1.3 * eR)) ** 2)
            z += 0.35 * eH * trough(r - eR, 1.0) * mask
        if P["borderWidth"] > 0.005:
            bw = P["borderWidth"] * wh
            dist = np.minimum(xm - ax, np.where(-y > par, (1 - front) * Le, 1e9))
            z -= 0.8 * F * trough(dist - bw, 0.8 + 0.2 * bw)
            z += 0.35 * F * plateau(dist, 0.45 * bw, 0.4 * bw)
        z_th = margin + (h - margin) * vault(np.clip(ax / w0, 0, 1), P)
        rear_band = 1 - smoothstep(flap + 1.0, flap + 4.0, -y)
        z = np.maximum(z, z_th * rear_band)
        if arc > 0.5:
            hood = smoothstep(flap - 0.5, flap + 2.0, y)
            z = np.maximum(z, hood * (z_th + P["bladeCamber"] * h + t + c + 0.3))
        occ = ring_top(P) * plateau(x, a, 1.0) * (1 - smoothstep(0.10 * Lc, 0.16 * Lc, -y))
        return np.maximum(z, occ)
    # crescent arms (genal horns / harpetid prolongations): swept bands of width W_s along the user's path
    yr_root = float(cheek(np.array([1.0]))[0]); S0 = -0.12
    def arm_outline(u, v):
        s = S0 + (1 - S0) * 0.5 * (u + 1); sc = np.clip(s, 0, 1)
        xc = float(np.interp(sc, S_, xc_s)); w = float(np.interp(sc, S_, W_s))
        ds = 1e-3; xa = float(np.interp(np.clip(sc + ds, 0, 1), S_, xc_s)); xb = float(np.interp(np.clip(sc - ds, 0, 1), S_, xc_s))
        tx, ty = (xa - xb), Lg * 2 * ds; nrm = math.hypot(tx, ty); tx, ty = tx / nrm, ty / nrm
        nx, ny = ty, -tx
        off = (v - 0.5) * w
        return xc + nx * off, yr_root + Lg * s + ny * off
    def arm_z(x, y):
        ax = np.abs(x)
        z_h = max(P["marginHeight"] * h, 0.0) + t + 0.6 + c
        z_in = margin + (h - margin) * vault(np.clip(ax / w0, 0, 1), P) + P["bladeCamber"] * h + t + c + 0.3
        inboard = ax < w0 + 1.0
        z = np.where(inboard & (y > yr_root - 0.5), np.maximum(z_h, z_in), z_h)
        body = y < yr_root + 0.5
        zb = zfun(np.asarray(x, float), np.asarray(y, float)) - 0.3
        blend = smoothstep(yr_root - 0.5, yr_root + 2.5, y)
        return np.where(body, zb, (1 - blend) * zb + blend * z)
    return dict(outline=outline, zfun=zfun, xmax=xmax, arm_outline=arm_outline, arm_z=arm_z, Lg=Lg, t=t, c=c, h=h, Lc=Lc,
                wh=wh, a=a, margin=margin, u_a=a / X_tip, eye=G)

def eye_solid(R, H, slope_deg, shade, arc_deg, lensD, lensGap, lensRise, embed=1.0, max_lenses=400, clip_x=None):
    """The eye as its own solid (eye_solid.py on Manifold): revolved drum over the visual arc, a palpebral lobe inward,
    spherical-cap lenses on a hex lattice within +-arc/2 of the outward (+x) direction. Frame: axis z, outward +x,
    base at z = -embed. Returns (mesh, n_lenses)."""
    from manifold3d import Manifold, CrossSection
    slope = math.radians(slope_deg); arc = math.radians(arc_deg); rb = R + H * math.tan(slope)
    drum = Manifold.revolve(CrossSection([[(0, -embed), (rb + embed * math.tan(slope), -embed), (R, H), (0, H)]]), 64)
    if shade > 0.005:
        drum = drum + M.to_manifold(M.cylinder(R + shade * R, 0.6 * lensD * R + 0.8, at=(0, 0, H - 0.5 * (0.6 * lensD * R + 0.8))))
    a_keep = math.degrees(arc) + 24; big = 6 * rb + 6 * H
    ak = math.radians(a_keep / 2)
    sector = Manifold.extrude(CrossSection([[(0, 0), (big * math.cos(ak), -big * math.sin(ak)), (big, -big * 0.2), (big, big * 0.2),
                                             (big * math.cos(ak), big * math.sin(ak))]]), 2 * H + 4).translate((0, 0, -embed - 1))
    lobe_pts = [(0, -embed), (1.8 * R, -embed)] + [(R + 0.8 * R * math.sin(tt), -embed + (H + embed) * math.cos(tt))
                                                     for tt in [math.radians(aa) for aa in (80, 65, 50, 35, 20, 8, 0)]] + [(0, H)]
    lobe = Manifold.revolve(CrossSection([lobe_pts]), 64) - sector
    if clip_x is not None:
        lobe = lobe ^ M.to_manifold(M.box(2 * big, 4 * big, 4 * big, at=(clip_x - big, 0, 0)))
    body = (drum ^ sector) + lobe
    D = lensD * R; rise = lensRise * D / 2; rho = max((rise ** 2 + (D / 2) ** 2) / (2 * rise), 0.35)
    if D < 0.5: return M.from_manifold(body), 0                       # holochroal facets below FDM relief: smooth band
    nx, nz = math.cos(slope), math.sin(slope)
    pitch_ = D * (1 + lensGap); rows = max(1, int(H / (0.87 * pitch_))); cs = []
    for i in range(rows):
        s = (i + 0.5) * 0.87 * pitch_
        if s > H - 0.5 * D: break
        r = R + (H - s) * math.tan(slope); n = max(1, int(arc * r / pitch_)); off = 0.5 * (i % 2)
        for k in range(n):
            th = -arc / 2 + (k + off + 0.5) * arc / (n + 0.5)
            if abs(th) <= arc / 2: cs.append((th, s, r))
    cs = cs[:max_lenses]
    import trimesh
    sph = trimesh.creation.icosphere(subdivisions=2, radius=rho)
    spheres = []
    for th, s, r in cs:
        depth = rho - rise
        c = ((r - depth * nx) * math.cos(th), (r - depth * nx) * math.sin(th), s + depth * nz)
        spheres.append(M.to_manifold(sph.copy().apply_translation(c)))
    if spheres: body = body + Manifold.batch_boolean(spheres, __import__("manifold3d").OpType.Add)
    return M.from_manifold(body), len(cs)

def eye_params(P, eR):
    return dict(R=eR, H=P.get("eyeHeight", 1.7) * eR, slope_deg=P.get("eyeSlope", 15.0), shade=P.get("eyeShade", 0.0),
                arc_deg=P.get("eyeArc", 110.0), lensD=P.get("lensD", 0.16), lensGap=P.get("lensGap", 0.3), lensRise=P.get("lensRise", 0.35))

def cephalon(P, bevel_deg=None, grid=(121, 61), notes=None):
    """The head as ONE solid: shell, rear hinge (wide), crescent arms, solid eyes (if eyeSolid), occipital spine."""
    S = cephalon_plan(P, notes)
    head = M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=grid[0], nv=grid[1])
    env = M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1])
    head = add_hinge(head, env, P, 0.0, rear=True, wide=True, bevel_deg=bevel_deg, halfwidth=S["wh"])
    if S["Lg"] > 0.5:
        arm = M.heightfield_shell(S["arm_outline"], S["arm_z"], S["t"], nu=81, nv=13, symmetric=False)
        head = M.union(head, arm, M.mirror_x(arm))
    if P.get("eyeSolid", 0) > 0.5 and P["eyeSize"] > 0.01:
        G = S["eye"]; EP = eye_params(P, G["eR"])
        xm = float(S["xmax"](np.array([G["ye"]]))[0]) - 0.4
        eye, n_lens = eye_solid(**EP, clip_x=xm - G["xe"])
        zb = float(S["zfun"](np.array([G["xe"]]), np.array([G["ye"]]))[0]) - 0.3
        e = eye.copy().apply_translation((G["xe"], G["ye"], zb))
        # union the RIGHT eye only, cut the head at the sagittal plane, mirror, weld: symmetric by construction.
        # (union(head, e, mirror(e)) left a ~1 mm asymmetric patch under the left eye's base — a Manifold artifact on
        # a near-coincident lobe/shell intersection; the half-and-mirror route cannot.)
        right = M.union(head, e)
        big = 4 * max(abs(v) for v in np.asarray(right.bounds).ravel()) + 10
        half = M.intersection(right, M.box(big, 2 * big, 2 * big, at=(0, 0, 0), align=("min", "c", "c")))
        head = M.union(half, M.mirror_x(half))
        if notes is not None: notes.append(("head", "eye solid", f"{n_lens} lenses/eye"))
    if P["occipitalSpine"] > 0.02:
        head = M.union(head, spine_solid(0.6 * S["margin"], 0.5, P["occipitalSpine"] * S["Lc"], (0, -0.07 * S["Lc"], ring_top(P) - 1.0), 0, pitch_deg=55))
    if int(P.get("headProngs", 0)) > 0:                       # anterior prong: d tines fanning forward from the head front
        yf = float(S["outline"](0.0, 1.0)[1])                 # front margin at the axis (u = 0, v = 1)
        head = M.union(head, prong(int(P["headProngs"]), P["headProngLen"] * S["Lc"], P["headProngSplay"],
                                   P.get("headProngWidth", 0.45) * S["margin"] + 0.6, 0.45, (0, yf + 1.5, 0.6 * S["margin"] + 0.5),
                                   yaw_deg=180.0, pitch_deg=8.0, stem_frac=P.get("headProngStem", 0.0), center_bias=P.get("headProngCenter", 1.0),
                                   curl_deg=P.get("headProngCurl", 0.0)))
    return head

def cephalon_cells(P, lap=0.06, grid=(121, 61)):
    """b1 (glabella + occipital ring, |u| <= a/X_tip) / a1 (cheek, eye, genal angle) / c1 = mirror a1."""
    S = cephalon_plan(P); ua = S["u_a"]
    ring_out = lambda u, v: S["outline"](u * ua, v)
    pl_out = lambda u, v: S["outline"]((ua - lap) + 0.5 * (u + 1) * (1 - (ua - lap)), v)
    axis = M.heightfield_shell(ring_out, S["zfun"], S["t"], nu=grid[0] // 3 | 1, nv=grid[1])
    cheek = M.heightfield_shell(pl_out, S["zfun"], S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False)
    return dict(axis=axis, cheek_right=cheek, cheek_left=M.mirror_x(cheek))

# ---------------------------------------------------------------- PRONGS (the d-tine spine family from the sketch)
def prong(d, length, splay_deg, base_r, tip_r, at, yaw_deg=0.0, pitch_deg=0.0, root_r=None, stem_frac=0.0, center_bias=1.0, curl_deg=0.0):
    """A spine that splits into d tines: d = 1 a single spine, 2 a fork, 3 a trident, ... n.
      length      total reach from the root to the tip of an outer tine
      stem_frac   fraction of that length that is a shared shaft before the split (0 = tines from the root, Walliserops ~0.5)
      splay_deg   total fan angle of the tines in the plan, about the (yaw, pitch) direction
      center_bias length of the middle tine / outer tines (odd d only; 1.3 = a longer middle prong)
      base_r/tip_r  radius at the root and at each tip; the stem tapers from base_r to the tine base
      curl_deg    extra pitch applied at the split (tines lift or dip relative to the stem)
    Pointing +y (rear) at yaw 0; yaw > 0 turns toward +x, pitch > 0 lifts. Returns one closed solid."""
    import trimesh
    d = max(1, int(d)); R = trimesh.transformations.rotation_matrix
    stem_len = max(0.0, min(stem_frac, 0.9)) * length; tine_len = length - stem_len
    r_split = base_r - (base_r - tip_r) * (stem_len / max(length, 1e-6)) if stem_len > 0 else base_r
    solids = []
    if stem_len > 0.5:
        solids.append(spine_solid(base_r, r_split, stem_len, at, yaw_deg, pitch_deg))
    # the split point in the world frame: `at` moved stem_len along the (yaw, pitch) direction
    dirv = np.array([0.0, 1.0, 0.0, 1.0])
    Mrot = R(math.radians(-yaw_deg), (0, 0, 1)) @ R(math.radians(pitch_deg), (1, 0, 0))
    split = np.asarray(at, float) + stem_len * (Mrot @ dirv)[:3]
    for k in range(d):
        off = 0.0 if d == 1 else splay_deg * (k / (d - 1) - 0.5)
        L = tine_len * (center_bias if (d % 2 == 1 and k == d // 2) else 1.0)
        solids.append(spine_solid(r_split, tip_r, L, split, yaw_deg + off, pitch_deg + curl_deg))
    stub_r = root_r if root_r is not None else 1.15 * max(base_r, r_split)
    knot = trimesh.creation.icosphere(subdivisions=2, radius=1.15 * r_split); knot.apply_translation(split)
    root = trimesh.creation.icosphere(subdivisions=2, radius=stub_r); root.apply_translation(at)
    return M.union(root, knot, *solids)
