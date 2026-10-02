"""
anatomy/head_crescent.py: the crescent head (27 Sep 2026) as a generator head. It is scripts/head3d.py's head,
unchanged in shape: the one-piece crescent outline (round front, straight sides running on into the horns, the U
notch between them) with head3d's surface (dome, ovoid glabella, axial and glabellar furrows, occipital ring, border
rim and furrow, posterior border along the notch, the horns as a tapering ridge down to rim height). Nothing is
raised over anything: the edge stays low all the way round, horns included.

Same contract as anatomy/head.py (plan / shell / solid / ornaments / ports / cells). Chosen with assemble's
head="crescent"; the classic head stays the default and the instrument's head.

Frame and size: head frame as every head (the rear port at y = 0, the notch bottom on the axis; head toward -y).
The shape is head3d's at its defaults (R 20, arm width 4, arm length 20, notch 2 mm behind the horn line, relief 10),
scaled uniformly by head_halfwidth(P) / 24 so it is as wide as the animal's head; nothing else is read from P except
the wall, clearance and the eye settings the generator's eye solid uses.
"""
import numpy as np
from scipy.spatial import cKDTree
import mesh as M
from fields import head_halfwidth
from anatomy import head as CLASSIC
from anatomy.common import smoothstep, plateau, trough
from anatomy.port import Port

eye_geometry, fov, eye_solid, lens_centres, eye_params, FOV_VERSION = (CLASSIC.eye_geometry, CLASSIC.fov, CLASSIC.eye_solid,
                                                                       CLASSIC.lens_centres, CLASSIC.eye_params, CLASSIC.FOV_VERSION)
NAME = "crescent"
GRID = (241, 81)          # head3d's grid: 0.2 mm columns at scale 1, so the 2 mm horn tips resolve

# ---------------------------------------------------------------- head3d's shape, at scale 1 (mm)
R, ARM_W, ARM_LEN, DIP = 20.0, 4.0, 20.0, 2.0
RELIEF, RIM, FURROW = 10.0, 1.5, 1.1
DOME_EXP, DOME_FILL = 1.5, 0.82
AXIS_FRAC, GLAB_INFLATE, GLAB_RISE, GLAB_FRONT, GLAB_LOBES = 0.28, 1.35, 0.18, 2.5, 2
GLAB_RISE_GAIN = 2.0      # bold-glabella lever (2 Oct 2026): amplify glab_rise so the slider sculpts a distinct, raised
                          # glabella instead of a gentle broad bulge. 2.0 = bold-but-rounded across the slider; 3.0 slabs out
                          # at the horn-height envelope by ~0.3. Print-model only (isopod); the measured pin head is unchanged.
BORDER_W = 0.10
EYE_SIZE, EYE_POS, EYE_HEIGHT = 0.16, 0.62, 0.85
# the face, as one settable set (27 Sep 2026): every _z / _glab_half / _eye call takes face=dict(...) overriding these
FACE = dict(eye_size=EYE_SIZE, eye_pos=EYE_POS, eye_height=EYE_HEIGHT, eye_lat=0.0,   # eye radius / half-width, place along
            axis_frac=AXIS_FRAC, glab_inflate=GLAB_INFLATE,                # the head (0 rear .. 1 front), height / radius;
            glab_rise=GLAB_RISE, glab_front=GLAB_FRONT, glab_lobes=GLAB_LOBES,  # glabella width, forward swelling, rise,
            furrow=FURROW, border_w=BORDER_W,                              # nose shape, lobes; furrow depth (mm at
            suture_end=0.0, suture_depth=0.0, eye_ridge=0.0,               # 2 Oct 2026: the facial suture and the eye ridge
            # ---- sculpt keys (2 Oct 2026, from the guide's order fact sheets). Every default is "off": the approved head.
            glab_len=1.0,        # glabella length / the approved one: 1.15 reaches the front border (Phacops, Lichida,
                                 # Corynexochida), 0.7 leaves a wide preglabellar field (Harpes, Dalmanitoidea)
            glab_bulge=0.10,     # side bulge of the glabella: 0.10 ovoid (approved), 0 straight, -0.15 concave / pestle
            glab_round=0.0,      # cross-section of the glabella: 0 flat-topped (approved), 1 a rounded vault
            glab_boss=0.0,       # frontal lobe as a round boss, radius / half-width (Olenelloidea ~0.2, Deiphon ~0.4)
            boss_rise=0.35,      # ... its height / relief
            furrow_cross=0.0,    # glabellar furrows: 0 paired (approved), 1 transglabellar (Paradoxides, Agnostina)
            furrow_splay=0.0,    # splayed furrows: hind pair slants back, front pairs forward (Corynexochida), 0..1
            bullae=0.0,          # a pair of bullar lobes beside the glabella, radius / half-width (Lichoidea ~0.18)
            node=0.0,            # median tubercle on the rear of the glabella, mm (Asaphoidea's preoccipital tubercle)
            eye_len=1.0,         # fore-aft stretch of the eye lobe: 2+ is the long crescentic / band eye
            caeca=0.0,           # genal caeca: low ridges radiating over the cheek, mm (Olenidae)
            brim=0.0,            # a brim round the front, width / half-width (Harpetida ~0.35, Trinucleioidea)
            brim_drop=0.8,       # ... the step between the brim and the head inside it, mm
            brim_pits=0,         # ... rows of pits on it (the harpetid / trinucleid fringe)
            tubercles=0.0,       # tubercle height, mm (Encrinurus, Lichida); 0 = smooth
            tub_all=0,           # 0 = on the glabella only, 1 = cheeks too
            efface=0.0)          # effacement of this head: furrows AND the glabella's rise fade (Illaenina, Asaphida)
                                                                           # (schema sutureEnd / sutureDepth / eyeRidge), mm at
                                                                           # scale 1; 0 = not drawn, the head is unchanged
                                                                           # scale 1); border width / half-width.
                                                                           # eye_lat: eye centre / half-width, the
                                                                           # schema's eyeLat (0 = hug the glabella)
def face_of(face=None):
    f = dict(FACE, **(face or {}))
    unknown = set(f) - set(FACE)
    if unknown: raise KeyError(f"unknown face keys {sorted(unknown)} (have {sorted(FACE)})")
    return f
FILL = 0.75
C = R * (2 * FILL - 1)                 # horn line (widest point) in head3d's frame
RO = R + ARM_W                         # half-width
RT = 0.5 * ARM_W                       # horn tip radius
A_N = R + RT                           # notch half-width (the horns' centrelines)
D_N = ARM_LEN - DIP                    # notch depth
Y0 = C + DIP                           # notch bottom on the axis: this becomes y = 0
LC = Y0 - (C - RO)                     # head length on the axis


def _y_front(ax):
    return C - np.sqrt(np.clip(RO ** 2 - np.asarray(ax, float) ** 2, 0, None))

def _y_rear(ax):
    ax = np.abs(np.asarray(ax, float))
    base = np.where(ax <= R, np.maximum(C + ARM_LEN - RT, C + np.sqrt(np.clip(R ** 2 - ax ** 2, 0, None))),
                    C + ARM_LEN - RT + np.sqrt(np.clip(RT ** 2 - (ax - A_N) ** 2, 0, None)))
    notch = C + ARM_LEN - D_N * np.sqrt(np.clip(1 - (ax / A_N) ** 2, 0, 1))
    return np.where(ax < A_N, np.minimum(base, notch), base)

_xs = np.linspace(-A_N, A_N, 801)
_NOTCH = cKDTree(np.column_stack([_xs, C + ARM_LEN - D_N * np.sqrt(np.clip(1 - (_xs / A_N) ** 2, 0, 1))]))

def _glab_half(f, face=None):
    F_ = face_of(face); a = F_["axis_frac"] * RO; gf = F_["glab_front"]
    w = a * (1 + (F_["glab_inflate"] - 1) * f) * (1 + F_["glab_bulge"] * np.sin(np.pi * np.clip(f, 0, 1)))
    fn = np.clip((f - 0.70) / 0.30, 0, 1)
    w = w * np.clip(1 - fn ** gf, 0, 1) ** (1 / gf)
    w = w * (0.75 + 0.25 * smoothstep(0.0, 0.10, f))
    return np.maximum(w, 0.35 * a)

def _eye(face=None):
    F_ = face_of(face); eR = F_["eye_size"] * RO
    xe = F_["eye_lat"] * RO if F_["eye_lat"] > 0.01 else float(_glab_half(F_["eye_pos"], face)) + eR + 1.0
    return dict(eR=eR, ye=Y0 - F_["eye_pos"] * LC, xe=xe, eH=F_["eye_height"] * eR)
_EYE = _eye()

def _arc_s(ax, y):
    """Distance along the head's outer margin from the front of the axis (round the front, then down the side)."""
    th = np.arctan2(ax, -(np.minimum(y, C) - C) + 1e-12)
    return RO * th + np.maximum(y - C, 0.0)

def _z_base(x, y, face=None, geom=None):
    """head3d.make_zfun, at scale 1, in head3d's frame. face: overrides of FACE. Everything except the two fine
    details (suture groove, eye ridge), which _detail() adds. The sculpt keys (FACE) each add one term, guarded so a
    face that leaves them at their defaults evaluates exactly the approved expressions.
    geom (isopod_model/shaped_head.py): another OUTLINE's fields at these points, replacing the crescent's own: do / dn
    (distance to the outer and the rear margin), behind / taper / hw (the genal spine: where, how tall, half-width),
    front (1 on the head proper), arc (distance along the margin), dome_fill, horn_h. None = the crescent."""
    F_ = face_of(face); h, rim, F = RELIEF, RIM, F_["furrow"]; AF = F_["axis_frac"]
    eff = float(F_["efface"]); gl = float(F_["glab_len"])
    if eff > 0: F = F * (1 - eff)
    ax = np.abs(np.asarray(x, float)); y = np.broadcast_to(np.asarray(y, float), ax.shape)
    f = np.clip((Y0 - y) / LC, 0, 1)
    fg = f if gl == 1.0 else f / gl                              # position along the glabella (1 = its approved front)
    sg = 1.0 / LC if gl == 1.0 else 1.0 / (LC * gl)             # 1 mm along the head, in fg
    dfill = DOME_FILL if geom is None else geom["dome_fill"]
    yc = Y0 - 0.19 * LC; aD = dfill * RO; bD = 0.88 * dfill * LC; bw = F_["border_w"] * RO
    rD = (ax / aD) ** DOME_EXP + (np.abs(y - yc) / bD) ** DOME_EXP
    z = rim + (h - rim) * np.clip(1 - rD, 0, 1) ** (1 / DOME_EXP)
    g = _glab_half(fg, face)
    rise = GLAB_RISE_GAIN * F_["glab_rise"] * h * ((1 - 0.85 * eff) if eff > 0 else 1.0)
    gr = float(F_["glab_round"])
    top = plateau(ax, g) if gr == 0.0 else (1 - gr) * plateau(ax, g) + gr * np.sqrt(np.clip(1 - (ax / (g + 0.6)) ** 2, 0, 1))
    ramp = f if gl == 1.0 else np.clip(fg, 0, 1)                 # the approved glabella rises toward its front, then stops short
    nose = 1 - smoothstep(0.80, 0.92, fg)
    if gr > 0:                                                   # rounded: a hump along its length too, easing down to the front
        ramp = (1 - gr) * ramp + gr * (0.45 + 0.55 * np.sin(0.9 * np.pi * np.clip(fg, 0, 1)))
        nose = (1 - gr) * nose + gr * (1 - smoothstep(0.62, 1.0, fg))
    z = z + rise * ramp * top * nose
    z = z - F * trough(ax - (g + 0.7), 0.9) * (fg < 0.9)
    z = z - 0.8 * F * trough(f - 0.13, 0.8 / LC) * plateau(ax, g + 1.5, 1.0)
    n_l = int(F_["glab_lobes"]); cross = float(F_["furrow_cross"]); splay = float(F_["furrow_splay"])
    step = 0.16 if n_l <= 4 else 0.62 / n_l
    for k in range(n_l):
        sk = 0.28 + step * k
        if cross == 0.0 and splay == 0.0:
            z = z - 0.7 * F * trough(fg - sk, 0.9 * sg) * trough(ax - (g - 1.2), 2.2) * (ax > 0.3 * g)
        else:
            lean = splay * 0.11 * (2.0 * k / max(n_l - 1, 1) - 1.0) * np.clip(ax / np.maximum(g, 1e-6), 0, 1)   # hind back, front forward
            paired = trough(ax - (g - 1.2), 2.2) * (ax > 0.3 * g)
            z = z - 0.7 * F * trough(fg - (sk + lean), 0.9 * sg) * ((1 - cross) * paired + cross * plateau(ax, g - 0.4, 0.8))
    if F_["glab_boss"] > 0:                                      # the frontal lobe as a boss, its front at the glabella's front
        Rb = F_["glab_boss"] * RO; fc = 0.90 * gl - Rb / LC
        rb = np.hypot(ax, (f - fc) * LC)
        z = z + F_["boss_rise"] * h * np.sqrt(np.clip(1 - (rb / Rb) ** 2, 0, 1))
    if F_["bullae"] > 0:                                         # bullar lobes: ovals against the glabella's sides
        Rb = F_["bullae"] * RO; fc = 0.60 * gl; xc = float(_glab_half(0.60, face)) + 0.55 * Rb
        q = np.hypot((ax - xc) / Rb, (f - fc) * LC / (1.35 * Rb))
        z = z + 0.22 * h * np.sqrt(np.clip(1 - q ** 2, 0, 1))
    if F_["node"] > 0:
        z = z + F_["node"] * np.exp(-(np.hypot(ax, (f - 0.21) * LC) / 0.9) ** 2)
    E = _eye(face); el = float(F_["eye_len"])
    if E["eR"] > 1e-6:
        ye_ = (y - E["ye"]) if el == 1.0 else (y - E["ye"]) / el
        r = np.hypot(ax - E["xe"], ye_); ang = np.degrees(np.arctan2(ye_, ax - E["xe"]))
        z = z + 0.15 * E["eH"] * np.exp(-(r / (1.3 * E["eR"])) ** 2) + 0.35 * E["eH"] * trough(r - E["eR"], 1.0) * plateau(ang, 75, 12)
    do = RO - np.hypot(ax, np.minimum(y - C, 0.0)) if geom is None else geom["do"]
    if F_["caeca"] > 0:                                          # genal caeca: ridges radiating from the eye (or mid-cheek)
        cx, cy = (E["xe"], E["ye"]) if E["eR"] > 1e-6 else (0.55 * RO, Y0 - 0.5 * LC)
        rr = np.hypot(ax - cx, y - cy); aa = np.arctan2(y - cy, ax - cx)
        win = smoothstep(1.3 * max(E["eR"], 1.5), 2.4 * max(E["eR"], 1.5), rr) * smoothstep(bw + 0.5, bw + 2.5, do) * smoothstep(g + 1.0, g + 2.5, ax)
        z = z + F_["caeca"] * (0.5 + 0.5 * np.cos(13 * aa)) ** 2 * win
    if F_["tubercles"] > 0:                                      # a hexagonal lattice of bumps (no seed: the same every build)
        tb = float(F_["tubercles"]); p = 2.6 * tb + 1.3; rowh = 0.866 * p
        j = np.round(y / rowh); xo = 0.5 * p * (np.mod(j, 2))
        i = np.round((ax - xo) / p); d2 = (ax - (i * p + xo)) ** 2 + (y - j * rowh) ** 2
        zone = plateau(ax, g - 0.6, 0.6) * (fg < 0.93) * (f > 0.14)
        if int(F_["tub_all"]): zone = np.maximum(zone, smoothstep(bw + 0.8, bw + 2.2, do) * (f > 0.05))
        z = z + tb * np.exp(-d2 / (0.42 * tb + 0.28) ** 2) * zone
    z = z - 0.8 * F * trough(do - bw, 0.8 + 0.2 * bw) + 0.35 * F * plateau(do, 0.45 * bw, 0.4 * bw)
    dn = _NOTCH.query(np.column_stack([ax.ravel(), y.ravel()]))[0].reshape(ax.shape) if geom is None else geom["dn"]
    cheek = smoothstep(0.0, 1.5, ax - (AF * RO + 1.0))
    z = z + cheek * (0.35 * F * plateau(dn, 0.45 * bw, 0.4 * bw) - 0.8 * F * trough(dn - bw, 0.8 + 0.2 * bw))
    if F_["brim"] > 0:                                           # the brim: a step down round the front, with rows of pits
        b2 = F_["brim"] * RO; front = (1 - smoothstep(C - 2.0, C + 3.0, y)) if geom is None else geom["front"]
        # the skin this surface is laid on is thin (crescent_head.SKIN), so the brim cannot be cut down into the head:
        # everything inside it is raised by brim_drop instead, which leaves the same step at the brim's inner edge
        z = z + F_["brim_drop"] * (smoothstep(0.85 * b2, 1.15 * b2, do) * front + (1 - front))
        rows = int(F_["brim_pits"])
        if rows > 0:
            pd = 0.86 * b2 / rows; u = do / pd; jr = np.clip(np.floor(u), 0, rows - 1)
            s_ = (_arc_s(ax, y) if geom is None else geom["arc"]) / pd + 0.5 * np.mod(jr, 2)
            d2 = ((u - (jr + 0.5)) ** 2 + (s_ - np.round(s_)) ** 2) * pd ** 2
            z = z - 0.5 * np.exp(-d2 / (0.30 * pd) ** 2) * (do < 0.9 * b2) * (do > 0.08 * b2) * front
    z = np.maximum(z, h * plateau(ax, AF * RO, 1.0) * (1 - smoothstep(0.06, 0.12, f)))
    dh = np.minimum(do, dn)
    if geom is None:
        behind = smoothstep(C - 3.0, C + 2.0, y)
        taper = 1 - 0.55 * smoothstep(C, C + ARM_LEN, y)
        horn = rim + (0.32 * h - rim) * taper * np.clip(dh / (0.5 * (RO - R)), 0, 1) ** 0.6
    else:
        behind = geom["behind"]
        horn = rim + (geom["horn_h"] * h - rim) * geom["taper"] * np.clip(dh / np.maximum(geom["hw"], 0.3), 0, 1) ** 0.6
    z = np.where(behind > 0, np.maximum(z, behind * horn + (1 - behind) * z), z)
    return np.maximum(z, 0.6 * rim)


# ---------------------------------------------------------------- the facial suture and the eye ridge (2 Oct 2026)
# The classic head's two fine details (anatomy/head.py, 27 Sep 2026; Gon 2009 pp. 21, 41) on the crescent, so both
# heads read sutureEnd / sutureDepth / eyeRidge. Paths are drawn at scale 1 in the classic head frame (y = Y - Y0:
# notch bottom on the axis at 0, head toward -y), from the crescent's own outline, glabella and eye.
#   front branch + the loop round the eye's axial side: the classic head's construction, unchanged
#   rear branch, by sutureEnd:
#     -1 .. -0.5   out to the lateral margin, from the eye's rear level back to the widest line   (proparian)
#     -0.5 .. 0    into the horn's root and down the horn's centreline, to its tip at 0           (gonatoparian:
#                  the horn is the genal spine, and a gonatoparian suture bisects the genal angle)
#      0 .. +1     to the notch (the head's rear margin), from beside the horn in to 0.6 of the
#                  notch's half-width                                                              (opisthoparian)
# Sigmas are the classic head's, in scale-1 mm; plan() divides depths by its scale so the generator's crescent head
# cuts sutureDepth mm, as the classic head does. The isopod model passes the preset's mm as scale-1 mm, so its
# larger head carries a proportionally larger groove.
def _S(face=None):
    """The crescent as the classic head's path builders see a plan (scale 1, classic frame)."""
    F_ = face_of(face); E = _eye(face)
    eye = None if F_["eye_size"] <= 0.01 else dict(xe=E["xe"], ye=E["ye"] - Y0, eR=E["eR"])
    return dict(wh=RO, Lc=LC, a=F_["axis_frac"] * RO, eye=eye,
                y_front=lambda ax: _y_front(ax) - Y0,
                cheek=lambda u: _y_rear(np.asarray(u, float) * RO) - Y0,
                xmax=lambda y: np.where(np.asarray(y, float) + Y0 < C, np.sqrt(np.clip(RO ** 2 - (np.asarray(y, float) + Y0 - C) ** 2, 0, None)), RO),
                glab_half=lambda y: _glab_half(np.clip(-np.asarray(y, float) / LC, 0, 1), face))

def suture_path(face=None):
    """Right-side polyline (N, 2), scale 1, classic frame."""
    F_ = face_of(face); end = float(np.clip(F_["suture_end"], -1.0, 1.0)); S = _S(face)
    pts = [tuple(p) for p in CLASSIC.suture_path(dict(sutureEnd=-1.0), S)[:-1]]     # front margin + the palpebral loop
    y_er = pts[-1][1]                                                               # rear of the loop (or the blind head's mid-cheek point)
    y_root, y_tip = C - Y0, C + ARM_LEN - RT - Y0                                   # the widest line; the horn tip's centre
    if end <= -0.5:
        yE = y_er + 2.0 * (1.0 + end) * (y_root - y_er)
        pts.append((float(S["xmax"](yE)) - 0.3, yE))
    elif end <= 0.0:
        pts.append((A_N, y_root))
        pts.append((A_N, y_root + 2.0 * (0.5 + end) * (y_tip + RT - 0.3 - y_root)))
    else:
        xE = A_N * (1.0 - 0.4 * end)
        pts.append((xE, C + ARM_LEN - D_N * np.sqrt(max(1 - (xE / A_N) ** 2, 0.0)) - Y0 - 0.3))
    return np.array(pts, float)

def eye_ridge_path(face=None):
    """Right-side polyline (2, 2), scale 1, classic frame: beside the glabella's frontal lobe -> the eye's front."""
    return CLASSIC.eye_ridge_path({}, _S(face))

def _detail(x, y, face=None):
    """Eye ridge (raised) minus suture groove, at scale 1, in head3d's frame (as _z_base)."""
    F_ = face_of(face); ax = np.abs(np.asarray(x, float)); yc = np.broadcast_to(np.asarray(y, float), ax.shape) - Y0
    d = np.zeros(ax.shape)
    if F_["eye_ridge"] > 1e-9:
        d = d + F_["eye_ridge"] * np.exp(-0.5 * (CLASSIC._polyline_dist(ax, yc, eye_ridge_path(face)) / CLASSIC.RIDGE_SIGMA_MM) ** 2)
    if F_["suture_depth"] > 1e-9:
        d = d - F_["suture_depth"] * np.exp(-0.5 * (CLASSIC._polyline_dist(ax, yc, suture_path(face)) / CLASSIC.SUTURE_SIGMA_MM) ** 2)
    return d

def has_detail(face=None):
    F_ = face_of(face); return F_["eye_ridge"] > 1e-9 or F_["suture_depth"] > 1e-9

def _z(x, y, face=None):
    """The whole surface: _z_base, plus the fine details when the face asks for them (else _z_base, bit for bit)."""
    z = _z_base(x, y, face)
    return z + _detail(x, y, face) if has_detail(face) else z


# ---------------------------------------------------------------- the contract
def plan(P, notes=None):
    """head3d's head in the head frame, scaled to the animal's head width: x = k X, y = k (Y - Y0), z = k Z."""
    k = head_halfwidth(P) / RO
    def outline(u, v):
        X = u * RO; yr = float(_y_rear(abs(X))); yf = float(_y_front(abs(X)))
        return k * X, k * (yr - v * (yr - yf) - Y0)
    face = dict(suture_end=float(P.get("sutureEnd", 0.0)), suture_depth=float(P.get("sutureDepth", 0.0)) / k,
                eye_ridge=float(P.get("eyeRidge", 0.0)) / k)                 # mm in P -> scale-1 mm, so the built depth is P's
    fine = has_detail(face)
    def zfun_base(x, y): return k * _z_base(np.asarray(x, float) / k, np.asarray(y, float) / k + Y0)
    def detail(x, y): return k * _detail(np.asarray(x, float) / k, np.asarray(y, float) / k + Y0, face)
    zfun = (lambda x, y: zfun_base(x, y) + detail(x, y)) if fine else zfun_base
    def xmax(y):
        Y = np.asarray(y, float) / k + Y0
        return k * np.where(Y < C, np.sqrt(np.clip(RO ** 2 - (Y - C) ** 2, 0, None)), RO)
    eye = dict(xe=k * _EYE["xe"], ye=k * (_EYE["ye"] - Y0), eR=k * _EYE["eR"], eH=k * _EYE["eH"], arc_deg=float(P["eyeArc"]),
               exponent=float(P.get("eyeProfile", 4.0)), glab_half=k * float(_glab_half(EYE_POS)),
               head_halfwidth=k * RO, head_length=k * LC, blind=False)
    return dict(outline=outline, zfun=zfun, xmax=xmax, t=P["wall"] * P["headWall"], c=P["clearance"], h=k * RELIEF,
                Lc=k * LC, wh=k * RO, a=k * AXIS_FRAC * RO, margin=k * RIM, Lg=0.0, u_a=AXIS_FRAC,
                eye=eye, scale=k, horn_tip_y=k * (C + ARM_LEN - Y0),
                zfun_base=zfun_base, detail=detail if fine else None,
                suture=k * suture_path(face), eye_ridge=k * eye_ridge_path(face))

def shell(P, S, grid=GRID):
    g = (max(grid[0], GRID[0]), max(grid[1], GRID[1]))
    zb, det = CLASSIC._base(S)
    return (M.heightfield_shell(S["outline"], zb, S["t"], nu=g[0], nv=g[1], detail=det),
            M.under_envelope(S["outline"], zb, nu=g[0], nv=g[1], detail=det))

def solid(P, S, grid=GRID):
    g = (max(grid[0], GRID[0]), max(grid[1], GRID[1]))
    zb, det = CLASSIC._base(S)
    return M.under_envelope(S["outline"], zb, nu=g[0], nv=g[1], floor=0.0, detail=det)

def ornaments(P, S, notes=None):
    """The classic head's ornaments on this plan: the generator's eye solid (when P's eyeSolid is on), and the
    occipital spine and prongs if P asks for them. No genal arms: the horns are part of the outline."""
    return CLASSIC.ornaments(P, S, notes)

def ports(P, S):
    return dict(rear=Port(y=0.0, rear=True, wide=True, halfwidth=S["wh"], ring_half=S["a"], kind="head"))

def cells(P, lap=0.06, grid=GRID):
    S = plan(P); ua = S["u_a"]
    ring_out = lambda u, v: S["outline"](u * ua, v)
    pl_out = lambda u, v: S["outline"]((ua - lap) + 0.5 * (u + 1) * (1 - (ua - lap)), v)
    zb, det = CLASSIC._base(S)
    axis = M.heightfield_shell(ring_out, zb, S["t"], nu=grid[0] // 3 | 1, nv=grid[1], detail=det)
    cheek = M.heightfield_shell(pl_out, zb, S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False, detail=det)
    return dict(axis=axis, cheek_right=cheek, cheek_left=M.mirror_x(cheek))
