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
BORDER_W = 0.10
EYE_SIZE, EYE_POS, EYE_HEIGHT = 0.16, 0.62, 0.85
# the face, as one settable set (27 Sep 2026): every _z / _glab_half / _eye call takes face=dict(...) overriding these
FACE = dict(eye_size=EYE_SIZE, eye_pos=EYE_POS, eye_height=EYE_HEIGHT, eye_lat=0.0,   # eye radius / half-width, place along
            axis_frac=AXIS_FRAC, glab_inflate=GLAB_INFLATE,                # the head (0 rear .. 1 front), height / radius;
            glab_rise=GLAB_RISE, glab_front=GLAB_FRONT, glab_lobes=GLAB_LOBES,  # glabella width, forward swelling, rise,
            furrow=FURROW, border_w=BORDER_W)                              # nose shape, lobes; furrow depth (mm at
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
    w = a * (1 + (F_["glab_inflate"] - 1) * f) * (1 + 0.10 * np.sin(np.pi * f))
    fn = np.clip((f - 0.70) / 0.30, 0, 1)
    w = w * np.clip(1 - fn ** gf, 0, 1) ** (1 / gf)
    w = w * (0.75 + 0.25 * smoothstep(0.0, 0.10, f))
    return np.maximum(w, 0.35 * a)

def _eye(face=None):
    F_ = face_of(face); eR = F_["eye_size"] * RO
    xe = F_["eye_lat"] * RO if F_["eye_lat"] > 0.01 else float(_glab_half(F_["eye_pos"], face)) + eR + 1.0
    return dict(eR=eR, ye=Y0 - F_["eye_pos"] * LC, xe=xe, eH=F_["eye_height"] * eR)
_EYE = _eye()

def _z(x, y, face=None):
    """head3d.make_zfun, at scale 1, in head3d's frame. face: overrides of FACE."""
    F_ = face_of(face); h, rim, F = RELIEF, RIM, F_["furrow"]; AF = F_["axis_frac"]
    ax = np.abs(np.asarray(x, float)); y = np.broadcast_to(np.asarray(y, float), ax.shape)
    f = np.clip((Y0 - y) / LC, 0, 1)
    yc = Y0 - 0.19 * LC; aD = DOME_FILL * RO; bD = 0.88 * DOME_FILL * LC; bw = F_["border_w"] * RO
    rD = (ax / aD) ** DOME_EXP + (np.abs(y - yc) / bD) ** DOME_EXP
    z = rim + (h - rim) * np.clip(1 - rD, 0, 1) ** (1 / DOME_EXP)
    g = _glab_half(f, face)
    z = z + F_["glab_rise"] * h * f * plateau(ax, g) * (1 - smoothstep(0.80, 0.92, f))
    z = z - F * trough(ax - (g + 0.7), 0.9) * (f < 0.9)
    z = z - 0.8 * F * trough(f - 0.13, 0.8 / LC) * plateau(ax, g + 1.5, 1.0)
    for k in range(int(F_["glab_lobes"])):
        z = z - 0.7 * F * trough(f - (0.28 + 0.16 * k), 0.9 / LC) * trough(ax - (g - 1.2), 2.2) * (ax > 0.3 * g)
    E = _eye(face)
    if E["eR"] > 1e-6:
        r = np.hypot(ax - E["xe"], y - E["ye"]); ang = np.degrees(np.arctan2(y - E["ye"], ax - E["xe"]))
        z = z + 0.15 * E["eH"] * np.exp(-(r / (1.3 * E["eR"])) ** 2) + 0.35 * E["eH"] * trough(r - E["eR"], 1.0) * plateau(ang, 75, 12)
    do = RO - np.hypot(ax, np.minimum(y - C, 0.0))
    z = z - 0.8 * F * trough(do - bw, 0.8 + 0.2 * bw) + 0.35 * F * plateau(do, 0.45 * bw, 0.4 * bw)
    dn = _NOTCH.query(np.column_stack([ax.ravel(), y.ravel()]))[0].reshape(ax.shape)
    cheek = smoothstep(0.0, 1.5, ax - (AF * RO + 1.0))
    z = z + cheek * (0.35 * F * plateau(dn, 0.45 * bw, 0.4 * bw) - 0.8 * F * trough(dn - bw, 0.8 + 0.2 * bw))
    z = np.maximum(z, h * plateau(ax, AF * RO, 1.0) * (1 - smoothstep(0.06, 0.12, f)))
    dh = np.minimum(do, dn)
    behind = smoothstep(C - 3.0, C + 2.0, y)
    taper = 1 - 0.55 * smoothstep(C, C + ARM_LEN, y)
    horn = rim + (0.32 * h - rim) * taper * np.clip(dh / (0.5 * (RO - R)), 0, 1) ** 0.6
    z = np.where(behind > 0, np.maximum(z, behind * horn + (1 - behind) * z), z)
    return np.maximum(z, 0.6 * rim)


# ---------------------------------------------------------------- the contract
def plan(P, notes=None):
    """head3d's head in the head frame, scaled to the animal's head width: x = k X, y = k (Y - Y0), z = k Z."""
    k = head_halfwidth(P) / RO
    def outline(u, v):
        X = u * RO; yr = float(_y_rear(abs(X))); yf = float(_y_front(abs(X)))
        return k * X, k * (yr - v * (yr - yf) - Y0)
    def zfun(x, y): return k * _z(np.asarray(x, float) / k, np.asarray(y, float) / k + Y0)
    def xmax(y):
        Y = np.asarray(y, float) / k + Y0
        return k * np.where(Y < C, np.sqrt(np.clip(RO ** 2 - (Y - C) ** 2, 0, None)), RO)
    eye = dict(xe=k * _EYE["xe"], ye=k * (_EYE["ye"] - Y0), eR=k * _EYE["eR"], eH=k * _EYE["eH"], arc_deg=float(P["eyeArc"]),
               exponent=float(P.get("eyeProfile", 4.0)), glab_half=k * float(_glab_half(EYE_POS)),
               head_halfwidth=k * RO, head_length=k * LC, blind=False)
    return dict(outline=outline, zfun=zfun, xmax=xmax, t=P["wall"] * P["headWall"], c=P["clearance"], h=k * RELIEF,
                Lc=k * LC, wh=k * RO, a=k * AXIS_FRAC * RO, margin=k * RIM, Lg=0.0, u_a=AXIS_FRAC,
                eye=eye, scale=k, horn_tip_y=k * (C + ARM_LEN - Y0))

def shell(P, S, grid=GRID):
    g = (max(grid[0], GRID[0]), max(grid[1], GRID[1]))
    return (M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=g[0], nv=g[1]),
            M.under_envelope(S["outline"], S["zfun"], nu=g[0], nv=g[1]))

def solid(P, S, grid=GRID):
    g = (max(grid[0], GRID[0]), max(grid[1], GRID[1]))
    return M.under_envelope(S["outline"], S["zfun"], nu=g[0], nv=g[1], floor=0.0)

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
    axis = M.heightfield_shell(ring_out, S["zfun"], S["t"], nu=grid[0] // 3 | 1, nv=grid[1])
    cheek = M.heightfield_shell(pl_out, S["zfun"], S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False)
    return dict(axis=axis, cheek_right=cheek, cheek_left=M.mirror_x(cheek))
