"""
anatomy/common.py: the vocabulary every part is drawn with. Nothing here knows about joints.

Derived scalars (pitch, ring_top), the surface primitives (smoothstep, plateau, trough, vault), the grafted solids
(spine_solid, prong), the safe evaluator for user path formulas, and the sampling grids. Transcribed unchanged from
parts.py (11 Sep 2026) so every mesh stays bit-identical.
"""
import ast
import math
import operator
import numpy as np
import mesh as M
from fields import seg_halfwidth, furrow_amp

GRID_SEG = (121, 61)      # (nu, nv), 10 Sep verdict: 120x60 per part; nu odd so the axis is a vertex column
GRID_TAIL = (121, 61)
GRID_HEAD = (121, 61)

# ---------------------------------------------------------------- derived scalars (trilobite.py, unchanged)
def pitch(P): return P["length"] * (1 - P["cephFrac"] - P["pygFrac"]) / P["segCount"]
def ring_top(P): return P["relief"] * (1 + P["axisRise"])

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
    m = P.get("vaultRound", 1.0)                       # 27 Sep 2026: < 1 rounds the flank into a helmet (0.62 = a quarter circle,
    if m != 1.0: outer = 0.65 * np.clip(outer / 0.65, 0, 1) ** m   # lower = fuller top, steeper sides). 1.0 = the v4 profile, untouched.
                                                       # (u runs past 1 at pleural tips and genal arms, where the legacy form goes negative)
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

# ---------------------------------------------------------------- CURVED SPINES (23 Oct 2026)
# A spine as a curve, not a line. Every spine family (occipital, genal, pleural, pygidial, prongs' tines) is a straight
# tapered frustum placed by yaw and pitch; a recurved horn (Dicranurus, Ceratarges), a hooked genal spine or a tine that
# arches cannot be. curved_spine() lays the same taper along an arc that bends by `bend_deg` over its length, in a plane
# turned `roll_deg` about the spine's own axis (0 = bends upward, 90 = bends toward +x, 180 = bends down). With
# bend_deg = 0 it returns spine_solid() exactly, so every existing spine is unchanged. Two numbers per family (bend, roll)
# cover a horn, a hook and an arch; the arc is a circle, which is what the limits of a spine curling under its own
# growth look like in the fossils and is the smallest description that has a tip direction.
SPINE_SEGMENTS = 24      # 24 Oct 2026: was 10 (a visibly beaded arc at print size); the joints are finer spheres too

def curved_spine(base_r, tip_r, length, at, yaw_deg, pitch_deg=0.0, bend_deg=0.0, roll_deg=0.0, n=SPINE_SEGMENTS, flare=0.0, plate=None):
    """A tapered spine along a circular arc: `bend_deg` is the total turn of the tangent from root to tip (0 = straight,
    180 = a half circle, tip pointing back at the root), `roll_deg` the plane it bends in about the spine's own axis
    (0 up, 90 toward +x, -90 toward -x, 180 down). Root at `at`, leaving along (yaw, pitch) as spine_solid does.
    Built as one swept tube (sweep_tube). bend_deg = 0 returns spine_solid().
    flare (24 Oct 2026): a root collar, `flare` x base_r wide at the root narrowing to base_r over the first 0.12 of the
    length, so a horn grows out of the shield instead of standing on it (0 = none, as every spine was).
    plate (24 Oct 2026, later): the thickness of the plate the spine leaves, model mm. The root is then an ELLIPSE no
    thicker than the plate (vertically) and `flare` wide across it, turning round over the first 0.35 of the length: a
    spine is the plate continuing, not a rod stood on it (a 14 mm collar on a 3 mm shield edge looked unnatural)."""
    if abs(bend_deg) < 1e-9 and flare <= 1.0 and plate is None: return spine_solid(base_r, tip_r, length, at, yaw_deg, pitch_deg)
    import trimesh
    R = trimesh.transformations.rotation_matrix
    n = max(2, int(n)); phi = math.radians(bend_deg); rad = length / abs(phi) if abs(phi) > 1e-9 else 0.0
    # the arc in its own frame: along +y from the origin, bending toward +z (roll 0); then rolled about y, pitched, yawed
    ts = np.linspace(0.0, 1.0, n + 1); ang = ts * phi
    pts = np.c_[np.zeros(n + 1), rad * np.sin(np.abs(ang)), np.sign(phi) * rad * (1 - np.cos(ang))] if abs(phi) > 1e-9 else np.c_[np.zeros(n + 1), ts * length, np.zeros(n + 1)]
    Mrot = R(math.radians(-yaw_deg), (0, 0, 1)) @ R(math.radians(pitch_deg), (1, 0, 0)) @ R(math.radians(roll_deg), (0, 1, 0))
    W = (np.c_[pts, np.ones(n + 1)] @ Mrot.T)[:, :3] + np.asarray(at, float)
    radii = base_r + (tip_r - base_r) * ts; ra = radii.copy()
    if flare > 1.0:                                                       # the collar: flare x base_r at the root -> base_r at 0.12 of the length (in the plane only)
        c = np.clip(ts / 0.12, 0, 1); ra = np.maximum(ra, base_r * (flare - (flare - 1.0) * (3 * c ** 2 - 2 * c ** 3)))
    if plate is None: return sweep_tube(W, ra)
    c = np.clip(ts / 0.35, 0, 1); c = 3 * c ** 2 - 2 * c ** 3                # the vertical half-thickness: the plate's at the root -> round by 0.35
    rb = np.minimum(radii, 0.5 * float(plate)) * (1 - c) + radii * c
    return sweep_tube(W, ra, rb=rb)

def sweep_tube(path, radii, rb=None, sections=32):
    """One closed surface along a polyline: a ring of `sections` points at every path point (radius radii[i]), frames
    carried by parallel transport so the rings never twist, flat caps at both ends. Replaces the frustum-and-sphere
    chain (24 Oct 2026): that was one solid per segment with a seam at every joint, a visibly ribbed horn at print size.
    rb: a second radius per point for an elliptical section -- radii[i] across the HORIZONTAL (in-plane) axis, rb[i] up
    the vertical one; the frames are then horizontal-anchored (no transport) so "vertical" stays vertical."""
    import trimesh
    P = np.asarray(path, float); r = np.asarray(radii, float); m = len(P)
    T = np.gradient(P, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1), 1e-12)[:, None]
    if rb is not None:                                                    # horizontal-anchored frames: N = T x z (the in-plane direction)
        rb = np.asarray(rb, float); frames = []; prev = None
        for i in range(m):
            N = np.cross(T[i], [0, 0, 1.0]); nn = np.linalg.norm(N)
            N = N / nn if nn > 0.1 else (prev if prev is not None else np.array([1.0, 0, 0]))
            N = N - T[i] * np.dot(N, T[i]); N /= max(np.linalg.norm(N), 1e-12); frames.append(N); prev = N
    else:
        N = np.cross(T[0], [1.0, 0, 0]);
        if np.linalg.norm(N) < 1e-6: N = np.cross(T[0], [0, 1.0, 0])
        N /= np.linalg.norm(N); frames = [N]
        for i in range(1, m):                                             # parallel transport of N along the tangent
            b = np.cross(T[i - 1], T[i]); sb = np.linalg.norm(b)
            if sb > 1e-9:
                b /= sb; th = math.atan2(sb, float(np.dot(T[i - 1], T[i])))
                N = N * math.cos(th) + np.cross(b, N) * math.sin(th) + b * np.dot(b, N) * (1 - math.cos(th))
            N = N - T[i] * np.dot(N, T[i]); N /= max(np.linalg.norm(N), 1e-12); frames.append(N)
    th = np.linspace(0, 2 * np.pi, sections, endpoint=False); V = []
    for i in range(m):
        Bv = np.cross(T[i], frames[i]); rbi = r[i] if rb is None else rb[i]
        V.append(P[i] + r[i] * np.cos(th)[:, None] * frames[i] + rbi * np.sin(th)[:, None] * Bv)
    V = np.vstack(V); F = []
    for i in range(m - 1):
        for j in range(sections):
            a, b = i * sections + j, i * sections + (j + 1) % sections; c, d = a + sections, b + sections
            F += [[a, b, d], [a, d, c]]
    c0, c1 = len(V), len(V) + 1; V = np.vstack([V, P[0], P[-1]])
    for j in range(sections):
        F.append([c0, (j + 1) % sections, j]); F.append([c1, (m - 1) * sections + j, (m - 1) * sections + (j + 1) % sections])
    mesh = trimesh.Trimesh(V, np.asarray(F), process=True)
    if mesh.volume < 0: mesh.invert()
    return mesh

def spine_tip(length, at, yaw_deg, pitch_deg=0.0, bend_deg=0.0, roll_deg=0.0):
    """Where curved_spine()'s tip lands (for clearances and the sheet)."""
    import trimesh
    R = trimesh.transformations.rotation_matrix
    if abs(bend_deg) < 1e-9: p = np.array([0.0, length, 0.0, 1.0])
    else:
        phi = math.radians(bend_deg); rad = length / abs(phi)
        p = np.array([0.0, rad * math.sin(abs(phi)), math.copysign(rad * (1 - math.cos(phi)), phi), 1.0])
    Mrot = R(math.radians(-yaw_deg), (0, 0, 1)) @ R(math.radians(pitch_deg), (1, 0, 0)) @ R(math.radians(roll_deg), (0, 1, 0))
    return (Mrot @ p)[:3] + np.asarray(at, float)

# genalPath is a user-supplied formula from the website; it MUST NOT be eval()'d.
# Even eval() with an empty __builtins__ is remote code execution (a whitelisted
# ufunc's __globals__ reaches os/import). Instead we parse to an AST and walk it,
# permitting only numeric literals, the variables s/pi/e, arithmetic, comparisons,
# and the whitelisted numpy functions below. Anything else raises -> fallback s**2.
_SAFE_FUNCS = {k: getattr(np, k) for k in (
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "arctan",
    "minimum", "maximum", "clip", "where")}
_SAFE_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
                ast.FloorDiv: operator.floordiv}
_SAFE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_SAFE_CMPOPS = {ast.Lt: operator.lt, ast.Gt: operator.gt, ast.LtE: operator.le,
                ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}

def _eval_node(node, names):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"unknown name {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        lhs = _eval_node(node.left, names); rhs = _eval_node(node.right, names)
        if isinstance(node.op, ast.Pow) and abs(rhs) > 64: raise ValueError("exponent too large")   # 9**9**9 would hang inside the build lock
        return _SAFE_BINOPS[type(node.op)](lhs, rhs)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_eval_node(node.operand, names))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, names); result = None
        for op, right_node in zip(node.ops, node.comparators):
            if type(op) not in _SAFE_CMPOPS:
                raise ValueError("comparison operator not allowed")
            right = _eval_node(right_node, names)
            cmp = _SAFE_CMPOPS[type(op)](left, right)
            result = cmp if result is None else (result & cmp)
            left = right
        return result
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
            raise ValueError("only positional calls to whitelisted functions")
        return _SAFE_FUNCS[node.func.id](*[_eval_node(a, names) for a in node.args])
    raise ValueError(f"disallowed expression element: {type(node).__name__}")

def safe_expr(expr, s, notes=None):
    """Evaluate a user formula in s with numpy math only, via an AST whitelist (no
    eval/exec). Bad or unsafe input -> s**2 (trilobite.safe_expr)."""
    names = dict(_SAFE_FUNCS); names.update(pi=np.pi, e=np.e, s=s)
    try:
        v = _eval_node(ast.parse(str(expr), mode="eval"), names)
        v = np.broadcast_to(np.asarray(v, float), s.shape).copy()
        if not np.all(np.isfinite(v)): raise ValueError("non-finite")
        return v
    except Exception as ex:
        if notes is not None: notes.append(("head", "genalPath rejected", f"{expr!r}: {str(ex)[:40]}"))
        return s ** 2

# ---------------------------------------------------------------- PRONGS (the d-tine spine family from the sketch)
def prong(d, length, splay_deg, base_r, tip_r, at, yaw_deg=0.0, pitch_deg=0.0, root_r=None, stem_frac=0.0, center_bias=1.0, curl_deg=0.0,
          bend_deg=0.0, roll_deg=0.0):
    """A spine that splits into d tines: d = 1 a single spine, 2 a fork, 3 a trident, ... n.
      length      total reach from the root to the tip of an outer tine
      stem_frac   fraction of that length that is a shared shaft before the split (0 = tines from the root, Walliserops ~0.5)
      splay_deg   total fan angle of the tines in the plan, about the (yaw, pitch) direction
      center_bias length of the middle tine / outer tines (odd d only; 1.3 = a longer middle prong)
      base_r/tip_r  radius at the root and at each tip; the stem tapers from base_r to the tine base
      curl_deg    extra pitch applied at the split (tines lift or dip relative to the stem)
      bend_deg, roll_deg (23 Oct 2026)  each tine arches along its length (curved_spine); 0 = straight, as before
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
        solids.append(curved_spine(r_split, tip_r, L, split, yaw_deg + off, pitch_deg + curl_deg, bend_deg, roll_deg))
    stub_r = root_r if root_r is not None else 1.15 * max(base_r, r_split)
    knot = trimesh.creation.icosphere(subdivisions=2, radius=1.15 * r_split); knot.apply_translation(split)
    root = trimesh.creation.icosphere(subdivisions=2, radius=stub_r); root.apply_translation(at)
    return M.union(root, knot, *solids)
