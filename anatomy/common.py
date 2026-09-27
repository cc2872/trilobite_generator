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
        return _SAFE_BINOPS[type(node.op)](_eval_node(node.left, names), _eval_node(node.right, names))
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
