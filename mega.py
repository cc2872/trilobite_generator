"""
mega.py — the composite family: real anatomy blended ONTO the sculpt mesh, so every generation is a sculpt.

The Olenoides sculpt (full resolution) is the base. Library patches are turned into DISPLACEMENT fields in
body coordinates — Phacops-minus-Olenoides for the glabella, segments and tail; an eye turret; the harpetid
brim — sampled at every vertex and applied along z, weighted by how much the surface faces up. Effacement is
Laplacian smoothing of the mesh. Then the displaced mesh goes through the sculpt pipeline (wrap → slice at
its furrows → hinges → instrument), so proportions, segment count and spines come from there.
"""
import os, json, numpy as np, trimesh
from scipy.interpolate import RegularGridInterpolator
import schema, skin, wrap
HERE = os.path.dirname(os.path.abspath(__file__)); ASSETS = os.path.join(HERE, "assets")
_LIB = None

def library():
    global _LIB
    if _LIB is None:
        raw = np.load(os.path.join(ASSETS, "parts_library.npz")); meta = json.load(open(os.path.join(ASSETS, "parts_library_meta.json")))
        lib = {}
        for key, arr in raw.items():
            n, part, field = key.split("__"); lib.setdefault(n, {}).setdefault(part, {})[field] = arr
        for n in lib: lib[n]["meta"] = meta[n]
        _LIB = lib
    return _LIB

def _interp(z):
    """Bilinear sampler over a patch z(t,u) with t,u in [0,1]x[-1,1]; zero outside."""
    t = np.linspace(0, 1, z.shape[0]); u = np.linspace(-1, 1, z.shape[1])
    return RegularGridInterpolator((t, u), z, bounds_error=False, fill_value=0.0)

def _smooth(z, k=2):
    for _ in range(k):
        zp = np.pad(z, 1, mode="edge"); z = 0.25 * zp[1:-1, :-2] + 0.5 * zp[1:-1, 1:-1] + 0.25 * zp[1:-1, 2:]
        zp = np.pad(z, 1, mode="edge"); z = 0.25 * zp[:-2, 1:-1] + 0.5 * zp[1:-1, 1:-1] + 0.25 * zp[2:, 1:-1]
    return z

def displacement(body, P, pr):
    """Per-vertex dz (mm) for the composite knobs, in the base sculpt's own frame."""
    L = library(); ole, pha, har = L["olenoides"], L["phacops"], L["harpetid"]; lm = ole["meta"]["landmarks"]
    v = body.vertices; s = np.clip((v[:, 1] - pr["y0"]) / pr["L"], 0, 1); u = np.clip(v[:, 0] / np.maximum(pr["W_of"](s), 1e-6), -1, 1)
    up = np.clip(body.vertex_normals[:, 2], 0, 1) ** 0.5
    H = P["relief"] if "relief" in P else float(v[:, 2].max()); dz = np.zeros(len(v))
    s_h, s_t = lm["s_h"], lm["s_t"]
    # glabella / head: (Phacops - Olenoides) head patches, confined to the axial zone, scaled by glabBlend
    if P["glabBlend"] > 0.001 or abs(P["glabRise"] - 0.18) > 0.005:
        dh = _smooth(pha["head"]["z"] - ole["head"]["z"]); f = _interp(dh)
        ih = s <= s_h; t = s[ih] / max(s_h, 1e-6); axis = np.exp(-(u[ih] / 0.45) ** 4)
        dz[ih] += H * 0.55 * (P["glabBlend"] * f(np.c_[t, u[ih]]) * axis + (P["glabRise"] - 0.18) * axis * 0.8)
    # eyes: catalog. type 0 flattens the base's own eye lobes; type 2 adds the Phacops turret (minus baseline)
    if int(P["eyeType"]) != 1 or abs(P["eyeSize"] - 0.11) > 0.005:
        src = {0: ole, 1: ole, 2: pha}[int(P["eyeType"])]["eye"]; e = src["z"] - np.percentile(src["z"], 10); e = np.clip(_smooth(e), 0, None); fe = _interp(e)
        es, eu = 0.24 * (P["eyePos"] / 0.45), 0.30; ds, du = 0.055, 0.18
        for side in (1, -1):
            w = np.exp(-(((s - es) / ds) ** 2 + ((u - side * eu) / du) ** 2))
            tt = np.clip((s - es) / (2 * ds) + 0.5, 0, 1); uu = np.clip((u - side * eu) / du, -1, 1)
            amp = {0: -0.20, 1: 0.0, 2: 0.55}[int(P["eyeType"])] + 0.9 * (P["eyeSize"] - 0.11) * (1 if int(P["eyeType"]) else 0)
            dz += H * amp * w * fe(np.c_[tt, uu])
    # thorax: (Phacops - Olenoides) segment patch, tiled over the base's OWN furrows
    if P["segBlend"] > 0.001:
        dsg = _smooth(pha["segment"]["z"] - ole["segment"]["z"]); f = _interp(dsg); fur = lm["furrows"]
        for k in range(len(fur) - 1):
            sel = (s > fur[k]) & (s <= fur[k + 1])
            if sel.any():
                t = (s[sel] - fur[k]) / (fur[k + 1] - fur[k]); dz[sel] += H * 0.45 * P["segBlend"] * f(np.c_[t, u[sel]])
    # tail
    if P["pygBlend"] > 0.001:
        dp = _smooth(pha["pygidium"]["z"] - ole["pygidium"]["z"]); f = _interp(dp); it = s > s_t
        dz[it] += H * 0.45 * P["pygBlend"] * f(np.c_[(s[it] - s_t) / max(1 - s_t, 1e-6), u[it]])
    # brim: the harpetid rim around the head margin
    if P["brim"] >= 0.5:
        bw = P["brimWidth"]; rim = _smooth(har["brim"]["z"]); fr = _interp(rim); ih = s <= s_h + 0.08
        m = np.clip((np.abs(u[ih]) - (1 - bw)) / bw, 0, 1) ** 0.8
        dz[ih] += H * 0.35 * m * (0.6 + 0.4 * fr(np.c_[s[ih] / (s_h + 0.08), (np.abs(u[ih]) - (1 - bw)) / bw * 2 - 1]))
    return dz * up

def composite_body(P):
    """The base sculpt with the composite displacement and effacement applied (full resolution)."""
    parts, _ = skin.load(); body = parts["Trilobite_spine"].copy(); pr = wrap.profiles(body)
    P0 = skin.sculpt_reference(); Q = {**P0, **P, "relief": P0["relief"]}
    dz = displacement(body, Q, pr); v = body.vertices.copy(); v[:, 2] = np.maximum(v[:, 2] + dz, 0.0); body.vertices = v
    if P["effacement"] > 0.01:
        trimesh.smoothing.filter_laplacian(body, lamb=0.5, iterations=int(1 + 12 * P["effacement"]), volume_constraint=False)
    return body

def build_mega(P=None, verbose=False):
    P = schema.coerce(P or {})
    body = composite_body(P)
    keys = ("length", "width", "relief", "cephFrac", "pygFrac", "segCount", "widthMaxPos", "widthHeadFront", "widthThoraxFront",
            "widthThoraxRear", "widthTail", "spineBase", "genalSpine", "maxAngle", "clearance", "wall", "barrelR", "nKnuckles", "boreDia")
    return skin.build_skinned({k: P[k] for k in keys if k in P}, legs=bool(int(P.get("legs", 0))), verbose=verbose, body_override=body)
