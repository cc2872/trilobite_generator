"""
instrument.py — the enrollment ruler.

Given a built animal (as meshes), measure — deterministically, on meshes, in milliseconds per pose:
  e_max            largest uniform enrollment in [0, 1] with no collision ANYWHERE (all pairs)
  free_curl_deg    e_max × joints × stop angle
  closure_gap_mm   tail-to-head distance at e_max
  enroll_class     "none" | "partial" | "complete"
  print_valid      printability verdict, kept separate from enrollability
Versioned. Shared volume below OVERLAP_TOL is touching, not collision (mating faces touch at e=0).
"""
import math, json, time
import numpy as np
import trimesh
from trimesh.collision import CollisionManager
import schema, fields

INSTRUMENT_VERSION = "1.0"
MESH_TOL = 0.15           # tessellation tolerance for the collision meshes

# ---------------------------------------------------------------- kinematics (numpy twin of trilobite.chain_transforms)
def _rot_x(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])

def _trans(x, y, z):
    T = np.eye(4); T[:3, 3] = (x, y, z); return T

def transforms(P, enroll, joint_angles=None):
    """4x4 transform per part. Uniform enrollment unless joint_angles (degrees, one per joint) is given."""
    import trilobite as T
    zh = T.hinge_z(P); offs = T.joint_offsets(P)
    n_joints = len(offs)
    angles = joint_angles if joint_angles is not None else [enroll * P["maxAngle"]] * n_joints
    mats = [np.eye(4)]; M = np.eye(4)
    for i in range(n_joints):
        M = M @ _trans(0, offs[i], 0) @ _trans(0, 0, zh) @ _rot_x(-angles[i]) @ _trans(0, 0, -zh)
        mats.append(M)
    return mats

# ---------------------------------------------------------------- meshes
def part_meshes(P, parts=None):
    """Tessellate built parts (or build them) into trimesh objects."""
    import trilobite as T
    parts = parts or T.parts_list(P)
    return [T.to_trimesh(p, MESH_TOL, 0.3) for p in parts]

# ---------------------------------------------------------------- collision (exact overlap volume)
OVERLAP_TOL = 0.5         # mm^3 of shared volume below which two parts merely touch

def _posed(meshes, mats):
    out = []
    for m, M in zip(meshes, mats):
        p = m.copy(); p.apply_transform(M); out.append(p)
    return out

def _aabb_hit(a, b, pad=0.5):
    return bool(np.all(a.bounds[0] - pad <= b.bounds[1]) and np.all(b.bounds[0] - pad <= a.bounds[1]))

def overlap_volume(a, b):
    r = trimesh.boolean.intersection([a, b], engine="manifold")
    return float(abs(r.volume)) if r is not None and len(r.faces) else 0.0

def collisions(P, meshes, enroll, joint_angles=None):
    """List of (i, j, mm^3) for every pair of parts sharing more than OVERLAP_TOL of volume."""
    posed = _posed(meshes, transforms(P, enroll, joint_angles))
    out = []
    for i in range(len(posed)):
        for j in range(i + 1, len(posed)):
            if not _aabb_hit(posed[i], posed[j]): continue
            v = overlap_volume(posed[i], posed[j])
            if v > OVERLAP_TOL: out.append((i, j, v))
    return out

def e_max(P, meshes, tol=0.004):
    """Largest collision-free uniform enrollment in [0, 1] by bisection (checks e=1 first)."""
    if not collisions(P, meshes, 1.0): return 1.0
    if collisions(P, meshes, 0.0): return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if collisions(P, meshes, mid): hi = mid
        else: lo = mid
    return lo

def closure_gap(P, meshes, enroll):
    """Minimum distance between head and tail at this pose (0 if they overlap)."""
    posed = _posed(meshes, transforms(P, enroll))
    if _aabb_hit(posed[0], posed[-1]) and overlap_volume(posed[0], posed[-1]) > OVERLAP_TOL: return 0.0
    cm = CollisionManager(); cm.add_object("head", posed[0])
    return float(cm.min_distance_single(posed[-1]))

# ---------------------------------------------------------------- verdicts
def print_validity(P):
    P = schema.coerce(P)
    import trilobite as T
    d = T.pitch(P); zh, Wh = T.hinge_z(P), T.hinge_width(P)
    knuckle = Wh / P["nKnuckles"] - P["clearance"]; pin_wall = P["barrelR"] - P["boreDia"] / 2
    margin = P["marginHeight"] * P["relief"]
    v = []
    if P["wall"] < 1.5: v.append(f"wall {P['wall']:.1f} mm < 1.5")
    if pin_wall < 1.2: v.append(f"pin wall {pin_wall:.1f} mm < 1.2")
    if knuckle < 3.0: v.append(f"knuckle {knuckle:.1f} mm < 3.0")
    if zh < P["barrelR"] + 3: v.append(f"hinge axis {zh:.1f} mm too low")
    if d < 8: v.append(f"pitch {d:.1f} mm < 8")
    if Wh < 6: v.append(f"hinge width {Wh:.1f} mm < 6")
    if 0.5 * margin < 1.5 and (max(fields.pleural_spine_field(P)) > 0 or P["genalSpine"] > 0 or P["pygSpine"] > 0):
        v.append(f"spine base {0.5*margin:.1f} mm < 1.5")
    return dict(print_valid=len(v) == 0, violations=v, pitch=round(d, 2), hinge_z=round(zh, 2),
                hinge_width=round(Wh, 2), knuckle=round(knuckle, 2), pin_wall=round(pin_wall, 2))

def measure(P, meshes=None, parts=None):
    """The full ruler. Returns a flat dict suitable as one row of a morphospace dataset."""
    P = schema.coerce(P)
    t0 = time.time()
    meshes = meshes or part_meshes(P, parts)
    joints = P["segCount"] + 1
    em = e_max(P, meshes)
    gap = closure_gap(P, meshes, em)
    first = collisions(P, meshes, min(1.0, em + 0.01))
    cls = "none" if em < 0.05 else ("complete" if gap < 3.0 else "partial")
    out = dict(instrument=INSTRUMENT_VERSION, params=schema.param_hash(P),
               e_max=round(em, 3), free_curl_deg=round(em * joints * P["maxAngle"], 1),
               total_curl_deg=round(joints * P["maxAngle"], 1), closure_gap_mm=round(gap, 2),
               enroll_class=cls, stopped_by=[(a, b, round(v, 1)) for a, b, v in first][:6],
               touching_at_zero=len(collisions(P, meshes, 0.0)) > 0, seconds=round(time.time() - t0, 1))
    out.update(print_validity(P))
    return out

if __name__ == "__main__":
    import sys, pickle, os
    P = schema.defaults()
    if os.path.exists("parts_mesh.pkl"):
        meshes = pickle.load(open("parts_mesh.pkl", "rb"))
    else:
        meshes = part_meshes(P)
    r = measure(P, meshes)
    print(json.dumps(r, indent=1))
