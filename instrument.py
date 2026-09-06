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
    """Exact shared volume (Manifold). If a mesh is not a closed volume, fall back to FCL contact
    with a penetration threshold — approximate, but never silently 'no collision'."""
    if a.is_volume and b.is_volume:
        r = trimesh.boolean.intersection([a, b], engine="manifold")
        return float(abs(r.volume)) if r is not None and len(r.faces) else 0.0
    # Monte-Carlo estimate: volume samples of the smaller part tested against the other. Approximate
    # (±~10% at this sample count) but a real mm^3 number, not a binary guess from FCL contact depth.
    small, big = (a, b) if abs(a.volume) <= abs(b.volume) else (b, a)
    try:
        pts = trimesh.sample.volume_mesh(small, 6000)
        if len(pts) == 0: return 0.0
        inside = np.concatenate([big.contains(pts[k:k + 1000]) for k in range(0, len(pts), 1000)])   # chunked: no embree here
        return float(np.mean(inside)) * abs(small.volume)
    except Exception:
        cm = CollisionManager(); cm.add_object("a", a)
        hit, data = cm.in_collision_single(b, return_data=True)
        depth = max((d.depth for d in data), default=0.0)
        return OVERLAP_TOL + 1.0 if depth > 0.5 else 0.0

class MeasureTimeout(Exception):
    """Raised when a measurement pass can't finish within its time budget (see measure())."""

def collisions(P, meshes, enroll, joint_angles=None, deadline=None):
    """List of (i, j, mm^3) for every pair of parts sharing more than OVERLAP_TOL of volume."""
    posed = _posed(meshes, transforms(P, enroll, joint_angles))
    out = []
    for i in range(len(posed)):
        for j in range(i + 1, len(posed)):
            if deadline is not None and time.time() > deadline:
                raise MeasureTimeout(f"collisions() exceeded its time budget ({len(out)} pairs found so far)")
            if not _aabb_hit(posed[i], posed[j]): continue
            v = overlap_volume(posed[i], posed[j])
            if v > OVERLAP_TOL: out.append((i, j, v))
    return out

def e_max(P, meshes, tol=0.004, deadline=None):
    """Largest collision-free uniform enrollment in [0, 1] by bisection (checks e=1 first)."""
    if not collisions(P, meshes, 1.0, deadline=deadline): return 1.0
    if collisions(P, meshes, 0.0, deadline=deadline): return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if collisions(P, meshes, mid, deadline=deadline): hi = mid
        else: lo = mid
    return lo

def closure_gap(P, meshes, enroll, deadline=None):
    """Minimum distance between head and tail at this pose (0 if they overlap)."""
    if deadline is not None and time.time() > deadline:
        raise MeasureTimeout("closure_gap() exceeded its time budget")
    posed = _posed(meshes, transforms(P, enroll))
    if _aabb_hit(posed[0], posed[-1]) and overlap_volume(posed[0], posed[-1]) > OVERLAP_TOL: return 0.0
    cm = CollisionManager(); cm.add_object("head", posed[0])
    return float(cm.min_distance_single(posed[-1]))

# ---------------------------------------------------------------- verdicts
def volume_sanity(P, parts):
    """A failed OpenCascade fuse returns the tool, not an error — the shield 'vanished' this way once, and a
    segment once came back inside-out. Compare each part's volume with a crude expectation from its bounding
    box: a plate that lost its shell is well under 40 % of a shell-thick slab over its footprint; a bad body is
    negative or absurdly large. Returns a list of violation strings (empty = sane)."""
    import trilobite as T
    v = []
    for name, part in zip(T.PART_NAMES(P), parts):
        vol = float(part.volume); bb = part.bounding_box().size
        footprint = float(bb.X * bb.Y)
        expect = footprint * P["wall"] * 0.625           # a plate over ~62 % of the box; 40 % of that = 25 % footprint
        if vol <= 0: v.append(f"{name}: non-positive volume {vol:.0f} mm3 (failed boolean)")
        elif vol < 0.4 * expect: v.append(f"{name}: volume {vol:.0f} mm3 < 40 % of expected ~{expect:.0f} (part lost in a boolean?)")
        elif vol > footprint * bb.Z * 1.05: v.append(f"{name}: volume {vol:.0f} mm3 exceeds its bounding box (inside-out body)")
        n = len(part.solids())
        if n != 1: v.append(f"{name}: {n} solids")
    return v

def print_validity(P, parts=None):
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
    # spines are in-plane continuations of the plates (chord ≥ 1.4 mm, wall thickness): no separate base rule
    if parts is not None: v += volume_sanity(P, parts)
    return dict(print_valid=len(v) == 0, violations=v, pitch=round(d, 2), hinge_z=round(zh, 2),
                hinge_width=round(Wh, 2), knuckle=round(knuckle, 2), pin_wall=round(pin_wall, 2))

MEASURE_BUDGET_S = 60     # wall-clock ceiling for the whole measuring pass (see measure())

def measure(P, meshes=None, parts=None, budget_s=MEASURE_BUDGET_S):
    """The full ruler. Returns a flat dict suitable as one row of a morphospace dataset.

    Bounded by budget_s: a part that tessellates non-watertight (rare, but not fully eliminated by
    trilobite.py's build-time retry) forces overlap_volume()'s Monte-Carlo/FCL fallback, which can be
    ~50x slower per pair than the normal exact-boolean path. A bisection search plus several full
    collision passes can then compound that into minutes, and on a slow/shared host, worse. Rather than
    let one pathological build hang the whole single-process server indefinitely, give up cleanly within
    budget_s and report the measurement as incomplete instead of presenting a number that was never
    actually verified."""
    P = schema.coerce(P)
    t0 = time.time()
    deadline = t0 + budget_s
    meshes = meshes or part_meshes(P, parts)
    joints = P["segCount"] + 1
    try:
        em = e_max(P, meshes, deadline=deadline)
        gap = closure_gap(P, meshes, em, deadline=deadline)
        first = collisions(P, meshes, min(1.0, em + 0.01), deadline=deadline)
        touching = len(collisions(P, meshes, 0.0, deadline=deadline)) > 0
        cls = "none" if em < 0.05 else ("complete" if gap < 3.0 else "partial")
        out = dict(instrument=INSTRUMENT_VERSION, params=schema.param_hash(P), measure_timed_out=False,
                   e_max=round(em, 3), free_curl_deg=round(em * joints * P["maxAngle"], 1),
                   total_curl_deg=round(joints * P["maxAngle"], 1), closure_gap_mm=round(gap, 2),
                   enroll_class=cls, stopped_by=[(a, b, round(v, 1)) for a, b, v in first][:6],
                   touching_at_zero=touching, seconds=round(time.time() - t0, 1))
    except MeasureTimeout:
        out = dict(instrument=INSTRUMENT_VERSION, params=schema.param_hash(P), measure_timed_out=True,
                   e_max=None, free_curl_deg=None, total_curl_deg=round(joints * P["maxAngle"], 1),
                   closure_gap_mm=None, enroll_class="unknown", stopped_by=[], touching_at_zero=None,
                   seconds=round(time.time() - t0, 1))
    out.update(print_validity(P, parts))
    return out

def measure_worker(P, mesh_paths, budget_s, result_path):
    """Entry point for a measuring subprocess (see trilobite_web.py's _measure_hard_bounded()).
    measure()'s own budget_s can only check the clock between pairwise checks - it can't interrupt a
    single already-in-flight OpenCascade/trimesh C call, and in production that alone has hung the
    whole single-process server for over an hour on a pathological mesh. Running this in a subprocess
    lets the parent hard-kill it regardless of what it's stuck inside. Loads the STL files the caller
    already tessellated and exported to disk, rather than rebuilding the CAD model a second time.
    Must stay at module level (not a closure) so it can be pickled as a multiprocessing.Process target
    on platforms using the 'spawn' start method (Windows)."""
    import json
    import trimesh
    try:
        meshes = [trimesh.load(p) for p in mesh_paths]
        result = measure(P, meshes, parts=None, budget_s=budget_s)
    except Exception as ex:
        joints = P["segCount"] + 1
        result = dict(instrument=INSTRUMENT_VERSION, params=schema.param_hash(P), measure_timed_out=True,
                      e_max=None, free_curl_deg=None, total_curl_deg=round(joints * P["maxAngle"], 1),
                      closure_gap_mm=None, enroll_class="unknown", stopped_by=[], touching_at_zero=None,
                      seconds=None, error=f"measure worker failed: {ex}")
    with open(result_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    import sys, pickle, os
    P = schema.defaults()
    if os.path.exists("parts_mesh.pkl"):
        meshes = pickle.load(open("parts_mesh.pkl", "rb"))
    else:
        meshes = part_meshes(P)
    r = measure(P, meshes)
    print(json.dumps(r, indent=1))
