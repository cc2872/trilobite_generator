"""
instrument2.py — the enrollment ruler, version 2.0.

Why 2.0: version 1.0 swept a fraction e in [0, 1] of the printed stop angle (maxAngle, 18 deg/joint).
Its ceiling was therefore (segCount + 1) x 18 deg — a function of segment count, one of the
morphospace axes — and no preset ever collided before the stop (e_max = 1.0, stopped_by empty for
every animal). The ruler was measuring itself.

2.0 separates the measurement from the manufactured stop:
  * the measurement meshes are built with ONE fixed, wide ventral bevel (BOUND_DEG per joint, the
    same for every animal), so the stop cannot be the limiter inside the sweep;
  * the sweep is in degrees per joint over [0, BOUND_DEG], bisected to the first anatomical
    collision or to head-tail closure;
  * the reading is theta_joint_deg, total_deg = theta x joints, closure gap (mm and / length), and
    limited_by in {anatomy, closed, bound}. 'bound' means the ruler ran out before the animal did
    and the number is censored — it is reported as such, never as a measurement.
The printed stop is then a manufacturing choice made AFTER measurement (stop_recommended_deg).
Collision machinery (exact Manifold overlap, OVERLAP_TOL, MeasureTimeout) is inherited from 1.0.
"""
import math, json, time
import numpy as np
import schema
from instrument import (_rot_x, _trans, _posed, _aabb_hit, overlap_volume, OVERLAP_TOL,
                        MeasureTimeout, print_validity, MEASURE_BUDGET_S, MESH_TOL)
from trimesh.collision import CollisionManager

INSTRUMENT_VERSION = "2.0"
BOUND_DEG = 45.0          # per-joint bevel of the measurement build; the sweep's hard ceiling
CLOSED_GAP_MM = 3.0       # head-tail gap below which the animal is called closed
STOP_MARGIN_DEG = 2.0     # recommended printed stop = theta_joint - this (never above BOUND_DEG)

def measurement_params(P, bound_deg=BOUND_DEG):
    """The parameter set the measurement meshes are built from: identical to P except that the
    ventral bevel (maxAngle drives the wedge cut in trilobite.add_hinge) is opened to bound_deg."""
    P = schema.coerce(P)
    return schema.coerce(dict(P, maxAngle=min(bound_deg, schema.BY_KEY["maxAngle"].hi)))

def transforms_deg(P, theta_deg):
    """4x4 transform per part for a uniform flexion of theta_deg at every joint."""
    import trilobite as T
    zh = T.hinge_z(P); offs = T.joint_offsets(P)
    mats = [np.eye(4)]; M = np.eye(4)
    for i in range(len(offs)):
        M = M @ _trans(0, offs[i], 0) @ _trans(0, 0, zh) @ _rot_x(-theta_deg) @ _trans(0, 0, -zh)
        mats.append(M)
    return mats

_REST = {}                # id(meshes) -> {(i, j): overlap at theta = 0}; designed interference (stepped fronts, 0.5 mm^3 budget)

def rest_overlaps(P, meshes, deadline=None):
    """Pairwise shared volume at theta = 0. Mating faces touch by design (the tail's stepped front budgets 0.5 mm^3);
    a collision is overlap that GROWS from this baseline, not overlap that was built in."""
    key = id(meshes)
    if key not in _REST:
        posed = _posed(meshes, transforms_deg(P, 0.0)); base = {}
        for i in range(len(posed)):
            for j in range(i + 1, len(posed)):
                if deadline is not None and time.time() > deadline: raise MeasureTimeout("rest_overlaps() exceeded its budget")
                if _aabb_hit(posed[i], posed[j]): base[(i, j)] = overlap_volume(posed[i], posed[j])
        _REST[key] = base
    return _REST[key]

def collisions_deg(P, meshes, theta_deg, deadline=None):
    rest = rest_overlaps(P, meshes, deadline)
    posed = _posed(meshes, transforms_deg(P, theta_deg))
    out = []
    for i in range(len(posed)):
        for j in range(i + 1, len(posed)):
            if deadline is not None and time.time() > deadline:
                raise MeasureTimeout(f"collisions_deg() exceeded its budget ({len(out)} pairs so far)")
            if not _aabb_hit(posed[i], posed[j]): continue
            v = overlap_volume(posed[i], posed[j]) - rest.get((i, j), 0.0)      # 8 Sep 2026: baseline-subtracted
            if v > OVERLAP_TOL: out.append((i, j, v))
    return out

def head_body_mesh(Pm):
    """The cephalon with its genal arms / prolongations switched off: the surface the tail must reach
    for the animal to count as closed. A harpetid's brim prolongations reach most of the way to its
    tail at rest; measuring closure against them reads 'closed' when the tail merely taps an arm."""
    import trilobite as T
    if Pm["genalSpine"] * Pm["cephFrac"] * Pm["length"] <= 0.5: return None
    part = T._build_checked(lambda: T.build_cephalon(dict(Pm, genalSpine=0.0)), "head_body")
    return T.to_trimesh(part, *T.SANE_MESH_TOL)

def closure_gap_deg(P, meshes, theta_deg, deadline=None, head_body=None):
    """Minimum tail-to-head-BODY distance at this flexion (0 if they overlap)."""
    if deadline is not None and time.time() > deadline:
        raise MeasureTimeout("closure_gap_deg() exceeded its budget")
    mats = transforms_deg(P, theta_deg)
    head = (head_body if head_body is not None else meshes[0]).copy(); head.apply_transform(mats[0])
    tail = meshes[-1].copy(); tail.apply_transform(mats[-1])
    if _aabb_hit(head, tail) and overlap_volume(head, tail) > OVERLAP_TOL: return 0.0
    cm = CollisionManager(); cm.add_object("head", head)
    return float(cm.min_distance_single(tail))

def theta_max(P, meshes, bound_deg=BOUND_DEG, tol_deg=0.1, deadline=None):
    """Largest collision-free uniform flexion in degrees per joint, bisected on [0, bound_deg].
    Returns (theta, hit_at_bound) where hit_at_bound is False if even the bound is collision-free."""
    rest = rest_overlaps(P, meshes, deadline)                    # theta = 0 is collision-free by definition (baseline)
    if not collisions_deg(P, meshes, bound_deg, deadline=deadline): return bound_deg, False
    lo, hi = 0.0, bound_deg
    while hi - lo > tol_deg:
        mid = 0.5 * (lo + hi)
        if collisions_deg(P, meshes, mid, deadline=deadline): hi = mid
        else: lo = mid
    return lo, True

def measure(P, meshes, parts=None, bound_deg=BOUND_DEG, budget_s=MEASURE_BUDGET_S, head_body=None):
    """The full 2.0 ruler. `meshes` MUST come from measurement_params(P) — i.e. the wide-bevel build —
    or the bevel becomes the limiter again and the reading is 1.0's reading in different units."""
    P = schema.coerce(P)
    t0 = time.time(); deadline = t0 + budget_s
    joints = int(P["segCount"]) + 1; n_last = joints            # part index of the tail
    L = float(P["length"])
    base = dict(instrument=INSTRUMENT_VERSION, params=schema.param_hash(P), bound_deg=bound_deg,
                joints=joints, printed_stop_deg=P["maxAngle"], collision_rule=f"overlap - overlap_at_rest > {OVERLAP_TOL} mm3")
    try:
        th, hit = theta_max(P, meshes, bound_deg, deadline=deadline)
        gap = closure_gap_deg(P, meshes, th, deadline=deadline, head_body=head_body)
        first = collisions_deg(P, meshes, min(bound_deg, th + 0.2), deadline=deadline) if hit else []
        pairs = [(a, b) for a, b, _ in first]
        # closed only if the tail is at the head BODY; a head-tail collision on the arms is anatomy
        if not hit:                         limited = "bound"
        elif gap < CLOSED_GAP_MM:           limited = "closed"
        else:                               limited = "anatomy"
        # what 1.0 would have said about this animal, for the record
        e_at_print_stop = min(1.0, th / P["maxAngle"]) if P["maxAngle"] > 0 else None
        out = dict(base, measure_timed_out=False,
                   theta_joint_deg=round(th, 2), total_deg=round(th * joints, 1),
                   closure_gap_mm=round(gap, 2), gap_over_L=round(gap / L, 4),
                   limited_by=limited, stopped_by=[(a, b, round(v, 1)) for a, b, v in first][:6],
                   # closed: the head-tail touch IS the intended end pose, so the stop sits at theta;
                   # anatomy: back off a margin so the print never reaches the colliding pose; bound: unknown
                   stop_recommended_deg=(round(th, 1) if limited == "closed" else
                                         round(max(0.0, th - STOP_MARGIN_DEG), 1) if limited == "anatomy" else None),
                   v1_equivalent_e_max=round(e_at_print_stop, 3) if e_at_print_stop is not None else None,
                   seconds=round(time.time() - t0, 1))
    except MeasureTimeout:
        out = dict(base, measure_timed_out=True, theta_joint_deg=None, total_deg=None, closure_gap_mm=None,
                   gap_over_L=None, limited_by="unknown", stopped_by=[], stop_recommended_deg=None,
                   v1_equivalent_e_max=None, seconds=round(time.time() - t0, 1))
    out.update(print_validity(P, parts))
    return out

def sweep_gap(P, meshes, thetas, head_body=None):
    """Head-body-to-tail gap along a list of joint angles (the closure curve on the sheet)."""
    return [(float(t), round(closure_gap_deg(P, meshes, t, head_body=head_body), 2)) for t in thetas]
