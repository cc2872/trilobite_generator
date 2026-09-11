"""
instrument2.py — the enrollment ruler, version 2.1.

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

Why 2.1 (10 Sep 2026) — six audit findings, each of which moved a number or a claim:
  1. The bevel is no longer passed through P["maxAngle"] (schema clamps it at 40): every "bound 45" run
     was built at 40. The bevel now goes through trilobite.MEASURE_BEVEL_DEG via measurement_bevel(),
     and the row records bevel_built_deg = what was actually cut.
  2. The rest baseline is keyed on a content hash of the meshes, not id(meshes) (CPython reuses ids).
  3. Rest overlap above REST_BUDGET_MM3 on any pair is a build defect: the animal is `invalid`, not
     measured with the interference subtracted away.
  4. theta_max does a coarse forward scan (SCAN_STEP_DEG) before bisecting, so a tip that swings through
     a neighbour and out again is caught (bisection assumed collision was monotone in theta).
  5. The closed pose is classified (sphaeroidal / double / spiral / discoidal) from the tail margin's
     position in the head frame — the pre-registered enrolment class.
  6. row() flattens a result to the sweep's CSV columns, with instrument_version and param_hash on every row.
The printed stop is a manufacturing choice made AFTER measurement (stop_recommended_deg).
Collision machinery (exact Manifold overlap, OVERLAP_TOL, MeasureTimeout) is inherited from 1.0.
"""
import math, json, time, hashlib, contextlib
import numpy as np
import schema
from instrument import (_rot_x, _trans, _posed, _aabb_hit, overlap_volume, OVERLAP_TOL,
                        MeasureTimeout, print_validity, MEASURE_BUDGET_S, MESH_TOL)
from trimesh.collision import CollisionManager

INSTRUMENT_VERSION = "2.1"
BOUND_DEG = 45.0          # per-joint bevel of the measurement build; the sweep's hard ceiling
CLOSED_GAP_MM = 3.0       # head-tail gap below which the animal is called closed
STOP_MARGIN_DEG = 2.0     # recommended printed stop = theta_joint - this (never above BOUND_DEG)
SCAN_STEP_DEG = 2.5       # forward scan step before bisection (pre-reg §1)
RES_DEG = 0.1             # bisection resolution (pre-reg §1)
REST_BUDGET_MM3 = 1.0     # rest overlap above this on any pair = build defect, animal invalid (pre-reg §1, §4)
CLASS_TOL = 0.05          # |s_tail| below this (fraction of head length) = margins meet = sphaeroidal (pre-reg §3)
DISC_DEG = 150.0          # closed with less total curl than this = discoidal (pre-reg §3)

# ---------------------------------------------------------------- the ruler's bevel
@contextlib.contextmanager
def measurement_bevel(bound_deg=BOUND_DEG):
    """Every part built inside this block gets a ventral bevel of bound_deg per joint, whatever P['maxAngle']
    says. This is the only sanctioned way to build measurement meshes."""
    import trilobite as T
    prev = T.MEASURE_BEVEL_DEG
    T.MEASURE_BEVEL_DEG = float(bound_deg)
    try:
        yield float(bound_deg)
    finally:
        T.MEASURE_BEVEL_DEG = prev

def measurement_params(P, bound_deg=BOUND_DEG):
    """2.0 opened P['maxAngle'] to bound_deg here; schema.coerce clamped it to 40 on the way into the build.
    2.1: the bevel goes through measurement_bevel(); this returns the coerced morphology unchanged and is kept
    for callers that still pass a Pm around. Build inside `with measurement_bevel(bound_deg):`."""
    return schema.coerce(P)

def build_measurement(P, bound_deg=BOUND_DEG, export_dir=None):
    """Build the measurement meshes for P at a fixed bevel. Returns a dict:
    parts, names, meshes, head_body, bevel_built_deg, build_notes, unsane_parts, build_seconds."""
    import trilobite as T
    P = schema.coerce(P)
    T.BUILD_NOTES.clear(); T.GRID_JITTER = 0
    t0 = time.time()
    with measurement_bevel(bound_deg) as b:
        parts = T.parts_list(P); names = T.PART_NAMES(P)
        meshes = []
        for n, p in zip(names, parts):
            m = getattr(p, "_checked_mesh", None) or T.to_trimesh(p, *T.SANE_MESH_TOL)
            if export_dir is not None: m.export(f"{export_dir}/{n}.stl")
            meshes.append(m)
        hb = head_body_mesh(P)
    notes = list(T.BUILD_NOTES)
    return dict(parts=parts, names=names, meshes=meshes, head_body=hb, bevel_built_deg=b, build_notes=notes,
                unsane_parts=[n[0] for n in notes if n[1] == "UNSANE after retries"],
                build_seconds=round(time.time() - t0, 1))

def probe_bound(P, bounds=(45.0, 35.0, 25.0)):
    """The widest bevel in `bounds` at which the middle segment still builds as one solid (a wide wedge can
    sever pleural tips). Returns (bound, {bound: n_solids}). Falls back to the narrowest if none is clean."""
    import trilobite as T
    P = schema.coerce(P); probe = {}
    for b in bounds:
        with measurement_bevel(b):
            T.BUILD_NOTES.clear(); T.GRID_JITTER = 0
            try: n = len(T.build_segment(P, int(P["segCount"]) // 2).solids())
            except Exception: n = -1
        probe[b] = n
        if n == 1: return b, probe
    return bounds[-1], probe

# ---------------------------------------------------------------- kinematics
def transforms_deg(P, theta_deg):
    """4x4 transform per part for a uniform flexion of theta_deg at every joint."""
    import trilobite as T
    zh = T.hinge_z(P); offs = T.joint_offsets(P)
    mats = [np.eye(4)]; M = np.eye(4)
    for i in range(len(offs)):
        M = M @ _trans(0, offs[i], 0) @ _trans(0, 0, zh) @ _rot_x(-theta_deg) @ _trans(0, 0, -zh)
        mats.append(M)
    return mats

# ---------------------------------------------------------------- rest baseline (content-keyed)
_REST = {}                # mesh-set content hash -> {(i, j): overlap at theta = 0}

def mesh_set_key(meshes):
    h = hashlib.md5()
    for m in meshes:
        h.update(np.ascontiguousarray(m.vertices, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(m.faces, dtype=np.int64).tobytes())
    return h.hexdigest()

def rest_overlaps(P, meshes, deadline=None):
    """Pairwise shared volume at theta = 0. Mating faces touch by design (the tail's stepped front budgets 0.5 mm^3);
    a collision is overlap that GROWS from this baseline, not overlap that was built in. Keyed on mesh content."""
    key = mesh_set_key(meshes)
    if key not in _REST:
        if len(_REST) > 8: _REST.clear()
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

# ---------------------------------------------------------------- closure
def head_body_mesh(Pm):
    """The cephalon with its genal arms / prolongations switched off: the surface the tail must reach
    for the animal to count as closed. A harpetid's brim prolongations reach most of the way to its
    tail at rest; measuring closure against them reads 'closed' when the tail merely taps an arm.
    Must be called inside measurement_bevel() to get the same bevel as the rest of the build."""
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

def tail_margin_local(P):
    """The pygidial margin at the axis, in the tail's own frame: the tail is built with its front hinge at y = 0
    and its shield margin at y = pygFrac * length (spines are extra, so this point is spine-independent)."""
    P = schema.coerce(P)
    return np.array([0.0, P["pygFrac"] * P["length"], P["marginHeight"] * P["relief"] * P["tailRelief"], 1.0])

def closure_geometry(P, theta_deg):
    """Where the tail margin sits in the head's frame at this flexion. The head is built with its rear hinge at
    y = 0 and its front margin at y = -cephFrac * length (mats[0] is the identity, so head frame == world).
    s_tail = (y_tip + Lc) / Lc: 0 = tail margin meets the head's front margin; > 0 = tail margin is posterior of
    the front margin, i.e. inside the head's footprint (under the cephalon); < 0 = past the front of the head."""
    P = schema.coerce(P)
    Lc = P["cephFrac"] * P["length"]
    tip = transforms_deg(P, theta_deg)[-1] @ tail_margin_local(P)
    return dict(tail_margin_xyz=[round(float(v), 2) for v in tip[:3]],
                s_tail=round(float((tip[1] + Lc) / Lc), 4), head_front_y=round(-Lc, 2))

def classify(limited_by, total_deg, s_tail):
    """Pre-registered enrolment class (PREREG §3). Only `closed` animals get a shape class."""
    if limited_by == "closed":
        if total_deg < DISC_DEG: return "discoidal"
        if abs(s_tail) <= CLASS_TOL: return "sphaeroidal"
        return "double" if s_tail > 0 else "spiral"
    if limited_by == "anatomy": return "open"
    return "censored"

# ---------------------------------------------------------------- the sweep in theta
def theta_max(P, meshes, bound_deg=BOUND_DEG, tol_deg=RES_DEG, scan_step_deg=SCAN_STEP_DEG, deadline=None):
    """Largest collision-free uniform flexion in degrees per joint on [0, bound_deg].
    A coarse forward scan finds the FIRST colliding step (tips can pass through a neighbour and out again, so
    'the bound is clear' does not mean the interval is), then bisection refines inside that step.
    Returns (theta, hit, trace) where hit is False if every scan point up to and including the bound is clear."""
    rest_overlaps(P, meshes, deadline)                           # theta = 0 is collision-free by definition (baseline)
    steps = list(np.arange(scan_step_deg, bound_deg, scan_step_deg)) + [bound_deg]
    trace = []; lo, hi = 0.0, None
    for t in steps:
        c = collisions_deg(P, meshes, float(t), deadline=deadline)
        trace.append((round(float(t), 2), len(c)))
        if c: hi = float(t); break
        lo = float(t)
    if hi is None: return bound_deg, False, trace
    while hi - lo > tol_deg:
        mid = 0.5 * (lo + hi)
        if collisions_deg(P, meshes, mid, deadline=deadline): hi = mid
        else: lo = mid
    return lo, True, trace

def _rest_gate(P, meshes, deadline):
    rest = rest_overlaps(P, meshes, deadline)
    worst = max(rest.values(), default=0.0)
    over = [(i, j, round(v, 2)) for (i, j), v in rest.items() if v > REST_BUDGET_MM3]
    return worst, over

def measure(P, meshes, parts=None, bound_deg=BOUND_DEG, budget_s=MEASURE_BUDGET_S, head_body=None, bevel_built_deg=None, allow_unsane=False, unsane_parts=()):
    """The full 2.1 ruler. `meshes` MUST come from build_measurement() (or a build inside measurement_bevel()) —
    i.e. the wide-bevel build — or the bevel becomes the limiter again and the reading is 1.0's in different units.
    Pass bevel_built_deg from build_measurement so the row records what was cut, not what was asked for.
    allow_unsane=True measures non-watertight parts anyway (debugging only; the row is still flagged by row())."""
    P = schema.coerce(P)
    t0 = time.time(); deadline = t0 + budget_s
    joints = int(P["segCount"]) + 1
    L = float(P["length"])
    base = dict(instrument=INSTRUMENT_VERSION, params=schema.param_hash(P), bound_deg=bound_deg,
                bevel_built_deg=(bevel_built_deg if bevel_built_deg is not None else bound_deg),
                scan_step_deg=SCAN_STEP_DEG, res_deg=RES_DEG, rest_budget_mm3=REST_BUDGET_MM3,
                joints=joints, printed_stop_deg=P["maxAngle"], collision_rule=f"overlap - overlap_at_rest > {OVERLAP_TOL} mm3")
    empty = dict(theta_joint_deg=None, total_deg=None, closure_gap_mm=None, gap_over_L=None, stopped_by=[],
                 stop_recommended_deg=None, v1_equivalent_e_max=None, s_tail=None, tail_margin_xyz=None,
                 enroll_class="censored", scan_trace=[])
    try:
        # PREREG gate 2: a non-watertight part is a build defect. Measuring it anyway gave 2.0's phacopida reading
        # on a broken mesh, and Manifold's slow path on non-manifold input turns a 10 s measurement into 8 minutes.
        # PREREG gate 1+2: the build's own verdict (UNSANE after retries, volume lost in a boolean) counts even when
        # the mesh that came out happens to be watertight — redlichiida seg2 lost 60 % of its volume and still tessellated clean.
        import trilobite as T
        names = T.PART_NAMES(P)
        bad_names = [names[i] if i < len(names) else str(i) for i, m in enumerate(meshes) if not m.is_watertight]
        bad_names += [n for n in unsane_parts if n not in bad_names]
        pv = print_validity(P, parts)
        bad_names += [v.split(":")[0] for v in pv.get("violations", []) if ("boolean" in v or "solids" in v) and v.split(":")[0] not in bad_names]
        if bad_names and not allow_unsane:
            out = dict(base, **empty, measure_timed_out=False, limited_by="invalid",
                       reason="unsane_parts:" + ",".join(bad_names),
                       rest_max_mm3=None, rest_over_budget=[], seconds=round(time.time() - t0, 1))
            out.update(print_validity(P, parts)); return out
        rest_max, over = _rest_gate(P, meshes, deadline)
        if over:
            out = dict(base, **empty, measure_timed_out=False, limited_by="invalid", reason="rest_interference",
                       rest_max_mm3=round(rest_max, 2), rest_over_budget=over[:6], seconds=round(time.time() - t0, 1))
            out.update(print_validity(P, parts)); return out
        th, hit, trace = theta_max(P, meshes, bound_deg, deadline=deadline)
        gap = closure_gap_deg(P, meshes, th, deadline=deadline, head_body=head_body)
        first = collisions_deg(P, meshes, min(bound_deg, th + 0.2), deadline=deadline) if hit else []
        # closed only if the tail is at the head BODY; a head-tail collision on the arms is anatomy
        if not hit:                         limited = "bound"
        elif gap < CLOSED_GAP_MM:           limited = "closed"
        else:                               limited = "anatomy"
        geo = closure_geometry(P, th)
        total = th * joints
        cls = classify(limited, total, geo["s_tail"])
        # what 1.0 would have said about this animal, for the record
        e_at_print_stop = min(1.0, th / P["maxAngle"]) if P["maxAngle"] > 0 else None
        out = dict(base, measure_timed_out=False, limited_by=limited, reason="",
                   rest_max_mm3=round(rest_max, 2), rest_over_budget=[],
                   theta_joint_deg=round(th, 2), total_deg=round(total, 1),
                   closure_gap_mm=round(gap, 2), gap_over_L=round(gap / L, 4),
                   enroll_class=cls, s_tail=geo["s_tail"], tail_margin_xyz=geo["tail_margin_xyz"],
                   stopped_by=[(a, b, round(v, 1)) for a, b, v in first][:6], scan_trace=trace,
                   # closed: the head-tail touch IS the intended end pose, so the stop sits at theta;
                   # anatomy: back off a margin so the print never reaches the colliding pose; bound: unknown
                   stop_recommended_deg=(round(th, 1) if limited == "closed" else
                                         round(max(0.0, th - STOP_MARGIN_DEG), 1) if limited == "anatomy" else None),
                   v1_equivalent_e_max=round(e_at_print_stop, 3) if e_at_print_stop is not None else None,
                   seconds=round(time.time() - t0, 1))
    except MeasureTimeout:
        out = dict(base, **empty, measure_timed_out=True, limited_by="invalid", reason="measure_timeout",
                   rest_max_mm3=None, rest_over_budget=[], seconds=round(time.time() - t0, 1))
    out.update(print_validity(P, parts))
    return out

def sweep_gap(P, meshes, thetas, head_body=None):
    """Head-body-to-tail gap along a list of joint angles (the closure curve on the sheet)."""
    return [(float(t), round(closure_gap_deg(P, meshes, t, head_body=head_body), 2)) for t in thetas]

# ---------------------------------------------------------------- the sweep's CSV row
ROW_COLUMNS = ["preset", "param_hash", "instrument_version", "valid", "reason",
               "theta_joint_deg", "total_deg", "gap_mm", "gap_over_length", "limited_by", "limiting_pair",
               "enroll_class", "s_tail", "bevel_built_deg", "bound_deg", "rest_max_mm3",
               "segCount", "overlap", "ceph_pyg_ratio", "relief", "widthThoraxRear", "spine_scale", "eyeSize",
               "is_control", "print_valid", "n_violations", "unsane_parts", "schema_notes", "build_seconds", "measure_seconds"]

def row(result, P, preset="", is_control=0, spine_scale=None, unsane_parts=(), schema_notes=(), build_seconds=None):
    """Flatten a measure() result to the sweep's CSV columns (analyze_sweep.py reads exactly these).
    valid = the row carries a real reading (limited_by in {closed, anatomy, bound}); a `bound` row is valid but censored
    — analyze_sweep.py excludes it from H1–H3 and counts it in the censoring rate."""
    P = schema.coerce(P)
    valid = result.get("limited_by") in ("closed", "anatomy", "bound")
    unsane = list(unsane_parts) + [v.split(":")[0] for v in result.get("violations", []) if "solids" in v or "boolean" in v]
    reason = result.get("reason") or ""
    if valid and unsane and "unsane_parts" not in reason: reason = (reason + ";" if reason else "") + f"unsane_parts:{','.join(sorted(set(unsane)))}"
    first = result.get("stopped_by") or []
    return dict(preset=preset, param_hash=result.get("params", schema.param_hash(P)), instrument_version=result.get("instrument", INSTRUMENT_VERSION),
                valid=int(valid), reason=reason,
                theta_joint_deg=result.get("theta_joint_deg"), total_deg=result.get("total_deg"),
                gap_mm=result.get("closure_gap_mm"), gap_over_length=result.get("gap_over_L"),
                limited_by=result.get("limited_by"), limiting_pair=(f"{first[0][0]}-{first[0][1]}" if first else ""),
                enroll_class=result.get("enroll_class", "censored"), s_tail=result.get("s_tail"),
                bevel_built_deg=result.get("bevel_built_deg"), bound_deg=result.get("bound_deg"), rest_max_mm3=result.get("rest_max_mm3"),
                segCount=int(P["segCount"]), overlap=P["overlap"], ceph_pyg_ratio=round(P["cephFrac"] / P["pygFrac"], 4),
                relief=P["relief"], widthThoraxRear=P["widthThoraxRear"],
                spine_scale=(spine_scale if spine_scale is not None else round(max(P["spineBase"], P["genalSpine"], P["pygSpine"]), 4)),
                eyeSize=P["eyeSize"], is_control=int(is_control),
                print_valid=int(bool(result.get("print_valid"))), n_violations=len(result.get("violations", [])),
                unsane_parts=",".join(sorted(set(unsane))), schema_notes=";".join(schema_notes),
                build_seconds=build_seconds, measure_seconds=result.get("seconds"))
