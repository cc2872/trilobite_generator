"""
instrument.py — the enrollment ruler, version 2.1, on the mesh builder (prompt 6, 11 Sep 2026). One file.

History. 1.0 swept a fraction of the printed stop (its ceiling was a function of segment count: the ruler measured
itself). 2.0 built the measurement meshes at ONE fixed wide bevel per joint and swept degrees to first interference or
closure. 2.1 fixed six audit findings: the bevel actually built is recorded (it used to pass through the schema's
maxAngle clamp), the rest baseline is keyed on mesh content, rest interference above a budget censors the animal, a
coarse forward scan precedes bisection, the closed pose is classified, and every row carries the instrument version and
the parameter hash. This file is 2.1 with the OpenCascade builder replaced by parts.py; the readings are unchanged
(11 Sep 2026: proetida 22.81/22.81, corynexochida 22.66/22.58, harpetida 30.39/30.31 deg vs the frozen BREP runs).

The reading: theta_joint_deg (largest uniform flexion per joint before interference or closure), total_deg,
closure gap (mm and / length), limited_by in {closed, anatomy, bound, invalid}, enroll_class in
{sphaeroidal, double, spiral, discoidal, open, censored}, s_tail (tail margin in the head frame), limiting pair.
'bound' means the ruler ran out before the animal did: censored, never a measurement.
"""
import math, time, hashlib, csv, os
import numpy as np
import trimesh
from trimesh.collision import CollisionManager
import schema, parts, mesh as M

INSTRUMENT_VERSION = "2.1"
BOUND_DEG = 45.0          # per-joint bevel of the measurement build; the sweep's hard ceiling
BOUNDS_PROBE = (45.0, 35.0, 25.0)   # widest bevel at which the middle segment stays one body (a wide wedge can sever tips)
CLOSED_GAP_MM = 3.0       # head-tail gap below which the animal is called closed
STOP_MARGIN_DEG = 2.0     # recommended printed stop = theta - this (anatomy-limited animals)
SCAN_STEP_DEG = 2.5       # forward scan step before bisection (PREREG §1)
RES_DEG = 0.1             # bisection resolution
REST_BUDGET_MM3 = 1.0     # rest overlap above this on any pair = build defect (PREREG §1, §4)
OVERLAP_TOL = 0.5         # mm^3 of overlap growth that counts as a collision
CLASS_TOL = 0.05          # |s_tail| below this (fraction of head length) = margins meet = sphaeroidal (PREREG §3)
DISC_DEG = 150.0          # closed with less total curl than this = discoidal
MEASURE_BUDGET_S = 600
MEASURE_GRID = (121, 61)  # 11 Sep: 61x31 severed harpetid tips and arms at every bevel; measure on the print grid (~24 s/animal)
PRINT_GRID = (121, 61)

class MeasureTimeout(Exception): pass

# ---------------------------------------------------------------- building the animal
def part_names(P): return ["head"] + [f"seg{i}" for i in range(int(P["segCount"]))] + ["tail"]

def probe_bound(P, bounds=BOUNDS_PROBE, grid=MEASURE_GRID):
    """Widest bevel in `bounds` at which EVERY segment builds as one body (a wide wedge severs pleural tips — agnostida's
    middle segment at 45, harpetida's seg7 at 45). Returns (bound, {bound: max bodies over segments})."""
    P = schema.coerce(P); probe = {}
    for b in bounds:
        worst = 1
        for i in range(int(P["segCount"])):
            try: n = len(M.bodies(parts.segment(P, i, bevel_deg=b, grid=grid)))
            except Exception: n = -1
            worst = n if (n < 0 or n > worst) and worst != -1 else worst
            if worst != 1: break
        probe[b] = worst
        if worst == 1: return b, probe
    return bounds[-1], probe

def build_animal(P, bound_deg=BOUND_DEG, grid=MEASURE_GRID, export_dir=None):
    """Every part at one fixed bevel. Returns dict: names, meshes, head_body, bevel_built_deg, unsane_parts, notes, build_seconds.
    unsane_parts = parts that did not come out as one closed body (a severed pleural tip is a body of its own)."""
    P = schema.coerce(P); t0 = time.time(); notes = []; names = part_names(P); meshes = []; unsane = []
    builders = [lambda: parts.cephalon(P, bevel_deg=bound_deg, grid=grid, notes=notes)] + \
               [(lambda i=i: parts.segment(P, i, bevel_deg=bound_deg, grid=grid)) for i in range(int(P["segCount"]))] + \
               [lambda: parts.pygidium(P, bevel_deg=bound_deg, grid=grid)]
    for n, fn in zip(names, builders):
        try:
            m = fn()
        except Exception as ex:
            notes.append((n, "build failed", str(ex)[:80])); unsane.append(n); meshes.append(None); continue
        nb = len(M.bodies(m))
        if nb != 1 or not m.is_watertight:
            notes.append((n, "unsane", f"{nb} bodies, watertight {m.is_watertight}")); unsane.append(n)
        if export_dir is not None: m.export(os.path.join(export_dir, f"{n}.stl"))
        meshes.append(m)
    hb = None
    if P["genalSpine"] * P["cephFrac"] * P["length"] > 0.5:
        try: hb = parts.cephalon(dict(P, genalSpine=0.0), bevel_deg=bound_deg, grid=grid)
        except Exception as ex: notes.append(("head_body", "build failed", str(ex)[:80]))
    return dict(names=names, meshes=meshes, head_body=hb, bevel_built_deg=float(bound_deg), unsane_parts=unsane, notes=notes,
                build_seconds=round(time.time() - t0, 1), grid=grid)

# ---------------------------------------------------------------- kinematics
def _rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])
def _trans(x, y, z):
    m = np.eye(4); m[:3, 3] = (x, y, z); return m

def transforms_deg(P, theta_deg):
    """4x4 per part for a uniform flexion of theta_deg at every joint (head = identity)."""
    zh = parts.hinge_z(P); offs = parts.joint_offsets(P)
    mats = [np.eye(4)]; Mx = np.eye(4)
    for i in range(len(offs)):
        Mx = Mx @ _trans(0, offs[i], 0) @ _trans(0, 0, zh) @ _rot_x(-theta_deg) @ _trans(0, 0, -zh)
        mats.append(Mx)
    return mats

def _posed(meshes, mats): return [m.copy().apply_transform(T) for m, T in zip(meshes, mats)]
def _aabb_hit(a, b, pad=0.5):
    return bool(np.all(a.bounds[0] - pad <= b.bounds[1]) and np.all(b.bounds[0] - pad <= a.bounds[1]))

def overlap_volume(a, b):
    """Exact shared volume via Manifold, without a mesh round-trip (an empty intersection is a legitimate 0, not a
    broken mesh). A non-closed INPUT raises — the gates upstream are supposed to have caught it."""
    return float((M.to_manifold(a) ^ M.to_manifold(b)).volume())

# ---------------------------------------------------------------- rest baseline (content-keyed)
_REST = {}
def mesh_set_key(meshes):
    h = hashlib.md5()
    for m in meshes:
        h.update(np.ascontiguousarray(m.vertices, np.float64).tobytes()); h.update(np.ascontiguousarray(m.faces, np.int64).tobytes())
    return h.hexdigest()

def rest_overlaps(P, meshes, deadline=None):
    key = mesh_set_key(meshes)
    if key not in _REST:
        if len(_REST) > 8: _REST.clear()
        posed = _posed(meshes, transforms_deg(P, 0.0)); base = {}
        for i in range(len(posed)):
            for j in range(i + 1, len(posed)):
                if deadline is not None and time.time() > deadline: raise MeasureTimeout("rest_overlaps")
                if _aabb_hit(posed[i], posed[j]): base[(i, j)] = overlap_volume(posed[i], posed[j])
        _REST[key] = base
    return _REST[key]

def collisions_deg(P, meshes, theta_deg, deadline=None):
    rest = rest_overlaps(P, meshes, deadline); posed = _posed(meshes, transforms_deg(P, theta_deg)); out = []
    for i in range(len(posed)):
        for j in range(i + 1, len(posed)):
            if deadline is not None and time.time() > deadline: raise MeasureTimeout("collisions_deg")
            if not _aabb_hit(posed[i], posed[j]): continue
            v = overlap_volume(posed[i], posed[j]) - rest.get((i, j), 0.0)
            if v > OVERLAP_TOL: out.append((i, j, v))
    return out

# ---------------------------------------------------------------- closure
def closure_gap_deg(P, meshes, theta_deg, deadline=None, head_body=None):
    if deadline is not None and time.time() > deadline: raise MeasureTimeout("closure_gap_deg")
    mats = transforms_deg(P, theta_deg)
    head = (head_body if head_body is not None else meshes[0]).copy().apply_transform(mats[0])
    tail = meshes[-1].copy().apply_transform(mats[-1])
    if _aabb_hit(head, tail) and overlap_volume(head, tail) > OVERLAP_TOL: return 0.0
    cm = CollisionManager(); cm.add_object("head", head)
    return float(cm.min_distance_single(tail))

def tail_margin_local(P):
    P = schema.coerce(P)
    return np.array([0.0, P["pygFrac"] * P["length"], P["marginHeight"] * P["relief"] * P["tailRelief"], 1.0])

def closure_geometry(P, theta_deg):
    """s_tail = (y_tip + Lc) / Lc in the head frame: 0 margins meet, > 0 tail margin inside the head's footprint (under
    the cephalon), < 0 past the front of the head."""
    P = schema.coerce(P); Lc = P["cephFrac"] * P["length"]
    tip = transforms_deg(P, theta_deg)[-1] @ tail_margin_local(P)
    return dict(tail_margin_xyz=[round(float(v), 2) for v in tip[:3]], s_tail=round(float((tip[1] + Lc) / Lc), 4), head_front_y=round(-Lc, 2))

def classify(limited_by, total_deg, s_tail):
    if limited_by == "closed":
        if total_deg < DISC_DEG: return "discoidal"
        if abs(s_tail) <= CLASS_TOL: return "sphaeroidal"
        return "double" if s_tail > 0 else "spiral"
    if limited_by == "anatomy": return "open"
    return "censored"

# ---------------------------------------------------------------- the sweep in theta
def theta_max(P, meshes, bound_deg=BOUND_DEG, tol_deg=RES_DEG, scan_step_deg=SCAN_STEP_DEG, deadline=None):
    rest_overlaps(P, meshes, deadline)
    steps = list(np.arange(scan_step_deg, bound_deg, scan_step_deg)) + [bound_deg]
    trace = []; lo, hi = 0.0, None
    for t in steps:
        c = collisions_deg(P, meshes, float(t), deadline=deadline); trace.append((round(float(t), 2), len(c)))
        if c: hi = float(t); break
        lo = float(t)
    if hi is None: return bound_deg, False, trace
    while hi - lo > tol_deg:
        mid = 0.5 * (lo + hi)
        if collisions_deg(P, meshes, mid, deadline=deadline): hi = mid
        else: lo = mid
    return lo, True, trace

# ---------------------------------------------------------------- print validity (mesh version of 1.0's)
def print_validity(P, meshes=None):
    P = schema.coerce(P); v = []
    d = parts.pitch(P)
    if d < 8.0: v.append(f"pitch {d:.1f} mm < 8")
    Wh = parts.hinge_width(P); kw = Wh / int(P["nKnuckles"]) - P["clearance"]
    if kw < 3.0: v.append(f"knuckle {kw:.1f} mm < 3.0")
    if P["wall"] < 1.5: v.append(f"wall {P['wall']} mm < 1.5")
    return dict(print_valid=(len(v) == 0), violations=v)

# ---------------------------------------------------------------- the reading
def measure(P, meshes, bound_deg=BOUND_DEG, budget_s=MEASURE_BUDGET_S, head_body=None, bevel_built_deg=None, unsane_parts=(), allow_unsane=False):
    P = schema.coerce(P); t0 = time.time(); deadline = t0 + budget_s
    joints = int(P["segCount"]) + 1; L = float(P["length"]); names = part_names(P)
    base = dict(instrument=INSTRUMENT_VERSION, params=schema.param_hash(P), bound_deg=bound_deg,
                bevel_built_deg=(bevel_built_deg if bevel_built_deg is not None else bound_deg), scan_step_deg=SCAN_STEP_DEG,
                res_deg=RES_DEG, rest_budget_mm3=REST_BUDGET_MM3, joints=joints, printed_stop_deg=P["maxAngle"],
                collision_rule=f"overlap - overlap_at_rest > {OVERLAP_TOL} mm3")
    empty = dict(theta_joint_deg=None, total_deg=None, closure_gap_mm=None, gap_over_L=None, stopped_by=[], stop_recommended_deg=None,
                 v1_equivalent_e_max=None, s_tail=None, tail_margin_xyz=None, enroll_class="censored", scan_trace=[])
    pv = print_validity(P, meshes)
    def invalid(reason, **kw):
        out = dict(base, **empty, measure_timed_out=False, limited_by="invalid", reason=reason, rest_max_mm3=None, rest_over_budget=[],
                   seconds=round(time.time() - t0, 1)); out.update(kw); out.update(pv); return out
    bad = [n for n, m in zip(names, meshes) if m is None or not m.is_watertight or len(M.bodies(m)) != 1]
    bad += [n for n in unsane_parts if n not in bad]
    if bad and not allow_unsane: return invalid("unsane_parts:" + ",".join(bad))
    try:
        rest = rest_overlaps(P, meshes, deadline); rest_max = max(rest.values(), default=0.0)
        over = [(i, j, round(v, 2)) for (i, j), v in rest.items() if v > REST_BUDGET_MM3]
        if over: return invalid("rest_interference", rest_max_mm3=round(rest_max, 2), rest_over_budget=over[:6])
        th, hit, trace = theta_max(P, meshes, bound_deg, deadline=deadline)
        gap = closure_gap_deg(P, meshes, th, deadline=deadline, head_body=head_body)
        first = collisions_deg(P, meshes, min(bound_deg, th + 0.2), deadline=deadline) if hit else []
        limited = "bound" if not hit else ("closed" if gap < CLOSED_GAP_MM else "anatomy")
        geo = closure_geometry(P, th); total = th * joints; cls = classify(limited, total, geo["s_tail"])
        e_stop = min(1.0, th / P["maxAngle"]) if P["maxAngle"] > 0 else None
        out = dict(base, measure_timed_out=False, limited_by=limited, reason="", rest_max_mm3=round(rest_max, 2), rest_over_budget=[],
                   theta_joint_deg=round(th, 2), total_deg=round(total, 1), closure_gap_mm=round(gap, 2), gap_over_L=round(gap / L, 4),
                   enroll_class=cls, s_tail=geo["s_tail"], tail_margin_xyz=geo["tail_margin_xyz"],
                   stopped_by=[(a, b, round(v, 1)) for a, b, v in first][:6], scan_trace=trace,
                   stop_recommended_deg=(round(th, 1) if limited == "closed" else round(max(0.0, th - STOP_MARGIN_DEG), 1) if limited == "anatomy" else None),
                   v1_equivalent_e_max=round(e_stop, 3) if e_stop is not None else None, seconds=round(time.time() - t0, 1))
    except MeasureTimeout:
        return invalid("measure_timeout", measure_timed_out=True)
    out.update(pv); return out

def sweep_gap(P, meshes, thetas, head_body=None):
    return [(float(t), round(closure_gap_deg(P, meshes, t, head_body=head_body), 2)) for t in thetas]

def read(P, bound=None, grid=MEASURE_GRID, export_dir=None, budget_s=MEASURE_BUDGET_S):
    """Build + probe + measure in one call. Returns (result, build)."""
    P = schema.coerce(P)
    b, probe = (bound, {bound: 1}) if bound else probe_bound(P, grid=grid)
    B = build_animal(P, b, grid=grid, export_dir=export_dir)
    r = measure(P, B["meshes"], bound_deg=b, budget_s=budget_s, head_body=B["head_body"], bevel_built_deg=B["bevel_built_deg"], unsane_parts=B["unsane_parts"])
    r.update(bevel_probe={str(k): v for k, v in probe.items()}, build_notes=B["notes"], build_seconds=B["build_seconds"], unsane_parts=B["unsane_parts"], grid=list(grid))
    return r, B

def posed_stl(P, meshes, theta_deg, path):
    mats = transforms_deg(P, theta_deg)
    trimesh.util.concatenate([m.copy().apply_transform(T) for m, T in zip(meshes, mats) if m is not None]).export(path)

# ---------------------------------------------------------------- the sweep's CSV row
ROW_COLUMNS = ["preset", "param_hash", "instrument_version", "valid", "reason", "theta_joint_deg", "total_deg", "gap_mm", "gap_over_length",
               "limited_by", "limiting_pair", "enroll_class", "s_tail", "bevel_built_deg", "bound_deg", "rest_max_mm3",
               "segCount", "overlap", "ceph_pyg_ratio", "relief", "widthThoraxRear", "spine_scale", "eyeSize", "is_control",
               "print_valid", "n_violations", "unsane_parts", "schema_notes", "build_seconds", "measure_seconds"]

def row(result, P, preset="", is_control=0, spine_scale=None, unsane_parts=(), schema_notes=(), build_seconds=None):
    P = schema.coerce(P); valid = result.get("limited_by") in ("closed", "anatomy", "bound"); first = result.get("stopped_by") or []
    unsane = sorted(set(list(unsane_parts) + list(result.get("unsane_parts", []))))
    return dict(preset=preset, param_hash=result.get("params", schema.param_hash(P)), instrument_version=result.get("instrument", INSTRUMENT_VERSION),
                valid=int(valid), reason=result.get("reason") or "", theta_joint_deg=result.get("theta_joint_deg"), total_deg=result.get("total_deg"),
                gap_mm=result.get("closure_gap_mm"), gap_over_length=result.get("gap_over_L"), limited_by=result.get("limited_by"),
                limiting_pair=(f"{first[0][0]}-{first[0][1]}" if first else ""), enroll_class=result.get("enroll_class", "censored"), s_tail=result.get("s_tail"),
                bevel_built_deg=result.get("bevel_built_deg"), bound_deg=result.get("bound_deg"), rest_max_mm3=result.get("rest_max_mm3"),
                segCount=int(P["segCount"]), overlap=P["overlap"], ceph_pyg_ratio=round(P["cephFrac"] / P["pygFrac"], 4), relief=P["relief"],
                widthThoraxRear=P["widthThoraxRear"], spine_scale=(spine_scale if spine_scale is not None else round(max(P["spineBase"], P["genalSpine"], P["pygSpine"]), 4)),
                eyeSize=P["eyeSize"], is_control=int(is_control), print_valid=int(bool(result.get("print_valid"))), n_violations=len(result.get("violations", [])),
                unsane_parts=",".join(unsane), schema_notes=";".join(schema_notes), build_seconds=build_seconds if build_seconds is not None else result.get("build_seconds"),
                measure_seconds=result.get("seconds"))

def write_row(rw, path):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
        if new: w.writeheader()
        w.writerow({k: rw.get(k) for k in ROW_COLUMNS}); f.flush()
