"""
isopod_model/enroll.py: the enrollment test applied to the isopod model.

The ruler is instrument.py 2.1 and it is not changed. What changes is the joint: the instrument measures the pin
hinge (its kinematics, instrument.transforms_deg = joints/pin), and this model has the isopod's ball joints. So the
reading here runs instrument 2.1's own procedure and constants, unchanged,
  * rest baseline keyed on mesh content, rest interference above REST_BUDGET_MM3 censors,
  * forward scan in SCAN_STEP_DEG, bisection to RES_DEG, collision = overlap growth over rest > OVERLAP_TOL,
    every pair of pieces,
  * closure: head-tail gap < CLOSED_GAP_MM at the stopping angle,
  * classification: classify() verbatim (PREREG §3),
with the isopod model's kinematics in place of the pin's: each joint turns about its ball centre (pieces in their
rest placement, a uniform curl at every joint). And two geometric inputs the pin build takes from the schema are
taken from the model instead: the head's length (the crescent head is not the schema's head) and the tail margin
point (the tail's rear margin on the midline, in its rest placement).

This is a separate reading. It is never written to tests/references, the sweep CSV or anything pre-registered, and
every row says kinematics = "isopod ball (isopod_model)".

    python isopod_model/enroll.py [preset ...]   -> readings in isopod_model/readings/<preset>.json
"""
import os, sys, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import numpy as np, trimesh
import schema, mesh as M
import instrument as INS
import check as CHECK

KINEMATICS = "isopod ball (isopod_model)"


def reading(P, parts, PV, name="", budget_s=INS.MEASURE_BUDGET_S):
    """parts: the model's pieces in model mm (rest placement); PV: [(y, z)] pivots in model mm."""
    P = schema.coerce(P); n = len(parts)
    pose = lambda P_, th: CHECK.pose(PV, [th] * (n - 1))
    # the tail margin (midline, rear edge of the tail's own plan) and the head length, from the model
    tail = parts[-1]; yt = float(tail.bounds[1][1])
    loc, _, _ = tail.ray.intersects_location([(0.0, yt - 0.05, 200.0)], [(0, 0, -1.0)])
    zt = float(np.asarray(loc).reshape(-1, 3)[:, 2].max()) if len(loc) else 0.0
    tip_rest = np.array([0.0, yt, zt, 1.0])
    y_front = float(parts[0].bounds[0][1]); y_rear_port = PV[0][0]; Lc = y_rear_port - y_front
    saved = (INS.transforms_deg, INS.closure_geometry, INS.print_validity, INS.part_names)
    def closure_geometry(P_, th):
        tip = pose(P_, th)[-1] @ tip_rest
        return dict(tail_margin_xyz=[round(float(v), 2) for v in tip[:3]], s_tail=round(float((tip[1] - y_front) / Lc), 4),
                    head_front_y=round(y_front, 2))
    INS.transforms_deg = pose; INS.closure_geometry = closure_geometry
    INS.print_validity = lambda P_, meshes=None: dict(print_valid=True, violations=[])       # pin-hinge rules: n/a here
    INS.part_names = lambda P_: ["head"] + [f"seg{i}" for i in range(n - 2)] + ["tail"]
    INS._REST.clear()
    try:
        r = INS.measure(P, parts, bound_deg=INS.BOUND_DEG, budget_s=budget_s)
    finally:
        INS.transforms_deg, INS.closure_geometry, INS.print_validity, INS.part_names = saved
    names = INS.part_names(P) if len(INS.part_names(P)) == n else [f"p{k}" for k in range(n)]
    r["stopped_by"] = [(names[a] if isinstance(a, int) else a, names[b] if isinstance(b, int) else b, v) for a, b, v in r.get("stopped_by", [])]
    r.update(kinematics=KINEMATICS, preset=name, head_length_mm=round(Lc, 2), tail_margin_rest=[round(float(v), 2) for v in tip_rest[:3]],
             note="instrument 2.1 procedure and constants; isopod joints' kinematics; not a pre-registered reading")
    r.pop("printed_stop_deg", None); r.pop("v1_equivalent_e_max", None)
    return r


if __name__ == "__main__":
    import presets as PR
    names = sys.argv[1:] or PR.names()
    for nm in names:
        res = PR.run(nm); print(nm, res["reading"]["limited_by"], res["reading"].get("enroll_class"), res["reading"].get("theta_joint_deg"))
