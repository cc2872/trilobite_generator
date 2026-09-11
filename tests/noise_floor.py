"""tests/noise_floor.py — the instrument's repeatability floor: the same animal at different mesh resolutions and
scan/bisection settings. The spread in theta_joint_deg is the smallest difference between two animals worth reporting;
it sets THETA_TOL_DEG in test_regression.py and the retrodiction criterion in the pre-reg (§6).

    python tests/noise_floor.py [preset]           # ~3 min; prints the table and the floor

There is no build-to-build randomness any more (no grid jitter, no retries): the floor is entirely resolution."""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import schema, instrument as I

def floor(name="proetida", grids=((101, 51), (121, 61), (141, 71)), scan_steps=(2.5, 1.0), res=(0.1, 0.05)):
    P = schema.coerce(json.load(open(os.path.join(ROOT, "presets", f"{name}.json")))["params"])
    rows = []
    for g in grids:
        B = I.build_animal(P, 45.0, grid=g)
        for st in scan_steps:
            for rs in res:
                I.SCAN_STEP_DEG, I.RES_DEG = st, rs; I._REST.clear()
                r = I.measure(P, B["meshes"], bound_deg=45.0, head_body=B["head_body"], bevel_built_deg=45.0, unsane_parts=B["unsane_parts"])
                rows.append((g, st, rs, r["limited_by"], r["enroll_class"], r["theta_joint_deg"], r["s_tail"]))
                print(f"grid {g} scan {st} res {rs}: {r['limited_by']} {r['enroll_class']} theta {r['theta_joint_deg']} s_tail {r['s_tail']}  ({r['seconds']}s)", flush=True)
    I.SCAN_STEP_DEG, I.RES_DEG = 2.5, 0.1
    th = [x[5] for x in rows if x[5] is not None]
    spread = (max(th) - min(th)) if th else None
    print(f"\n{name}: theta spread over {len(rows)} settings = {spread} deg  -> report differences below this as noise")
    return rows, spread

if __name__ == "__main__":
    floor(sys.argv[1] if len(sys.argv) > 1 else "proetida")
