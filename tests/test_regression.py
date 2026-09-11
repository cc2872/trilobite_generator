"""tests/test_regression.py — a fresh build must reproduce the frozen 2.1 targets.

    python -m pytest tests/ -q          (or: python tests/test_regression.py)

For every folder in tests/references/ with a measure_v2.json:
  1. build the animal from params.json with the CURRENT builder at the frozen bevel_built_deg
  2. every part watertight
  3. surface distance, current part -> frozen part, median below SURF_TOL_MM (the frozen tessellation tolerance)
  4. instrument reading: limited_by and enroll_class equal; theta_joint_deg within THETA_TOL_DEG
THETA_TOL_DEG is a placeholder until tests/noise_floor.py has measured the repeatability floor (prompt 8)."""
import os, sys, json, glob
import numpy as np, pytest, trimesh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import schema, instrument as I

SURF_TOL_MM = 0.30     # frozen meshes were tessellated at (0.15, 0.3); domed heads are spline-biased (see CHANGES prompt 5) and skipped here
THETA_TOL_DEG = 0.2    # measured 11 Sep 2026 (tests/noise_floor.py): proetida reads 22.81 at every grid/scan/res setting; the floor is the bisection resolution (0.1)
REFS = sorted(d for d in glob.glob(os.path.join(ROOT, "tests", "references", "*")) if os.path.exists(os.path.join(d, "measure_v2.json")))

def surface_distance(a, b, n=4000):
    pts, _ = trimesh.sample.sample_surface(a, n)
    return float(np.median(np.abs(trimesh.proximity.signed_distance(b, pts))))

@pytest.mark.parametrize("ref", REFS, ids=[os.path.basename(r) for r in REFS])
def test_preset_reproduces(ref):
    frozen = json.load(open(os.path.join(ref, "measure_v2.json")))
    P = schema.coerce(json.load(open(os.path.join(ref, "params.json"))))
    bevel = float(frozen["bevel_built_deg"])
    B = I.build_animal(P, bevel)
    for name, m in zip(B["names"], B["meshes"]):
        assert m is not None and m.is_watertight, f"{name} not watertight"
        fpath = os.path.join(ref, f"{name}.stl")
        if os.path.exists(fpath) and name != "head":
            assert surface_distance(m, trimesh.load(fpath)) < SURF_TOL_MM, f"{name} surface drifted"
    r = I.measure(P, B["meshes"], bound_deg=bevel, head_body=B["head_body"], bevel_built_deg=B["bevel_built_deg"], unsane_parts=B["unsane_parts"])
    if frozen["limited_by"] == "invalid" and frozen.get("reason", "").startswith("unsane"):
        pytest.skip("frozen as a BREP build defect; the mesh builder has no target to match here")
    assert r["limited_by"] == frozen["limited_by"]
    if frozen["limited_by"] in ("closed", "anatomy"):
        assert r["enroll_class"] == frozen["enroll_class"]
        assert abs(r["theta_joint_deg"] - frozen["theta_joint_deg"]) <= THETA_TOL_DEG

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
