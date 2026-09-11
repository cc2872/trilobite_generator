"""tests/test_parts.py — thorax rows and pygidium on the mesh builder.

    python -m pytest tests/test_parts.py -q

Uses the frozen instrument-2.1 references (tests/references/<preset>/): proetida has no ornament, so its parts are the
clean comparison; the other frozen orders must simply build closed at the bevel they were frozen at."""
import os, sys, json, glob
import numpy as np, pytest, trimesh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import schema, parts, mesh as M

REF = os.path.join(ROOT, "tests", "references")
SURF_TOL_MM = 0.30

def _params(name): return schema.coerce(json.load(open(os.path.join(REF, name, "params.json"))))
def _bevel(name): return float(json.load(open(os.path.join(REF, name, "measure_v2.json")))["bevel_built_deg"])
def _dist(m, ref_path, n=4000):
    r = trimesh.load(ref_path)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=0)
    _, d, _ = trimesh.proximity.closest_point(r, pts)
    return float(np.median(d)), r.volume

HAVE_PROETIDA = os.path.exists(os.path.join(REF, "proetida", "measure_v2.json"))
FROZEN = sorted(os.path.basename(d) for d in glob.glob(os.path.join(REF, "*")) if os.path.exists(os.path.join(d, "params.json")))

@pytest.mark.skipif(not HAVE_PROETIDA, reason="proetida reference not frozen")
@pytest.mark.parametrize("i", [0, 3, 8])
def test_proetida_segment_matches_reference(i):
    P = _params("proetida"); s = parts.segment(P, i, bevel_deg=_bevel("proetida"))
    assert s.is_watertight and len(M.bodies(s)) == 1 and M.is_symmetric(s)
    med, vref = _dist(s, os.path.join(REF, "proetida", f"seg{i}.stl"))
    assert med < SURF_TOL_MM
    assert abs(s.volume - vref) / vref < 0.02

@pytest.mark.skipif(not HAVE_PROETIDA, reason="proetida reference not frozen")
def test_proetida_tail_matches_reference():
    P = _params("proetida"); t = parts.pygidium(P, bevel_deg=_bevel("proetida"))
    assert t.is_watertight and len(M.bodies(t)) == 1 and M.is_symmetric(t)
    med, vref = _dist(t, os.path.join(REF, "proetida", "tail.stl"))
    assert med < SURF_TOL_MM
    assert abs(t.volume - vref) / vref < 0.03          # the margin now stops 1.5 deg short of the front line

@pytest.mark.skipif(not HAVE_PROETIDA, reason="proetida reference not frozen")
def test_cells_fuse_to_the_row_shell():
    P = _params("proetida"); S = parts.segment_plan(P, 3)
    full = M.heightfield_shell(S["outline"], S["zfun"], S["t"])
    cells = parts.segment_cells(P, 3)
    for c in cells.values(): assert c.is_watertight and len(M.bodies(c)) == 1
    fused = M.union(*cells.values())
    assert fused.is_watertight and abs(fused.volume - full.volume) / full.volume < 0.01
    assert M.is_symmetric(fused)

@pytest.mark.parametrize("name", FROZEN)
def test_frozen_orders_build_closed(name):
    P = _params(name); b = _bevel(name)
    for i in (0, int(P["segCount"]) // 2, int(P["segCount"]) - 1):
        s = parts.segment(P, i, bevel_deg=b); assert s.is_watertight, f"{name} seg{i}"
    t = parts.pygidium(P, bevel_deg=b); assert t.is_watertight, f"{name} tail"

def test_agnostida_bevel_severs_tips_like_the_brep_builder():
    """The BREP bevel probe dropped agnostida to 35 deg because a 45 deg wedge cut its pleural tips off (3 solids).
    The mesh builder must show the same geometry: 3 bodies at 45, 1 at 35."""
    if "agnostida" not in FROZEN: pytest.skip("agnostida reference not frozen")
    P = _params("agnostida")
    assert len(M.bodies(parts.segment(P, 1, bevel_deg=45.0))) > 1
    assert len(M.bodies(parts.segment(P, 1, bevel_deg=35.0))) == 1

def test_pleura_lengths_sum_to_pleural_width():
    P = _params(FROZEN[0]); inner, blade, bend = parts.pleura_lengths(P, 2)
    from fields import seg_halfwidth
    w = seg_halfwidth(P, 2); a = P["axisFrac"] * w
    assert abs(inner + blade - (w - a)) < 1e-9 and 0 < bend < 90

def test_no_opencascade_in_parts():
    import parts as p, inspect
    src = inspect.getsource(p)
    assert "build123d" not in src and "OCP" not in src
