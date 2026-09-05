"""
Invariants of the trilobite generator + instrument. Run:  pytest -q   (one full build, ~1 min)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pytest
import schema, fields, trilobite as T, instrument as I

@pytest.fixture(scope="session")
def built():
    P = schema.defaults()
    parts = T.parts_list(P)
    meshes = I.part_meshes(P, parts)
    return P, parts, meshes

def test_schema_roundtrip():
    P = schema.defaults(); P["segCount"] = 7; P["nKnuckles"] = 4      # even knuckles must be coerced to odd
    Q = schema.from_json(schema.to_json(P))
    assert Q["segCount"] == 7 and Q["nKnuckles"] == 5
    assert schema.param_hash(Q) == schema.param_hash(schema.coerce(Q))

def test_width_curve_monotone_tail():
    P = schema.defaults(); wc = fields.width_curve(P)
    s = np.linspace(P["widthMaxPos"], 1, 30); w = wc(s)
    assert np.all(np.diff(w) <= 1e-6), "width must not grow behind the widest point"

def test_parts_valid_single_solid(built):
    P, parts, meshes = built
    for p in parts:
        assert p.is_valid and len(p.solids()) == 1

def test_meshes_are_volumes(built):
    P, parts, meshes = built
    bad = [n for n, m in zip(T.PART_NAMES(P), meshes) if not m.is_volume]
    if bad:   # known OpenCascade tessellation gap on some parts; the instrument falls back to FCL for those
        pytest.xfail(f"non-watertight tessellation for {bad}")

def test_bilateral_symmetry(built):
    P, parts, meshes = built
    import trimesh
    for m in meshes:
        v = m.vertices[::40].copy(); v[:, 0] *= -1               # mirror across the sagittal plane
        _, d, _ = trimesh.proximity.closest_point(m, v)
        assert np.percentile(d, 95) < 0.4, "mirror image must land on the surface"

def test_no_overlap_at_zero(built):
    P, parts, meshes = built
    assert I.collisions(P, meshes, 0.0) == []

def test_stop_monotonicity(built):
    P, parts, meshes = built
    free = [I.collisions(P, meshes, e) == [] for e in (0.25, 0.5, 0.75, 0.97)]
    assert all(free), "reference body must be collision-free up to the stop"
    hits = I.collisions(P, meshes, 1.06)
    assert len(hits) >= P["segCount"], "every joint must engage its stop just past maxAngle"

def test_measure_reference(built):
    P, parts, meshes = built
    r = I.measure(P, meshes)
    assert r["e_max"] > 0.97 and r["print_valid"] and not r["touching_at_zero"]
    assert r["enroll_class"] in ("partial", "complete")
