"""tests/test_mesh.py — mesh.py must produce closed, symmetric solids and exact booleans.

    python -m pytest tests/test_mesh.py -q

The last test compares a height-field shell against the OpenCascade builder's B-spline plate on the same outline and
zfun (median surface distance under 0.30 mm; the spline fit is biased ~0.25 mm low on domes — see the test body). It is skipped when
build123d is not installed — after prompt 6 that will be the normal case and the frozen references take over."""
import os, sys, math
import numpy as np, pytest, trimesh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import mesh as M

OUT = M.superellipse_outline(60, 40, pinch_rear=False)
def ZF(x, y): return 8.0 * np.clip(1 - (x / 20) ** 2, 0, 1) * np.sin(np.pi * np.clip(y / 60, 0, 1)) + 8.0   # base 8: the barrel cut must not sever the flap
T_SHELL = 3.0
def ZF_FLAT(x, y): return np.full(np.shape(x), 10.0)                 # flat plate for the hinge tests: top 10, bottom 6 with T_HINGE
T_HINGE = 4.0
HINGE = dict(y_axis=54.0, rear=True, zh=7.2, Wh=14.0, barrel_r=1.6, clearance=0.35, n_knuckles=5, ring_top=10.0,
             wall=1.6, bore_d=1.75, band=22.0, reach=3.0, bevel_deg=45)

@pytest.fixture(scope="module")
def shell(): return M.heightfield_shell(OUT, ZF, T_SHELL)

@pytest.fixture(scope="module")
def env(): return M.under_envelope(OUT, ZF)

@pytest.fixture(scope="module")
def flat(): return M.heightfield_shell(OUT, ZF_FLAT, T_HINGE)

@pytest.fixture(scope="module")
def flat_env(): return M.under_envelope(OUT, ZF_FLAT)

def test_shell_closed_and_symmetric(shell):
    assert shell.is_watertight and shell.is_winding_consistent
    assert len(M.bodies(shell)) == 1
    assert M.is_symmetric(shell)
    assert shell.volume > 0

def test_pinched_both_ends_is_closed():
    m = M.heightfield_shell(M.superellipse_outline(60, 40), ZF, 1.6)
    assert m.is_watertight and len(M.bodies(m)) == 1

def test_shell_thickness_is_t():
    """Bottom surface sits t below the top where the plate is thick enough."""
    m = M.heightfield_shell(OUT, ZF, T_SHELL)
    zmax = m.vertices[:, 2].max()
    # the crown of the dome: the highest bottom vertex should be ~t below the highest top vertex
    top = m.vertices[m.vertices[:, 2] > zmax - 0.05]
    xy = top[0, :2]
    near = m.vertices[np.linalg.norm(m.vertices[:, :2] - xy, axis=1) < 0.6]
    assert abs((near[:, 2].max() - near[:, 2].min()) - T_SHELL) < 0.15

def test_envelope_reaches_floor(env):
    assert env.is_watertight and abs(env.vertices[:, 2].min() + 1.0) < 1e-9

def test_boolean_volumes():
    b = M.box(20, 20, 20); c = M.cylinder(5, 30, axis="z", sections=96)
    assert abs(M.difference(b, c).volume - (8000 - math.pi * 25 * 20)) < 8      # 96-gon vs circle
    assert abs(M.union(b, M.box(20, 20, 20, at=(10, 0, 0))).volume - 12000) < 1e-6
    assert abs(M.intersection(b, M.box(20, 20, 20, at=(10, 0, 0))).volume - 4000) < 1e-6

def test_primitives_closed():
    for p in (M.box(1, 2, 3), M.cylinder(1, 5, "x"), M.cylinder(1, 5, "y"), M.frustum(2, 0.3, 25), M.wedge(22, 80, 22.5, True, 60, 3.5)):
        assert p.is_watertight

def test_mirror_union_equals_symmetric_build(shell):
    half = M.superellipse_outline(60, 40, pinch_rear=False)
    def half_out(u, v): return half(0.5 * (u + 1), v)
    hs = M.heightfield_shell(half_out, ZF, T_SHELL, symmetric=False)
    full = M.union(hs, M.mirror_x(hs))
    assert full.is_watertight and M.is_symmetric(full)
    assert abs(full.volume - shell.volume) / shell.volume < 0.02

def test_hinge_single_body_symmetric(flat, flat_env):
    shell, env = flat, flat_env
    h = M.hinge(shell, env, **HINGE)
    assert h.is_watertight and len(M.bodies(h)) == 1
    assert M.is_symmetric(h)
    removed = M.difference(shell, h).volume; added = M.difference(h, shell).volume
    assert removed > 20 and added > 20                      # the bevel took the flap underside, the knuckles/block added

def test_wedge_angle_convention():
    """The rear wedge is the box y in [0, L], z in [-L, 0] rotated by -phi about the hinge line: its far top corner must land at
    (y + L cos phi, z - L sin phi) — descending rearward, add_hinge's convention."""
    L = 80.0
    for phi in (10.0, 22.5):
        w = M.wedge(22, L, phi, rear=True, y=54.0, z=7.2)
        target = np.array([54.0 + L * math.cos(math.radians(phi)), 7.2 - L * math.sin(math.radians(phi))])
        assert np.min(np.linalg.norm(w.vertices[:, 1:] - target, axis=1)) < 1e-6

def test_hinge_bevel_depends_on_angle(flat, flat_env):
    shell, env = flat, flat_env
    v45 = M.difference(shell, M.hinge(shell, env, **HINGE)).volume
    v20 = M.difference(shell, M.hinge(shell, env, **dict(HINGE, bevel_deg=20))).volume
    assert abs(v45 - v20) > 1.0                             # the cut changes with the angle (its sign is geometry-dependent)

def test_unsane_input_raises():
    open_mesh = trimesh.Trimesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]), process=False)
    with pytest.raises(ValueError):
        M.to_manifold(open_mesh)

@pytest.mark.skipif(not __import__("importlib").util.find_spec("build123d"), reason="build123d not installed (legacy/ builder)")
def test_matches_brep_plate():
    sys.path.insert(0, os.path.join(ROOT, 'legacy')); import trilobite as T
    brep = T.to_trimesh(T.plate(OUT, ZF, T_SHELL, nu=53, nv=29), *T.SANE_MESH_TOL)
    ours = M.heightfield_shell(OUT, ZF, T_SHELL)
    pts, _ = trimesh.sample.sample_surface(ours, 5000, seed=0)
    _, d, _ = trimesh.proximity.closest_point(brep, pts)
    # Measured 10 Sep 2026: the B-spline least-squares fit (tol 0.08) sits ~0.25 mm BELOW the exact height field on a
    # 12 mm dome, sides agree to 0.05 mm. The mesh builder is the faithful one; 0.30 is the frozen-reference tolerance.
    assert np.median(d) < 0.30
    assert abs(ours.volume - brep.volume) / brep.volume < 0.05
