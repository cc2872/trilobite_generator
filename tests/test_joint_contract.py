"""tests/test_joint_contract.py: what EVERY joint module must satisfy (joints/base.py). Parametrised over the joints
in joints/__init__.py, so a new joint file is tested the moment it is registered.

    python -m pytest tests/test_joint_contract.py -q

A joint that passes here can be swapped in without touching anatomy/, assemble.py, the instrument or the site.
"""
import itertools, numpy as np, pytest, trimesh
import schema, mesh as M, assemble, joints

JOINTS = ["pin", "flexi"]
ANIMAL = dict(segCount=4, maxAngle=18)                      # small default animal: fast, has every port type


@pytest.fixture(scope="module", params=JOINTS)
def built(request):
    J = joints.get(request.param); P = schema.coerce(ANIMAL)
    parts = assemble.animal(P, J)
    return J, P, parts


def test_template_surface(built):
    J, P, _ = built
    for name in ("NAME", "MEASURED", "BASE", "fits", "pivot", "stop_deg", "cut", "overhang", "pose", "transforms_deg"):
        assert hasattr(J, name), f"{J.__name__} lacks {name}"
    assert J.BASE in ("shell", "solid")
    assert J.fits(P) is True
    pv = J.pivot(P); assert "z" in pv and "y_beyond_plane" in pv
    assert 0 < J.stop_deg(P) <= 90


def test_every_part_is_one_watertight_body(built):
    J, P, parts = built
    assert len(parts) == int(P["segCount"]) + 2
    for k, m in enumerate(parts):
        assert m.is_watertight, f"{J.NAME} part {k} not watertight"
        assert len(m.split(only_watertight=False)) == 1, f"{J.NAME} part {k} is {len(m.split(only_watertight=False))} bodies"
        assert M.is_symmetric(m), f"{J.NAME} part {k} is not mirror-symmetric"


def test_pose_shape_and_identity(built):
    J, P, parts = built
    n = len(parts); mats = J.pose(P, [0.0] * (n - 1))
    assert len(mats) == n and np.allclose(mats[0], np.eye(4))
    for T in mats: assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
    u = J.transforms_deg(P, 7.5); v = J.pose(P, [7.5] * (n - 1))
    assert all(np.allclose(a, b) for a, b in zip(u, v))


def test_no_overlap_at_rest_or_at_the_stop(built):
    """Adjacent AND non-adjacent pairs: nothing shares volume flat or fully curled (the joint's own tolerance is
    OVERLAP_TOL of the instrument, 0.5 mm3; a print joint must be tighter than that)."""
    J, P, parts = built
    n = len(parts)
    for deg in (0.0, J.stop_deg(P)):
        posed = [M.to_manifold(m.copy().apply_transform(T)) for m, T in zip(parts, J.pose(P, [deg] * (n - 1)))]
        for i, j in itertools.combinations(range(n), 2):
            ov = (posed[i] ^ posed[j]).volume()
            assert ov < 0.5, f"{J.NAME}: parts {i},{j} overlap {ov:.2f} mm3 at {deg} deg"


def test_curls_to_its_stop_without_jamming(built):
    """Bending every joint together from flat to the stop never makes two adjacent parts intersect beyond the
    tolerance: the joint's stop, not a collision, is what ends the curl on the default animal."""
    J, P, parts = built
    n = len(parts); stop = J.stop_deg(P)
    for deg in np.linspace(0, stop, 4):
        posed = [M.to_manifold(m.copy().apply_transform(T)) for m, T in zip(parts, J.pose(P, [deg] * (n - 1)))]
        for i in range(n - 1):
            assert (posed[i] ^ posed[i + 1]).volume() < 0.5, f"{J.NAME}: parts {i},{i+1} jam at {deg:.1f} deg"


def test_pose_is_a_rigid_chain(built):
    """Each part's transform is rigid, and joint k only moves parts k+1.. (the head never moves)."""
    J, P, parts = built
    n = len(parts); v = [0.0] * (n - 1); v[1] = 10.0
    mats = J.pose(P, v)
    for T in mats:
        R = T[:3, :3]; assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    flat = J.pose(P, [0.0] * (n - 1))
    assert np.allclose(mats[1], flat[1]) and not np.allclose(mats[2], flat[2])
