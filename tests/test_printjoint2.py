"""Print joint (printjoint2): every part one watertight body, no rest overlap, captured under pull, stop at maxAngle."""
import numpy as np, trimesh, schema, parts, mesh as M, instrument as I, printjoint2 as J2

def _pen(a, b, T): return (M.to_manifold(a) ^ M.to_manifold(b.copy().apply_transform(T))).volume()

def test_tall_dorsal_spine_carries_into_the_flexi_build():
    """A tall back-swept axial spine must reach into the flexi segment, not be sliced off by the per-segment run trim
    (the 21 Sep bug: flexi thorax read ~28 mm vs the pin's ~53). Watertightness alone did not catch it — the spine
    was silently trimmed. Assert the flexi segment keeps most of the pin's height, and the part stays one body."""
    P = schema.coerce(dict(maxAngle=18, axialSpine=2.0))
    pin = parts.segment(P, 3); fx = J2.clean(J2.print_segment(P, 3))
    assert len(M.bodies(fx)) == 1 and fx.is_watertight
    assert fx.bounds[1][2] > 0.85 * pin.bounds[1][2], (fx.bounds[1][2], pin.bounds[1][2])

def test_coupon_is_captured_and_stops_at_maxangle():
    P = schema.coerce(dict(maxAngle=30)); a, b = J2.chain(P, deg=0.0, n=2); J, d, zj, y_piv = J2.geometry(P)
    assert len(M.bodies(a)) == 1 and len(M.bodies(b)) == 1 and a.is_watertight and b.is_watertight
    assert _pen(a, b, np.eye(4)) < 0.005                                                  # no rest overlap
    assert _pen(a, b, trimesh.transformations.translation_matrix((0, 0.6, 0))) > 0.05       # pull 0.6 mm: held
    ok = _pen(a, b, trimesh.transformations.rotation_matrix(np.radians(-28), (1, 0, 0), (0, y_piv, zj))) < 0.005
    hit = _pen(a, b, trimesh.transformations.rotation_matrix(np.radians(-34), (1, 0, 0), (0, y_piv, zj))) > 0.05
    assert ok and hit                                                                        # free at 28, stopped by 34

def test_default_animal_builds_and_curls():
    P = schema.coerce(dict(maxAngle=24)); pieces = J2.print_animal(P)
    assert all(len(M.bodies(p)) == 1 and p.is_watertight for p in pieces)
    for deg in (0.0, 22.0):
        posed = I._posed(pieces, J2.transforms_deg(P, deg)); mans = [M.to_manifold(p) for p in posed]
        for i in range(len(posed)):
            for j in range(i + 1, len(posed)):
                if I._aabb_hit(posed[i], posed[j]): assert (mans[i] ^ mans[j]).volume() < 0.01, (deg, i, j)
