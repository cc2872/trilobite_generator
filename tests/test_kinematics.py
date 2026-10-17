"""The enrolment bend profile (kinematics.py): Esteve et al. 2017's expression theta(i) = Ln[a i] b, and the sheet /
print-model paths that use it. Instrument 2.1 is not touched by it: its transforms stay uniform."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import kinematics as KIN

# Esteve et al. 2017 Fig. 5A (spiral model, 15 segments), read off the figure: joint 1 = cephalon / first segment, then joints 2..15
ESTEVE_SPIRAL_15 = [45, 23.0, 20.5, 19.0, 17.9, 17.1, 16.4, 15.8, 15.3, 14.9, 14.5, 14.2, 13.9, 13.6, 13.3]


def test_the_expression_and_its_constants():
    assert abs(KIN.A - 0.0019) < 0.0002 and abs(KIN.B + 3.69) < 0.02           # the paper's a and b (spiral model), as fitted
    assert abs(KIN.c_first() - 23.0) < 0.1                                      # c = b ln a: the first thoracic joint
    for i in (1, 3, 7, 14):                                                     # theta(i) = b ln(a i) = c + b ln i
        assert abs(KIN.theta(i) - (KIN.c_first() + KIN.B * math.log(i))) < 1e-9
    assert KIN.theta(10_000) == 0.0                                             # never negative


def test_profile_reproduces_esteve_2017_spiral_curve():
    p = KIN.profile(15)
    assert len(p) == 15 and p[0] == 45.0
    assert max(abs(a - b) for a, b in zip(p[1:], ESTEVE_SPIRAL_15[1:])) < 0.3   # within the reading error of the figure
    assert abs(sum(p) - 274) < 1.5                                              # their total: 14 c + b ln(14!) + 45
    assert all(p[k] > p[k + 1] for k in range(1, 14))                           # monotone decay behind the head


def test_the_count_dependence_that_decides_closure():
    # same a, b: a shorter chain ends higher and sums lower (the 15-vs-14-segment mechanism)
    assert abs(KIN.profile(10)[-1] - 14.9) < 0.1 and abs(KIN.total(10, 45.0) - 205) < 1.5
    assert abs(KIN.total(15, 45.0) - 274) < 1.5


def test_angles_scale_with_the_head_and_degenerate_chains_work():
    a = KIN.angles(10, 30.0)
    assert abs(a[0] - 30.0) < 1e-9 and abs(a[1] - KIN.theta(1) * 30 / 45) < 1e-9
    assert KIN.angles(10, 0.0) == [0.0] * 10
    assert KIN.profile(0) == [] and KIN.profile(1) == [45.0] and KIN.profile(2) == [45.0, KIN.theta(1)]


def test_sheet_helpers_take_a_head_angle_or_a_list_and_agree():
    import blueprint as B
    piv = [(20.0 * k, 7.5) for k in range(6)]
    T1 = B._curl_mats(piv, 30.0); T2 = B._curl_mats(piv, KIN.angles(6, 30.0))
    for m1, m2 in zip(T1, T2): assert np.allclose(m1, m2)
    ang = B._angles(piv, 30.0); assert ang[0] == 30.0 and ang[-1] < ang[1] < ang[0]
    secs = [[np.array([[20.0 * k, 0.0], [20.0 * k + 10, 5.0]])] for k in range(7)]
    P1 = B._posed_sections(secs, piv, 30.0); P2 = B._posed_sections(secs, piv, KIN.angles(6, 30.0))
    for a, b in zip(P1, P2):
        for q1, q2 in zip(a, b): assert np.allclose(q1, q2)


def test_instrument_2_1_is_still_uniform():
    import instrument as I, schema
    import joints.pin as PIN
    P = schema.defaults(); mats = I.transforms_deg(P, 20.0)
    assert np.allclose(np.array(mats), np.array(PIN.pose(P, [20.0] * len(PIN.joint_offsets(P)))))


def test_dorsal_pivots_sit_at_the_plate_edge_above_the_ball():
    import trimesh
    # a toy chain: boxes 20 long, 10 tall, each overlapping the next by 4 (plate over lip), on the bed
    pieces = [trimesh.creation.box((30, 20, 10)).apply_translation((0, 10 + 16 * k, 5)) for k in range(4)]
    dp = KIN.dorsal_pivots(pieces)
    assert len(dp) == 3
    for k, (y, z) in enumerate(dp):
        assert abs(y - (20 + 16 * k)) < 1e-6          # the rear edge of piece k
        assert abs(z - 0.0) < 1e-6                     # the underside at that edge (a box has no plate: its bottom)
