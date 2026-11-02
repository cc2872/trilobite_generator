"""tests/test_organic.py -- the organic protocol (anatomy/organic.py, 1 Nov 2026): what each rule guarantees, and what
stays at 0 (the instrument, the landmarks)."""
import os, json
import numpy as np
import pytest
import schema, mesh as M
from anatomy import organic as OR, genal as GN, thorax as T
from anatomy.common import curved_spine, spine_solid, prong

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _preset(n): return schema.coerce(json.load(open(os.path.join(ROOT, "presets", f"{n}.json")))["params"])

def _width_at(m, y):
    s = m.section(plane_origin=(0, y, 0), plane_normal=(0, 1, 0)); return float(np.ptp(s.vertices[:, 0])) if s is not None else 0.0

def test_a_long_thin_spine_becomes_a_horn_with_a_round_tip():
    """Rules 1 and 2: a 100 mm spine asked for at 0.5 mm root / 0.1 mm tip."""
    s0 = curved_spine(0.5, 0.1, 100.0, (0, 0, 0), 0.0, 0.0, 0.0, organic=0.0)
    s1 = curved_spine(0.5, 0.1, 100.0, (0, 0, 0), 0.0, 0.0, 0.0, organic=1.0)
    assert s1.is_watertight and len(M.bodies(s1)) == 1
    r0 = 100.0 / OR.SLENDER; rt = OR.TIP_FRAC * r0
    assert _width_at(s1, 50.0) >= 0.95 * 2 * (rt + (r0 - rt) * 0.5 ** OR.TAPER_EXP)   # the full, convex taper
    assert _width_at(s1, 97.0) >= 0.9 * 2 * rt                                       # no needle at the end
    assert s1.vertices[:, 1].max() > 100.0 + 0.5 * rt                                 # a round cap past the end
    assert s1.volume > 5 * s0.volume

def test_off_is_the_old_frustum():
    a = spine_solid(2.0, 0.4, 30.0, (1, 2, 3), 25.0, 10.0, organic=0.0)
    assert len(a.vertices) < 200 and a.is_watertight                                  # the plain frustum, as since 6.0

def test_prong_stays_one_body():
    p = prong(3, 40.0, 40.0, 1.0, 0.3, (0, 0, 0), stem_frac=0.4, bend_deg=50.0)
    assert p.is_watertight and len(M.bodies(p)) == 1

def test_pleural_spine_is_no_thinner_than_its_length_allows():
    """Rules 1, 2, 4 in the thorax's plan: the root floor, a round cap, a ridge standing above the edges."""
    P = _preset("odontopleurida")
    S0 = T.plan(dict(P, organic=0.0), 3); S1 = T.plan(dict(P, organic=1.0), 3)
    w, xt = S1["w"], S1["X_tip"]; Ls = xt - w
    yf, yr = S1["edges"](np.array([w + 0.05 * Ls]))
    assert 0.5 * float(yr[0] - yf[0]) >= 0.9 * Ls / OR.SLENDER_PLAN
    yf0, yr0 = S0["edges"](np.array([w + 0.05 * Ls])); assert float(yr[0] - yf[0]) > float(yr0[0] - yf0[0])
    x = np.array([w + 0.5 * Ls]); a, b = S1["edges"](x); mid = 0.5 * (a + b)
    assert float(S1["zfun"](x, mid)[0]) > float(S1["zfun"](x, a + 0.05 * (b - a))[0]) + 0.1   # crest above the edge
    assert T.solid(dict(P, organic=1.0), S1).is_watertight

def test_a_kinked_genal_piece_bends_through_an_arc():
    p0 = GN.Piece((10, 0), 40, 2.0, 75, kink=0.3, kink_angle=22, organic=0.0)
    p1 = GN.Piece((10, 0), 40, 2.0, 75, kink=0.3, kink_angle=22, organic=1.0)
    assert [b[0] for b in p0.bands] == ["root", "segment1", "segment2"]
    assert [b[0] for b in p1.bands] == ["root", "segment1"] and p1.path is not None
    assert p1.polygon().is_valid and p1.tip_round >= OR.TIP_FRAC - 1e-9

def test_widening_keeps_the_inner_edge():
    """A genal piece widened by rule 1 grows on its outer side: its inner edge at the root stays where it was."""
    p0 = GN.Piece((10, 0), 60, 2.0, 0.0, organic=0.0)
    turn = OR.CURVE_PLAN; OR.CURVE_PLAN = 0.0                                          # rule 1 alone (rule 7 would bow it)
    try: p1 = GN.Piece((10, 0), 60, 2.0, 0.0, organic=1.0)
    finally: OR.CURVE_PLAN = turn
    assert p1.W > p0.W
    inner = lambda p: p.bands[1][1](-0.9, 0.0)[0]                                      # segment1, near its root, v = 0 (inner side)
    assert abs(inner(p1) - inner(p0)) < 0.3

def test_the_instrument_and_the_landmarks_stay_at_zero():
    import instrument, landmarks
    P = _preset("redlichiida")
    assert instrument._measured(P)["organic"] == 0.0
    a = landmarks.cephalon(dict(P, organic=1.0))["lm"]; b = landmarks.cephalon(dict(P, organic=0.0))["lm"]
    assert all(np.allclose(a[k], b[k]) for k in a)

def test_a_straight_spine_is_a_curve_toward_the_rear():
    """Rule 7: asked for straight, a spine pointing out (+x) bends so its tip turns toward the rear (+y), in the plane."""
    s = curved_spine(1.0, 0.4, 40.0, (5, 0, 2), 90.0, 0.0, 0.0, organic=1.0)
    far = s.vertices[np.argmax(s.vertices[:, 0])]
    assert far[1] > 3.0 and abs(far[2] - 2.0) < 2.0
    kept = curved_spine(1.0, 0.4, 40.0, (5, 0, 2), 90.0, 0.0, 10.0, 90.0, organic=1.0)   # a bend the user set is kept
    assert kept.vertices[np.argmax(kept.vertices[:, 0])][1] < 3.0

def test_the_genal_piece_is_one_curve():
    """Rule 7: unkinked, an arc on the same chord (the tip end stays where it was asked for); kinked, a smooth curve."""
    p = GN.Piece((10, 0), 40, 2.0, 0.0, organic=1.0)
    assert p.path is not None and np.allclose(p.path[-1], p.end) and p.path[:, 0].max() > 11.0   # bowed outward
    q = GN.Piece((10, 0), 40, 2.0, 75, kink=0.3, kink_angle=22, organic=1.0)
    assert q.path is not None and OR.min_bend_radius(q.path) > 0.5 * q.W
    assert GN.Piece((10, 0), 40, 2.0, 0.0, organic=0.0).path is None

def test_a_hook_turns_early_and_keeps_its_length():
    """hook > 0: most of the turn near the root, then a straight run; the tip is where spine_tip says; hook 0 = the arc."""
    from anatomy.common import spine_tip, hook_turn
    assert hook_turn(0.25, 0.8) > 0.6 and abs(hook_turn(0.25, 0.0) - 0.25) < 1e-12
    s = curved_spine(1.5, 0.5, 60.0, (0, 0, 0), 0.0, 30.0, 150.0, 90.0, hook=0.6, organic=0.0)
    t = spine_tip(60.0, (0, 0, 0), 0.0, 30.0, 150.0, 90.0, hook=0.6)
    assert s.is_watertight and len(M.bodies(s)) == 1 and np.min(np.linalg.norm(s.vertices - t, axis=1)) < 1.0
    a = spine_tip(60.0, (0, 0, 0), 0.0, 30.0, 150.0, 90.0); b = spine_tip(60.0, (0, 0, 0), 0.0, 30.0, 150.0, 90.0, hook=1e-12)
    assert np.allclose(a, b, atol=1e-6)
