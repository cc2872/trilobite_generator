"""tests/test_crescent_detail.py (2 Oct 2026): the facial suture and the eye ridge on the crescent head, and the
order both heads draw the suture in. Head modules only; nothing here builds a joint or an animal."""
import numpy as np, pytest
import schema, mesh as M
from anatomy import head as CLASSIC, head_crescent as HC

X, Y = np.meshgrid(np.linspace(-23.5, 23.5, 141), np.linspace(-13.5, 29.5, 121))
ON = dict(suture_depth=0.3, suture_end=1.0, eye_ridge=0.8)

def test_default_face_is_the_approved_head_bit_for_bit():
    assert not HC.has_detail()
    assert np.array_equal(HC._z(X, Y), HC._z_base(X, Y))
    for end in (-1.0, 0.0, 1.0):                                  # sutureEnd alone (depth 0) draws nothing
        assert np.array_equal(HC._z(X, Y, dict(suture_end=end)), HC._z_base(X, Y))
    S = HC.plan(schema.coerce({})); assert S["detail"] is None and S["zfun"] is S["zfun_base"]

def test_groove_and_ridge_reach_their_height_on_the_line_and_vanish_off_it():
    d = lambda p, f: float(HC._detail(np.array([p[0]]), np.array([p[1] + HC.Y0]), f)[0])
    s = HC.suture_path(ON); r = HC.eye_ridge_path(ON)
    mid = 0.5 * (s[-2] + s[-1])                                   # middle of the rear branch, clear of the ridge
    assert d(mid, dict(suture_depth=0.3, suture_end=1.0)) == pytest.approx(-0.3, abs=0.02)
    assert d(0.5 * (r[0] + r[1]), dict(eye_ridge=0.8)) == pytest.approx(0.8, abs=0.02)
    assert abs(d((0.0, -10.0), ON)) < 1e-6                        # the axis is untouched
    assert abs(d(mid + np.array([3.0, 0.0]), dict(suture_depth=0.3, suture_end=1.0))) < 1e-6

def test_detail_is_mirror_symmetric():
    assert np.allclose(HC._detail(X, Y, ON), HC._detail(-X, Y, ON))

@pytest.mark.parametrize("mod", ["classic", "crescent"])
@pytest.mark.parametrize("end", [-1.0, -0.6, -0.25, 0.0, 0.5, 1.0])
def test_suture_runs_front_to_rear_without_doubling_back(mod, end):
    """The reversed eye loop (fixed 2 Oct 2026) sent the path from the front margin to the eye's REAR, forward round
    the loop, then back across the eye. y must never decrease along the path (head frame: front is -y)."""
    if mod == "classic":
        pts = CLASSIC.plan(schema.coerce(dict(sutureDepth=0.3, sutureEnd=end)))["suture"]
    else:
        pts = HC.suture_path(dict(suture_depth=0.3, suture_end=end))
    assert np.all(np.diff(pts[:, 1]) >= -1e-9)

def test_each_end_lands_where_it_says():
    tip_y, root_y = HC.C + HC.ARM_LEN - HC.Y0, HC.C - HC.Y0
    pro = HC.suture_path(dict(suture_depth=0.3, suture_end=-1.0))[-1]
    gon = HC.suture_path(dict(suture_depth=0.3, suture_end=0.0))[-1]
    opi = HC.suture_path(dict(suture_depth=0.3, suture_end=1.0))[-1]
    assert pro[1] < root_y and pro[0] > 0.8 * HC.RO              # lateral margin, forward of the widest line
    assert gon[0] == pytest.approx(HC.A_N) and gon[1] > tip_y - 1.0   # the horn's tip
    assert opi[0] < 0.7 * HC.A_N and root_y < opi[1] < tip_y     # the notch, inboard of the horn

def test_built_crescent_shell_carries_the_detail_and_stays_closed():
    P = schema.coerce(dict(sutureDepth=0.3, sutureEnd=1.0, eyeRidge=0.8)); S = HC.plan(P)
    assert S["detail"] is not None and len(S["suture"]) > 3
    sh, env = HC.shell(P, S); so = HC.solid(P, S)
    for m in (sh, env, so): assert m.is_watertight and M.is_symmetric(m)
    P0 = schema.coerce({}); sh0, _ = HC.shell(P0, HC.plan(P0))
    assert abs(sh.volume - sh0.volume) / sh0.volume < 0.02
    mid = 0.5 * (S["suture"][-2] + S["suture"][-1])               # built depth = the schema's mm (plan divides by its scale)
    assert float(S["detail"](np.array([mid[0]]), np.array([mid[1]]))[0]) == pytest.approx(-0.3, abs=0.03)

def test_isopod_face_keeps_its_cache_key_when_nothing_is_set():
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isopod_model"))
    import body
    f0 = body.face_of(schema.coerce({})); assert set(f0) == {"eye_size", "eye_pos", "eye_height", "eye_lat", "glab_inflate", "glab_rise", "glab_front", "glab_lobes"}
    f1 = body.face_of(schema.coerce(dict(sutureDepth=0.3, sutureEnd=-1.0, eyeRidge=0.6)))
    assert f1["suture_depth"] == 0.3 and f1["suture_end"] == -1.0 and f1["eye_ridge"] == 0.6
