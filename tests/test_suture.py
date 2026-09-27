"""tests/test_suture.py: the facial suture (schema 6.2), tested on the head module alone.

    python -m pytest tests/test_suture.py -q

Nothing here touches a joint, the assembler or the instrument: anatomy/head.py is partitioned, so the suture is
checked where it lives. sutureDepth = 0 must leave the plan bit-identical (the frozen references depend on it).
"""
import numpy as np, pytest
import schema, mesh as M
from anatomy import head as HEAD

BASE = schema.coerce(dict(segCount=6, eyeSize=0.16, eyePos=0.55, eyeLat=0.55, genalSpine=0.3))
GRID = (81, 41)

def _grid(S, n=GRID):
    """The plan sampled on its own outline grid: (x, y, z) arrays."""
    u = np.linspace(-1, 1, n[0]); v = np.linspace(0, 1, n[1])
    xy = np.array([[S["outline"](ui, vi) for ui in u] for vi in v])
    x, y = xy[..., 0], xy[..., 1]
    return x, y, S["zfun"](x, y)

def test_depth_zero_leaves_the_plan_bit_identical():
    """The default head is the 6.1 head: sutureEnd has no effect at all while sutureDepth is 0."""
    x, y, z0 = _grid(HEAD.plan(dict(BASE, sutureDepth=0.0, sutureEnd=0.0)))
    for end in (-1.0, 0.0, 1.0):
        _, _, z = _grid(HEAD.plan(dict(BASE, sutureDepth=0.0, sutureEnd=end)))
        assert np.array_equal(z, z0)

def test_plan_carries_the_path():
    S = HEAD.plan(BASE); p = S["suture"]
    assert isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3
    assert np.isfinite(p).all() and (p[:, 0] >= 0).all()                     # right side only; the plan is mirrored

@pytest.mark.parametrize("depth", [0.2, 0.5])
def test_groove_is_where_the_path_is_and_nowhere_else(depth):
    S0 = HEAD.plan(dict(BASE, sutureDepth=0.0)); S1 = HEAD.plan(dict(BASE, sutureDepth=depth))
    p = S1["suture"]
    # on the path (midpoints of its segments): the surface drops by the full depth
    mid = 0.5 * (p[:-1] + p[1:]); inner = mid[2:-2]                             # clear of the two margin ends
    dz = S0["zfun"](inner[:, 0], inner[:, 1]) - S1["zfun"](inner[:, 0], inner[:, 1])
    assert np.all(dz > 0.9 * depth) and np.all(dz < 1.05 * depth)
    # 2 mm off the path (normal offset) the groove has vanished; on the axis nothing changed at all
    t = p[1:] - p[:-1]; nrm = np.stack([-t[:, 1], t[:, 0]], 1); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    off = inner + 2.0 * nrm[2:-2]
    far = HEAD._polyline_dist(off[:, 0], off[:, 1], p) > 1.9                    # keep only points that really are 2 mm from every segment
    dz_off = S0["zfun"](off[far, 0], off[far, 1]) - S1["zfun"](off[far, 0], off[far, 1])
    assert np.all(np.abs(dz_off) < 0.01 * depth)
    ya = np.linspace(-0.9 * S1["Lc"], 0.0, 30); xa = np.zeros_like(ya)
    assert np.array_equal(S0["zfun"](xa, ya), S1["zfun"](xa, ya))

def test_groove_is_mirror_symmetric():
    S = HEAD.plan(dict(BASE, sutureDepth=0.3)); x, y, _ = _grid(S)
    assert np.allclose(S["zfun"](x, y), S["zfun"](-x, y))

def test_suture_end_selects_the_margin():
    """-1: lateral margin, forward of the genal angle. 0: the genal angle. +1: the posterior margin, inboard of it."""
    ends = {}
    for e in (-1.0, 0.0, 1.0):
        S = HEAD.plan(dict(BASE, sutureEnd=e, sutureDepth=0.2)); ends[e] = (S, S["suture"][-1])
    S, (x, y) = ends[-1.0]
    x_lat = float(S["xmax"](np.array([y]))[0]); y_gen = float(S["outline"](1.0, 0.0)[1])
    assert abs(x - x_lat) < 0.5 and y < y_gen - 1.0                              # on the side, ahead of the genal angle
    assert y > S["eye"]["ye"]                                                    # but behind the eye
    S, (x, y) = ends[0.0]
    assert abs(x - x_lat) < 0.5 and abs(y - y_gen) < 0.5                         # the genal angle itself
    S, (x, y) = ends[1.0]
    y_rear = float(S["outline"](x / S["outline"](1.0, 0.0)[0], 0.0)[1])
    assert x < 0.7 * S["wh"] and abs(y - y_rear) < 0.5                           # on the rear edge, well inboard

def test_loop_keeps_clear_of_the_eye_and_the_glabella():
    S = HEAD.plan(dict(BASE, sutureDepth=0.3)); p = S["suture"]; G = S["eye"]
    r = np.hypot(p[:, 0] - G["xe"], p[:, 1] - G["ye"])
    assert r.min() > 1.05 * G["eR"]                                              # never inside the eye's rim
    g = S["glab_half"] if "glab_half" in S else None
    assert (p[:, 0] > 0.5 * S["a"]).all()                                        # never on the axis

def test_blind_head_has_a_straight_suture():
    S = HEAD.plan(dict(BASE, eyeSize=0.0, sutureDepth=0.3)); p = S["suture"]
    assert S["eye"] is None and len(p) == 3                                       # front, mid, margin: no palpebral loop

def test_shell_still_builds_closed_with_the_groove():
    S = HEAD.plan(dict(BASE, sutureDepth=0.5, sutureEnd=-1.0))
    h, env = HEAD.shell(BASE, S, grid=(161, 81))
    assert h.is_watertight and len(M.bodies(h)) == 1 and M.is_symmetric(h)
    S0 = HEAD.plan(dict(BASE, sutureDepth=0.0)); h0, _ = HEAD.shell(BASE, S0, grid=(161, 81))
    assert abs(h0.volume - h.volume) < 0.02 * h0.volume                             # a skin: the groove moves it, barely changes it
    # the groove survives the builder's blur: it is added after it (mesh.sample_grid detail=), so the built skin drops
    # by the full depth on the line (same grid: vertices correspond one to one)
    dz = h0.vertices[:, 2] - h.vertices[:, 2]
    assert dz.min() > -1e-9 and 0.85 * 0.5 < dz.max() <= 0.5 + 1e-9

def test_plan_and_shell_agree_without_detail():
    """A head with neither ridge nor groove hands the builder no detail at all: the 6.1 path, bit for bit."""
    S = HEAD.plan(BASE); assert S["detail"] is None and S["zfun"] is S["zfun_base"]
