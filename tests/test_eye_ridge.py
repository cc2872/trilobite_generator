"""tests/test_eye_ridge.py: the eye ridge (schema 6.2), tested on the head module alone.

    python -m pytest tests/test_eye_ridge.py -q

Same contract as the suture: eyeRidge = 0 leaves the plan bit-identical; the ridge is the stated height on its line,
absent off it, mirror-symmetric, runs from the glabella to the front of the eye, and the shell still builds closed.
"""
import numpy as np, pytest
import schema, mesh as M
from anatomy import head as HEAD

BASE = schema.coerce(dict(segCount=6, eyeSize=0.16, eyePos=0.55, eyeLat=0.55, genalSpine=0.3))
GRID = (81, 41)

def _grid(S, n=GRID):
    u = np.linspace(-1, 1, n[0]); v = np.linspace(0, 1, n[1])
    xy = np.array([[S["outline"](ui, vi) for ui in u] for vi in v])
    return xy[..., 0], xy[..., 1], S["zfun"](xy[..., 0], xy[..., 1])

def test_height_zero_leaves_the_plan_bit_identical():
    x, y, z0 = _grid(HEAD.plan(dict(BASE, eyeRidge=0.0)))
    _, _, z = _grid(HEAD.plan(dict(BASE, eyeRidge=0.0, eyePos=BASE["eyePos"])))
    assert np.array_equal(z, z0)
    S = HEAD.plan(BASE); assert "eye_ridge" in S and S["eye_ridge"].shape == (2, 2)

@pytest.mark.parametrize("hgt", [0.4, 1.2])
def test_ridge_is_on_its_line_and_nowhere_else(hgt):
    S0 = HEAD.plan(dict(BASE, eyeRidge=0.0)); S1 = HEAD.plan(dict(BASE, eyeRidge=hgt)); p = S1["eye_ridge"]
    on = p[0] + np.outer(np.linspace(0.2, 0.8, 7), p[1] - p[0])                   # interior points of the line
    dz = S1["zfun"](on[:, 0], on[:, 1]) - S0["zfun"](on[:, 0], on[:, 1])
    assert np.all(dz > 0.97 * hgt) and np.all(dz < 1.03 * hgt)
    t = p[1] - p[0]; nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)
    off = on + 3.0 * nrm                                                          # 3 mm to the side = 5 sigma
    dz_off = S1["zfun"](off[:, 0], off[:, 1]) - S0["zfun"](off[:, 0], off[:, 1])
    assert np.all(np.abs(dz_off) < 0.01 * hgt)
    ya = np.linspace(-0.9 * S1["Lc"], 0.0, 30); xa = np.zeros_like(ya)              # the axis is untouched
    assert np.array_equal(S0["zfun"](xa, ya), S1["zfun"](xa, ya))

def test_ridge_is_mirror_symmetric():
    S = HEAD.plan(dict(BASE, eyeRidge=0.8)); x, y, _ = _grid(S)
    assert np.allclose(S["zfun"](x, y), S["zfun"](-x, y))

def test_ridge_runs_from_the_glabella_to_the_front_of_the_eye():
    S = HEAD.plan(dict(BASE, eyeRidge=0.8)); (x0, y0), (x1, y1) = S["eye_ridge"]; G = S["eye"]
    g = float(S["glab_half"](np.array([y0]))[0]) if "glab_half" in S else None
    assert x0 > 0.5 * S["a"] and x0 < G["xe"]                                     # leaves beside the glabella, inboard of the eye
    assert y0 < y1                                                                # from the frontal lobe (front = -y) back to the eye
    assert abs(np.hypot(x1 - G["xe"], y1 - G["ye"]) - 1.1 * G["eR"]) < 0.3 * G["eR"]   # ends at the eye's front edge
    assert y1 < G["ye"]                                                           # ahead of the eye centre

def test_blind_head_still_has_a_ridge():
    S = HEAD.plan(dict(BASE, eyeSize=0.0, eyeRidge=0.8)); p = S["eye_ridge"]
    assert S["eye"] is None and p.shape == (2, 2) and p[0, 1] < p[1, 1]

def test_ridge_and_suture_coexist():
    """Both on: the suture's groove still reaches its depth where it crosses nothing, the ridge its height."""
    S0 = HEAD.plan(dict(BASE)); S = HEAD.plan(dict(BASE, eyeRidge=0.8, sutureDepth=0.3))
    q = S["suture"]; mid = 0.5 * (q[-2] + q[-1])                                    # on the posterior branch, far from the ridge
    assert abs((S0["zfun"](mid[:1], mid[1:]) - S["zfun"](mid[:1], mid[1:]))[0] - 0.3) < 0.03
    p = S["eye_ridge"]; m = 0.5 * (p[0] + p[1])
    assert abs((S["zfun"](m[:1], m[1:]) - S0["zfun"](m[:1], m[1:]))[0] - 0.8) < 0.08

def test_shell_still_builds_closed_with_the_ridge():
    S = HEAD.plan(dict(BASE, eyeRidge=1.2)); h, _ = HEAD.shell(BASE, S, grid=(161, 81))   # a 3 mm ridge needs the finer grid to sample
    assert h.is_watertight and len(M.bodies(h)) == 1 and M.is_symmetric(h)
    S0 = HEAD.plan(BASE); h0, _ = HEAD.shell(BASE, S0, grid=(161, 81))
    # a constant-thickness skin: the ridge lifts top and bottom together, so the volume barely moves while the skin does
    assert abs(h.volume - h0.volume) < 0.02 * h0.volume
    dz = h.vertices[:, 2] - h0.vertices[:, 2]                                       # same grid: vertices correspond one to one
    assert dz.min() > -1e-9 and 0.8 * 1.2 < dz.max() <= 1.2 + 1e-9                 # nothing lowered; the ridge line reaches its height
    p = S["eye_ridge"]; m = 0.5 * (p[0] + p[1])
    v = h.vertices[np.argmax(dz), :2].copy(); v[0] = abs(v[0])                    # either side: the plan is mirrored
    assert np.linalg.norm(v - m) < 0.6 * np.linalg.norm(p[1] - p[0])                # and the highest lift is on the line
