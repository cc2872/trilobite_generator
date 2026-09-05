"""Skin system invariants (fast: works on the cached .npz skins, plus one CAD plate)."""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pytest
import skins

SK = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "skins", "*.npz")))

@pytest.mark.parametrize("path", SK)
def test_skin_arrays_well_formed(path):
    s = skins.load_skin(path)
    assert s["z"].shape == (skins.NV, skins.NU) and s["w"].shape == (skins.NV,)
    assert np.isfinite(s["z"]).all() and np.isfinite(s["w"]).all()
    assert (s["z"] >= 0).all() and 0.15 < s["z"].max() < 1.0            # heights as a fraction of length
    assert 0 < s["w"].min() and s["w"].max() <= 1.2                    # silhouette as a fraction of max half-width
    assert np.allclose(s["z"], s["z"][:, ::-1])                          # bilateral
    assert s["w"][0] > s["w"][-1]                                        # heads are wider at the rear than at the tip

def test_blend_identity_and_normalisation():
    a, b = [skins.load_skin(p) for p in SK[:2]]
    assert np.allclose(skins.blend([a, a], [3, 7])["z"], a["z"])
    m = skins.blend([a, b], [1, 1])["z"]
    assert np.allclose(m, 0.5 * (a["z"] + b["z"]))

def test_skin_functions_hit_the_grid():
    s = skins.load_skin(SK[0]); L, W = 40.0, 30.0
    outline, zfun = skins.skin_functions(s, L, W)
    x, y = outline(0.0, 0.0); assert abs(x) < 1e-9 and abs(y) < 1e-9        # rear midline is the origin
    x, y = outline(1.0, 1.0); assert y == -L and abs(x - s["w"][-1] * W) < 1e-9
    z0 = zfun(np.array([0.0]), np.array([0.0]))[0]; assert abs(z0 - s["z"][0, skins.NU // 2] * L) < 1e-6

def test_skin_plate_builds():
    import trilobite as T
    s = skins.load_skin(SK[0]); outline, zfun = skins.skin_functions(s, 40.0, 30.0)
    plate = T.plate(outline, zfun, 2.0, nu=41, nv=29)
    assert plate.is_valid and len(plate.solids()) == 1
