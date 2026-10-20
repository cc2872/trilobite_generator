"""20 Oct 2026: the anatomical default head ("plain"), the crescent kept by name, the named styles unchanged, and the traced margin."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "isopod_model")); sys.path.insert(0, os.path.join(ROOT, "web"))
import numpy as np, pytest
import shaped_head as SH, head_styles as HS

TRACE = [[0, -1.3], [0.3, -1.25], [0.6, -1.0], [0.85, -0.6], [1.0, -0.2], [1.0, 0.0]]


def test_default_is_anatomical_not_the_kit():
    o = SH.OUTLINE
    assert o["front_exp"] < 2.0 and o["genal_tip"] < 0.3 and o["notch"] < 0.2 and o["genal_w"] < 0.15       # pointed, tapered spine, straight rear
    k = SH.CRESCENT_OUTLINE
    assert k["notch"] == 0.9 and k["genal_tip"] == 1.0 and k["front_exp"] == 2.0                             # the kit silhouette, kept
    assert HS.outline("crescent") == k and HS.outline("plain") == {} and "plain" in HS.STYLES


def test_named_styles_say_every_old_default_explicitly():
    for n, o in HS.OUTLINES.items():
        if n == "plain": continue
        for key in HS._OLD: assert key in o, (n, key)          # so a change of OUTLINE's defaults never moves a named style


def test_every_style_draws():
    for n in HS.OUTLINES: SH.Shape(HS.outline(n))               # cryptolithus (genal 2.6) was outside the range until today


def test_trace_builds_a_shape_and_drops_the_margin_handles():
    sh = SH.Shape({"trace": TRACE, "genal": 0.5})
    assert abs(sh.L1 / SH.RO - 1.3) < 1e-6 and sh.horn and abs(sh.xg / SH.RO - 1.0) < 1e-6
    assert np.all(sh.yr >= sh.yf) and np.isfinite(sh.yf).all() and np.isfinite(sh.yr).all()
    D = SH.diagram({"trace": TRACE, "genal": 0.5})
    assert D["trace"].shape == (6, 2) and not {"front", "widest", "angle", "notch"} & set(D["handles"]) and {"spine", "root", "eye", "glabella"} <= set(D["handles"])
    assert SH.diagram({})["trace"] is None


def test_trace_validation():
    with pytest.raises(ValueError): SH.outline_of({"trace": [[0, 0], [1, 0]]})                 # too few
    with pytest.raises(ValueError): SH.outline_of({"trace": [[0, 0], [1, -0.5], [1, -1]]})     # first point not the front
    with pytest.raises(ValueError): SH.outline_of({"trace": [[0, -1], [float("nan"), 0], [1, 0]]})
    assert SH.outline_of({"trace": []}).get("trace") is None
    assert SH.outline_of({"trace": [[0, -1], [-0.8, -0.5], [1, 0]]})["trace"][1][0] == 0.8       # folded


def test_trace_through_the_api_and_a_saved_head():
    import app as A
    A.app.config["TESTING"] = True
    with A.app.test_client() as c:
        r = c.post("/api/head_diagram", json={"style": "plain", "outline": {"trace": TRACE}, "face": {}})
        assert r.status_code == 200 and len(r.get_json()["outline"]["trace"]) == 6 and "front" not in r.get_json()["diagram"]["handles"]
        assert c.post("/api/head_diagram", json={"style": "plain", "outline": {"trace": [[0, 0], [1, 0]]}}).status_code == 400
        names = [s["name"] for s in c.get("/api/head_styles").get_json()]
        assert names[:2] == ["plain", "crescent"]
        r = c.post("/api/heads", json={"name": "pytest-traced", "style": "plain", "outline": {"trace": TRACE, "genal": 0.5}, "face": {}})
        try:
            assert r.status_code == 200 and sorted(r.get_json()["saved"]["outline"]) == ["genal", "trace"]
            d = c.get("/api/head_diagram?style=pytest-traced").get_json()
            assert len(d["outline"]["trace"]) == 6 and d["saved"]["name"] == "pytest-traced"
            assert c.get("/api/head_diagram?style=cryptolithus").status_code == 200
        finally:
            c.delete("/api/heads/pytest-traced")
