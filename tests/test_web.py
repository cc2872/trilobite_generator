"""tests/test_web.py — the Flask API on the mesh builder: schema/cells, presets, build (+cache, GLB), measure."""
import os, sys, json, time
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "web"))
pytest.importorskip("flask")
import app as A

@pytest.fixture(scope="module")
def c(): return A.app.test_client()

def test_schema_endpoint(c):
    s = c.get("/api/schema").get_json()
    assert s["version"] == "6.0" and s["instrument"] == "2.1" and len(s["params"]) == 95
    assert set(s["cells"]) == {"frame", "b1", "a1", "b2", "a2", "b3", "a3", "ruler"}

def test_preset_and_index(c):
    assert c.get("/").status_code == 200
    assert c.get("/api/preset/nope").status_code == 404
    P = c.get("/api/preset/proetida").get_json(); assert len(P) == 95

def test_build_caches_and_serves_glb(c):
    P = c.get("/api/preset/proetida").get_json(); P["segCount"] = 4
    m = c.post("/api/build", json={"P": P}).get_json()
    assert len(m["parts"]) == 6 and all(p.get("bodies") == 1 for p in m["parts"]) and m["print"]["print_valid"] in (True, False)
    assert c.post("/api/build", json={"P": P}).get_json()["key"] == m["key"]
    g = c.get(m["parts"][1]["url"]); assert g.status_code == 200 and len(g.data) > 10000
    assert set(m["kinematics"]) == {"hinge_z", "offsets", "names"} and len(m["kinematics"]["offsets"]) == 5

@pytest.mark.slow
def test_measure_endpoint(c):
    P = c.get("/api/preset/proetida").get_json(); P["segCount"] = 4
    r = c.post("/api/measure", json={"P": P}).get_json()
    assert r["instrument"] == "2.1" and r["limited_by"] in ("closed", "anatomy", "bound", "invalid") and "kinematics" in r
