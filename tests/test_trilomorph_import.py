"""scripts/trilomorph_import.py (20 Oct 2026): a TriloMorph cephalon's landmarks and margin curves become a saved head."""
import os, sys, json
import numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "isopod_model"))
import trilomorph_import as TI, shaped_head as SH


def _synthetic():
    """A semicircular head 20 mm long, 24 mm half-width, with a 10 mm genal spine swept 20 deg outward and an eye, in a rotated,
    translated frame (the photo's), so the import has to find the axis."""
    th = np.radians(25.0); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); T = np.array([300.0, 120.0])
    W = lambda p: (R @ np.asarray(p, float)) + T                           # head frame (x across, y: 0 rear, -20 front) -> photo
    phi = np.linspace(np.pi / 2, 0, 60); ant = [W([24 * np.cos(a), -20 * np.sin(a)]) for a in phi]     # LM13 (front) -> LM15 (genal angle)
    post = [W([x, 0.0]) for x in np.linspace(0, 24, 30)]                                                  # LM4 -> LM15
    lm = {"LM1": W([0, -20]), "LM13": W([0, -20]), "LM4": W([0, 0]), "LM15": W([24, 0]), "LM16": W([24 + 10 * np.sin(np.radians(20)), 10 * np.cos(np.radians(20))]),
          "LM8": W([14, -14]), "LM9": W([14, -8]), "LM14": W([17, -11]), "LM5": W([9, -12]), "LM6": W([7, -3]), "LM2": W([0, -18]), "LM3": W([0, -3]), "LM7": W([7, 0]),
          "LM10": W([0, -20]), "LM11": W([22, -1]), "LM12": W([23, 0])}
    return dict(lm={k: tuple(v) for k, v in lm.items()}, curves=dict(anterior=np.array(ant), posterior=np.array(post)))


def test_synthetic_head_round_trips():
    h = TI.head_from(_synthetic()); o = h["outline"]; f = h["face"]
    tr = np.array(o["trace"]); assert tr[0].tolist() == [0.0, pytest.approx(-20 / 24, abs=0.01)] and tr[-1].tolist() == [0.0, 0.0]
    assert abs(tr[:, 0].max() - 1.0) < 1e-6 and (tr[:, 1] <= 0.99).all()
    assert abs(o["genal"] - 10 / 20) < 0.02 and abs(o["genal_spread"] - 20) < 1.0                       # spine length / sagittal length; spread recovered through the rotation
    assert abs(f["eye_size"] - 3 / 24) < 0.01 and abs(f["eye_pos"] - 11 / 20) < 0.01 and abs(f["eye_lat"] - (17 - 3) / 24) < 0.01
    assert abs(f["axis_frac"] - 7 / 24) < 0.01 and abs(f["glab_inflate"] - 9 / 7) < 0.01
    assert not h["blind"]
    SH.outline_of(o)                                                                                      # the head editor accepts it
    sh = SH.Shape(o); assert abs(sh.L1 / SH.RO - 20 / 24) < 0.02 and sh.horn


def test_blind_and_no_spine():
    rec = _synthetic(); rec["lm"]["LM9"] = rec["lm"]["LM8"]; rec["lm"]["LM16"] = rec["lm"]["LM15"]
    h = TI.head_from(rec); assert h["blind"] and h["face"]["eye_size"] == 0.0 and h["outline"]["genal"] == 0.0


def test_tps_and_txt_readers_find_landmarks_and_curves(tmp_path):
    tps = "LM=16\n" + "\n".join(f"{i}.0 {2 * i}.0" for i in range(16)) + "\nCURVES=4\n" + "".join("POINTS=3\n1 1\n2 2\n3 3\n" for _ in range(4)) + "IMAGE=x.png\nSCALE=0.1\nID=x\n"
    p = tmp_path / "x_C.tps"; p.write_text(tps); r = TI.read_tps(str(p))
    assert len(r["lm"]) == 16 and set(r["curves"]) == set(TI.CURVES) and r["curves"]["posterior"].shape == (3, 2)
    txt = ("<shapes type=list>\n\t<landmarks.scaled type=matrix rownames=TRUE colnames=FALSE nrow=2 ncol=2 as.numeric=TRUE >\n\t\tLM1\t0\t-1\n\t\tLM4\t0\t0\n\t</landmarks.scaled>\n"
           "\t<curves.scaled type=list>\n\t\t<anterior type=matrix rownames=FALSE colnames=FALSE nrow=2 ncol=2 as.numeric=TRUE >\n\t\t\t0\t-1\n\t\t\t1\t0\n\t\t</anterior>\n\t</curves.scaled>\n</shapes>\n")
    q = tmp_path / "y_C.txt"; q.write_text(txt); r = TI.read_txt(str(q))
    assert r["lm"]["LM4"] == (0.0, 0.0) and r["curves"]["anterior"].shape == (2, 2)


def test_import_endpoint_reads_a_tps_upload(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "web")); import app as A; import io
    A.app.config["TESTING"] = True
    rec = _synthetic(); L = rec["lm"]
    order = [f"LM{i}" for i in range(1, 17)]
    lines = ["LM=16"] + ["%.3f %.3f" % L[k] for k in order] + ["CURVES=4"]
    for name in ("glabella", "suture", "anterior", "posterior"):
        pts = rec["curves"].get(name, np.array([L["LM2"], L["LM7"]]))
        lines += ["POINTS=%d" % len(pts)] + ["%.3f %.3f" % tuple(p) for p in pts]
    lines += ["IMAGE=syn.png", "SCALE=1.0", "ID=syn"]
    with A.app.test_client() as c:
        r = c.post("/api/heads/import", data={"file": (io.BytesIO("\n".join(lines).encode()), "syn_C.tps")}, content_type="multipart/form-data")
        assert r.status_code == 200, r.get_json()
        j = r.get_json(); assert j["style"] == "plain" and len(j["outline"]["trace"]) == 72 and abs(j["outline"]["genal"] - 0.5) < 0.02 and j["landmarks"] == 16
        assert c.post("/api/heads/import", data=b"hello").status_code == 400
        assert c.post("/api/heads/import", data=b"LM=2\n0 0\n1 1\n").status_code == 400
