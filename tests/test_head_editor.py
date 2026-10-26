"""tests/test_head_editor.py — the head editor (web/head.html, 9 Oct 2026): /head, /static, /api/head_diagram, and the
custom-head cache key. No build here (a shaped-head build is minutes); the build path is exercised by
test_web.test_build_with_a_head_style and by the editor itself."""
import os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "web"))
pytest.importorskip("flask")
import schema
import app as A

@pytest.fixture(scope="module")
def c(): return A.app.test_client()

def test_pages_and_static(c):
    assert c.get("/head").status_code == 200
    r = c.get("/static/p5.min.js"); assert r.status_code == 200 and len(r.data) > 500_000
    assert c.get("/static/../app.py").status_code == 404                                  # no traversal out of web/static

def test_diagram_of_a_style_matches_the_build_face(c):
    d = c.get("/api/head_diagram?style=harpes").get_json()
    assert d["style"] == "harpes" and set(d["diagram"]) >= {"outline", "glabella", "section", "profile", "handles", "size"}
    assert set(d["diagram"]["handles"]) >= {"front", "widest", "glabella"} and d["handles"]["front"][0] == "length"
    # the face the drawing uses is the one the build uses: harpes sets eyeSize in its schema params, body.face_of maps it
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import head_styles as HS, body as B
    P = schema.coerce(dict(schema.preset(schema.DEFAULT_PRESET), **HS.style("harpes")[1]), base=schema.table_defaults())
    assert d["face"]["eye_size"] == pytest.approx(B.face_of(P, HS.style("harpes")[0])["eye_size"])
    assert d["face"]["eye_size"] < 0.1                                                      # harpes: eyes reduced to tubercles

def test_edits_redraw_and_bad_keys_are_refused(c):
    base = c.get("/api/head_diagram?style=phacops").get_json()
    d = c.post("/api/head_diagram", json={"style": "phacops", "outline": {"genal": 0.8}, "face": {"glab_len": 0.5}}).get_json()
    assert d["outline"]["genal"] == 0.8 and d["face"]["glab_len"] == 0.5 and d["id"] != base["id"]
    assert "spine" in d["diagram"]["handles"] and "spine" not in base["diagram"]["handles"]   # the spine handle appears with the spine
    assert d["diagram"]["handles"]["glabella"][1] > base["diagram"]["handles"]["glabella"][1]  # a shorter glabella ends further back
    assert c.post("/api/head_diagram", json={"style": "phacops", "outline": {"bogus": 1}}).status_code == 400
    assert c.post("/api/head_diagram", json={"style": "phacops", "face": {"bogus": 1}}).status_code == 400

def test_custom_head_key_is_its_own(c):
    P = schema.defaults()
    k_style = A._iso_key(P, "harpes"); k_same = A._iso_key(P, {"style": "harpes"}); k_edit = A._iso_key(P, {"style": "harpes", "outline": {"genal": 1.5}})
    assert k_style != k_same and "-h-custom-" in k_same and k_same != k_edit
    assert A._iso_key(P, {"style": "harpes", "outline": {"genal": 1.5}}) == k_edit          # deterministic
    h = A._head_style({"style": "harpes", "face": {"glab_len": 0.5}})
    assert h["outline"]["genal"] == pytest.approx(1.3) and h["face"]["glab_len"] == 0.5     # the style's outline, the edit on its face

def test_saved_heads_round_trip(c, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HEADS", str(tmp_path))                                              # the test's own store
    r = c.post("/api/heads", json={"name": "t-harpes", "style": "harpes", "outline": {"genal": 1.6}, "face": {"glab_len": 0.5}, "doc": "test"})
    assert r.status_code == 200 and r.get_json()["saved"]["style"] == "harpes" and r.get_json()["saved"]["outline"] == {"genal": 1.6}
    assert c.post("/api/heads", json={"name": "harpes", "style": "harpes"}).status_code == 400          # a guide style's name
    assert c.post("/api/heads", json={"name": "../x", "style": "harpes"}).status_code == 400            # not a file name
    assert c.post("/api/heads", json={"name": "t-bad", "style": "harpes", "outline": {"bogus": 1}}).status_code == 400
    assert [h["name"] for h in c.get("/api/heads").get_json()] == ["t-harpes"]
    assert any(h.get("saved") and h["name"] == "t-harpes" and h["style"] == "harpes" for h in c.get("/api/head_styles").get_json())
    # the saved name resolves wherever a style name goes, to the same geometry as the live dict
    d = c.get("/api/head_diagram?style=t-harpes").get_json()
    assert d["base"] == "harpes" and d["saved"]["name"] == "t-harpes" and d["outline"]["genal"] == 1.6 and d["face"]["glab_len"] == 0.5
    P = schema.defaults()
    assert A._iso_key(P, "t-harpes") == A._iso_key(P, {"style": "harpes", "outline": {"genal": 1.6}, "face": {"glab_len": 0.5}})
    assert "-h-custom-" in A._iso_key(P, "t-harpes")
    # edits on top of a saved head; saving from a saved head flattens onto the guide style
    r = c.post("/api/heads", json={"name": "t-harpes2", "style": "t-harpes", "outline": {"genal_spread": 15}})
    assert r.get_json()["saved"]["style"] == "harpes" and r.get_json()["saved"]["outline"] == {"genal": 1.6, "genal_spread": 15.0}
    assert c.delete("/api/heads/t-harpes2").status_code == 200 and c.delete("/api/heads/t-harpes2").status_code == 404
    assert [h["name"] for h in c.get("/api/heads").get_json()] == ["t-harpes"]

def test_opening_page_links_and_share(c):
    front = c.get("/").get_data(as_text=True)                                                    # 12 Oct: the front page is now the combined editor (studio.html)
    assert 'id="stage"' in front and 'id="build"' in front                                       # its canvas + build button
    html = c.get("/generator").get_data(as_text=True)                                            # the parametric generator moved to /generator
    assert 'id="editor"' in html and 'id="familytree"' in html                                   # the generator's nav row
    body = c.get("/body").get_data(as_text=True); head = c.get("/head").get_data(as_text=True)
    assert 'id="home"' in body and 'id="headeditor"' in body and 'id="home"' in head and 'id="bodyeditor"' in head   # 11 Oct: each editor links home and to the other
    assert "&head=" in html and "&joint=" in html                                                 # the share link carries the head and joint

# ---- the body editor (10 Oct 2026): rows, fusion, the diagram, saved bodies
def test_thorax_model_is_the_generators_own_build():
    """The default body model (10 Oct 2026, evening): an unedited spec gives the generator's own thorax and tail plans,
    bit for bit; a row edit changes only that row's plan; short rows fuse; the pygidial rows are the tail."""
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW
    from anatomy import thorax as THORAX, tail as TAIL
    from anatomy.common import pitch
    import numpy as np
    P = schema.coerce(schema.preset("textured")); P["printLength"] = 0.0; R = RW.plans(P, {})   # the canonical body under the old pitch rule (the default is now 'diagram' at printLength 160, which fuses short rows)
    assert R["model"] == "thorax" and R["scale"] == pytest.approx(18.0 / pitch(P)) and [n for n, *_ in R["parts"]][-1] == "tail"
    xs = np.linspace(-30, 30, 31); ys = np.linspace(0, 12, 13); X, Y = np.meshgrid(xs, ys)
    for i in range(int(P["segCount"])):
        S0, S1 = THORAX.plan(P, i), R["parts"][i][2]
        assert np.abs(S0["zfun"](np.abs(X), Y) - S1["zfun"](np.abs(X), Y)).max() < 1e-9 and R["offsets"][i] == pytest.approx(i * pitch(P))
        assert all(abs(a - b) < 1e-9 for a, b in zip(S0["outline"](0.7, 0.3), S1["outline"](0.7, 0.3)))
    T0, T1 = TAIL.plan(P), R["parts"][-1][2]
    assert np.abs(T0["zfun"](np.abs(X), Y) - T1["zfun"](np.abs(X), Y)).max() < 1e-9 and T1["wp"] == pytest.approx(T0["wp"]) and T1["Lp"] == pytest.approx(T0["Lp"])
    # one row edited: that row's plan moves, its neighbours' do not
    R2 = RW.plans(P, {"rows": {"4": {"width": 40.0, "sweep": 35.0, "tip": 10.0}}})
    S4 = R2["parts"][4][2]; assert S4["w"] == pytest.approx(40.0) and S4["X_tip"] > 40.0 + 5
    assert np.abs(R2["parts"][3][2]["zfun"](np.abs(X), Y) - THORAX.plan(P, 3)["zfun"](np.abs(X), Y)).max() < 1e-9
    # 26 Oct 2026: the thorax never fuses -- short rows stay their own pieces (flagged small for the editor); only the pygidium is fused
    R3 = RW.plans(P, {"rows": {"3": {"len": 4.0}, "4": {"len": 4.0}}})
    assert [3] in R3["groups"] and [4] in R3["groups"] and len(R3["parts"]) == len(R["parts"])
    D = RW.diagram(P, {"rows": {"3": {"len": 4.0}, "4": {"len": 4.0}}})
    assert D["model"] == "thorax" and [r["i"] for r in D["rows"] if r["small"]] == [3, 4] and len(D["side"]) == len(D["pieces"])
    assert D["side"][0]["bottom"][0][1] == 0.0 and D["side"][0]["bottom"][-1][1] > 0.0        # solid to the bed, a plate at the rear
    with pytest.raises(ValueError): RW.plans(P, {"model": "bogus"})
    with pytest.raises(ValueError): RW.plans(P, {"list": [dict(RW.boring(P)[-1])]})           # a pygidium with no thorax

def test_fidelity_and_macro_hold_for_every_preset():
    """The rows model, over EVERY order preset (not just the default, review G61): an unedited body is the generator's
    own thorax and tail to machine precision, and the macro projects the width back within the projection's tolerance."""
    import glob, json, numpy as np
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW
    from anatomy import thorax as THORAX, tail as TAIL
    xs = np.linspace(-30, 30, 31); ys = np.linspace(0, 12, 13); X, Y = np.meshgrid(xs, ys)
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, "presets", "*.json"))) if not os.path.basename(f).startswith("_")]
    assert len(files) >= 8
    for f in files:
        name = os.path.basename(f)[:-5]; P = schema.coerce(json.load(open(f))["params"], base=schema.table_defaults()); P["printLength"] = 0.0; R = RW.plans(P, {})   # 25 Oct: pin the old pitch rule so each thorax row is its own piece
        for i in range(int(P["segCount"])):
            d = np.abs(THORAX.plan(P, i)["zfun"](np.abs(X), Y) - R["parts"][i][2]["zfun"](np.abs(X), Y)).max()
            assert d < 1e-6, f"{name} thorax row {i}: {d:.2e}"
        T0, T1 = TAIL.plan(P), R["parts"][-1][2]
        assert np.abs(T0["zfun"](np.abs(X), Y) - T1["zfun"](np.abs(X), Y)).max() < 1e-6, f"{name} tail fidelity"
        mw = RW.macro(P, {})["width"]
        assert abs(mw - P["width"]) < 0.12 * P["width"], f"{name} macro width {mw:.2f} vs {P['width']:.2f}"   # macro is a lossy projection (agnostida ~5.6%)

def test_mixed_kinds_refused():
    """A thoracic row after a pygidial one is refused, not silently folded into the tail (review B14)."""
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW
    b = RW.boring(schema.defaults())
    with pytest.raises(ValueError): RW.validate_list([dict(b[0]), dict(b[-1]), dict(b[0])])

def test_rows_model_defaults_and_fusion():
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW
    P = schema.coerce(schema.preset("textured")); rs = RW.rows_from_P(P, model="slab"); sc = RW.default_scale(rs)   # the canonical fusion body (the default is now 'diagram', 25 Oct)
    th = [r for r in rs if r["kind"] == "thorax"]; py = [r for r in rs if r["kind"] == "pyg"]
    assert len(th) == int(P["segCount"]) and th[0]["len"] * sc == pytest.approx(18.0)            # the preset's rows, at the isopod pitch
    g = RW.groups(rs, sc)
    assert g[:len(th)] == [[i] for i in range(len(th))]                                          # every thoracic row its own piece
    assert [r for grp in g[len(th):] for r in grp] == list(range(len(th), len(rs))) and len(g) < len(rs)   # the pygidium rows fuse (pairwise since 25 Oct) into the trailing pieces, in order
    assert RW.groups(RW.apply(rs, {"rows": {"3": {"len": 3.0}, "4": {"len": 3.0}}}), sc)[3] == [3, 4, 5] or \
           RW.groups(RW.apply(rs, {"rows": {"3": {"len": 3.0}, "4": {"len": 3.0}}}), sc)[2] == [2, 3, 4]          # short rows fuse with a neighbour
    with pytest.raises(KeyError): RW.apply(rs, {"rows": {"0": {"bogus": 1}}})
    with pytest.raises(KeyError): RW.apply(rs, {"bogus": 1})
    D = RW.diagram(P, {"model": "slab", "thorax_rows": 11, "pyg_rows": 20, "pyg_len": 95, "term_spine": 12, "print_scale": 1.5})   # a hand-size scale so the short ribs fuse (the slab model still fuses; the thorax model no longer does, 26 Oct)
    assert D["body"]["thorax_rows"] == 11 and D["body"]["pyg_rows"] == 20 and D["term_spine"] == 12 and len(D["rows"]) == 31
    assert [r for pc in D["pieces"] for r in pc["rows"] if r >= 11] == list(range(11, 31)) and len(D["pieces"]) < 31   # the 20 ribs become the trailing fused pieces, in order (pairwise since the 25-Oct rule)

def test_body_endpoints(c, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "BODIES", str(tmp_path))
    assert c.get("/body").status_code == 200
    d = c.get("/api/body_diagram?preset=textured&head=dalmanites").get_json()
    dg = d["diagram"]; pyg = int(A._presets()["textured"]["pygRings"])
    assert d["preset"] == "textured" and d["head"]["style"] == "dalmanites"
    assert dg["pieces"][-1]["name"] == "tail" and dg["pieces"][-1]["rows"] == list(range(len(dg["rows"]) - pyg, len(dg["rows"])))   # the preset's pygRings ribs are the tail piece's rows (fused since 25 Oct)
    spec = {"thorax_rows": 11, "pyg_rows": 20, "pyg_len": 95, "term_spine": 12, "rows": {"2": {"tip": 40, "sweep": 35}}}
    e = c.post("/api/body_diagram", json={"preset": "textured", "spec": spec}).get_json()
    assert e["diagram"]["rows"][2]["row"]["tip"] == 40 and e["id"] != d["id"]
    assert c.post("/api/body_diagram", json={"preset": "textured", "spec": {"rows": {"2": {"bogus": 1}}}}).status_code == 400
    assert c.post("/api/bodies", json={"name": "t-body", "preset": "textured", "spec": spec}).status_code == 200
    assert c.post("/api/bodies", json={"name": "textured", "preset": "textured", "spec": {}}).status_code == 400   # a preset's name
    assert [b["name"] for b in c.get("/api/bodies").get_json()] == ["t-body"]
    P = schema.defaults()
    assert A._iso_key(P, None, "t-body") == A._iso_key(P, None, spec) and "-b-" in A._iso_key(P, None, spec)
    assert A._iso_key(P, None, None) != A._iso_key(P, None, {})                                      # the rows model is its own build even unedited
    assert c.delete("/api/bodies/t-body").status_code == 200 and c.get("/api/bodies").get_json() == []

def test_plain_start_and_explicit_row_list(c):
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW
    P = schema.defaults(); P["printLength"] = 0.0; b = RW.boring(P)   # 25 Oct: old pitch rule so the plain body is 10 pieces (9 thorax + the fused pygidium), not re-fused
    assert all(r["tip"] == 0 for r in b) and sum(r["kind"] == "thorax" for r in b) == 9                  # plain: no spines, 9 thoracic rows
    D = RW.diagram(P, {"list": b}); assert D["mode"] == "list" and len(D["pieces"]) == 10 and D["pieces"][-1]["rows"] == list(range(9, 15))
    lst = b + [dict(b[-1])]                                                                           # "+": a copy of the row before
    assert len(RW.diagram(P, {"list": lst})["rows"]) == 16
    lst2 = b[:4] + [dict(b[3], len=4.0)] + b[4:]                                                      # a short copy fuses with its neighbour
    g = RW.groups(RW.apply(b, {"list": lst2}), RW.default_scale(b)); assert any(4 in grp and len(grp) > 1 for grp in g)
    with pytest.raises(ValueError): RW.validate_list([{"len": -1}])
    with pytest.raises(KeyError): RW.validate_list([{"bogus": 1}])
    d = c.get("/api/body_diagram?preset=textured&start=boring").get_json()
    assert d["diagram"]["mode"] == "list" and len(d["diagram"]["list"]) == 15 and d["diagram"]["list"][0]["tip"] == 0
    e = c.post("/api/body_diagram", json={"preset": "textured", "spec": {"list": d["diagram"]["list"] + [d["diagram"]["list"][-1]], "term_spine": 5}}).get_json()
    assert len(e["diagram"]["rows"]) == 16 and e["diagram"]["term_spine"] == 5

def test_body_diagram_side_section(c):
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW
    P = schema.coerce(schema.preset("textured")); P["printLength"] = 0.0   # old pitch rule: lips/plates lie 1:1 per row (the endpoint clamps printLength to its range, so build the diagram directly)
    d = RW.diagram(P, {"list": RW.boring(P)})
    assert len(d["side"]) == len(d["pieces"]) and len(d["pivots"]) == len(d["pieces"]) - 1
    p0, p1 = d["side"][0], d["side"][1]
    assert p0["y_end"] > p0["y1"]                                                                    # the plate over the next piece
    import numpy as np                                                                               # the lip clears the plate ahead, pointwise
    by, bz = zip(*p0["bottom"]); under = [q for q in p1["top"] if q[0] <= p0["y_end"]]
    assert under and all(q[1] <= np.interp(q[0], by, bz) - d["gap_z"] + 1e-3 for q in under)

@pytest.mark.slow
def test_rows_model_builds_sane_plain_body():
    """The plain trilobite through the rows path: every piece one watertight body, nothing touching at rest or at
    20 deg per joint, plates WALL thick (~90 s)."""
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW, body as BODY, check as CHECK
    P = schema.defaults(); P["printLength"] = 0.0                                                # 25 Oct: old pitch rule (the new default fuses the plain body to fewer pieces)
    parts, plans, PV, K = BODY.build(P, "isopod", rows={"list": RW.boring(P)})
    assert len(parts) == 11 and all(p.is_watertight and len(p.split(only_watertight=False)) == 1 for p in parts)   # head + 9 thorax + the fused pygidium (old pitch rule, pinned above)
    pv = [(y, z) for y, z in PV]
    assert CHECK.contacts(parts, pv, 0.0) == [] and CHECK.contacts(parts, pv, 20.0) == []

# ---- guards (11 Oct 2026): nothing invalid is built or served
def test_spec_bounds_refused(c):
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW
    P = schema.defaults(); b = RW.boring(P)
    for bad in ({"rows": {"0": {"len": -3}}}, {"rows": {"0": {"width": 1e6}}}, {"rows": {"0": {"sweep": float("nan")}}}, {"rows": {"0": {"len": "x"}}},
                {"print_scale": 0}, {"min_joint_mm": 1e9}, {"thorax_rows": 500}, {"list": b * 10}, {"rows": {"0": "no"}}, {"model": "cube"}):
        with pytest.raises((ValueError, KeyError)): RW.plans(P, bad)
        assert c.post("/api/body_diagram", json={"preset": "textured", "spec": bad}).status_code == 400
    assert c.post("/api/body_diagram", json={"preset": "textured", "spec": {"rows": {"0": {"len": 12, "sweep": 40}}}}).status_code == 200
    assert c.post("/api/build", json={"P": {}, "joint": "isopod", "body": {"rows": {"0": {"width": 1e6}}}}).status_code == 400          # refused before any build
    assert c.post("/api/sheet", json={"P": {}, "joint": "isopod", "body": {"rows": {"0": {"len": -1}}}}).status_code == 400

def test_clean_and_validate_parts():
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import body as BODY
    import trimesh, numpy as np
    big = trimesh.creation.box((20, 20, 20)); speck = trimesh.creation.box((0.3, 0.3, 0.3)); speck.apply_translation((30, 0, 0))
    sliver = trimesh.creation.box((5, 5, 0.1)); sliver.apply_translation((0, 40, 0))
    parts = [trimesh.util.concatenate([big, speck, sliver])]; plans = [("seg0",)]
    dropped = BODY.clean_parts(parts, plans)
    assert len(parts[0].split(only_watertight=False)) == 1 and dropped[0] == pytest.approx(0.027 + 2.5, abs=1e-3)   # the speck (< 2 mm3) and the two-face sliver (0.1 mm thin) go
    BODY.validate_parts(parts, plans)                                                                                 # and what is left is valid
    two = trimesh.util.concatenate([big, trimesh.creation.box((10, 10, 10)).apply_translation((40, 0, 0))])
    with pytest.raises(BODY.MeshError): BODY.validate_parts([two], plans)                                            # two real bodies: refused
    holed = big.copy(); holed.update_faces(np.arange(len(holed.faces)) > 0)
    with pytest.raises(BODY.MeshError): BODY.validate_parts([holed], plans)                                          # not watertight: refused

# ---- the editors mirrored on the macro model (11 Oct 2026)
def test_macro_projection_and_saved_body_head(c, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "BODIES", str(tmp_path / "b")); monkeypatch.setattr(A, "HEADS", str(tmp_path / "h"))
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW, head_styles as HS
    P = schema.coerce(schema.preset("textured")); P["printLength"] = 0.0   # 25 Oct: the macro round-trip is tight for the textbook textured body under the old pitch rule
    m0 = RW.macro(P, {})                                                                                   # an unedited body: the preset back, near enough
    assert all(abs(float(m0[k]) - float(P[k])) < 0.01 for k in m0 if isinstance(P[k], (int, float)))
    b = RW.boring(P); b[4]["width"] = 40.0; b[4]["sweep"] = 35.0; b[2]["tip"] = 12.0
    m = RW.macro(P, {"list": b})
    assert m["segCount"] == 9 and m["width"] == pytest.approx(80.0) and 0.3 < m["widthMaxPos"] < 0.8 and m["macroIndex"] == 2 and m["macroAmp"] > 0.2
    hm = HS.macro(P, dict(HS.outline("harpes"), genal=1.0, genal_curve=15), {"eye_size": 0.2}, HS.STYLES["harpes"]["params"])
    assert hm["genalSpine"] == pytest.approx(1.0) and hm["genalCurve"] == -15 and hm["eyeSize"] == pytest.approx(0.2) and hm["cephFrac"] > P["cephFrac"]
    # through the site: a saved head, a saved body that remembers it, and /api/macro on both
    assert c.post("/api/heads", json={"name": "t-mh", "style": "harpes", "outline": {"genal": 1.0}, "face": {"eye_size": 0.2}}).status_code == 200
    assert c.post("/api/bodies", json={"name": "t-mb", "preset": "textured", "head": "t-mh", "spec": {"list": b}}).status_code == 200
    assert c.post("/api/bodies", json={"name": "t-mb2", "preset": "textured", "head": "no-such-head", "spec": {}}).status_code == 404
    assert c.get("/api/bodies").get_json()[0]["head"] == "t-mh"
    r = c.post("/api/macro", json={"preset": "textured", "head": "t-mh", "body": "t-mb"}); assert r.status_code == 200
    o = r.get_json()["P"]; assert o["genalSpine"] == pytest.approx(1.0) and o["segCount"] == 9 and o["width"] == pytest.approx(80.0) and o["eyeSize"] == pytest.approx(0.2)
    assert c.post("/api/macro", json={"preset": "textured", "head": "dalmanites"}).get_json()["P"].get("genalSpine") is None   # a guide style: its values only
    assert c.post("/api/macro", json={"preset": "textured", "body": {"rows": {"0": {"len": -1}}}}).status_code == 400
    assert c.post("/api/macro", json={"preset": "textured", "head": "nope"}).status_code == 404
    html = c.get("/generator").get_data(as_text=True); assert "/api/macro" in html and "applyMacro" in html   # the generator moved to /generator (12 Oct)

def test_isopod_progress_counts_parts():
    """The loading count for the isopod model is in parts (head + pieces + tail), not 0 of 1 (11 Oct 2026)."""
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW
    P = schema.coerce(schema.preset("textured")); P["printLength"] = 0.0   # 25 Oct: the canonical body under the old pitch rule (count is head + every row + tail; the new default fuses short rows)
    assert A._iso_n_parts(P, None) == int(P["segCount"]) + 2
    b = RW.boring(P); b[3]["len"] = 4.0; b[4]["len"] = 4.0                                              # 26 Oct: the thorax no longer fuses -- short rows stay their own pieces
    assert A._iso_n_parts(P, {"list": b}) == 9 + 2
    assert "0 parts out of 1" not in A.app.test_client().get("/api/progress").get_data(as_text=True)

# ---- the stress battery's lessons (11 Oct 2026): refused before building, drawn with a warning
def test_unbuildable_bodies_and_heads_are_refused_early(c):
    sys.path.insert(0, os.path.join(ROOT, "isopod_model")); import rows as RW, shaped_head as SH
    P = schema.defaults(); P["printLength"] = 0.0   # 25 Oct: old pitch rule so the oversized cases exceed the bed as the thresholds were set (the new smaller scale would pass the term-spine case)
    assert RW.plans(P, {})["problems"] == [] and RW.plans(P, {"list": RW.boring(P)})["problems"] == []
    needle = {"rows": {str(i): {"width": 2.0} for i in range(9)}}; flat = {"rows": {str(i): {"relief": 0.5} for i in range(9)}}
    big = {"print_scale": 20}; spine = {"list": [{"len": 8, "width": 30, "kind": "thorax"}] * 3, "term_spine": 500, "print_scale": 5}   # print_scale pins the size so the oversized check fires at any printLength (the endpoint clamps printLength to its range)
    for bad, word in ((needle, "wide"), (flat, "tall"), (big, "print would be"), (spine, "print would be")):
        with pytest.raises(ValueError, match=word): RW.plans(P, bad)                                  # the build path refuses
        assert RW.plans(P, bad, strict=False)["problems"] and RW.diagram(P, bad)["problems"]           # the editor draws it and says why
        r = c.post("/api/body_diagram", json={"preset": "textured", "spec": bad}); assert r.status_code == 200 and r.get_json()["diagram"]["problems"]
        assert c.post("/api/build", json={"P": {}, "joint": "isopod", "body": bad}).status_code == 400
    for bad in ({"width": 0.3}, {"length": 3.0}, {"genal": float("nan")}, {"front_exp": "x"}):
        with pytest.raises(ValueError): SH.outline_of(bad)
    assert c.post("/api/build", json={"P": {}, "joint": "isopod", "head": {"style": "harpes", "outline": {"length": 2.5, "width": 2.5, "genal": 2.5}}}).status_code == 400   # a 1.6 m print
    assert A._grid_step(0.25, 80, 40) == 0.25 and A._grid_step(0.25, 600, 600) > 0.25 if hasattr(A, "_grid_step") else True

