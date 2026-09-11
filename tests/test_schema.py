"""tests/test_schema.py — schema 6.0: the cell contract covers every key, presets carry no dead keys, prongs build,
and a whole mesh-built animal reads the same on instrument 2.1 as the frozen OpenCascade one."""
import os, sys, json, glob
import numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import schema, parts, mesh as M
REF = os.path.join(ROOT, "tests", "references")

def test_version_and_coverage():
    assert schema.SCHEMA_VERSION == "6.0"
    missing, dup = schema.cells_check(); assert missing == [] and dup == []
    for c, d in schema.CELLS.items(): assert len(d["primary"]) == 3, c

def test_presets_are_v6():
    for f in glob.glob(os.path.join(ROOT, "presets", "*.json")):
        d = json.load(open(f)); assert d.get("schema") == "6.0", f
        assert not (set(d["params"]) & schema.DEAD_KEYS), f
        Q, notes = schema.coerce_report(d["params"], base=schema.table_defaults())
        assert not any("unknown key" in n for n in notes), (f, notes)

def test_migrate_reports_dead_keys():
    Q, notes = schema.migrate({"tubercles": 0.5, "seed": 3, "length": 100})
    assert Q == {"length": 100} and len(notes) == 2

def test_named_presets_still_load():
    for n in schema.PRESETS: schema.preset(n)

def test_prongs_build_on_head_and_tail():
    P = schema.preset("textured"); P = schema.coerce(dict(P, headProngs=3, headProngLen=0.5, headProngSplay=40, tailProngs=2, tailProngLen=0.4))
    h = parts.cephalon(P, bevel_deg=45.0); t = parts.pygidium(P, bevel_deg=45.0)
    assert h.is_watertight and len(M.bodies(h)) == 1 and M.is_symmetric(h)
    assert t.is_watertight and len(M.bodies(t)) == 1
    h0 = parts.cephalon(schema.coerce(dict(P, headProngs=0)), bevel_deg=45.0)
    assert h.volume > h0.volume + 5 and h.vertices[:, 1].min() < h0.vertices[:, 1].min() - 5   # the trident reaches forward

FROZEN_READ = [n for n in ("proetida", "corynexochida", "harpetida") if os.path.exists(os.path.join(REF, n, "measure_v2.json"))]

@pytest.mark.parametrize("name", FROZEN_READ)
def test_whole_animal_reads_like_the_frozen_one(name):
    """The instrument on mesh-built parts must reproduce the frozen 2.1 reading (bevel, class, theta within 0.2 deg).
    Measured 11 Sep 2026: proetida 22.81/22.81, corynexochida 22.66/22.58, harpetida 30.39/30.31."""
    import instrument as I2
    P = schema.coerce(json.load(open(os.path.join(ROOT, "presets", f"{name}.json")))["params"])
    frozen = json.load(open(os.path.join(REF, name, "measure_v2.json"))); b = float(frozen["bevel_built_deg"])
    meshes = [parts.cephalon(P, bevel_deg=b)] + [parts.segment(P, i, bevel_deg=b) for i in range(int(P["segCount"]))] + [parts.pygidium(P, bevel_deg=b)]
    hb = parts.cephalon(dict(P, genalSpine=0.0), bevel_deg=b) if P["genalSpine"] * P["cephFrac"] * P["length"] > 0.5 else None
    r = I2.measure(P, meshes, bound_deg=b, budget_s=600, head_body=hb, bevel_built_deg=b)
    assert r["limited_by"] == frozen["limited_by"] and r["enroll_class"] == frozen["enroll_class"]
    assert abs(r["theta_joint_deg"] - frozen["theta_joint_deg"]) <= 0.2
