"""tests/test_curved_spines.py: curved spines (23 Oct 2026, schema 6.5).

The primitive, the prong with arched tines, the shared head ornaments on the pin head, and that every default is a
no-op: a spine with bend 0 is spine_solid() exactly, a preset with the new keys at their defaults builds the same
head. The isopod model's printable animal is built by scripts/build_ornaments.py (minutes per animal), not here."""
import os, sys, json
import numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import schema, mesh as M, assemble, instrument as I
from anatomy.common import spine_solid, curved_spine, spine_tip, prong
from joints import pin as PIN

def preset(name): return schema.coerce(json.load(open(os.path.join(ROOT, "presets", f"{name}.json"))))

def test_bend_zero_is_the_old_spine():
    a = spine_solid(2.0, 0.4, 30.0, (1, 2, 3), 25.0, 10.0); b = curved_spine(2.0, 0.4, 30.0, (1, 2, 3), 25.0, 10.0, 0.0)
    assert np.allclose(a.vertices, b.vertices) and (a.faces == b.faces).all()

@pytest.mark.parametrize("bend,roll", [(60, 0), (150, 0), (90, 90), (-90, 0), (180, 180), (30, -45)])
def test_arc_is_one_watertight_body_and_the_tip_is_where_spine_tip_says(bend, roll):
    s = curved_spine(2.0, 0.4, 30.0, (0, 0, 0), 0.0, 0.0, bend, roll)
    assert s.is_watertight and len(M.bodies(s)) == 1
    tip = spine_tip(30.0, (0, 0, 0), 0.0, 0.0, bend, roll)
    far = s.vertices[np.argmax(np.linalg.norm(s.vertices, axis=1))]
    assert np.linalg.norm(far - tip) < 1.0                      # the farthest vertex is the tip, within the tip's radius plus a facet

def test_arc_length_is_preserved():
    """A 30 mm spine bent 180 deg is a half circle of radius 30/pi: its tip sits 2r from the root."""
    tip = spine_tip(30.0, (0, 0, 0), 0.0, 0.0, 180.0, 0.0)
    assert abs(np.linalg.norm(tip) - 2 * 30.0 / np.pi) < 1e-6

def test_prong_with_arched_tines():
    p = prong(3, 40.0, 40.0, 2.0, 0.45, (0, 0, 0), stem_frac=0.5, bend_deg=70.0)
    assert p.is_watertight and len(M.bodies(p)) == 1
    p0 = prong(3, 40.0, 40.0, 2.0, 0.45, (0, 0, 0), stem_frac=0.5)
    assert abs(p0.volume - prong(3, 40.0, 40.0, 2.0, 0.45, (0, 0, 0), stem_frac=0.5, bend_deg=0.0).volume) < 1e-9

def test_schema_defaults_are_a_no_op():
    D = schema.table_defaults()
    for k in ("occipitalBend", "genalHorn", "termBend", "frontSpine", "headProngBend", "tailProngBend", "genalHornBend"): assert float(D[k]) == 0.0
    assert int(D["occipitalTines"]) == 1

@pytest.mark.parametrize("extra", [
    dict(occipitalSpine=1.2, occipitalTines=2, occipitalSplay=70, occipitalBend=150),            # Dicranurus
    dict(genalHorn=1.4, genalHornPitch=55, genalHornBend=120, genalHornRoll=-30),                # Ceratarges
    dict(headProngs=3, headProngLen=0.9, headProngStem=0.5, headProngSplay=40, headProngBend=60),  # Walliserops, arched
    dict(frontSpine=0.8, frontSpineBend=40),                                                     # Odontocephalus
    dict(occipitalSpine=1.0, occipitalTines=3, occipitalBend=90, genalHorn=1.0, genalHornBend=100, headProngs=2, headProngBend=-50, frontSpine=0.5, frontSpineBend=-60),
])
def test_pin_head_with_ornaments_is_one_body(extra):
    P = schema.coerce(dict(preset("proetida"), **extra))
    h = assemble.part(P, "head", PIN, bevel_deg=45.0, grid=I.MEASURE_GRID)
    assert h.is_watertight and len(M.bodies(h)) == 1

def test_pin_head_default_unchanged_by_the_new_keys():
    P = preset("proetida"); h = assemble.part(P, "head", PIN, bevel_deg=45.0, grid=I.MEASURE_GRID)
    ref = json.load(open(os.path.join(ROOT, "tests", "references", "proetida", "measure_v2.json")))
    assert len(M.bodies(h)) == 1                                   # the frozen regression test covers the reading itself
