"""Free spines (24 Oct 2026): a spine rooted anywhere stands on a plate -- its root is moved off joints, lips and
notches onto one surface of one part, thinned when the plate has no room for the preset's collar, and a snapped root
is a fixed point (resolving it again moves nothing), so the studio's dots sit where the build puts the spine."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "isopod_model"))
import schema, head_styles as HS
import rows as RW


def _P():
    P = schema.coerce(schema.defaults()); P, face = HS.apply(P, "plain"); return schema.coerce(P)


def _info(P, spines):
    R = RW.plans(P, dict(spines=spines)); return RW._free_info(R, P), R


def test_roots_land_on_one_plate():
    P = _P()
    info, R = _info(P, [dict(x=19.15, y=44.0, len=20.5), dict(x=16.0, y=32.0, len=20.5), dict(x=10.0, y=36.0, len=20.5), dict(x=0.0, y=36.0, len=20.5)])
    for f in info:
        assert f["cover"] >= 1.0, f                                      # every collar wholly on its plate
        assert 0 <= f["part"] < len(R["parts"])
        assert f["z"] is not None and f["z"] > 0.05
    assert info[3]["root"][0] == 0.0                                      # a median spine stays on the axis (moves fore-aft only)
    assert info[2]["moved"] > 1.0                                         # (10, 36) is on row 4's lip, under row 3's plate: moved back onto the open top


def test_snapped_root_is_a_fixed_point():
    P = _P()
    info, _ = _info(P, [dict(x=19.15, y=44.0, len=20.5), dict(x=16.0, y=32.0, len=20.5), dict(x=10.0, y=36.0, len=20.5)])
    again, _ = _info(P, [dict(x=f["root"][0], y=f["root"][1], len=20.5) for f in info])
    for f, g in zip(info, again):
        assert g["moved"] < 1e-6 and g["cover"] >= 1.0 and abs(g["thin"] - f["thin"]) < 1e-9, (f, g)


def test_thickness_fits_the_room():
    P = _P()
    info, _ = _info(P, [dict(x=16.0, y=32.0, len=20.5)])                    # a joint between two 8 mm rows: no room for a 4.4 mm collar
    assert info[0]["thin"] < 1.0
    for fac in RW.FREE_THIN: assert 0 < fac <= 1.0
    assert RW.free_spine_angles(45.0) == (30.0, 15.0)                      # leaves at >= 30 degrees
    assert RW.free_spine_angles(20.0) == (20.0, 0.0)
    p, b = RW.free_spine_angles(100.0); assert p == 35.0 and abs(p + b - 100.0) < 1e-9


def test_off_the_animal_is_refused_gracefully():
    P = _P()
    info, R = _info(P, [dict(x=300.0, y=36.0, len=20.5)])
    assert info[0]["cover"] < 1.0 or info[0]["z"] is None                 # nothing to stand on within reach: reported, not crashed


def test_prong_keys_validate_and_default():
    P = _P()
    info, R = _info(P, [dict(x=12.0, y=38.0, len=30.0, tines=3, splay=95.0, stem=0.46, curl=10.0), dict(x=0.0, y=-14.0, len=20.0)])
    assert R["free_spines"][0]["tines"] == 3 and R["free_spines"][1]["tines"] == 1       # a plain spine stays a plain spine
    assert R["free_spines"][1]["splay"] == 40.0 and R["free_spines"][1]["stem"] == 0.4     # defaults filled
    assert info[0]["cover"] >= 1.0
    import pytest
    with pytest.raises(ValueError): RW.validate_free_spines([dict(x=1, y=1, tines=9)])
