"""10 Nov 2026: the hand-sized print. printLength 0 scales the whole animal, joint kit included, to HAND_MM on its longest
side; the kit never goes below KIT_MIN of the isopod's; a body's own print_scale or an explicit printLength keeps the old
rules. Fast checks only (no build); tests/test_hand_size.py::test_agnostid_builds_hand_sized is the slow one."""
import os, sys, json
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "isopod_model"), os.path.join(ROOT, "web")]
import schema


def _body():
    import body as BODY
    return BODY


def test_hand_ratio_targets_hand_size_with_a_joint_floor():
    B = _body()
    assert B.hand_ratio(320.0) == pytest.approx(B.HAND_MM / 320.0)            # a 320 mm animal prints at half size, joints at half
    assert B.hand_ratio(100.0) == 1.0                                          # a small animal is not enlarged
    assert B.hand_ratio(1000.0) == pytest.approx(B.KIT_MIN)                    # the joints stop at the floor (the animal stays bigger)
    assert B.hand_ratio(1000.0, fit18=0.5) == pytest.approx(2 * B.KIT_MIN)      # a short segment holds a smaller kit: the floor comes sooner
    assert B.hand_kit(0.72, 1.06, 0.6) == pytest.approx(0.6)                    # the joints keep their size while the segment holds them
    assert B.hand_kit(0.5, 1.06, 0.6) == pytest.approx(0.53)                    # then shrink with the segments
    assert B.hand_kit(0.2, 1.06, 0.6) == pytest.approx(B.KIT_MIN)               # never below the floor
    assert B.KIT_MIN == pytest.approx(1 / 3)                                    # the isopod printed whole at 1/3 still articulates


def test_rows_plans_take_the_kit_it_is_given():
    import rows as RW
    P = schema.coerce(schema.preset("textured"))
    a = RW.plans(P, {}, strict=False); b = RW.plans(P, {}, strict=False, kit=0.4)
    s18 = RW.default_scale(a["rows"])
    assert a["min_joint_mm"] == pytest.approx(RW.MIN_JOINT_MM * P["jointScale"]) and b["min_joint_mm"] == pytest.approx(RW.MIN_JOINT_MM * 0.4)
    # the hand-sized print is self-similar: a kit that fits at the isopod's pitch fits at any ratio (no new problems)
    assert RW.problems(a["rows"], s18, kit=1.0) == RW.problems(a["rows"], s18 * 0.4, kit=0.4)


def test_presets_are_hand_sized_or_held_by_the_joint_floor():
    """Each order preset's longest side at the isopod's pitch, as built 10 Nov 2026 (the uploaded exports), gives a print
    of at most HAND_MM unless the joint floor binds -- and then no bigger than 1/3 of the old print."""
    B = _body()
    E18 = dict(agnostida=223, asaphida=367, corynexochida=309, harpetida=510, lichida=596, phacopida=583, proetida=343, ptychopariida=439, redlichiida=520)
    for name, e in E18.items():
        r = B.hand_ratio(e); size = e * r                                        # fit18 1 (rows at the isopod's pitch)
        assert size <= B.HAND_MM + 1e-6 or r == pytest.approx(B.KIT_MIN), name
        assert size <= e / 3 + 1e-6 or size <= B.HAND_MM + 1e-6, name


def test_size_key_ignores_the_print_scale():
    B = _body(); P = schema.defaults()
    assert B._size_key(P, "shaped", {}, {}, {"rows": {"0": {"width": 20}}}) == B._size_key(P, "shaped", {}, {}, {"rows": {"0": {"width": 20}}, "print_scale": 0.7})


@pytest.mark.slow
def test_agnostid_builds_hand_sized():
    """The agnostid's drawing through the studio path (~30 s): longest side within SIZE_TOL of HAND_MM, every piece
    watertight, the kit at the ratio it was printed at, nothing touching at rest."""
    os.environ.setdefault("TRILO_RECORD", "0")
    import app as A, check as CHECK, trimesh
    R = A._drawing_resolve("agnostida", A._preset_record("agnostida")[1]["drawing"]); IB = A._isopod_body()
    parts, plans, PV, K = IB.build(R["P"], "shaped", face=R["head"]["face"], outline=R["head"]["outline"], rows=R["body"])
    s = K["s"]; whole = trimesh.util.concatenate([p.copy().apply_scale(1.0 / s) for p in parts])
    assert abs(max(whole.extents[:2]) / IB.HAND_MM - 1) <= IB.SIZE_TOL + 0.01
    assert IB.KIT_MIN <= K["kit"] <= 1.0 and IB.ROWS_INFO["hand"]
    assert all(p.is_watertight for p in parts)
    assert CHECK.contacts(parts, [(y, z) for y, z in PV], 0.0) == []


def test_cut_at_bed_removes_what_is_below_the_bed():
    """10 Nov 2026: a part reaching under z = 0 (a spine root, a tail spine rolled down) is cut flat at the bed."""
    import trimesh
    B = _body()
    a = trimesh.creation.box(extents=(10, 10, 6)); a.apply_translation((0, 0, 2))      # z from -1 to 5
    b = trimesh.creation.box(extents=(10, 10, 4)); b.apply_translation((20, 0, 2))     # z from 0 to 4: untouched
    parts = [a, b]; B.cut_at_bed(parts)
    assert parts[0].bounds[0][2] >= -1e-6 and abs(parts[0].volume - 500.0) < 1e-3 and parts[0].is_watertight
    assert parts[1] is b and set(B.LAST_BED_CUT) == {0} and abs(B.LAST_BED_CUT[0] - 100.0) < 1e-3
