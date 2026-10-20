"""landmarks.py (20 Oct 2026): the TriloMorph protocol read off the head and tail plans, and the TPS it writes."""
import os, sys, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import schema, landmarks as LM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _presets():
    out = {"default": schema.coerce(schema.table_defaults())}
    for f in sorted(glob.glob(os.path.join(ROOT, "presets", "*.json"))):
        n = os.path.basename(f)[:-5]
        if not n.startswith("_"): out[n] = schema.coerce(json.load(open(f))["params"], base=schema.table_defaults())
    return out


def test_every_preset_gives_the_full_protocol():
    for name, P in _presets().items():
        c, p = LM.cephalon(P), LM.pygidium(P)
        assert set(c["lm"]) == set(LM.NAMES_C) and set(p["lm"]) == set(LM.NAMES_P), name
        assert all(np.isfinite(v).all() for v in list(c["lm"].values()) + list(p["lm"].values())), name
        assert set(c["curves"]) == set(LM.CURVES_C) and set(p["curves"]) == set(LM.CURVES_P), name
        # anterior is +y and the right half is +x: LM1 ahead of LM4, LM16/LM15 lateral of the axis, glabella inside the cheek
        assert c["lm"]["LM1"][1] > c["lm"]["LM4"][1] and c["lm"]["LM15"][0] > 0 and c["lm"]["LM5"][0] < c["lm"]["LM15"][0], name
        assert p["lm"]["LM1"][1] > p["lm"]["LM4"][1] and p["lm"]["LM7"][0] > p["lm"]["LM5"][0] > 0, name
        # the sagittal points sit on the axis; LM2 lies between LM1 and LM3
        for k in ("LM1", "LM2", "LM3", "LM4", "LM10", "LM13"): assert abs(c["lm"][k][0]) < 1e-9, (name, k)
        assert c["lm"]["LM1"][1] >= c["lm"]["LM2"][1] > c["lm"]["LM3"][1] > c["lm"]["LM4"][1], name


def test_blind_and_sighted_eyes():
    P = _presets()["proetida"]; c = LM.cephalon(P)
    assert not c["blind"] and c["lm"]["LM8"][1] > c["lm"]["LM9"][1] and c["lm"]["LM14"][0] > c["lm"]["LM8"][0]
    B = schema.coerce(dict(P, eyeSize=0.0)); cb = LM.cephalon(B)
    assert cb["blind"] and cb["lm"]["LM8"] == cb["lm"]["LM9"] == cb["lm"]["LM14"] == cb["lm"]["LM10"]   # collapsed, as TriloMorph's files do


def test_genal_spine_and_border_landmarks():
    P = _presets()["proetida"]
    c0 = LM.cephalon(schema.coerce(dict(P, genalSpine=0.0, borderWidth=0.0)))
    assert c0["lm"]["LM16"] == c0["lm"]["LM15"] and c0["lm"]["LM11"] == c0["lm"]["LM12"]        # no spine: tip = angle; no border: LM11 = LM12
    c1 = LM.cephalon(schema.coerce(dict(P, genalSpine=0.8, borderWidth=0.1)))
    L = c1["head_length"]
    assert 0.6 < np.hypot(*(np.array(c1["lm"]["LM16"]) - c1["lm"]["LM15"])) / L < 1.0 and c1["lm"]["LM11"] != c1["lm"]["LM12"]


def test_preglabellar_field_moves_lm2_and_default_matches_the_old_constants():
    P = _presets()["proetida"]; L = LM.cephalon(P)["head_length"]
    c10 = LM.cephalon(schema.coerce(dict(P, preglabField=0.10))); c30 = LM.cephalon(schema.coerce(dict(P, preglabField=0.30)))
    f10 = (c10["lm"]["LM1"][1] - c10["lm"]["LM2"][1]) / L; f30 = (c30["lm"]["LM1"][1] - c30["lm"]["LM2"][1]) / L
    assert 0.05 < f10 < 0.15 < 0.25 < f30 < 0.35
    # the head plan at the default is the 6.0 head: the glabella's relief ends where it always did (0.80-0.92 Lc fade, furrow to 0.90)
    from anatomy import head as H
    S = H.plan(schema.coerce(dict(P, preglabField=0.10))); Lc = S["Lc"]
    z_in, z_out = S["zfun"](np.array([0.0]), np.array([-0.85 * Lc]))[0], S["zfun"](np.array([0.0]), np.array([-0.95 * Lc]))[0]
    S3 = H.plan(schema.coerce(dict(P, preglabField=0.30)))
    z3_in, z3_mid = S3["zfun"](np.array([0.0]), np.array([-0.60 * Lc]))[0], S3["zfun"](np.array([0.0]), np.array([-0.80 * Lc]))[0]
    assert z_in > z_out and z3_in > z3_mid                                                    # a field in front of the glabella is lower than the glabella


def test_tps_is_readable_and_has_their_template():
    P = _presets()["phacopida"]
    txt = LM.tps_string([("abc", P)], "C")
    assert txt.startswith("LM=16") and "CURVES=4" in txt and txt.count("POINTS=") == 4 and "IMAGE=abc_C.png" in txt and "SCALE=1.0" in txt
    assert [int(m) for m in re.findall(r"POINTS=(\d+)", txt)] == list(LM.SEMI_C)
    body = [l for l in txt.splitlines()[1:17]]
    assert all(len(l.split()) == 2 for l in body)
    txtP = LM.tps_string([("abc", P)], "P")
    assert txtP.startswith("LM=7") and "CURVES=3" in txtP and "IMAGE=abc_P.png" in txtP
