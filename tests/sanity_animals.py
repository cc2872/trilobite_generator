"""tests/sanity_animals.py — animals the instrument must read a known way (PREREG §4 "sanity animals"; also a pytest).

    python -m pytest tests/sanity_animals.py -q      (~2 min: four whole-animal readings)

- two segments                      -> bound (the ruler runs out before a 3-joint chain can close or interfere)
- enormous lateral pleural spines   -> open, limited mid-thorax (spine on spine), not head-tail
- tail wider than the head          -> if it closes, the class is not sphaeroidal (margins cannot meet edge to edge)
- eyeSize moved on the same animal  -> identical theta (the null axis)"""
import os, sys, json
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import schema, instrument as I

def _base():
    P = schema.preset("textured"); P.update(segCount=6, length=120); return schema.coerce(P)

def test_two_segments_is_bound():
    r, _ = I.read(schema.coerce(dict(_base(), segCount=2)), bound=45.0)
    assert r["limited_by"] == "bound" and r["enroll_class"] == "censored"

def test_huge_lateral_spines_limit_mid_thorax():
    P = schema.coerce(dict(_base(), spineBase=1.4, spineGrad=0.0, spineSweep=60, tipTaper=0.3))
    r, _ = I.read(P, bound=45.0)
    assert r["limited_by"] in ("anatomy", "invalid")
    if r["limited_by"] == "anatomy":
        i, j, _ = r["stopped_by"][0]; assert 0 < i and j < int(P["segCount"]) + 1     # a seg-seg pair, not head or tail

def test_wide_tail_is_not_sphaeroidal():
    P = schema.coerce(dict(_base(), pygWidth=1.25, pygFrac=0.34, cephFrac=0.24))
    r, _ = I.read(P, bound=45.0)
    if r["limited_by"] == "closed": assert r["enroll_class"] != "sphaeroidal"

def test_null_axis_does_not_move_theta():
    a, _ = I.read(schema.coerce(dict(_base(), eyeSize=0.05)), bound=45.0)
    b, _ = I.read(schema.coerce(dict(_base(), eyeSize=0.35)), bound=45.0)
    assert a["limited_by"] == b["limited_by"] and (a["theta_joint_deg"] is None or abs(a["theta_joint_deg"] - b["theta_joint_deg"]) < 1e-6)
