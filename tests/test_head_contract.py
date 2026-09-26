"""tests/test_head_contract.py: what a head module (anatomy/head.py, or a replacement) must provide so every joint
fits it and the assembler can build it. The same shape of contract holds for anatomy/thorax.py and anatomy/tail.py
(checked at the end).

    python -m pytest tests/test_head_contract.py -q
"""
import numpy as np, pytest
import schema, mesh as M, assemble
from anatomy import head as HEAD, thorax as THORAX, tail as TAIL
from anatomy.port import Port

P = schema.coerce(dict(segCount=3, eyeSolid=1, genalSpine=0.5, occipitalSpine=0.3, headProngs=2))


def test_head_provides_the_contract():
    for name in ("plan", "shell", "solid", "ornaments", "ports", "cells"):
        assert callable(getattr(HEAD, name)), f"head lacks {name}"
    S = HEAD.plan(P)
    for k in ("outline", "zfun", "t", "Lc", "wh", "a", "margin"): assert k in S
    x, y = S["outline"](0.3, 0.5); assert np.isfinite([x, y]).all()
    z = S["zfun"](np.array([0.0, 5.0]), np.array([-10.0, -10.0])); assert np.isfinite(z).all() and (z > 0).all()


def test_head_ports_are_ports():
    S = HEAD.plan(P); ports = HEAD.ports(P, S)
    assert set(ports) == {"rear"}
    p = ports["rear"]; assert isinstance(p, Port) and p.rear and p.wide and p.kind == "head" and p.halfwidth > p.ring_half > 0


def test_head_plate_and_solid_are_closed():
    S = HEAD.plan(P)
    sh, env = HEAD.shell(P, S); so = HEAD.solid(P, S)
    for m in (sh, env, so): assert m.is_watertight and M.is_symmetric(m)
    assert so.volume > sh.volume                                       # the solid fills under the shell


def test_head_ornaments_are_named_closed_solids_in_build_order():
    S = HEAD.plan(P); orn = HEAD.ornaments(P, S)
    names = [o[0] for o in orn]
    assert names == ["genalArms", "eyes", "occipitalSpine", "headProngs"]
    for name, solids, how in orn:
        assert how in ("union", "mirror")
        for m in solids: assert m.is_watertight and m.volume > 0


def test_head_builds_with_every_joint_and_without_ornaments():
    for joint in ("pin", "flexi"):
        h = assemble.part(P, "head", joint)
        assert h.is_watertight and len(h.split(only_watertight=False)) == 1
    bare = assemble.part(P, "head", "pin", skip=("genalArms", "eyes", "occipitalSpine", "headProngs"))
    full = assemble.part(P, "head", "pin")
    assert full.volume > bare.volume


def test_head_body_drops_the_genal_spines():
    hb = assemble.head_body(P); h = assemble.part(P, "head", "pin")
    assert hb.is_watertight and hb.bounds[1][1] < h.bounds[1][1]      # the arms reached further back


def test_thorax_and_tail_follow_the_same_contract():
    for mod, args in ((THORAX, (P, 1)), (TAIL, (P,))):
        for name in ("plan", "shell", "solid", "ornaments", "ports", "cells"):
            assert callable(getattr(mod, name)), f"{mod.__name__} lacks {name}"
        S = mod.plan(*args); ports = mod.ports(P, S, 1) if mod is THORAX else mod.ports(P, S)
        for p in ports.values(): assert isinstance(p, Port)
        sh, env = mod.shell(P, S); so = mod.solid(P, S)
        assert sh.is_watertight and env.is_watertight and so.is_watertight
        for entry in mod.ornaments(P, S): assert entry[1].is_watertight
    assert set(THORAX.ports(P, THORAX.plan(P, 1), 1)) == {"front", "rear"}
    assert set(TAIL.ports(P, TAIL.plan(P))) == {"front"}
