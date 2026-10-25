"""Schema 6.4 tail controls (21 Oct 2026): tailOutlineExp, pygAxisLen, pygAxisFrac, pygBorderWidth.
Each default must reproduce the tail as built since v4; each must also actually move the tail."""
import math
import numpy as np
import schema, landmarks
from anatomy import tail as T


def _P(**kw):
    P = schema.table_defaults(); P.update(kw); return schema.coerce(P, quiet=True)


def test_defaults_are_the_v4_tail():
    P = _P(); S = T.plan(P)
    assert S["outline_exp"] == 2.0 and S["axis_len"] == 0.85
    assert math.isclose(S["a"], P["axisFrac"] * S["wp"])                 # pygAxisFrac 0 -> the thoracic axis fraction
    assert math.isclose(S["border_frac"], P["borderWidth"])              # pygBorderWidth < 0 -> the head's border
    Sb = T.plan(_P(pygSpine=0.0, pygMarginal=0))                          # bare margin (the default carries paired spines)
    for u in (0.0, 0.3, 0.7, 1.0):                                        # outline exponent 2 -> the half-ellipse
        x, y = Sb["outline"](u, 1.0); S = Sb
        assert abs((x / S["wp"]) ** 2 + (y / S["Lp"]) ** 2 - 1) < 0.02 or y < 1e-6


def test_explicit_values_equal_inherited_ones():
    P = _P(); A = landmarks.pygidium(P)["lm"]
    B = landmarks.pygidium(_P(pygAxisFrac=P["axisFrac"], pygBorderWidth=P["borderWidth"], pygAxisLen=0.85, tailOutlineExp=2.0))["lm"]
    for k in A: assert np.allclose(A[k], B[k]), k


def test_controls_move_the_tail():
    base = landmarks.pygidium(_P())["lm"]
    assert landmarks.pygidium(_P(pygAxisLen=0.4))["lm"]["LM2"][1] > base["LM2"][1] + 1      # shorter axis (y is negated)
    assert landmarks.pygidium(_P(pygAxisFrac=0.5))["lm"]["LM5"][0] > base["LM5"][0] + 1     # wider axis at the front
    assert landmarks.pygidium(_P(pygBorderWidth=0.3))["lm"]["LM3"][1] != base["LM3"][1]     # own border
    pt = T.plan(_P(tailOutlineExp=1.0)); el = T.plan(_P())
    xp, yp = pt["outline"](0.5, 1.0); xe, ye = el["outline"](0.5, 1.0)
    assert math.hypot(xp, yp) < math.hypot(xe, ye) - 0.5                                     # pointed margin sits inside the ellipse


def test_pointed_tail_builds_one_body():
    import mesh as M
    P = _P(tailOutlineExp=1.0, pygAxisLen=0.6, pygBorderWidth=0.15, pygAxisFrac=0.4)
    sh, env = T.shell(P, T.plan(P))
    assert len(M.bodies(sh)) == 1 and sh.is_watertight


def test_tail_lobe_width_and_flat_list():
    """25 Oct 2026: a tail spine list carries a per-lobe width; a list of flat lobes (blade 1, no rise) stays a flat margin."""
    import rows as RW
    P = _P(pygSpineWidth=0.55, pygMarginal=3)
    S = T.plan(P)
    assert all(abs(s["width"] - 1.0) < 1e-9 for s in S["spines"][:1]) and S["spines"][1]["width"] > 1.0   # the pair in its own units; marginals in the pair's
    lst = RW.validate_tail_spines([dict(phi=70, len=10, rise=0, roll=0, blade=1, width=1.5), dict(phi=85, len=9, rise=0, roll=0, blade=1)])
    assert lst[1]["width"] == 1.0
    Pt = dict(P, _tailSpines=lst); S2 = T.plan(Pt)
    assert not S2["tube"] and abs(S2["spines"][0]["hh"] / S2["spines"][1]["hh"] - 1.5) < 1e-6
