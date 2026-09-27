"""tests/test_sphere_eye.py: the spherical eye and the curved stalk (schema 6.2), head module only.

    python -m pytest tests/test_sphere_eye.py -q

A stalked eye is always a sphere on a curved stalk; eyeSphere=1 asks for a sphere on a sessile eye. The sessile
default is still the drum, so nothing frozen moves.
"""
import math, numpy as np, pytest
import schema, mesh as M
from anatomy import head as HEAD

R = 5.0

def test_sessile_default_is_still_the_drum():
    drum, _ = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.05, 0.3, 0.35)                # lensD R = 0.25 mm: no facets
    sph, _ = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.05, 0.3, 0.35, sphere=True)
    assert drum.is_watertight and sph.is_watertight
    assert abs(drum.volume - sph.volume) > 0.1 * sph.volume                             # a different shape, not a rename

def test_sphere_is_a_sphere():
    m, n = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.05, 0.3, 0.35, sphere=True)
    assert n == 0 and m.is_watertight and len(M.bodies(m)) == 1
    assert abs(m.volume - 4 / 3 * math.pi * R ** 3) < 0.03 * 4 / 3 * math.pi * R ** 3      # an icosphere, subdivisions 4
    c = m.center_mass; assert abs(c[0]) < 0.05 and abs(c[1]) < 0.05 and abs(c[2] - (0.45 * R - 1.0)) < 0.05
    assert m.bounds[0][2] < -1.0                                                          # sunk below the base: it overlaps the shell

def test_stalked_eye_is_a_sphere_on_a_curved_stalk():
    m, _ = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.05, 0.3, 0.35, stalk=3 * R, stalk_r=0.45, bend_deg=40)
    assert m.is_watertight and len(M.bodies(m)) == 1
    top = m.vertices[m.vertices[:, 2] > m.bounds[1][2] - 2 * R]                          # the sphere, roughly
    assert top[:, 0].mean() > 0.3 * R                                                     # carried outward by the bend
    root = m.vertices[m.vertices[:, 2] < m.bounds[0][2] + 0.5]
    assert abs(root[:, 0].mean()) < 0.3                                                   # leaves the root vertically
    pts, tang = HEAD.stalk_arc(3 * R, 40, -1 - 3 * R)
    assert abs(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() - 3 * R) < 0.02 * 3 * R   # the arc has the stalk's length
    assert abs(math.degrees(math.atan2(tang[0], tang[2])) - 40) < 1e-6                    # and ends tilted by the bend

def test_straight_stalk_when_bend_is_zero():
    m, _ = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.05, 0.3, 0.35, stalk=3 * R, bend_deg=0)
    assert m.is_watertight and abs(m.center_mass[0]) < 0.05

def test_lenses_sit_on_the_sphere_within_the_arc():
    m, n = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.16, 0.3, 0.35, sphere=True)
    assert n > 20
    plain, _ = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.05, 0.3, 0.35, sphere=True)
    assert m.volume > plain.volume                                                        # caps add material
    far = m.vertices[np.linalg.norm(m.vertices - plain.center_mass, axis=1) > R + 0.05]  # anything outside the sphere = a lens
    ang = np.degrees(np.arctan2(far[:, 1], far[:, 0]))
    assert np.abs(ang).max() < 110 / 2 + 8 and far[:, 0].min() > 0                       # all within +-arc/2 of outward

def test_lean_tips_the_stalked_eye_forward():
    a, _ = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.05, 0.3, 0.35, stalk=3 * R, bend_deg=30, lean=0)
    b, _ = HEAD.eye_solid(R, 1.7 * R, 15, 0.0, 110, 0.05, 0.3, 0.35, stalk=3 * R, bend_deg=30, lean=45)
    assert b.center_mass[1] < a.center_mass[1] - R                                        # forward = -y in the head frame

def test_head_ornaments_carry_the_stalked_sphere():
    P = schema.coerce(dict(segCount=6, eyeSize=0.14, eyeSolid=1, eyeStalk=2.0, eyeStalkBend=35, eyeLat=0.6))
    S = HEAD.plan(P); orn = dict((e[0], e) for e in HEAD.ornaments(P, S))
    assert "eyes" in orn and orn["eyes"][2] == "mirror"
    eye = orn["eyes"][1][0]; assert eye.is_watertight and len(M.bodies(eye)) == 1
    zc = float(S["zfun"](np.array([S["eye"]["xe"]]), np.array([S["eye"]["ye"]]))[0])
    assert eye.bounds[1][2] > zc + 2.0 * S["eye"]["eR"]                                   # it stands well above the cheek
    assert eye.bounds[0][2] < zc                                                          # and its root is in the shell

# ---- pelagic eyes (27 Sep 2026): the character is a setting of the sphere eye, not a new shape
def test_pelagic_eye_sits_in_the_flank_of_a_helmet_head():
    """After Carolinites: a helmet head curving down at the sides, the globe set into the flank with its top near the
    apex and its bottom below the margin, out past the head's width, at most 15 % into the glabella, one closed body."""
    P = schema.express(schema.coerce(dict(segCount=6)), "pelagicEyes")
    assert "pelagicEyes" in schema.characters(P)
    S = HEAD.plan(P); G = S["eye"]; h, _ = HEAD.shell(P, S, grid=(121, 61))
    eye = {e[0]: e for e in HEAD.ornaments(P, S)}["eyes"][1][0]
    assert eye.is_watertight and len(M.bodies(eye)) == 1
    ext = eye.bounds[1] - eye.bounds[0]; assert ext[2] > 1.1 * ext[0]                       # taller than wide
    assert eye.bounds[1][0] > G["head_halfwidth"] + 0.3 * G["eR"]                            # well out past the head's width
    assert G["xe"] - G["eR"] >= G["glab_half"] - 0.15 * G["eR"] - 1e-6                       # at most 15 % into the glabella
    zc = float(S["zfun"](np.array([G["xe"]]), np.array([G["ye"]]))[0]); apex = S["h"]
    assert 0.2 * apex < zc < 0.6 * apex                                                      # the flank: the head has curved down here
    assert 0.6 * apex < eye.bounds[1][2] < apex and eye.bounds[0][2] < -0.3 * apex          # top tucked under the crest, bottom hanging well below
    whole = M.union(h, eye, M.mirror_x(eye)); assert whole.is_watertight and len(M.bodies(whole)) == 1

def test_collar_is_zero_by_default_and_rings_the_eye():
    P0 = schema.coerce(dict(segCount=6, eyeSolid=1, eyeSphere=1, eyeSize=0.2, eyeLat=0.6)); P1 = dict(P0, eyeCollar=0.4)
    S0, S1 = HEAD.plan(P0), HEAD.plan(P1); G = S0["eye"]
    ring = np.array([[G["xe"] + G["eR"], G["ye"]], [G["xe"] - G["eR"], G["ye"]]])
    dz = S1["zfun"](ring[:, 0], ring[:, 1]) - S0["zfun"](ring[:, 0], ring[:, 1])
    assert np.all(dz > 0.9 * 0.4 * G["eR"])                                                   # full collar height on the rim
    far = np.array([[G["xe"] + 3 * G["eR"], G["ye"]]]); assert abs((S1["zfun"](far[:, 0], far[:, 1]) - S0["zfun"](far[:, 0], far[:, 1]))[0]) < 0.01
    assert np.array_equal(S0["zfun"](ring[:, 0], ring[:, 1]), HEAD.plan(dict(P0, eyeCollar=0.0))["zfun"](ring[:, 0], ring[:, 1]))

def test_small_sphere_eye_is_unchanged_by_the_sink_rule():
    """Below R = 7.5 mm the sink is 0.55 R, exactly what it was."""
    m, _ = HEAD.eye_solid(5.0, 8.5, 15, 0.0, 110, 0.05, 0.3, 0.35, sphere=True)
    assert abs(m.center_mass[2] - (0.45 * 5.0 - 1.0)) < 0.05

def test_eye_geometry_glabella_rule():
    """An eye that clears the glabella is where eyeLat puts it; one that overlaps it by more than 15 % of its radius
    is moved out to exactly that overlap, never further. (Every preset clears it: none moved.)"""
    P = schema.coerce(dict(segCount=6, eyeSize=0.12, eyeLat=0.80))
    G = HEAD.eye_geometry(P); assert abs(G["xe"] - 0.80 * G["head_halfwidth"]) < 1e-9
    P = schema.coerce(dict(segCount=6, eyeSize=0.20, eyeLat=0.40, glabInflate=1.7, eyePos=0.75))
    G = HEAD.eye_geometry(P); assert G["xe"] > 0.40 * G["head_halfwidth"]
    assert abs((G["xe"] - G["eR"]) - (G["glab_half"] - 0.15 * G["eR"])) < 1e-9

def test_vault_round_default_is_the_v4_profile_bit_for_bit():
    from anatomy.common import vault
    P = schema.coerce(dict(segCount=6)); u = np.linspace(-1, 1, 401)
    f = P["fulcrum"]; au = np.abs(u)
    legacy = np.where(au <= f, 1 - 0.35 * (au / f) ** 2, 0.65 * (1 - (np.maximum(au - f, 0) / (1 - f)) ** 1.6))
    assert np.array_equal(vault(u, dict(P, tent=0.0)), legacy)
    assert np.array_equal(vault(u, dict(P, tent=0.0, vaultRound=1.0)), legacy)
    r = vault(u, dict(P, tent=0.0, vaultRound=0.45))
    assert np.all(r[au > f] >= legacy[au > f]) and r[np.argmin(np.abs(au - 0.9))] > 1.5 * legacy[np.argmin(np.abs(au - 0.9))]   # fuller flank, same edge
