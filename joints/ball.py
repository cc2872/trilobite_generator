"""
joints/ball.py: a print-in-place BALL-AND-SOCKET joint, on the joints/base.py template. PRINT ONLY, never measured.

The retention BALL is the actual ball of itbefred's Articulated Giant Isopod (Thingiverse #4777921, CC-BY-NC-SA,
used with permission) — extracted from the source STL to joints/assets/isopod_ball.stl and grafted onto each part.
It is a proven, durable print-in-place hinge; we reuse its exact geometry rather than re-deriving it.

The joint is SELF-SIMILAR: every dimension is a fixed fraction of the segment pitch, at the isopod's own ratios
(ball diameter ~0.70 x pitch, the reason it prints strong). So the joint always fits and always looks the same
relative to the body, at any animal size — no per-animal scaling floors, no minimum pitch to snap under.

Form (joints/flexi's recessed shape): the FRONT of a part is a convex spherical lobe (radius Rc about the pivot)
holding a true-sphere socket + round mouth; the REAR of the part ahead carries the isopod ball on a round neck
reaching into it. The spherical socket lets the ball rotate freely; the ball (radius r > mouth) is captured; the
neck swings until it meets the mouth rim (stop_deg). BASE = "solid".

To match the isopod's absolute scale (a 13 mm ball prints durably), model_scale(P) reports 18/pitch — the factor
the download applies so the printed animal is isopod-sized. The on-screen model keeps the ratio; the viewer auto-fits.

Parameters (mostly derived from pitch; opts may override):
  jointZ      0.55              pivot height as a fraction of the ring top at the joint
  mouthFrac   0.80              socket mouth radius as a fraction of r (< 1 so the ball is captured)
  overhangs   True              restore the anatomy the rear trim removed (pleural spines, genal arms, ...)
"""
import math, os, numpy as np, trimesh
import mesh as M
from anatomy.common import pitch, ring_top, spine_solid
from anatomy import thorax as THORAX
from manifold3d import Manifold, OpType

NAME = "ball"; MEASURED = False; BASE = "solid"
DEFAULTS = dict(jointZ=0.55, mouthFrac=0.80, overhangs=True)
RESTORED_ORNAMENTS = ("genalArms", "occipitalSpine")   # supplied by overhang() instead (they reach over seg0)
MIN_PITCH = 3.0        # a sanity floor only; the joint is self-similar and fits at any pitch above it
ISO_PITCH = 18.0       # the isopod's own segment pitch (mm); the download scales the animal to this so the ball is ~13 mm
# self-similar joint ratios (fraction of pitch), taken from the isopod's proportions
BALL_FRAC = 0.70; NECK_FRAC = 0.28; LIP_FRAC = 0.16
GAP_AXIAL_FRAC = 0.045; GAP_VERT_FRAC = 0.020; GAP_LAT_FRAC = 0.015

# the actual isopod ball, extracted from the source STL, centred at the origin, scaled at graft time to bound-radius r
_ASSET = trimesh.load(os.path.join(os.path.dirname(__file__), "assets", "isopod_ball.stl"), force="mesh")
_ASSET.apply_translation(-_ASSET.centroid)
_ASSET_R = float(np.linalg.norm(_ASSET.vertices, axis=1).max())   # bounding radius: scale so the ball fits exactly in r

def clean(m):
    """Drop print-debris shells: any connected component under 5 mm^3 OR thinner than 0.5 mm on its shortest axis."""
    keep = [b for b in m.split(only_watertight=False) if b.volume >= 5.0 and b.extents.min() >= 0.5]
    return trimesh.util.concatenate(keep) if keep else m

def _ell(r, at):
    e = trimesh.creation.icosphere(subdivisions=3, radius=r); e.apply_translation(at); return M.to_manifold(e)
def _iso(r, at):
    """The isopod ball, scaled so its bounding radius is r and placed at `at`."""
    m = _ASSET.copy(); m.apply_scale(r / _ASSET_R); m.apply_translation(at); return M.to_manifold(m)
def _force(body): return M.to_manifold(M.from_manifold(body))   # materialise: guard the manifold3d lazy-CSG drop

def model_scale(P):
    """Factor the download applies so the built animal reaches the isopod's absolute scale (a ~13 mm ball)."""
    return round(ISO_PITCH / pitch(P), 4)

def fits(P):
    geometry(P); return True
def pivot(P):
    J, d, zj, y_piv = geometry(P); return dict(z=round(zj, 2), y_beyond_plane=round(y_piv - d, 2), scale=model_scale(P),
                                              ball_d=round(J["ballD"], 2), mouth_r=round(J["mouthFrac"] * 0.5 * J["ballD"], 2))
def stop_deg(P):
    """A ball joint has no enrolment stop; this is the free travel per joint, set by the neck meeting the mouth rim:
    asin((mouth_r - neckR) / r)."""
    J, d, zj, y_piv = geometry(P); r = 0.5 * J["ballD"]
    return round(math.degrees(math.asin(np.clip((J["mouthFrac"] * r - 0.5 * J["neckD"]) / r, 0.0, 1.0))), 1)

def geometry(P, J=None):
    """Resolve the joint. Every solid dimension is a fixed fraction of pitch (self-similar), so the joint fits at any
    size. Pivot is kept a wall below the shell top so the socket has a roof, and above the bed so the ball clears it."""
    J = dict(DEFAULTS, **(J or {})); d = pitch(P)
    if d < MIN_PITCH: raise ValueError(f"pitch {d:.2f} mm < {MIN_PITCH}: animal too small for a ball joint")
    J["ballD"] = BALL_FRAC * d; J["neckD"] = NECK_FRAC * d; J["lip"] = LIP_FRAC * d
    J["gap_axial"] = GAP_AXIAL_FRAC * d; J["gap_vertical"] = GAP_VERT_FRAC * d; J["gap_lateral"] = GAP_LAT_FRAC * d
    r = 0.5 * J["ballD"]
    S = THORAX.plan(P, 0); ztop = float(S["zfun"](np.array([0.0]), np.array([0.5 * d]))[0])
    lo = r + 0.5; hi = ztop - (r + J["gap_vertical"] + J["lip"]) - 0.3          # ball clears the bed; wall above the socket
    if hi < lo: raise ValueError(f"segment too short ({ztop:.1f} mm) for a {J['ballD']:.1f} mm ball joint")
    zj = float(np.clip(J["jointZ"] * ring_top(P), lo, hi))
    y_piv = d + 0.5 * J["gap_axial"] + J["lip"] + J["gap_axial"] + r            # ball centre = the next part's socket centre
    return J, d, zj, y_piv

def _rear_joint(body, P, J, zj, y_piv):
    """Concave SPHERE wrapping the next part's convex lobe, plus the isopod retention ball on a round neck."""
    ga = J["gap_axial"]; r = 0.5 * J["ballD"]; Rc = J["lip"] + ga + r
    body = _force(body)
    body = body - _ell(Rc + ga, (0, y_piv, zj))                                          # concave rear (dome)
    body = _force(body)
    y_rear = y_piv - (Rc + ga); neckR = 0.5 * J["neckD"]
    neck = M.to_manifold(M.cylinder(neckR, (y_piv - y_rear) + 2.0, axis="y", at=(0, 0.5 * (y_piv + y_rear) - 1.0, zj)))
    return body + neck + _iso(r, (0, y_piv, zj))                                          # <-- the actual isopod ball

def _front_joint(body, P, J, zj, yc, front_face_y):
    """Convex SPHERE lobe about the pivot with a spherical socket + round mouth (captures the previous part's ball).
    No enrolment stop: the front face below the lobe band is cut flat."""
    ga, gv, gl = J["gap_axial"], J["gap_vertical"], J["gap_lateral"]; r = 0.5 * J["ballD"]; Rc = J["lip"] + ga + r; big = 400.0
    body = _force(body)
    ztop = zj - Rc - ga
    front_zone = M.to_manifold(M.box(big, 2 * Rc + 2.0, big, at=(0, yc - Rc - 1.0, ztop), align=("c", "min", "min")))
    lobe = _ell(Rc, (0, yc, zj))
    behind = M.to_manifold(M.box(big, big, big, at=(0, yc, ztop), align=("c", "min", "min")))
    body = ((body - front_zone) + ((body ^ front_zone) ^ lobe)) + (body ^ behind)
    body = _force(body)
    body = body - M.to_manifold(M.box(big, (yc - front_face_y) + 1.0, ztop + 1.0, at=(0, front_face_y - 1.0, -1.0), align=("c", "min", "min")))
    body = _force(body)
    socket = _ell(r + gv, (0, yc, zj))
    mouthR = J["mouthFrac"] * r; L = yc + 2.0
    mouth0 = M.cylinder(mouthR, L, axis="y", at=(0, yc - 0.5 * L, zj))                    # cavity centre forward through the face
    th = math.radians(stop_deg(P) + 3.0); fan = [M.to_manifold(mouth0)]
    for f in np.linspace(-th, th, 9):
        fan.append(M.to_manifold(mouth0.copy().apply_transform(trimesh.transformations.rotation_matrix(f, (1, 0, 0), (0, yc, zj)))))
    return body - socket - Manifold.batch_boolean(fan, OpType.Add)

# ---------------------------------------------------------------- the template: cut / overhang / pose
def _trim_rear(body, P, port, J, d):
    """A segment's run compressed into [gap/2, pitch - gap/2] (as flexi). Returns (body, T), T the plan->part map."""
    ga = J["gap_axial"]; big = 400.0
    ovl = port.shingle; run = (d + max(ovl - 2.0, 1.0)) - ovl
    body = body ^ M.to_manifold(M.box(big, run, big, at=(0, ovl, -1), align=("c", "min", "min")))
    body = body.translate((0, -ovl, 0)).scale((1.0, (d - ga) / run, 1.0)).translate((0, 0.5 * ga, 0))
    T = np.diag([1.0, (d - ga) / run, 1.0, 1.0]); T[1, 3] = -ovl * (d - ga) / run + 0.5 * ga
    return _force(body), T

def cut(body, env, P, port, **opts):
    """One edge of one solid part."""
    J, d, zj, y_piv = geometry(P, opts); ga = J["gap_axial"]; r = 0.5 * J["ballD"]; big = 400.0
    man = body if isinstance(body, Manifold) else M.to_manifold(body); T = np.eye(4)
    if port.rear:
        if port.kind == "seg":
            man, T = _trim_rear(man, P, port, J, d)
            man = _rear_joint(man, P, J, zj, y_piv)
        else:                                                       # head rear: cut at the joint plane, ball beyond it
            man = man ^ M.to_manifold(M.box(big, big, big, at=(0, -0.5 * ga, 0), align=("c", "max", "c")))
            man = _rear_joint(man, P, J, zj, y_piv - d)
    else:
        yc = 0.5 * ga + J["lip"] + ga + r
        if port.kind == "tail":
            man = man ^ M.to_manifold(M.box(big, big, big, at=(0, 0.5 * ga, 0), align=("c", "min", "c")))
        man = _front_joint(man, P, J, zj, yc, front_face_y=0.5 * ga)
    return M.from_manifold(_force(man)), T

def overhang(P, part, S, port, opts=None):
    """What the rear trim removed, in the part frame, continuous with the rear face (same restore pass as flexi)."""
    J, d, zj, y_piv = geometry(P, opts); ga = J["gap_axial"]; big = 400.0
    if part == "tail": return None
    if part == "head":
        occ = None
        if P["occipitalSpine"] > 0.02:
            occ = M.to_manifold(spine_solid(0.6 * S["margin"], 0.5, P["occipitalSpine"] * S["Lc"],
                                (0, -0.07 * S["Lc"], ring_top(P) - 1.0), 0, pitch_deg=55))
        if S["Lg"] <= 0.5: return None if occ is None else M.from_manifold(occ)
        arm = M.to_manifold(M.heightfield_shell(S["arm_outline"], S["arm_z"], S["t"], nu=81, nv=13, symmetric=False))
        arm = arm + arm.mirror((1, 0, 0))
        shell = M.to_manifold(M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=121, nv=61))
        side = M.to_manifold(M.box(big, big, big, at=(S["a"] + 0.5, -0.5 * ga - 0.6, -1), align=("min", "min", "min")))
        out = arm + (shell ^ (side + side.mirror((1, 0, 0))))
        return M.from_manifold(out + occ if occ is not None else out)
    lap = 0.3; ovl = S["ovl"]; run = (d + max(ovl - 2.0, 1.0)) - ovl; y_cut = ovl + run
    env = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=121, nv=61, floor=0.0))
    x0 = S["a"] + 0.5
    sides = M.to_manifold(M.box(big, big, big, at=(x0, y_cut - lap, -1), align=("min", "min", "min")))
    sides = sides + sides.mirror((1, 0, 0))
    return M.from_manifold((env ^ sides).translate((0, (d - 0.5 * ga) - y_cut, 0)))

def _offsets(P):
    from joints.pin import joint_offsets; return joint_offsets(P)

def pose(P, angles_deg, J=None):
    """4x4 per part, joint k bent by angles_deg[k] about the ball centre. A ball joint poses in any direction; the
    chain pose bends about x (the enrolment axis) so the site/instrument have one representative motion."""
    J, d, zj, y_piv = geometry(P, J); offs = _offsets(P); yp = y_piv - d
    mats = [np.eye(4)]; T = np.eye(4)
    for k in range(len(offs)):
        T = T @ trimesh.transformations.translation_matrix((0, offs[k], 0)) @ trimesh.transformations.rotation_matrix(math.radians(-float(angles_deg[k])), (1, 0, 0), (0, yp, zj))
        mats.append(T)
    return mats

def transforms_deg(P, theta_deg, J=None): return pose(P, [theta_deg] * len(_offsets(P)), J)
