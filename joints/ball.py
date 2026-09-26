"""
joints/ball.py: a print-in-place BALL-AND-SOCKET joint, on the joints/base.py template. PRINT ONLY, never measured.

Inspired by the appendage joints of itbefred's Articulated Giant Isopod (Thingiverse #4777921, CC-BY-NC-SA) — the
MECHANISM only (a ball on a neck captured in a socket is a generic hinge, not copyrightable); none of that model's
geometry is used. Every dimension here is our own, parametrized on the segment pitch.

Unlike joints/flexi (a one-axis ventral hinge for enrolment), a ball joint poses in ANY direction and has no hard
enrolment stop — so it does NOT reproduce the trilobite's rolling; it makes a freely poseable articulated model.
BASE = "solid": each part is a solid wedge on a flat base.

Form: the REAR of a part carries a ball (radius r = ballD/2) on a round neck reaching back to the next part's socket
centre. The FRONT of the part ahead is a spherical socket (cavity r + gap_ball) opening through a round MOUTH
(radius mouthFrac*r < r) in its front face — so the ball, printed in place, is captured (mouth < ball) yet free to
swing until the neck meets the mouth rim (stop_deg). Pivot: the ball centre (x=0, y = lip+gap+r beyond the joint
plane, z = jointZ*ring, clamped 6-10 mm). Pose print parts with this module's pose(), not the pin's.

Parameters (all print-only, mm unless noted):
  jointZ      0.55              pivot height as a fraction of the ring top at the joint (abs-clamped 6-10 mm)
  ballD       3.8               ball diameter -> r = ballD/2
  neckD       1.6               round neck diameter
  mouthFrac   0.82              socket mouth radius as a fraction of r (< 1 so the ball is captured)
  gap_ball    0.35              spherical clearance between ball and socket (print-in-place)
  gap_axial   0.60              gap between the parts' flat faces (also the run-trim end gap)
  overhangs   True              restore the anatomy the rear trim removed (pleural spines, genal arms, ...)
"""
import math, numpy as np, trimesh
import mesh as M
from anatomy.common import pitch, ring_top, spine_solid
from anatomy import thorax as THORAX
from manifold3d import Manifold, OpType

NAME = "ball"; MEASURED = False; BASE = "solid"
DEFAULTS = dict(jointZ=0.55, ballD=3.8, neckD=1.6, mouthFrac=0.82, gap_ball=0.35, gap_axial=0.60, overhangs=True)
RESTORED_ORNAMENTS = ("genalArms", "occipitalSpine")   # supplied by overhang() instead (they reach over seg0)
MIN_PITCH = 4.5

def clean(m):
    """Drop print-debris shells: any connected component under 5 mm^3 OR thinner than 0.5 mm on its shortest axis."""
    keep = [b for b in m.split(only_watertight=False) if b.volume >= 5.0 and b.extents.min() >= 0.5]
    return trimesh.util.concatenate(keep) if keep else m

def _ball(r, at):
    e = trimesh.creation.icosphere(subdivisions=3, radius=r); e.apply_translation(at); return M.to_manifold(e)

def fits(P):
    geometry(P); return True
def pivot(P):
    J, d, zj, y_piv = geometry(P); return dict(z=round(zj, 2), y_beyond_plane=round(y_piv - d, 2), scale=J.get("scaled", 1.0),
                                              gap_ball=J["gap_ball"], mouth_r=round(J["mouthFrac"] * 0.5 * J["ballD"], 2))
def stop_deg(P):
    """The neck swings until it meets the mouth rim: asin((mouth_r - neckR) / r). A ball joint has no enrolment stop;
    this is the free travel per joint in any direction."""
    J, d, zj, y_piv = geometry(P); r = 0.5 * J["ballD"]
    return round(math.degrees(math.asin(np.clip((J["mouthFrac"] * r - 0.5 * J["neckD"]) / r, 0.0, 1.0))), 1)

def geometry(P, J=None):
    """Resolve the joint. Short-pitch animals scale the ball/neck down (gaps do not) so the joint stays printable
    instead of breaking; below MIN_PITCH -> ValueError. Pivot is an absolute bed height, clamped and kept a wall
    below the shell top so the socket has a roof."""
    J = dict(DEFAULTS, **(J or {})); d = pitch(P)
    if d < MIN_PITCH: raise ValueError(f"pitch {d:.2f} mm < {MIN_PITCH}: no ball joint fits (fewer segments or a longer animal)")
    lip = 1.0
    need = 2 * lip + 3 * J["gap_axial"] + J["ballD"]
    if d < need + 1.0:
        k = max((d - 1.0 - 3 * J["gap_axial"]) / (need - 3 * J["gap_axial"]), 0.45)
        for key in ("ballD", "neckD"): J[key] = J[key] * k
        J["scaled"] = round(k, 3)
    r = 0.5 * J["ballD"]
    zj = float(np.clip(J["jointZ"] * ring_top(P), 6.0, 10.0))
    S = THORAX.plan(P, 0); ztop = float(S["zfun"](np.array([0.0]), np.array([0.5 * d]))[0])
    zj = min(zj, ztop - (r + J["gap_ball"] + 1.0) - 0.3)               # keep a wall above the socket
    y_piv = d + 0.5 * J["gap_axial"] + lip + J["gap_ball"] + r          # ball centre = the next part's socket centre
    return J, d, zj, y_piv

def _ball_rear(body, P, J, zj, y_piv, y_rear):
    """Neck (axis y) from the rear face out to the ball, plus the ball at the pivot."""
    r = 0.5 * J["ballD"]; neckR = 0.5 * J["neckD"]
    body = M.to_manifold(M.from_manifold(body))                                          # force: guard the lazy-CSG drop
    neck = M.to_manifold(M.cylinder(neckR, (y_piv - y_rear) + 1.0, axis="y", at=(0, 0.5 * (y_rear - 1.0 + y_piv), zj)))
    return body + neck + _ball(r, (0, y_piv, zj))

def _socket_front(body, P, J, zj, yc):
    """Spherical cavity (r + gap) at the socket centre, opened to the front face by a round mouth (radius < r, so the
    printed-in-place ball is captured) swept through the swing so the neck is free."""
    r = 0.5 * J["ballD"]; gb = J["gap_ball"]; mouthR = J["mouthFrac"] * r; big = 400.0
    body = M.to_manifold(M.from_manifold(body))                                          # force: guard the lazy-CSG drop
    cavity = _ball(r + gb, (0, yc, zj))
    mouth0 = M.cylinder(mouthR, yc + big, axis="y", at=(0, yc, zj), align=("c", "max", "c"))   # from the cavity forward through the face
    th = math.radians(stop_deg(P) + 3.0); fan = [M.to_manifold(mouth0)]
    for f in np.linspace(-th, th, 9):
        fan.append(M.to_manifold(mouth0.copy().apply_transform(trimesh.transformations.rotation_matrix(f, (1, 0, 0), (0, yc, zj)))))
    return body - cavity - Manifold.batch_boolean(fan, Manifold.OpType.Add if hasattr(Manifold, "OpType") else 0) if False else _sub_fan(body, cavity, fan)

def _sub_fan(body, cavity, fan):
    from manifold3d import OpType
    return body - cavity - Manifold.batch_boolean(fan, OpType.Add)

# ---------------------------------------------------------------- the template: cut / overhang / pose
def _trim_rear(body, P, port, J, d):
    """A segment's run compressed into [gap/2, pitch - gap/2] (as flexi). Returns (body, T), T the plan->part map."""
    ga = J["gap_axial"]; big = 400.0
    ovl = port.shingle; run = (d + max(ovl - 2.0, 1.0)) - ovl
    body = body ^ M.to_manifold(M.box(big, run, big, at=(0, ovl, -1), align=("c", "min", "min")))
    body = body.translate((0, -ovl, 0)).scale((1.0, (d - ga) / run, 1.0)).translate((0, 0.5 * ga, 0))
    T = np.diag([1.0, (d - ga) / run, 1.0, 1.0]); T[1, 3] = -ovl * (d - ga) / run + 0.5 * ga
    return M.to_manifold(M.from_manifold(body)), T

def cut(body, env, P, port, **opts):
    """One edge of one solid part."""
    J, d, zj, y_piv = geometry(P, opts); ga = J["gap_axial"]; r = 0.5 * J["ballD"]; big = 400.0
    man = body if isinstance(body, Manifold) else M.to_manifold(body); T = np.eye(4)
    if port.rear:
        if port.kind == "seg":
            man, T = _trim_rear(man, P, port, J, d)
            man = _ball_rear(man, P, J, zj, y_piv, y_rear=d - 0.5 * ga)
        else:                                                       # head rear: cut at the joint plane, ball beyond it
            man = man ^ M.to_manifold(M.box(big, big, big, at=(0, -0.5 * ga, 0), align=("c", "max", "c")))
            man = _ball_rear(man, P, J, zj, y_piv - d, y_rear=-0.5 * ga)
    else:
        yc = 0.5 * ga + 1.0 + J["gap_ball"] + r                     # socket centre (lip=1.0 beyond the front face)
        if port.kind == "tail":
            man = man ^ M.to_manifold(M.box(big, big, big, at=(0, 0.5 * ga, 0), align=("c", "min", "c")))
        man = _socket_front(man, P, J, zj, yc)
    return M.from_manifold(M.to_manifold(M.from_manifold(man))), T

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
