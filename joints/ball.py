"""
joints/ball.py: a print-in-place BALL-AND-SOCKET joint, on the joints/base.py template. PRINT ONLY, never measured.

Mechanism inspired by the appendage joints of itbefred's Articulated Giant Isopod (Thingiverse #4777921,
CC-BY-NC-SA) — the ball-in-socket idea only (a generic hinge, not copyrightable); none of that model's geometry is
used. Every dimension here is our own, parametrized on the segment pitch.

It is joints/flexi's recessed form with every CYLINDER replaced by a SPHERE, so the joint bears and rotates on any
axis instead of one: the FRONT of a part is a convex spherical lobe (radius Rc about the pivot) with a spherical
socket + round mouth inside it; the REAR of the part ahead is the matching concave sphere with a ball on a round
neck reaching into that socket. Concentric spheres rotate freely about the pivot; the ball (radius r > mouth) is
captured; the round neck swings until it meets the mouth rim (stop_deg). There is NO enrolment stop — a ball joint
poses in any direction and does NOT reproduce the trilobite's rolling. BASE = "solid".

Parameters (all print-only, mm unless noted):
  jointZ      0.55              pivot height as a fraction of the ring top at the joint (abs-clamped 6-10 mm)
  ballD       6.0               retention ball diameter -> r = ballD/2
  neckD       3.0               round neck diameter (swings in the mouth) -- the load-bearing member; kept thick
  mouthFrac   0.80              socket mouth radius as a fraction of r (< 1 so the ball is captured)
  lip         2.0               wall between the socket and the concave face
  gap_axial / gap_vertical / gap_lateral   0.80 / 0.30 / 0.25   (axial: face gap & run end; vertical: ball/socket clearance)
  overhangs   True              restore the anatomy the rear trim removed (pleural spines, genal arms, ...)

Durability: short-pitch animals scale the solids down, but SCALE_FLOOR caps how far, and NECK_MIN/LIP_MIN put an
absolute floor under the two thin members so the printed joint never falls below what an FDM print can survive. An
animal too short-pitched to hold a durable joint raises ValueError rather than emit a joint that snaps.
"""
import math, numpy as np, trimesh
import mesh as M
from anatomy.common import pitch, ring_top, spine_solid
from anatomy import thorax as THORAX
from manifold3d import Manifold, OpType

NAME = "ball"; MEASURED = False; BASE = "solid"
DEFAULTS = dict(jointZ=0.55, ballD=6.0, neckD=3.0, mouthFrac=0.80, lip=2.0,
                gap_axial=0.80, gap_vertical=0.30, gap_lateral=0.25, overhangs=True)
RESTORED_ORNAMENTS = ("genalArms", "occipitalSpine")   # supplied by overhang() instead (they reach over seg0)
SCALE_FLOOR = 0.62     # never shrink the solids past this fraction of their default (below it the neck is too thin to print)
NECK_MIN = 1.8         # absolute floor (mm) on the load-bearing neck: thinner than this snaps on an FDM print
LIP_MIN = 1.0          # absolute floor (mm) on the socket wall
MIN_PITCH = 7.5        # below this the floored joint self-intersects -> ValueError (verified clean to ~7.7; use fewer segments, a longer animal, or flexi)

def clean(m):
    """Drop print-debris shells: any connected component under 5 mm^3 OR thinner than 0.5 mm on its shortest axis."""
    keep = [b for b in m.split(only_watertight=False) if b.volume >= 5.0 and b.extents.min() >= 0.5]
    return trimesh.util.concatenate(keep) if keep else m

def _ell(r, at):
    e = trimesh.creation.icosphere(subdivisions=3, radius=r); e.apply_translation(at); return M.to_manifold(e)
def _force(body): return M.to_manifold(M.from_manifold(body))   # materialise: guard the manifold3d lazy-CSG drop

def fits(P):
    geometry(P); return True
def pivot(P):
    J, d, zj, y_piv = geometry(P); return dict(z=round(zj, 2), y_beyond_plane=round(y_piv - d, 2), scale=J.get("scaled", 1.0),
                                              ball_d=round(J["ballD"], 2), mouth_r=round(J["mouthFrac"] * 0.5 * J["ballD"], 2))
def stop_deg(P):
    """A ball joint has no enrolment stop; this is the free travel per joint in any direction, set by the neck meeting
    the mouth rim: asin((mouth_r - neckR) / r)."""
    J, d, zj, y_piv = geometry(P); r = 0.5 * J["ballD"]
    return round(math.degrees(math.asin(np.clip((J["mouthFrac"] * r - 0.5 * J["neckD"]) / r, 0.0, 1.0))), 1)

def geometry(P, J=None):
    """Resolve the joint. Short-pitch animals scale the ball/neck/lip down (gaps do not); below MIN_PITCH -> ValueError.
    Pivot is an absolute bed height, clamped and kept a wall below the shell top so the socket has a roof."""
    J = dict(DEFAULTS, **(J or {})); d = pitch(P)
    if d < MIN_PITCH: raise ValueError(f"pitch {d:.2f} mm < {MIN_PITCH}: no ball joint fits (fewer segments or a longer animal)")
    need = 2 * J["lip"] + 3 * J["gap_axial"] + J["ballD"]
    if d < need + 1.0:
        k = max((d - 1.0 - 3 * J["gap_axial"]) / (need - 3 * J["gap_axial"]), SCALE_FLOOR)
        for key in ("ballD", "neckD", "lip"): J[key] = J[key] * k
        J["scaled"] = round(k, 3)
    J["neckD"] = max(J["neckD"], NECK_MIN); J["lip"] = max(J["lip"], LIP_MIN)   # absolute floors: never print a member too thin to survive
    zj = float(np.clip(J["jointZ"] * ring_top(P), 6.0, 10.0))
    S = THORAX.plan(P, 0); ztop = float(S["zfun"](np.array([0.0]), np.array([0.5 * d]))[0])
    zj = min(zj, ztop - (0.5 * J["ballD"] + J["gap_vertical"] + J["lip"]) - 0.3)   # keep a wall above the socket
    y_piv = d + 0.5 * J["gap_axial"] + J["lip"] + J["gap_axial"] + 0.5 * J["ballD"]   # ball centre = the next part's socket centre
    return J, d, zj, y_piv

def _rear_joint(body, P, J, zj, y_piv):
    """Concave SPHERE wrapping the next part's convex lobe, plus the retention ball on a round neck reaching into it."""
    ga = J["gap_axial"]; r = 0.5 * J["ballD"]; Rc = J["lip"] + ga + r
    body = _force(body)
    body = body - _ell(Rc + ga, (0, y_piv, zj))                                          # concave rear (dome)
    body = _force(body)
    y_rear = y_piv - (Rc + ga); neckR = 0.5 * J["neckD"]
    neck = M.to_manifold(M.cylinder(neckR, (y_piv - y_rear) + 2.0, axis="y", at=(0, 0.5 * (y_piv + y_rear) - 1.0, zj)))
    return body + neck + _ell(r, (0, y_piv, zj))

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
