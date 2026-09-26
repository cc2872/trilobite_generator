"""
joints/flexi.py: the print-in-place joint (v3, 21 Sep 2026), on the joints/base.py template. PRINT ONLY, never
measured; the instrument keeps joints/pin.

Thingiverse #3839472 form, 0.4 mm faces: each part is a SOLID wedge on a flat base (BASE = "solid"). One rounded
joint per pair: the FRONT of a part is a convex lobe (cylinder about the pivot line) with a round through-bore; the
REAR of the part ahead is the matching concave face with a barrel on a round neck reaching into that bore. The
stop is a back-leaning face below the lobe band, cut so the faces meet at maxAngle. Pivot (x = 0, y = lip+gap+r
beyond the joint plane, z = jointZ*ring, clamped 6-10 mm) sits behind the pin's joint plane: pose print parts with
this module's pose(), not the pin's.

What this joint does to a part (cut): the rear cut trims the plan to one pitch (the plate between y = ovl and the
rear edge is compressed into [gap/2, pitch - gap/2]) and adds the concave face + barrel; the front cut adds the
lobe, bore, slot fan and stop. Everything the trim removed that reaches back over later parts (pleurae beyond the
pitch, the head's genal arms and rear flap, the occipital spine) is returned by overhang() and put back by
assemble.restore(), cleared against the parts behind through the curl.

Parameters (all print-only, mm unless noted):
  jointZ      0.55              pivot height as a fraction of the ring top at the joint (abs-clamped 6-10 mm)
  knobW/H     3.6 / 3.6         barrel length across the body / barrel diameter -> lobe/bore radius r = knobH/2
  neckH       1.2               round neck through the bore
  lip         1.2               wall between the bore and the concave face
  gap_axial / gap_vertical / gap_lateral   0.80 / 0.30 / 0.25
  overhangs   True              restore the trimmed anatomy (False: the old stand-in genal horn, no pleural spines)
"""
import math, numpy as np, trimesh
import mesh as M
from anatomy.common import pitch, ring_top, spine_solid
from anatomy import thorax as THORAX, head as HEAD
from manifold3d import Manifold, OpType

NAME = "flexi"; MEASURED = False; BASE = "solid"
DEFAULTS = dict(jointZ=0.55, knobW=3.6, knobH=3.6, neckH=1.2, lip=1.2,
                gap_axial=0.80, gap_vertical=0.30, gap_lateral=0.25, baseChamfer=0.0, overhangs=True)
RESTORED_ORNAMENTS = ("genalArms", "occipitalSpine")   # the head ornaments overhang() supplies instead (they reach over seg0)
MIN_PITCH = 4.5          # below this no joint fits; the export should refuse rather than emit a fused chain

def clean(m):
    """Drop print-debris shells: any connected component under 5 mm^3 OR thinner than 0.5 mm on its shortest axis.
    Catches boolean ghosts and the 1-3 mm^3 border flakes a thin, low anterolateral head envelope sheds. Returns the
    concatenated real body/bodies; if nothing clears the bar, returns the input untouched."""
    keep = [b for b in m.split(only_watertight=False) if b.volume >= 5.0 and b.extents.min() >= 0.5]
    return trimesh.util.concatenate(keep) if keep else m


def fits(P):
    """Raises if no flexi joint fits (pitch under MIN_PITCH)."""
    geometry(P); return True
def pivot(P):
    J, d, zj, y_piv = geometry(P); return dict(z=round(zj, 2), y_beyond_plane=round(y_piv - d, 2), scale=J.get("scaled", 1.0),
                                              gaps={k: J[k] for k in ("gap_axial", "gap_vertical", "gap_lateral")})
def stop_deg(P): return float(P["maxAngle"])

def geometry(P, J=None):
    """Resolve the joint for this animal. Two smoothing rules (20 Sep 2026, from the slider study):
      * the joint needs 2*lip + 3*gap_axial + knobH of pitch; when the pitch is shorter than that plus 1 mm the
        barrel, neck and lip scale down together (gaps do not), so short-pitch animals get a smaller joint
        instead of a broken one; below MIN_PITCH -> ValueError
      * the pivot is an absolute height above the bed, clamped to [6, 10] mm, not a fraction of relief: a flat
        animal keeps a lip's worth of material above the pocket, a tall one does not lose its whole base to the V"""
    J = dict(DEFAULTS, **(J or {})); d = pitch(P)
    if d < MIN_PITCH: raise ValueError(f"pitch {d:.2f} mm < {MIN_PITCH}: no print joint fits (fewer segments or a longer animal)")
    need = 2 * J["lip"] + 3 * J["gap_axial"] + J["knobH"]
    if d < need + 1.0:
        k = max((d - 1.0 - 3 * J["gap_axial"]) / (need - 3 * J["gap_axial"]), 0.45)
        for key in ("knobW", "knobH", "neckH", "lip"): J[key] = J[key] * k
        J["scaled"] = round(k, 3)
    zj = float(np.clip(J["jointZ"] * ring_top(P), 6.0, 10.0))
    S = THORAX.plan(P, 0); ztop = float(S["zfun"](np.array([0.0]), np.array([0.5 * d]))[0])
    zj = min(zj, ztop - (0.5 * J["knobH"] + J["gap_vertical"] + J["lip"]) - 0.3)   # keep a lip above the pocket
    y_piv = d + 0.5 * J["gap_axial"] + J["lip"] + J["gap_axial"] + 0.5 * J["knobH"]   # barrel centre = the next segment's bore centre
                                                                                        # (its front face sits half a gap past the joint plane)
    return J, d, zj, y_piv

def _rear_joint(body, P, J, zj, y_piv):
    """Concave rear face wrapping the next segment's lobe, plus the barrel on a round neck reaching into it."""
    ga = J["gap_axial"]; r = 0.5 * J["knobH"]; Rc = J["lip"] + ga + r; big = 400.0
    body = M.to_manifold(M.from_manifold(body))                                          # force: guard the lazy-CSG drop
    body = body - M.to_manifold(M.cylinder(Rc + ga, big, axis="x", at=(0, y_piv, zj)))
    body = M.to_manifold(M.from_manifold(body))                                          # force (lazy-CSG drop otherwise)
    y_rear = y_piv - (Rc + ga); neckR = 0.5 * J["neckH"]
    neck = M.to_manifold(M.cylinder(neckR, (y_piv - y_rear) + 2.0, axis="y", at=(0, 0.5 * (y_piv + y_rear) - 1.0, zj)))
    knob = M.to_manifold(M.cylinder(r, J["knobW"], axis="x", at=(0, y_piv, zj)))
    return body + neck + knob

def _front_joint(body, P, J, zj, yc, y_prev_rear, front_face_y):
    """Convex lobe about the pivot (yc, zj) with a round through-bore and a rounded one-sided slot; below the lobe
    band the front face is cut so the stop lands at maxAngle for a rotation about (y_prev_rear + ..)."""
    ga, gv, gl = J["gap_axial"], J["gap_vertical"], J["gap_lateral"]; r = 0.5 * J["knobH"]; Rc = J["lip"] + ga + r; big = 400.0
    body = M.to_manifold(M.from_manifold(body))                                          # force: guard the lazy-CSG drop below the lobe band
    ztop = zj - Rc - ga
    front_zone = M.to_manifold(M.box(big, 2 * Rc + 2.0, big, at=(0, yc - Rc - 1.0, ztop), align=("c", "min", "min")))
    lobe = M.to_manifold(M.cylinder(Rc, big, axis="x", at=(0, yc, zj)))
    behind = M.to_manifold(M.box(big, big, big, at=(0, yc, ztop), align=("c", "min", "min")))
    body = ((body - front_zone) + ((body ^ front_zone) ^ lobe)) + (body ^ behind)
    body = M.to_manifold(M.from_manifold(body))
    # the stop: front face below the lobe band, a plane through the pivot line leaning back by alpha
    th = math.radians(P["maxAngle"]); delta = yc - y_prev_rear                         # pivot -> previous part's rear face
    t_alpha = (math.sin(th) - delta / zj) / math.cos(th)
    if t_alpha > 0.005:
        alpha = math.atan(t_alpha); y_e = yc + (zj - ztop) * t_alpha
        hs = M.box(big, big, big, at=(0, y_e, ztop), align=("c", "max", "c"))
        hs.apply_transform(trimesh.transformations.rotation_matrix(alpha, (1, 0, 0), (0, y_e, ztop)))
        body = body - (M.to_manifold(hs) ^ M.to_manifold(M.box(big, big, big, at=(0, 0, ztop), align=("c", "c", "max"))))
    else:
        body = body - M.to_manifold(M.box(big, (yc - front_face_y) + 1.0, ztop + 1.0, at=(0, front_face_y - 1.0, -1.0), align=("c", "min", "min")))
    body = M.to_manifold(M.from_manifold(body))
    bore = M.to_manifold(M.cylinder(r + gv, big, axis="x", at=(0, yc, zj)))
    neckR = 0.5 * J["neckH"]
    slot0 = M.cylinder(neckR + max(gl, gv), Rc + 2.0, axis="y", at=(0, yc - 0.5 * (Rc + 2.0) - 0.5, zj))
    th2 = math.radians(P["maxAngle"] + 2.0); fan = [M.to_manifold(slot0)]
    for f in np.linspace(math.radians(-2.0), th2, 9):
        fan.append(M.to_manifold(slot0.copy().apply_transform(trimesh.transformations.rotation_matrix(f, (1, 0, 0), (0, yc, zj)))))
    return body - bore - Manifold.batch_boolean(fan, OpType.Add)

# ---------------------------------------------------------------- the template: cut / overhang / pose
def _trim_rear(body, P, port, J, d):
    """A segment's run: the plate from y = ovl (the shingle band) to the rear edge, compressed into
    [gap/2, pitch - gap/2]. Returns (body, T) with T the plan -> part transform, so ornaments follow."""
    ga = J["gap_axial"]; big = 400.0
    ovl = port.shingle; run = (d + max(ovl - 2.0, 1.0)) - ovl
    body = body ^ M.to_manifold(M.box(big, run, big, at=(0, ovl, -1), align=("c", "min", "min")))
    body = body.translate((0, -ovl, 0)).scale((1.0, (d - ga) / run, 1.0)).translate((0, 0.5 * ga, 0))
    T = np.diag([1.0, (d - ga) / run, 1.0, 1.0]); T[1, 3] = -ovl * (d - ga) / run + 0.5 * ga
    return M.to_manifold(M.from_manifold(body)), T

def cut(body, env, P, port, **opts):
    """One edge of one solid part. body: trimesh solid (plan frame) or the Manifold from the other edge's cut."""
    J, d, zj, y_piv = geometry(P, opts); ga = J["gap_axial"]; r = 0.5 * J["knobH"]; big = 400.0
    man = body if isinstance(body, Manifold) else M.to_manifold(body); T = np.eye(4)
    if port.rear:
        if port.kind == "seg":                                      # segment rear: trim the run, then the concave face + barrel
            man, T = _trim_rear(man, P, port, J, d)
            man = _rear_joint(man, P, J, zj, y_piv)
        else:                                                       # head rear: cut at the joint plane, barrel beyond it
            man = man ^ M.to_manifold(M.box(big, big, big, at=(0, -0.5 * ga, 0), align=("c", "max", "c")))
            man = _rear_joint(man, P, J, zj, y_piv - d)
    else:
        yc = 0.5 * ga + J["lip"] + ga + r
        if port.kind == "tail":
            man = man ^ M.to_manifold(M.box(big, big, big, at=(0, 0.5 * ga, 0), align=("c", "min", "c")))
        man = _front_joint(man, P, J, zj, yc, y_prev_rear=-0.5 * ga, front_face_y=0.5 * ga)
    return M.from_manifold(M.to_manifold(M.from_manifold(man))), T

def overhang(P, part, S, port, opts=None):
    """What the rear trim removed, in the part frame, continuous with the part's rear face. None for the tail."""
    J, d, zj, y_piv = geometry(P, opts); ga = J["gap_axial"]; big = 400.0
    if part == "tail": return None
    if part == "head":
        occ = None
        if P["occipitalSpine"] > 0.02:                              # the same solid the head's ornaments carry
            occ = M.to_manifold(spine_solid(0.6 * S["margin"], 0.5, P["occipitalSpine"] * S["Lc"],
                                (0, -0.07 * S["Lc"], ring_top(P) - 1.0), 0, pitch_deg=55))
        if S["Lg"] <= 0.5: return None if occ is None else M.from_manifold(occ)
        arm = M.to_manifold(M.heightfield_shell(S["arm_outline"], S["arm_z"], S["t"], nu=81, nv=13, symmetric=False))
        arm = arm + arm.mirror((1, 0, 0))
        # the arm hangs off the head's rear flap (the shingle over seg0), which the rear cut removes: restore the flap
        # outside the axial ring as the pin build's shell, so the arm has something to hang from.
        shell = M.to_manifold(M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=121, nv=61))
        side = M.to_manifold(M.box(big, big, big, at=(S["a"] + 0.5, -0.5 * ga - 0.6, -1), align=("min", "min", "min")))
        out = arm + (shell ^ (side + side.mirror((1, 0, 0))))
        return M.from_manifold(out + occ if occ is not None else out)
    # segment: plan material behind the run, outside the axial ring (the ring is the joint's), shifted to the rear face
    lap = 0.3; ovl = S["ovl"]; run = (d + max(ovl - 2.0, 1.0)) - ovl; y_cut = ovl + run
    env = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=121, nv=61, floor=0.0))
    x0 = S["a"] + 0.5
    sides = M.to_manifold(M.box(big, big, big, at=(x0, y_cut - lap, -1), align=("min", "min", "min")))
    sides = sides + sides.mirror((1, 0, 0))
    return M.from_manifold((env ^ sides).translate((0, (d - 0.5 * ga) - y_cut, 0)))

def _offsets(P):
    from joints.pin import joint_offsets; return joint_offsets(P)

def pose(P, angles_deg, J=None):
    """4x4 per part, joint k bent by angles_deg[k] about the PRINT pivot (barrel centre: lip + gap + r beyond each
    joint plane, zj above the bed). Same part order and offsets as the pin, different pivot."""
    J, d, zj, y_piv = geometry(P, J); offs = _offsets(P); yp = y_piv - d
    mats = [np.eye(4)]; T = np.eye(4)
    for k in range(len(offs)):
        T = T @ trimesh.transformations.translation_matrix((0, offs[k], 0)) @ trimesh.transformations.rotation_matrix(math.radians(-float(angles_deg[k])), (1, 0, 0), (0, yp, zj))
        mats.append(T)
    return mats

def transforms_deg(P, theta_deg, J=None): return pose(P, [theta_deg] * len(_offsets(P)), J)

def stand_in_genal_horn(P, S, J=None):
    """With overhangs off: the old lifted tapered horn instead of the real arm (rooted forward of the trim so it
    fuses, pitched up/out so it clears seg0 through the curl)."""
    J, d, zj, y_piv = geometry(P, J); ga = J["gap_axial"]; y_rear = -0.5 * ga; out = []
    if P["genalSpine"] * S["Lc"] > 0.5:
        y_att = y_rear - 3.0
        xw = float(S["xmax"](np.array([y_att]))[0]) - 1.0
        z_att = float(S["zfun"](np.array([xw]), np.array([y_att]))[0])
        for sx in (1, -1):
            out.append(spine_solid(0.6 * S["margin"], 0.5, P["genalSpine"] * S["Lc"], (sx * xw, y_att, z_att), yaw_deg=sx * 22.0, pitch_deg=26.0))
    return out
