"""printjoint2.py — the Flexi-#2 joint, parametrized on our segment geometry. PRINT ONLY; the instrument keeps its pin.

What Flexi #2 (Thingiverse 3839472, 8 bodies) does, and what this reproduces:
  * every segment is a SOLID wedge with a flat base on the bed (no hollow shell, no nesting) -> prints with no supports
  * one joint per pair: a small ellipsoid knob on a short neck at the rear midline of segment k, captured in a pocket
    INSIDE segment k+1, entered through a slot in k+1's front face. The pocket is only a few mm wide, so its roof is a
    short bridge and prints free at ~0.3 mm gaps
  * the two faces meeting at each joint are cut as a V (half of maxAngle each) on the side away from the pivot,
    so the stop is the faces touching at maxAngle
Pivot: (x=0, y = pitch + lip + gap_axial + knobL/2, z = jointZ * local ring height). Note the pivot sits ~2.5 mm behind
the instrument's joint plane (y = pitch): the print's flexion axis is not the measured one; state it, don't hide it.

Parameters (all print-only, mm unless noted):
  jointZ      0.55   pivot height as a fraction of the ring top at the joint
  knobW/H     3.0 / 2.6         barrel length across the body / barrel diameter (knobL unused)
  neckW/H     1.8 / 1.4         neck section (must be < knobW-2*gap_lateral and < knobH-2*gap_vertical for retention)
  lip         1.4               front wall of the pocket
  gap_axial / gap_vertical / gap_lateral   0.20 / 0.30 / 0.25
  baseChamfer 0                 elephant-foot handling is left to the slicer's compensation setting, not cut into
                                the mesh: the in-geometry inset trick left loose slivers at thin tips (spine tips).
"""
import math, numpy as np, trimesh
import parts, mesh as M
from manifold3d import Manifold, OpType

DEFAULTS = dict(jointZ=0.55, knobW=3.6, knobH=3.0, knobL=1.6, neckW=1.8, neckH=1.4, lip=1.4,
                gap_axial=0.20, gap_vertical=0.30, gap_lateral=0.25, baseChamfer=0.0)


def _ell(rx, ry, rz, at):
    e = trimesh.creation.icosphere(subdivisions=3, radius=1.0); e.apply_scale([rx, ry, rz]); e.apply_translation(at); return M.to_manifold(e)


MIN_PITCH = 4.5          # below this no joint fits; the export should refuse rather than emit a fused chain


def geometry(P, J=None):
    """Resolve the joint for this animal. Two smoothing rules (20 Sep 2026, from the slider study):
      * the joint needs 2*lip + 3*gap_axial + knobH of pitch; when the pitch is shorter than that plus 1 mm the
        barrel, neck and lip scale down together (gaps do not), so short-pitch animals get a smaller joint
        instead of a broken one; below MIN_PITCH -> ValueError
      * the pivot is an absolute height above the bed, clamped to [6, 10] mm, not a fraction of relief: a flat
        animal keeps a lip's worth of material above the pocket, a tall one does not lose its whole base to the V"""
    J = dict(DEFAULTS, **(J or {})); d = parts.pitch(P)
    if d < MIN_PITCH: raise ValueError(f"pitch {d:.2f} mm < {MIN_PITCH}: no print joint fits (fewer segments or a longer animal)")
    need = 2 * J["lip"] + 3 * J["gap_axial"] + J["knobH"]
    if d < need + 1.0:
        k = max((d - 1.0 - 3 * J["gap_axial"]) / (need - 3 * J["gap_axial"]), 0.45)
        for key in ("knobW", "knobH", "neckW", "neckH", "lip"): J[key] = J[key] * k
        J["scaled"] = round(k, 3)
    zj = float(np.clip(J["jointZ"] * parts.ring_top(P), 6.0, 10.0))
    S = parts.segment_plan(P, 0); ztop = float(S["zfun"](np.array([0.0]), np.array([0.5 * d]))[0])
    zj = min(zj, ztop - (0.5 * J["knobH"] + J["gap_vertical"] + J["lip"]) - 0.3)   # keep a lip above the pocket
    y_piv = d + J["lip"] + J["gap_axial"] + 0.5 * J["knobH"]           # barrel centre, inside the next segment
    return J, d, zj, y_piv


def print_segment(P, i, J=None, pocket_on_first=False):
    J, d, zj, y_piv = geometry(P, J)
    S = parts.segment_plan(P, i); ga, gv, gl = J["gap_axial"], J["gap_vertical"], J["gap_lateral"]
    big = 400.0
    # ---- solid wedge body: everything under the dorsal surface down to the bed, flap removed, half-gap each end
    body = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=parts.GRID_SEG[0], nv=parts.GRID_SEG[1], floor=0.0))
    ovl = S["ovl"]; run = (d + max(ovl - 2.0, 1.0)) - ovl             # the full-height run is 2 mm short of a pitch
    body = body ^ M.to_manifold(M.box(big, run, big, at=(0, ovl, -1), align=("c", "min", "min")))
    L = d                                                              # stretch the run to a full pitch (~ +20 % in y,
    body = body.translate((0, -ovl, 0)).scale((1.0, (d - ga) / run, 1.0))   # ring furrows spread a little), half-gap each end
    body = body.translate((0, 0.5 * ga, 0))
    # ---- V faces below the pivot height (ventral curl closes the bottoms): half of maxAngle on each face
    th = math.radians(0.5 * P["maxAngle"])
    for rear in (True, False):
        y0 = (L - 0.5 * ga) if rear else 0.5 * ga
        wedge = M.box(big, big, big, at=(0, y0, zj), align=("c", ("min" if rear else "max"), "max"))
        wedge.apply_transform(trimesh.transformations.rotation_matrix((-1 if rear else 1) * th, (1, 0, 0), (0, y0, zj)))
        body = body - M.to_manifold(wedge)
    # ---- bed chamfer removed: the inset trick left 1-3 mm^3 slivers at thin tips (genal spines), so the STL came
    #      back as extra loose shells. Elephant-foot compensation belongs in the slicer, not cut into the mesh.
    # ---- knob + neck at the rear midline, reaching into the next segment
    y_rear = L - 0.5 * ga
    neck = M.to_manifold(M.box(J["neckW"], (y_piv - y_rear) + 1.0, J["neckH"], at=(0, y_rear - 1.0, zj), align=("c", "min", "c")))
    # knob = a short BARREL across the body (axis x): its y-z section is a circle, so flexion about x never changes
    # its fore-aft extent (the sphere's cap wedged into the slot and the disc's corners both cost axial play)
    knob = M.to_manifold(M.cylinder(0.5 * J["knobH"], J["knobW"], axis="x", at=(0, y_piv, zj)))
    body = body + neck + knob
    # ---- pocket + slot in the FRONT of this segment, for the knob of the segment ahead (same offsets, y measured
    #      from this segment's front face at y = 0.5*ga): pocket centre at y = 0.5*ga - d + y_piv = lip + ga + knobL/2 + 0.5*ga
    if i > 0 or pocket_on_first:
        yc = 0.5 * ga + J["lip"] + ga + 0.5 * J["knobH"]
        # pocket is a BOX (flat inner lip face = real retention across the knob's shoulders; an ellipsoid pocket lets
        # the knob wedge forward on thin shoulders). Slot runs from the face to the pocket's front wall only.
        pocket = M.to_manifold(M.box(J["knobW"] + 2 * gl, J["knobH"] + 2 * ga, J["knobH"] + 2 * gv, at=(0, yc, zj)))
        y_lip_in = yc - (0.5 * J["knobH"] + ga)
        slot = M.to_manifold(M.box(J["neckW"] + 2 * gl, y_lip_in + 1.0 + 0.05, J["neckH"] + 2 * gv, at=(0, -1.0, zj), align=("c", "min", "c")))
        # the slot must let the neck swing through the flexion range: sweep it
        th2 = math.radians(P["maxAngle"] + 6.0); fan = [slot]
        for f in np.linspace(-th2, th2, 7):
            fan.append(M.to_manifold(M.from_manifold(slot).apply_transform(trimesh.transformations.rotation_matrix(f, (1, 0, 0), (0, yc, zj)))))
        body = body - pocket - Manifold.batch_boolean(fan, OpType.Add)
    return M.from_manifold(body)


def chain(P, J=None, deg=0.0, n=None):
    """All thoracic print segments posed at `deg` per joint about the print pivot."""
    J, d, zj, y_piv = geometry(P, J); n = n or int(P["segCount"])
    segs = [print_segment(P, i, J) for i in range(n)]
    out, T = [], np.eye(4)
    for s in segs:
        out.append(s.copy().apply_transform(T))
        T = T @ trimesh.transformations.translation_matrix((0, d, 0)) @ trimesh.transformations.rotation_matrix(math.radians(-deg), (1, 0, 0), (0, y_piv - d, zj))
    return out


# ---------------------------------------------------------------- whole animal: head + segments + tail
def _knob_and_pocket(P, J, d, zj, y_piv):
    """Shared helpers: knob+neck reaching back from a rear face at y_rear; pocket+slot entering a front face at y_front."""
    ga, gv, gl = J["gap_axial"], J["gap_vertical"], J["gap_lateral"]
    def knob_at(y_rear, y_c):
        neck = M.to_manifold(M.box(J["neckW"], (y_c - y_rear) + 1.0, J["neckH"], at=(0, y_rear - 1.0, zj), align=("c", "min", "c")))
        return neck + M.to_manifold(M.cylinder(0.5 * J["knobH"], J["knobW"], axis="x", at=(0, y_c, zj)))
    def pocket_at(y_front):
        yc = y_front + J["lip"] + ga + 0.5 * J["knobH"]
        pocket = M.to_manifold(M.box(J["knobW"] + 2 * gl, J["knobH"] + 2 * ga, J["knobH"] + 2 * gv, at=(0, yc, zj)))
        y_lip_in = yc - (0.5 * J["knobH"] + ga)
        slot0 = M.box(J["neckW"] + 2 * gl, (y_lip_in - y_front) + 1.0 + 0.05, J["neckH"] + 2 * gv, at=(0, y_front - 1.0, zj), align=("c", "min", "c"))
        th2 = math.radians(P["maxAngle"] + 6.0); fan = [M.to_manifold(slot0)]
        for f in np.linspace(-th2, th2, 7):
            fan.append(M.to_manifold(slot0.copy().apply_transform(trimesh.transformations.rotation_matrix(f, (1, 0, 0), (0, yc, zj)))))
        return pocket + Manifold.batch_boolean(fan, OpType.Add)
    return knob_at, pocket_at


def _vface(body, y0, zj, rear, half_deg):
    big = 400.0; th = math.radians(half_deg)
    wedge = M.box(big, big, big, at=(0, y0, zj), align=("c", ("min" if rear else "max"), "max"))
    wedge.apply_transform(trimesh.transformations.rotation_matrix((-1 if rear else 1) * th, (1, 0, 0), (0, y0, zj)))
    return body - M.to_manifold(wedge)


def print_head(P, J=None):
    """Cephalon as a solid on a flat base (eyes and prongs stay: they are above the joint), knob at the rear midline
    reaching into seg0. Head frame: hinge at y = 0, the shell runs to -y."""
    J, d, zj, y_piv = geometry(P, J); ga = J["gap_axial"]
    S = parts.cephalon_plan(P)
    body = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=parts.GRID_SEG[0], nv=parts.GRID_SEG[1], floor=0.0))
    if P.get("eyeSolid", 0) > 0.5 and P["eyeSize"] > 0.01:                      # eyes (and stalks) as solids from the tracked eye builder
        G = S["eye"]; EP = parts.eye_params(P, G["eR"])
        eye, _ = parts.eye_solid(**EP)
        zb = float(S["zfun"](np.array([G["xe"]]), np.array([G["ye"]]))[0]) - 0.3 + EP.get("stalk", 0.0)
        eR_ = eye.copy().apply_translation((G["xe"], G["ye"], zb)); eL_ = M.mirror_x(eR_)
        body = body + M.to_manifold(eR_) + M.to_manifold(eL_)
    y_rear = -0.5 * ga
    body = body ^ M.to_manifold(M.box(400, 400, 400, at=(0, y_rear, 0), align=("c", "max", "c")))   # half a gap short of the joint
    body = _vface(body, y_rear, zj, True, 0.5 * P["maxAngle"])
    knob_at, _ = _knob_and_pocket(P, J, d, zj, y_piv)
    body = body + knob_at(y_rear, y_piv - d)                                     # knob centre = lip + gap + r beyond the joint plane
    return M.from_manifold(body)


def print_tail(P, J=None):
    """Pygidium as a solid on a flat base with the pocket at its front. Tail frame: hinge at y = 0, shell runs to +y."""
    J, d, zj, y_piv = geometry(P, J); ga = J["gap_axial"]
    S = parts.pygidium_plan(P)
    body = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=parts.GRID_TAIL[0], nv=parts.GRID_TAIL[1], floor=0.0))
    y_front = 0.5 * ga
    body = body ^ M.to_manifold(M.box(400, 400, 400, at=(0, y_front, 0), align=("c", "min", "c")))
    body = _vface(body, y_front, zj, False, 0.5 * P["maxAngle"])
    _, pocket_at = _knob_and_pocket(P, J, d, zj, y_piv)
    body = body - pocket_at(y_front)
    return M.from_manifold(body)


def print_animal(P, J=None):
    """All parts in their own frames (head, seg0..segN-1, tail), so instrument.transforms_deg(P, 0) assembles them."""
    n = int(P["segCount"])
    return [print_head(P, J)] + [print_segment(P, i, J, pocket_on_first=True) for i in range(n)] + [print_tail(P, J)]


def transforms_deg(P, theta_deg, J=None):
    """4x4 per part for a uniform flexion about the PRINT pivot (barrel centre: lip + gap + r beyond each joint plane,
    zj above the bed). Same part order and offsets as instrument.transforms_deg, different pivot — the print
    geometry must be posed with this, not with the instrument's hinge line."""
    J, d, zj, y_piv = geometry(P, J); offs = parts.joint_offsets(P); yp = y_piv - d
    mats = [np.eye(4)]; T = np.eye(4)
    for k in range(len(offs)):
        T = T @ trimesh.transformations.translation_matrix((0, offs[k], 0)) @ trimesh.transformations.rotation_matrix(math.radians(-theta_deg), (1, 0, 0), (0, yp, zj))
        mats.append(T)
    return mats
