"""printjoint2.py — the Flexi-#2 joint (v3), parametrized on our segment geometry. PRINT ONLY; the instrument keeps its pin.

Joint v3 (21 Sep 2026, Thingiverse #3839472 form, 0.4 mm faces): each segment is a SOLID wedge on a flat base (no
shell, no supports). One rounded joint per pair: the FRONT of a segment is a convex lobe (cylinder about the pivot
line) with a round through-bore; the REAR of the segment ahead is the matching concave face with a barrel on a round
neck reaching into that bore (_front_joint / _rear_joint). The stop is a back-leaning face below the lobe band, cut so
the faces meet at maxAngle. Pivot (x=0, y = lip+gap+r beyond the joint plane, z = jointZ*ring, clamped 6-10 mm) sits
behind the instrument's joint plane — pose print parts with transforms_deg here, not the instrument's hinge.

Anatomy (layered on top of the joint, reusing parts.py builders in the print frames — parts.py / references untouched):
  * pleural spines, pyg fork, marginal spines, head outline: already in the segment/tail/head plans (S).
  * grafted solids: genal spine as a lifted horn, occipital spine, head/tail prongs, dorsal axial spine, terminal
    spine. A tall axial spine is unioned AFTER the run-trim so it keeps its full sweep and simply LIMITS the curl —
    biology: a spiny trilobite does not fully enrol. clean() drops boolean debris/flakes before export.

Parameters (all print-only, mm unless noted):
  jointZ      0.55              pivot height as a fraction of the ring top at the joint (abs-clamped 6-10 mm)
  knobW/H     3.6 / 3.6         barrel length across the body / barrel diameter -> lobe/bore radius r = knobH/2
  neckH       1.2               round neck through the bore
  lip         1.2               wall between the bore and the concave face
  gap_axial / gap_vertical / gap_lateral   0.80 / 0.30 / 0.25
  baseChamfer 0                 elephant-foot handling is left to the slicer, not cut into the mesh
"""
import math, numpy as np, trimesh
import parts, mesh as M
from manifold3d import Manifold, OpType

DEFAULTS = dict(jointZ=0.55, knobW=3.6, knobH=3.6, neckH=1.2, lip=1.2,
                gap_axial=0.80, gap_vertical=0.30, gap_lateral=0.25, baseChamfer=0.0)


def clean(m):
    """Drop print-debris shells: any connected component under 5 mm^3 OR thinner than 0.5 mm on its shortest axis.
    Catches boolean ghosts and the 1-3 mm^3 border flakes a thin, low anterolateral head envelope sheds. Returns the
    concatenated real body/bodies; if nothing clears the bar, returns the input untouched."""
    keep = [b for b in m.split(only_watertight=False) if b.volume >= 5.0 and b.extents.min() >= 0.5]
    return trimesh.util.concatenate(keep) if keep else m


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
        for key in ("knobW", "knobH", "neckH", "lip"): J[key] = J[key] * k
        J["scaled"] = round(k, 3)
    zj = float(np.clip(J["jointZ"] * parts.ring_top(P), 6.0, 10.0))
    S = parts.segment_plan(P, 0); ztop = float(S["zfun"](np.array([0.0]), np.array([0.5 * d]))[0])
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


def print_segment(P, i, J=None, pocket_on_first=False):
    """Thingiverse #3839472 form (21 Sep): the FRONT of each segment is a rounded lobe (cylinder about the pivot line)
    with a round through-bore; the REAR of the segment ahead is the matching concave face with a barrel on a round neck
    reaching into that bore. Solid to the bed, no boxes."""
    J, d, zj, y_piv = geometry(P, J)
    S = parts.segment_plan(P, i); ga = J["gap_axial"]; big = 400.0; r = 0.5 * J["knobH"]
    yc = 0.5 * ga + J["lip"] + ga + r
    body = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=parts.GRID_SEG[0], nv=parts.GRID_SEG[1], floor=0.0))
    ovl = S["ovl"]; run = (d + max(ovl - 2.0, 1.0)) - ovl
    body = body ^ M.to_manifold(M.box(big, run, big, at=(0, ovl, -1), align=("c", "min", "min")))
    body = body.translate((0, -ovl, 0)).scale((1.0, (d - ga) / run, 1.0)).translate((0, 0.5 * ga, 0))
    body = M.to_manifold(M.from_manifold(body))
    body = _rear_joint(body, P, J, zj, y_piv)
    body = M.to_manifold(M.from_manifold(body))
    if i > 0 or pocket_on_first:
        body = _front_joint(body, P, J, zj, yc, y_prev_rear=-0.5 * ga, front_face_y=0.5 * ga)
    if P["axialSpine"] > 0.02:                                        # dorsal axial spine, unioned AFTER the run-trim/joints
        r_ax = 0.45 * S["margin"]                                     # so a tall back-swept spine keeps its full reach; it
        sp = M.to_manifold(parts.spine_solid(0.6 * r_ax + 0.6, 0.5, P["axialSpine"] * S["h"],   # sweeps over the next segment
                           (0, S["ovl"] + 0.45 * (S["d"] - S["ovl"]), S["h"] + S["rise"] - 1.0), 0, pitch_deg=60))  # = spine-limited enrolment (biology)
        body = body + sp.translate((0, -ovl, 0)).scale((1.0, (d - ga) / run, 1.0)).translate((0, 0.5 * ga, 0))
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
def print_head(P, J=None):
    """Cephalon solid on a flat base (eyes from the tracked eye builder), concave rear + barrel into seg0's lobe.
    Head frame: joint plane at y = 0, shell runs to -y; its pivot is lip + gap + r beyond the plane, like a segment's."""
    J, d, zj, y_piv = geometry(P, J); ga = J["gap_axial"]; r = 0.5 * J["knobH"]; big = 400.0
    S = parts.cephalon_plan(P)
    body = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=parts.GRID_SEG[0], nv=parts.GRID_SEG[1], floor=0.0))
    if P.get("eyeSolid", 0) > 0.5 and P["eyeSize"] > 0.01:
        G = S["eye"]; EP = parts.eye_params(P, G["eR"]); eye, _ = parts.eye_solid(**EP)
        zb = float(S["zfun"](np.array([G["xe"]]), np.array([G["ye"]]))[0]) - 0.3 + EP.get("stalk", 0.0)
        eR_ = eye.copy().apply_translation((G["xe"], G["ye"], zb)); body = body + M.to_manifold(eR_) + M.to_manifold(M.mirror_x(eR_))
    body = body ^ M.to_manifold(M.box(big, big, big, at=(0, -0.5 * ga, 0), align=("c", "max", "c")))
    body = _rear_joint(body, P, J, zj, y_piv - d)
    # ---- unioned head anatomy. Genal spine as a lifted tapered horn (rooted forward of the trim so it fuses, pitched
    #      up/out so it clears seg0 through the curl), occipital spine, anterior prongs. Outline features (pleural
    #      spines, head arc) are already in S.
    y_rear = -0.5 * ga
    if P["genalSpine"] * S["Lc"] > 0.5:
        y_att = y_rear - 3.0
        xw = float(S["xmax"](np.array([y_att]))[0]) - 1.0
        z_att = float(S["zfun"](np.array([xw]), np.array([y_att]))[0])
        for sx in (1, -1):
            body = body + M.to_manifold(parts.spine_solid(0.6 * S["margin"], 0.5, P["genalSpine"] * S["Lc"],
                                        (sx * xw, y_att, z_att), yaw_deg=sx * 22.0, pitch_deg=26.0))
    if P["occipitalSpine"] > 0.02:
        body = body + M.to_manifold(parts.spine_solid(0.6 * S["margin"], 0.5, P["occipitalSpine"] * S["Lc"],
                                    (0, -0.07 * S["Lc"], parts.ring_top(P) - 1.0), 0, pitch_deg=55))
    if int(P.get("headProngs", 0)) > 0:
        yf = float(S["outline"](0.0, 1.0)[1])                                    # front margin at the axis
        body = body + M.to_manifold(parts.prong(int(P["headProngs"]), P["headProngLen"] * S["Lc"], P["headProngSplay"],
                                    P.get("headProngWidth", 0.45) * S["margin"] + 0.6, 0.45, (0, yf + 1.5, 0.6 * S["margin"] + 0.5),
                                    yaw_deg=180.0, pitch_deg=8.0, stem_frac=P.get("headProngStem", 0.0),
                                    center_bias=P.get("headProngCenter", 1.0), curl_deg=P.get("headProngCurl", 0.0)))
    return M.from_manifold(body)


def print_tail(P, J=None):
    """Pygidium solid on a flat base with the front lobe + bore. Tail frame: joint plane at y = 0, shell runs to +y."""
    J, d, zj, y_piv = geometry(P, J); ga = J["gap_axial"]; r = 0.5 * J["knobH"]; big = 400.0
    S = parts.pygidium_plan(P); yc = 0.5 * ga + J["lip"] + ga + r
    body = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=parts.GRID_TAIL[0], nv=parts.GRID_TAIL[1], floor=0.0))
    body = body ^ M.to_manifold(M.box(big, big, big, at=(0, 0.5 * ga, 0), align=("c", "min", "c")))
    body = _front_joint(body, P, J, zj, yc, y_prev_rear=-0.5 * ga, front_face_y=0.5 * ga)
    # ---- unioned tail anatomy: terminal spine and posterior prongs at the tip (+y). Pyg fork + marginal spines are in S.
    if P["termSpine"] > 0.02:
        body = body + M.to_manifold(parts.spine_solid(0.5 * S["margin"], 0.6, P["termSpine"] * S["Lp"],
                                    (0, 0.85 * S["Lp"], 0.5 * S["margin"]), 0))
    if int(P.get("tailProngs", 0)) > 0:
        body = body + M.to_manifold(parts.prong(int(P["tailProngs"]), P["tailProngLen"] * S["Lp"], P["tailProngSplay"],
                                    P.get("tailProngWidth", 0.45) * S["margin"] + 0.6, 0.45, (0, 0.90 * S["Lp"], 0.5 * S["margin"]),
                                    yaw_deg=0.0, pitch_deg=5.0, stem_frac=P.get("tailProngStem", 0.0),
                                    center_bias=P.get("tailProngCenter", 1.0), curl_deg=P.get("tailProngCurl", 0.0)))
    return M.from_manifold(body)


def print_animal(P, J=None):
    """All parts in their own frames (head, seg0..segN-1, tail), so instrument.transforms_deg(P, 0) assembles them."""
    n = int(P["segCount"])
    return [clean(print_head(P, J))] + [clean(print_segment(P, i, J, pocket_on_first=True)) for i in range(n)] + [clean(print_tail(P, J))]


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
