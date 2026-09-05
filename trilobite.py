"""
trilobite.py — parametric, physically-enrollable trilobite, calibrated to Weaver's reference STL.

WHAT THE REFERENCE STL TAUGHT US (measured with trimesh, see measure_stl()):
  * 6 thoracic segments, identical topology, tapering ~7 % in width per segment
  * segments SHINGLE: each overlaps the next by ~half a pitch (the articulating half-ring)
  * pleurae end at ~1/3 of the relief height in a vertical rim, then sweep back into spines
  * axial ring ~1/3 of the body width; cephalon rear half is parallel-sided, front rounded
  * pygidium = short shield + two long flat forked spines
  * the reference joints are snap ball-and-sockets on the midline; we keep our pin hinge
    (more robust to print) but adopt the shingling, which hides the hinge and closes the dorsal gap

COORDINATES (everywhere): X = across, Y = along the body (head at -Y), Z = up.
Each part's FRONT hinge (the head's REAR hinge) sits at its local y=0.
"""
from build123d import *
import math

SHOW = "animate"          # "segment" | "head" | "tail" | "animal" | "animate"
VIEWER_PORT = 3939

# =====================================================================
#  PARAMETERS — the knobs.  Defaults calibrated to trilobite.stl (66.1 x 35.3 x 8.7 units)
# =====================================================================
P = dict(
    # body
    length=130,        # mm, whole animal excluding tail spines
    width=65,          # mm, max body width (cephalon)
    relief=16,         # mm, dorsal height  (STL: H/W = 0.246)
    segCount=6,        # STL: 6
    cephFrac=0.33,     # STL: 22.2 / 66.1
    pygFrac=0.14,      # tail SHIELD only (spines added on top); STL shield ~9 / 66.1
    taper=0.93,        # width scale per segment; STL: 0.932
    overlap=0.5,       # rear flap = this fraction of a pitch (shingling); STL: ~0.5
    marginHeight=0.34, # pleural margin height / relief; STL: 2.9 / 8.6
    wall=2.0, axisFrac=0.33, axisRise=0.15, lipFrac=0.08,
    # spines
    pleuralSpine=0.45, # spine length / segment half-width; STL: ~5.5 / 12
    spineSweep=45,     # degrees back from transverse
    pygSpine=0.9,      # tail spine length / shield length; STL: ~9 / 9
    pygSplay=20,       # degrees outward
    # head
    cephParallel=0.45, # fraction of head length that is parallel-sided at the rear; STL: ~0.45
    glabInflate=1.0, eyeSize=0.09, eyePos=0.45,
    pygWidth=0.85,     # shield width / last segment width
    # hinge
    clearance=0.3, boreDia=1.95, barrelR=2.6, nKnuckles=3, maxAngle=18,   # 3 knuckles: two ears + a tongue, 3.7 mm each
)

# =====================================================================
#  HELPERS
# =====================================================================
def pitch(P):
    return P["length"] * (1 - P["cephFrac"] - P["pygFrac"]) / P["segCount"]

def ring_top(P):
    return 0.75 * 1.33 * P["relief"] * (1 + P["axisRise"])

def hinge_z(P):
    """Hinge axis: knuckle top sits (wall + clearance) below the ring crest so the flap clears it."""
    return ring_top(P) - P["barrelR"] - P["wall"] - P["clearance"]

def seg_halfwidth(P, i):
    return P["width"] / 2 * P["taper"] ** (i + 1)

def hinge_width(P):
    """Constant across all joints: sized to the narrowest axial ring so every hinge interleaves."""
    return 2 * P["axisFrac"] * seg_halfwidth(P, P["segCount"] - 1) - 2

def arch(half_w, height, margin=0.0):
    """Closed roof profile in the XZ plane: Bezier from the margin height up over the crest and down,
    then a vertical rim to the base. Crest lands exactly at `height`."""
    m = margin * height
    ctrl = (height - 0.25 * m) / 0.75
    roof = Bezier((-half_w, 0, m), (-0.6 * half_w, 0, ctrl), (0.6 * half_w, 0, ctrl), (half_w, 0, m))
    if m < 1e-6:
        return make_face([roof, Line((half_w, 0, 0), (-half_w, 0, 0))])
    return make_face([roof, Line((half_w, 0, m), (half_w, 0, 0)), Line((half_w, 0, 0), (-half_w, 0, 0)),
                      Line((-half_w, 0, 0), (-half_w, 0, m))])

def slab(P, half_w, y0, y1, inset, margin=None):
    """Roof + axial ring extruded from y0 to y1, both shrunk by `inset` (0 = outer skin)."""
    if half_w - inset <= 1 or y1 - y0 <= 0:
        return None
    margin = P["marginHeight"] if margin is None else margin
    h, a = P["relief"], P["axisFrac"] * (P["width"] / 2)
    body = extrude(arch(half_w - inset, h - inset, margin).moved(Location((0, y0, 0))), amount=y1 - y0, dir=(0, 1, 0))
    ring = extrude(arch(a - inset, h * (1 + P["axisRise"]) - inset).moved(Location((0, y0, 0))), amount=y1 - y0, dir=(0, 1, 0))
    return body + ring

def roof_z(x, half_w, height, margin=0.0):
    m = margin * height; ctrl = (height - 0.25 * m) / 0.75
    best, zb = 1e9, 0
    for i in range(201):
        t = i / 200
        bx = -half_w*(1-t)**3 + 3*(-0.6*half_w)*(1-t)**2*t + 3*(0.6*half_w)*(1-t)*t**2 + half_w*t**3
        bz = m*(1-t)**3 + 3*ctrl*(1-t)**2*t + 3*ctrl*(1-t)*t**2 + m*t**3
        if abs(bx - x) < best: best, zb = abs(bx - x), bz
    return zb

def ellipsoid(rx, ry, rz):
    return scale(Sphere(1.0), by=(rx, ry, rz))

def above_ground():
    return Box(400, 400, 200, align=(Align.CENTER, Align.CENTER, Align.MIN))

def add_hinge(part, envelope, P, y_axis, rear):
    """Pin hinge on the face at y=y_axis. rear=True: even-slot knuckles, own material at y<y_axis."""
    w = P["width"] / 2
    rB, c, nK = P["barrelR"], P["clearance"], P["nKnuckles"]
    zh, Wh = hinge_z(P), hinge_width(P)
    kw, x0 = Wh / nK, -Wh / 2

    def barrel(xc, length, r):
        return Cylinder(r, length).rotate(Axis.Y, 90).moved(Location((xc, y_axis, zh)))

    part -= barrel(0, Wh + 4, rB + c)
    webs = None
    for i in range(nK):
        if (i % 2 == 0) != rear:
            continue
        xc = x0 + (i + 0.5) * kw
        part += barrel(xc, kw - c, rB)
        web = Box(kw - c, rB + c + 1.0, 2 * rB + c + 3,
                  align=(Align.CENTER, Align.MAX if rear else Align.MIN, Align.CENTER)).moved(Location((xc, y_axis, zh)))
        webs = web if webs is None else webs + web
    part += webs & envelope
    part -= barrel(0, Wh + 10, P["boreDia"] / 2)
    # stop block: a ventral pillar under the hinge on the mating face; the bevel below turns it into
    # the stop, and because hinge width and height are the same on every part, blocks always meet.
    blk = Box(Wh + 2, 2 * P["wall"], zh - rB - c - 0.5,
              align=(Align.CENTER, Align.MAX if rear else Align.MIN, Align.MIN)).moved(Location((0, y_axis, 0)))
    part += blk
    phi, L = P["maxAngle"] / 2, 80
    if rear:
        wedge = Box(2 * w + 20, L, L, align=(Align.CENTER, Align.MIN, Align.MAX)).rotate(Axis.X, -phi)
    else:
        wedge = Box(2 * w + 20, L, L, align=(Align.CENTER, Align.MAX, Align.MAX)).rotate(Axis.X, phi)
    part -= wedge.moved(Location((0, y_axis, zh))) - barrel(0, Wh + 4, rB + c)
    return part

def spine(base_r, tip_r, length, at, yaw_deg, pitch_deg=0):
    """Tapered spine from `at`, pointing +Y (rear), yawed about Z (+ = toward +X) and pitched up."""
    s = Cone(base_r, tip_r, length, align=(Align.CENTER, Align.CENTER, Align.MIN)).rotate(Axis.X, -90)
    return s.rotate(Axis.X, pitch_deg).rotate(Axis.Z, -yaw_deg).moved(Location(at))

# =====================================================================
#  PARTS
# =====================================================================
def build_segment(P, i=0):
    """Thoracic segment i: front (stepped, under the previous flap) at y=0 .. rear flap to y=d+ovl."""
    t, c, d = P["wall"], P["clearance"], pitch(P)
    ovl = P["overlap"] * d
    w = seg_halfwidth(P, i)
    lipW = P["lipFrac"] * P["width"]
    outer = slab(P, w, 0, ovl, t + c) + slab(P, w, ovl, d + ovl, 0)          # stepped front, full body + flap
    inner = slab(P, w, 0, ovl, 2 * t + c) + slab(P, w, ovl - t, d + ovl, t)
    seg = outer - inner
    seg += Box(lipW, d, t, align=(Align.MAX, Align.MIN, Align.MIN)).moved(Location((w, 0, 0)))
    seg += Box(lipW, d, t, align=(Align.MIN, Align.MIN, Align.MIN)).moved(Location((-w, 0, 0)))
    if P["pleuralSpine"] > 0:
        m = P["marginHeight"] * P["relief"]
        Ls = P["pleuralSpine"] * w
        for s in (1, -1):
            seg += spine(0.5 * m, 0.6, Ls, (s * (w - 0.5 * m), d + 0.3 * ovl, 0.5 * m), s * P["spineSweep"])
    seg = add_hinge(seg, outer, P, d, rear=True)
    seg = add_hinge(seg, outer, P, 0, rear=False)
    return seg

def build_cephalon(P):
    """Head: parallel-sided rear, rounded domed front, glabella, eyes, rear flap over segment 0."""
    t, c, h = P["wall"], P["clearance"], P["relief"]
    w, a = P["width"] / 2, P["axisFrac"] * (P["width"] / 2)
    Lc = P["cephFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P)
    par = P["cephParallel"] * Lc
    hz = ring_top(P) * 1.15
    env = (Box(2 * w + 4, par + ovl + 1, 3 * hz, align=(Align.CENTER, Align.MAX, Align.MIN)).moved(Location((0, ovl + 1, 0)))
           + ellipsoid(w * 1.02, Lc - par, hz).moved(Location((0, -par, 0)))) & above_ground()
    outer = slab(P, w, -Lc, ovl, 0) & env
    inner = slab(P, w, -Lc, ovl, t) & (ellipsoid(w * 1.02 - t, Lc - par - t, hz - t).moved(Location((0, -par, 0)))
                                       + Box(2 * w - 2 * t, par + ovl + 1, 3 * hz, align=(Align.CENTER, Align.MAX, Align.MIN)).moved(Location((0, ovl + 1, 0))))
    head = outer - inner
    re = P["eyeSize"] * w
    xe = a * P["glabInflate"] + re + 1.5
    ye = -P["eyePos"] * Lc
    fy = 1.0 if -ye < par else math.sqrt(max(0.0, 1 - ((-ye - par) / (Lc - par)) ** 2))
    ze = roof_z(xe, w, h, P["marginHeight"]) * fy - 0.35 * re
    for s in (1, -1):
        head += Sphere(re).moved(Location((s * xe, ye, ze)))
    head = add_hinge(head, outer, P, 0, rear=True)
    return head

def build_pygidium(P):
    """Tail: stepped front under the last flap, domed shield, forked spines."""
    t, c, h = P["wall"], P["clearance"], P["relief"]
    w = seg_halfwidth(P, P["segCount"] - 1) * P["pygWidth"]
    Lp = P["pygFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P)
    hz = ring_top(P) * 1.15
    env = ellipsoid(w * 1.02, Lp, hz) & above_ground()
    outer = (slab(P, w, 0, ovl, t + c) + slab(P, w, ovl, Lp, 0)) & env
    inner = (slab(P, w, 0, ovl, 2 * t + c) + slab(P, w, ovl - t, Lp, t)) & ellipsoid(w * 1.02 - t, Lp - t, hz - t)
    tail = outer - inner
    if P["pygSpine"] > 0:
        Ls = P["pygSpine"] * Lp
        for s in (1, -1):
            tail += spine(2.4, 0.6, Ls, (s * 0.5 * w, Lp * 0.6, 2.4), s * P["pygSplay"])
    tail = add_hinge(tail, outer, P, 0, rear=False)
    return tail

# =====================================================================
#  ASSEMBLY, COLLISION CHECK, ANIMATION
# =====================================================================
def parts_list(P):
    return [build_cephalon(P)] + [build_segment(P, i) for i in range(P["segCount"])] + [build_pygidium(P)]

def joint_offsets(P):
    """Y distance from each part's front hinge to its rear hinge (head's rear hinge is its origin)."""
    d = pitch(P)
    return [0] + [d] * P["segCount"]

def assemble(P, enroll, parts=None):
    parts = parts or parts_list(P)
    zh = hinge_z(P)
    theta = enroll * P["maxAngle"]
    rot = Location((0, 0, zh)) * Location((0, 0, 0), (-theta, 0, 0)) * Location((0, 0, -zh))
    offs = joint_offsets(P)
    posed, loc = [], Location()
    for i, part in enumerate(parts):
        if i > 0:
            loc = loc * (Location((0, offs[i - 1], 0)) * rot)
        posed.append(part.moved(loc))
    return posed

def check_enrollment(P, samples=(0.0, 0.5, 0.95, 1.05, 1.3), parts=None):
    parts = parts or parts_list(P)
    print("enrollment check (overlap mm^3 per joint, head->tail):")
    for e in samples:
        posed = assemble(P, e, parts)
        vols = [((posed[i] & posed[i + 1]).volume if (posed[i] & posed[i + 1]) else 0.0) for i in range(len(posed) - 1)]
        print(f"  enroll={e:4.2f} ({e*P['maxAngle']:5.1f} deg): " + " ".join(f"{v:6.1f}" for v in vols)
              + f"  {'OK' if max(vols) < 1 else 'COLLIDES'}")

def build_chain(P, parts=None):
    parts = parts or parts_list(P)
    zh, offs = hinge_z(P), joint_offsets(P)
    names = ["head"] + [f"seg{i}" for i in range(P["segCount"])] + ["tail"]
    node = None
    for i in reversed(range(len(parts))):
        body = parts[i].moved(Location((0, 0, -zh))); body.label = "shell"
        kids = [body] + ([node] if node is not None else [])
        node = Compound(children=kids, label=names[i])
        if i > 0:
            node.location = Location((0, offs[i - 1], 0))
    return Compound(children=[node], label="trilobite").moved(Location((0, 0, zh))), names

def animate_enrollment(P, parts=None, seconds=2.5):
    from ocp_vscode import show, Animation
    chain, names = build_chain(P, parts)
    show(chain, render_joints=False)
    anim = Animation()
    times = [0, seconds / 2, seconds]
    for i in range(1, len(names)):
        anim.add_track("/trilobite/" + "/".join(names[:i + 1]), "rx", times, [0, -P["maxAngle"], 0])
    anim.animate(speed=1)

def measure_stl(path="trilobite.stl"):
    """Re-derive the calibration numbers from a reference STL (needs trimesh)."""
    import trimesh, numpy as np
    m = trimesh.load(path)
    parts = sorted([p for p in m.split(only_watertight=False) if len(p.faces) > 10], key=lambda p: -p.centroid[1])
    L = m.bounds[1][1] - m.bounds[0][1]; W = m.bounds[1][0] - m.bounds[0][0]; H = m.bounds[1][2] - m.bounds[0][2]
    segs = parts[1:-1]
    widths = [p.bounds[1][0] - p.bounds[0][0] for p in segs]
    print(f"L={L:.2f} W={W:.2f} H={H:.2f} H/W={H/W:.3f} | segments={len(segs)} | "
          f"cephFrac={(parts[0].bounds[1][1]-parts[0].bounds[0][1])/L:.3f} | "
          f"taper/segment={(widths[-1]/widths[0])**(1/(len(widths)-1)):.3f} | "
          f"pitch={-np.mean(np.diff([p.centroid[1] for p in segs])):.2f} vs segment length={np.mean([p.bounds[1][1]-p.bounds[0][1] for p in segs]):.2f}")

# =====================================================================
#  MAIN
# =====================================================================
if __name__ == "__main__":
    parts = parts_list(P)
    for name, p in zip(["head"] + [f"seg{i}" for i in range(P["segCount"])] + ["tail"], parts):
        print(f"{name:5s} solids: {len(p.solids())} valid: {p.is_valid} size: {tuple(round(v,1) for v in p.bounding_box().size)}")
    check_enrollment(P, parts=parts)

    export_stl(parts[0], "cephalon.stl"); export_stl(parts[-1], "pygidium.stl")
    for i in range(P["segCount"]):
        export_stl(parts[1 + i], f"segment{i}.stl")
    export_stl(Compound(assemble(P, 0.0, parts)), "animal_flat.stl")
    export_stl(Compound(assemble(P, 1.0, parts)), "animal_enrolled.stl")
    print("wrote cephalon.stl, pygidium.stl, segment0..%d.stl, animal_flat.stl, animal_enrolled.stl" % (P["segCount"] - 1))

    try:
        from ocp_vscode import show, set_port
        set_port(VIEWER_PORT)
        if SHOW == "segment":  show(parts[2])
        elif SHOW == "head":   show(parts[0])
        elif SHOW == "tail":   show(parts[-1])
        elif SHOW == "animal": show(*assemble(P, 1.0, parts))
        else:                  animate_enrollment(P, parts)
    except Exception as e:
        print("viewer not available:", e)
