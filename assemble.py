"""
assemble.py: the one place where parts meet joints (26 Sep 2026).

    part(P, name, joint, ...)      one part: plate -> joint at every port -> ornaments
    builders(P, joint, ...)        [(name, fn)] so a caller can build part by part with progress
    animal(P, joint, ...)          every part, then the overhang pass (restore) for joints that trim
    restore(P, parts, joint, ...)  put back what a trimming joint removed, cleared against the parts behind
    pose(P, joint, angles)         bend the chain

The anatomy modules (anatomy/head, thorax, tail) never see a joint; the joint modules (joints/pin, flexi) never
see a plan. This file is the only one that imports both. The instrument, the site, the fill and the blueprint call
this file, so swapping a head or a joint changes exactly one module and nothing here.

Build order is the 11 Sep order, kept so the pin meshes stay bit-identical to the frozen references:
    plate (shell + doublure)  ->  rear joint  ->  front joint  ->  ornaments in the anatomy's order.
"""
import math
import numpy as np
import trimesh
import mesh as M
import joints
from anatomy import head as HEAD, thorax as THORAX, tail as TAIL
from anatomy.common import GRID_SEG, GRID_TAIL, GRID_HEAD
from manifold3d import Manifold, OpType

# ---------------------------------------------------------------- naming
def part_names(P): return ["head"] + [f"seg{i}" for i in range(int(P["segCount"]))] + ["tail"]

def _module_and_plan(P, name, notes=None):
    """Which anatomy module builds `name`, its plan, its ports, its grid."""
    if name == "head":
        S = HEAD.plan(P, notes); return HEAD, S, HEAD.ports(P, S), GRID_HEAD, None
    if name == "tail":
        S = TAIL.plan(P); return TAIL, S, TAIL.ports(P, S), GRID_TAIL, None
    i = int(name[3:]); S = THORAX.plan(P, i)
    return THORAX, S, THORAX.ports(P, S, i), GRID_SEG, i

# ---------------------------------------------------------------- one part
def _finish(body, mod, P, S, T, notes=None, skip=()):
    """Union the anatomy's ornaments onto the joint-cut body, in the anatomy's order, through the joint's frame T."""
    orn = mod.ornaments(P, S, notes) if mod is HEAD else mod.ornaments(P, S)
    ident = np.allclose(T, np.eye(4))
    for entry in orn:
        name, solids, how = (entry if len(entry) == 3 else (entry[0], [entry[1]], "union"))
        if name in skip: continue
        solids = [s if ident else s.copy().apply_transform(T) for s in solids]
        if how == "mirror":
            # union the RIGHT eye only, cut the part at the sagittal plane, mirror, weld: symmetric by construction.
            # (union(head, e, mirror(e)) left a ~1 mm asymmetric patch under the left eye's base, a Manifold artifact
            # on a near-coincident lobe/shell intersection; the half-and-mirror route cannot.)
            right = M.union(body, *solids)
            big = 4 * max(abs(v) for v in np.asarray(right.bounds).ravel()) + 10
            half = M.intersection(right, M.box(big, 2 * big, 2 * big, at=(0, 0, 0), align=("min", "c", "c")))
            body = M.union(half, M.mirror_x(half))
        else:
            body = M.union(body, *solids)
    return body

def _skip_for(J, opts):
    """Ornaments a joint's overhang pass supplies instead of the part builder (they reach over later parts)."""
    if J.MEASURED: return ()
    if (opts or {}).get("overhangs", True): return tuple(getattr(J, "RESTORED_ORNAMENTS", ()))
    return ("genalArms",)                                              # legacy flexi: stand-in horn, arms not restored

def part(P, name, joint="pin", bevel_deg=None, grid=None, notes=None, opts=None, skip=None):
    """One part in its own frame. joint: a name or a joint module. bevel_deg: the pin's instrument bevel (None =
    printed stop). opts: joint options (flexi gaps ...). skip: ornament names left out; None = the joint's own rule
    (a trimming joint's overhang pass supplies the ornaments that reach over later parts)."""
    J = joints.get(joint) if isinstance(joint, str) else joint
    if skip is None: skip = _skip_for(J, opts)
    mod, S, ports, g, i = _module_and_plan(P, name, notes); grid = grid or g
    if J.BASE == "shell": body, env = mod.shell(P, S, grid)
    else:                 body = mod.solid(P, S, grid); env = None
    T = np.eye(4)
    for key in ("rear", "front"):                      # rear first, then front: the 11 Sep order
        if key not in ports: continue
        kw = dict(bevel_deg=bevel_deg) if J.MEASURED else dict(opts or {})
        body, T1 = J.cut(body, env, P, ports[key], **kw)
        T = T1 @ T
    body = _finish(body, mod, P, S, T, notes, skip)
    if name == "head" and not J.MEASURED and not (opts or {}).get("overhangs", True) and hasattr(J, "stand_in_genal_horn"):
        horns = J.stand_in_genal_horn(P, S, opts)                    # legacy: no overhang pass, a stub horn instead of the arm
        if horns: body = M.union(body, *horns)
    return body

def head_body(P, bevel_deg=None, grid=None, notes=None):
    """The head without its genal spines (the instrument's closure test uses it)."""
    return part(dict(P, genalSpine=0.0), "head", "pin", bevel_deg=bevel_deg, grid=grid, notes=notes)

# ---------------------------------------------------------------- the animal
def builders(P, joint="pin", bevel_deg=None, grid=None, notes=None, opts=None):
    """[(name, fn)] in chain order. For a trimming joint the ornaments it restores later are skipped here."""
    J = joints.get(joint) if isinstance(joint, str) else joint
    skip = _skip_for(J, opts)
    return [(n, (lambda n=n: part(P, n, J, bevel_deg=bevel_deg, grid=grid, notes=notes, opts=opts, skip=skip))) for n in part_names(P)]

def animal(P, joint="pin", bevel_deg=None, grid=None, notes=None, opts=None, report=None):
    """Every part in its own frame, overhangs restored for joints that trim. pose(P, joint, [0]*n) assembles them."""
    J = joints.get(joint) if isinstance(joint, str) else joint
    parts = [fn() for _, fn in builders(P, J, bevel_deg, grid, notes, opts)]
    if (opts or {}).get("overhangs", True) and J.overhang is not None and not J.MEASURED:
        parts = restore(P, parts, J, opts, report)
    return parts

def pose(P, joint, angles_deg):
    J = joints.get(joint) if isinstance(joint, str) else joint
    return J.pose(P, angles_deg)

def transforms_deg(P, joint, theta_deg):
    J = joints.get(joint) if isinstance(joint, str) else joint
    return J.transforms_deg(P, theta_deg)

# ---------------------------------------------------------------- overhangs (26 Sep 2026)
# A print joint trims each part so the joint fits (one pitch behind the front face, the joint plane on the head).
# The pin build has no such limit: its shells shingle over each other, so pleural spines, falcate pleural tips and
# genal arms reach back over 2-4 later parts. The trim was deleting them (06f24ab3bc: 87 mm wide vs 121.5).
# restore() asks the joint for what it removed (joint.overhang), puts it back, and removes from it every place a
# LATER part occupies at rest and anywhere through the curl (joints together at several angles and each alone at
# the stop), grown by the gaps. Flexion is rotation about axes parallel to x, so anything that clears the later
# parts over that sweep never jams. Genal arms keep the pin build's raised band; where they pass over the thorax
# they overhang it and the slicer needs supports under them.
REACH = 5                # an overhang is cleared against the next REACH parts (spines and genal arms span 2-4)

def _combos(nj, k, j, amax, lo=-2.0):
    """Joint-angle vectors that exercise joints k..j-1 (the ones between part k and part j): all together at a few
    angles, and each alone at the stop with the rest flat or at the stop."""
    out = []
    for a in (lo, amax / 3, 2 * amax / 3, amax):
        v = [0.0] * nj
        for q in range(k, j): v[q] = a
        out.append(v)
    for q in range(k, j):
        for base in (0.0, amax):
            v = [0.0] * nj
            for r in range(k, j): v[r] = base
            v[q] = amax if base == 0.0 else 0.0; out.append(v)
    return out

def _boxes_meet(bb, R, bx, m):
    c = np.array([[x, y, z, 1.0] for x in (bb[0], bb[3]) for y in (bb[1], bb[4]) for z in (bb[2], bb[5])]) @ np.asarray(R).T
    lo, hi = c[:, :3].min(0), c[:, :3].max(0)
    return bool(np.all(lo <= np.array(bx[3:]) + m) and np.all(hi >= np.array(bx[:3]) - m))

def restore(P, base, J, opts=None, report=None):
    """base: parts in their own frames. Returns them with the joint's overhangs restored (see the block comment)."""
    J = joints.get(J) if isinstance(J, str) else J
    opts = dict(getattr(J, "DEFAULTS", {}), **(opts or {}))
    ga, gv, gl = opts.get("gap_axial", 0.4), opts.get("gap_vertical", 0.3), opts.get("gap_lateral", 0.25)
    names = part_names(P); n = len(base); nj = n - 1; amax = J.stop_deg(P); g = max(0.5 * ga, gl)
    W0 = J.pose(P, [0.0] * nj)
    world = [M.to_manifold(base[k].copy().apply_transform(W0[k])) for k in range(n)]
    ext = []
    for k, name in enumerate(names):
        mod, S, ports, _, i = _module_and_plan(P, name)
        port = ports.get("rear")
        e = J.overhang(P, name, S, port, opts) if port is not None else None
        ext.append(None if e is None else M.to_manifold(e).translate(tuple(W0[k][:3, 3])))
    offsets = [(0, 0, 0), (g, 0, 0), (-g, 0, 0), (0, g, 0), (0, -g, 0), (0, 0, gv)]   # a cheap dilation by the gaps
    rep = []; bbw = [w.bounding_box() for w in world]
    for k in range(n - 2, -1, -1):                                                 # tail first: later parts are final
        e = ext[k]
        if e is None or e.is_empty(): continue
        v0 = e.volume(); bx = e.bounding_box(); m_ = 1.0 + g
        crop = M.to_manifold(M.box(bx[3] - bx[0] + 2 * m_, bx[4] - bx[1] + 2 * m_, bx[5] - bx[2] + 2 * m_,
                                   at=(0.5 * (bx[0] + bx[3]), bx[1] - m_, bx[2] - m_), align=("c", "min", "min")))
        ghosts = []
        for j in range(k + 1, min(n, k + 1 + REACH)):
            for v in _combos(nj, k, j, amax):
                Tv = J.pose(P, v)
                R = W0[k] @ np.linalg.inv(Tv[k]) @ Tv[j] @ np.linalg.inv(W0[j])  # j relative to a fixed k, world-at-rest frame
                if not _boxes_meet(bbw[j], R, bx, m_): continue
                pj = world[j].transform(R[:3, :].tolist()) ^ crop                  # only the bit that can touch this overhang
                if pj.is_empty(): continue
                ghosts += [pj.translate(o) for o in offsets]
        for gh in ghosts:                                                          # one at a time: the overhang stays small,
            e = e - gh                                                             # a union of ~300 overlapping copies is not
            if e.num_tri() == 0: break
        e = M.to_manifold(M.from_manifold(e))                                      # force (lazy-CSG drop otherwise)
        before = len(world[k].decompose())
        merged = M.from_manifold(world[k] + e)
        comps = sorted(merged.split(only_watertight=False), key=lambda b: -b.volume)
        keep = comps[:max(before, 1)]                                              # pieces cut off from the part fall away
        lost = sum(c.volume for c in comps[max(before, 1):])
        world[k] = M.to_manifold(trimesh.util.concatenate(keep))
        rep.append(dict(part=k, overhang_mm3=round(v0, 1), kept_mm3=round(e.volume(), 1), detached_dropped_mm3=round(float(lost), 1)))
    if report is not None: report.extend(sorted(rep, key=lambda r: r["part"]))
    clean = getattr(J, "clean", lambda m: m)
    return [clean(M.from_manifold(world[k]).apply_transform(np.linalg.inv(W0[k]))) for k in range(n)]
