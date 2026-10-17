"""
isopod_model/body.py (was scripts/iso_body.py): the print body built the isopod way (28 Sep 2026). A print build
beside assemble's, not a joint module: it has to see neighbouring parts, so it builds the whole chain at once. Nothing
in anatomy/, joints/, the instrument or the site changes. Normally run through isopod_model/build.py.

    python isopod_model/body.py [out_dir] [--preset NAME] [--head isopod|generator]

Why: the ball build (joints/ball.py through assemble) trims every part to one pitch with box cuts, replaces the lower
front of every part across the whole width, and then restores the trimmed plates by subtracting the posed
neighbours at a handful of angles. That is what made it jagged: square pleural tips, stair-stepped plates, knife-thin
shards, loose flakes.

Here every part is drawn once, whole, as a height-field solid over its own anatomy plan (the generator's surface,
untouched), in the ball build's rest placement:
  * behind its rear joint plane a part is a PLATE (its own surface, WALL thick), lying over the next part's lip;
  * the lip sits under the plate on an air gap that stays at least GAP_Z through the whole curl (the isopod's wedge,
    computed: the lip is swept about the pivot to CURL and kept under the plate at every step);
  * below the lip the part ahead ends in a column face, the part behind starts on a column face behind the pivot
    (the isopod's columns and the gap between them);
  * THE MIDDLE-BOTTOM IS THE ISOPOD'S, copied (28 Sep 2026): the strip |x| <= ISO_W below ISO_ZC of the isopod's
    segment piece (column, ball with its slit and neck, socket, fork), from assets/isopod, at the isopod's own size in
    the print (the build is scaled by 18 / pitch on export, the isopod's pitch). Each part gets the socket half at
    its rear and the ball half at its front, placed on the joint, so the ball sits in the socket exactly as in the
    isopod; the pivot is the isopod ball's centre.
Checks: every part watertight and one piece; neighbours apart at rest and through the curl.
"""
import os, sys, math, argparse, gc, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, trimesh
import schema, mesh as M
from anatomy import head as HEAD, thorax as THORAX, tail as TAIL
from anatomy.common import spine_solid, prong, pitch
from joints import ball as BALL
from manifold3d import Manifold, OpType, Mesh as _MMesh

WALL = 2.0          # plate thickness behind a part's rear joint plane (model mm)
GAP_Z = 0.5         # least air gap between a plate and the lip under it, through the curl (model mm)
CURL = 30.0         # every joint is CLEARED to this angle (the printed joint moves on its own; a hand can bend any joint). The enrolment
                    # POSE (reading, sheet, viewer) follows kinematics.profile instead: head joint most, log decay behind (17 Oct 2026)
CURL_CLEAR_STEP = 3.0   # angular step for the head/segment clearance sweep (deg). 1.5 was exact-ish but doubled the
                        # boolean count; 3.0 halves it. If the curl check ever flags a sliver, lower this or widen the dilation.
LIP_MIN = 0.8       # thinnest lip kept (model mm)
STRIP_PAST = 0.3    # the joint strip is cleared this far behind a part's rear face before the kit goes in (model mm)
HEAD_JOINT_DEG = 50.0   # the cephalon joint J1 is relieved to turn this far (deg): Esteve et al. 2017 measure about 45 at the head
                        # joint, the kit alone stops at 36 (the neck's foot swings into the column base of the piece ahead). The
                        # head piece alone is cut where seg0's kit sweeps through between CURL and this angle (17 Oct 2026).
BISECT_STEPS = 14   # the lip's curl bisection: 14 halvings of a ~20 mm range settle it to 0.002 mm (was 24, to 2e-6 mm;
                    # 11 Oct 2026, with Claire's verdict: the cutters and the lip pass cost 40 % less)
WORKERS = max(1, min(4, int(os.environ.get("TRILO_WORKERS", os.cpu_count() or 1))))   # forked workers for the cutters, the head sweep and the
                                                                                        # clearances; TRILO_WORKERS=1 runs everything inline
CUTTER_MAX_POINTS = 150_000   # the front cutter's grid: 0.1 mm steps up to this many points (the default preset's 140k),
                              # coarser beyond (11 Oct 2026: a 600 mm wide row made a 600k-point cutter and ran out of memory)
HEAD_MAX_POINTS = 400_000     # the head's ray grids (its rear cut, its underside), likewise (the default head: 13k and 38k)


def _grid_step(step, w, h, max_points=HEAD_MAX_POINTS):
    """The grid step that keeps a w x h mm grid under max_points: `step` itself for every ordinary animal, coarser
    for a giant one (a 600 mm head at 0.25 mm was 5.8 M rays: out of memory in the head, before any part)."""
    n = (w / step + 1) * (h / step + 1)
    return step if n <= max_points else math.sqrt(w * h / max_points)
GRID = {"head": (241, 121), "seg": (161, 61), "tail": (241, 121)}

# the isopod's middle-bottom (asset mm, the asset's work frame: midline x = 0, bed z = 0)
ISO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "isopod", "giant_isopod_main_itbefred.stl")
ISO_PIECE = 3       # the segment piece copied (ball at its front, socket at its rear)
ISO_W = 15.0        # the strip's half-width: the middle block (|x| <= 12.4) and the fork, cut in the empty slot beside
ISO_ZC = 20.0       # the strip's top: ball, neck, socket and fork are all below 13.5
ISO_YM = 91.0       # where piece 3's strip is split into its ball half and socket half (its column is prismatic here)
ISO_OVER = 0.4      # the halves overlap by this much past the split (the column is prismatic there)
ISO_PITCH = 18.0    # joints/ball.py's ISO_PITCH: the build is exported at 18 / pitch, so s = pitch / 18 below


_ISO_MESH = None
def _force(man):
    """Materialise a lazy manifold3d CSG node (the guard joints/ball._force gives), WITHOUT the trimesh round trip and
    the component decomposition it did on every step (11 Oct 2026: that was 42 s of a 103 s build). Slivers are
    dropped once per part at the end (from_manifold + clean_parts), not after every boolean."""
    man.num_vert(); return man


def _iso_load():
    """The 25 MB isopod asset, loaded from disk once and reused (iso_kit and isopod_head both need it). Returns a
    fresh copy each call so callers can translate/split without corrupting the cache."""
    global _ISO_MESH
    if _ISO_MESH is None: _ISO_MESH = trimesh.load(ISO)
    return _ISO_MESH.copy()


def _iso_frame():
    """The copied isopod pieces (piece ISO_PIECE and the next) and the two ball centres, in the asset's work frame."""
    m = _iso_load(); z0 = m.bounds[0][2]; xc = 0.5 * (m.bounds[0][0] + m.bounds[1][0])
    m.apply_translation((-xc, 0, -z0))
    parts = sorted(m.split(only_watertight=False), key=lambda p: p.bounds[0][1])
    def centre(p):                                                          # as crescent_head.ball_centre
        y_hi = p.bounds[0][1] + 13; o = []
        for dx in (3.5, 4.0, 4.5, 5.0):
            for d in p.section(plane_origin=(dx, 0, 0), plane_normal=(1, 0, 0)).discrete:
                c = d.mean(0)
                if d[:, 1].max() < y_hi and 2 < c[2] < 13: o.append(c[1:])
        return np.array([0.0, *np.mean(o, 0)])
    A, B = parts[ISO_PIECE], parts[ISO_PIECE + 1]
    cf, cr = centre(A), centre(B)
    # column faces, measured 8 mm off the midline (clear of the ball and fork): the socket piece's rear face and the
    # ball piece's front face
    def hits(p, x):
        loc, _, _ = p.ray.intersects_location([(x, cr[1] - 20, 3.0)], [(0, 1.0, 0)]); return np.sort(np.asarray(loc).reshape(-1, 3)[:, 1])
    rear = hits(A, 8.0); rear = rear[rear < cr[1] + 3].max() - cr[1]
    front = hits(B, 8.0)[0] - cr[1]
    return A, cf, cr, rear, front


def iso_kit(P, s=None):
    """The isopod strip halves, scaled to the model (s = pitch / 18, or the given model-mm-per-print-mm), each in its
    own joint frame (the ball centre at y = 0, z at scale): 'ball' = piece's front half (ball, neck, column front),
    'socket' = its rear half (column rear, socket, fork). Plus the joint numbers in model mm."""
    A, cf, cr, rear, front = _iso_frame()
    if s is None: s = 1.0 / BALL.model_scale(P)                            # = pitch / 18
    def cut(y0, y1):
        box = M.to_manifold(M.box(2 * ISO_W, y1 - y0, ISO_ZC + 1.0, at=(0, y0, -1.0), align=("c", "min", "min")))
        return M.from_manifold(M.to_manifold(A) ^ box)
    ball = cut(cf[1] - 12.0, ISO_YM + ISO_OVER); ball.apply_translation((0, -cf[1], 0)); ball.apply_scale(s)
    sock = cut(ISO_YM - ISO_OVER, cr[1] + 12.0); sock.apply_translation((0, -cr[1], 0)); sock.apply_scale(s)
    return dict(ball=ball, socket=sock, s=s, zp=cr[2] * s, col_rear=rear * s, col_front=front * s,
                ball_start=(-12.0) * s, ball_end=(ISO_YM + ISO_OVER - cf[1]) * s,
                sock_start=(ISO_YM - ISO_OVER - cr[1]) * s, sock_end=12.0 * s, w=ISO_W * s, zc=ISO_ZC * s)


def placed_plans(P):
    """[(name, module, plan, y_offset)] in chain order, at the rest placement of the ball pose."""
    n = int(P["segCount"]); T = BALL.pose(P, [0.0] * (n + 1))
    out = [("head", HEAD, HEAD.plan(P), T[0][1, 3])]
    for i in range(n): out.append((f"seg{i}", THORAX, THORAX.plan(P, i), T[i + 1][1, 3]))
    out.append(("tail", TAIL, TAIL.plan(P), T[n + 1][1, 3]))
    return out


def pivots(plans, J, K):
    """Joint k (between part k and k+1): the pivot (y, z), the ball centre. The isopod's socket-piece column rear face
    goes on part k's rear face (the joint plane less half the axial gap)."""
    ga = J["gap_axial"]
    return [(plans[k + 1][3] - 0.5 * ga - K["col_rear"], K["zp"]) for k in range(len(plans) - 1)]


def top_world(S, off):
    return lambda x, y: S["zfun"](np.abs(np.asarray(x, float)), np.asarray(y, float) - off)


def _rear_edge(S, off, ax):
    """y of a part's rear outline edge at |x| (world)."""
    us = np.linspace(0, 1, 201); pts = np.array([S["outline"](u, 1.0) for u in us]); pts2 = np.array([S["outline"](u, 0.0) for u in us])
    xr = np.abs(pts[:, 0]); yr = np.maximum(pts[:, 1], pts2[:, 1]) + off
    o = np.argsort(xr); return np.interp(ax, xr[o], yr[o], right=-1e9)


def _turn(y, z, pv, a):
    """(y, z) turned about the pivot by a curl of a degrees (the part behind goes down)."""
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a)); dy, dz = y - pv[0], z - pv[1]
    return pv[0] + dy * c + dz * s, pv[1] - dy * s + dz * c


def part_top(k, plans, J, PV, x, y, zs=None, U=None):
    """Part k's top at world (x, y): its anatomy surface (zs if given), and where it lies under the plate of the part
    ahead, the highest z that stays GAP_Z under that plate at every step of the curl to CURL (bisection).
    U: the underside of the part ahead as a function (x, y) -> z (+inf where there is none), when that part is not
    drawn from its plan (the isopod head)."""
    name, mod, S, off = plans[k]; ax = np.abs(x)
    top = top_world(S, off)(x, y) if zs is None else zs.copy()
    if k == 0: return top
    _, _, Sp, offp = plans[k - 1]; pv = PV[k - 1]; zp_top = top_world(Sp, offp)
    if U is None:
        y_end = _rear_edge(Sp, offp, ax) + 0.3
        under = y <= y_end
        if not under.any(): return top
        y_end_u = y_end[under]
        U = lambda x_, y_: np.where(y_ <= y_end_u, zp_top(x_, y_) - WALL, np.inf)
    else:
        under = np.isfinite(U(x, y))
        if not under.any(): return top
    xu, yu, tu = x[under], y[under], top[under]                            # the bisection runs on the lip points only
    def ok(z):                                                              # (11 Oct 2026: same arithmetic, a fifth of the points)
        good = np.ones_like(z, bool)
        for a in np.linspace(0.0, CURL, 11):
            ya, za = _turn(yu, z, pv, a)
            good &= za <= U(xu, ya) - GAP_Z                                 # +inf: not under the plate
        return good
    hi = tu.copy(); lo = np.full_like(tu, pv[1] - 5.0)
    fine = ok(hi)
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi); g = ok(mid)
        lo = np.where(g, mid, lo); hi = np.where(g, hi, mid)
    lim = np.where(fine, tu, lo)
    out = top.copy(); out[under] = np.minimum(tu, lim)
    return out


def sample_plan(S, nu, nv):
    """mesh.sample_grid for a plan, bit for bit, but vectorised when the plan's outline is u * X_tip across and the
    front/rear edges along (every thorax/rows plan: outline(u, v) = (u X_tip, yf + v (yr - yf))). The head and tail
    plans go through mesh.sample_grid itself. 11 Oct 2026: the per-point Python loop was 16 s of a 103 s build."""
    if "edges" not in S or "X_tip" not in S: return M.sample_grid(S["outline"], S["zfun"], nu, nv)
    if nu % 2 == 0: nu += 1
    passes = max(1, int(round(2 * ((nu - 1) / (M.REF_NU - 1)) ** 2)))
    mid = nu // 2
    us = np.array([-1 + 2 * i / (nu - 1) for i in range(mid, nu)]); x = us * S["X_tip"]
    yf, yr = S["edges"](np.abs(x)); yf = np.asarray(yf, float); yr = np.asarray(yr, float)
    v = np.array([j / (nv - 1) for j in range(nv)])[:, None]
    xs = np.zeros((nv, nu)); ys = np.zeros((nv, nu))
    xs[:, mid:] = x[None, :]; ys[:, mid:] = yf[None, :] + v * (yr[None, :] - yf[None, :])
    xs[:, :mid] = -xs[:, nu - 1:mid:-1]; ys[:, :mid] = ys[:, nu - 1:mid:-1]; xs[:, mid] = 0.0
    zs = S["zfun"](np.abs(xs.ravel()), ys.ravel()).reshape(nv, nu)
    for _ in range(passes):
        zp = np.pad(zs, 1, mode="edge"); zs = 0.25 * zp[1:-1, :-2] + 0.5 * zp[1:-1, 1:-1] + 0.25 * zp[1:-1, 2:]
        zp = np.pad(zs, 1, mode="edge"); zs = 0.25 * zp[:-2, 1:-1] + 0.5 * zp[1:-1, 1:-1] + 0.25 * zp[2:, 1:-1]
    zs[:, :mid] = zs[:, nu - 1:mid:-1]
    for j in (0, nv - 1):
        if np.ptp(xs[j]) < 1e-9 and np.ptp(ys[j]) < 1e-9: zs[j, :] = zs[j].mean()
    for i in (0, nu - 1):
        if np.ptp(xs[:, i]) < 1e-9 and np.ptp(ys[:, i]) < 1e-9: zs[:, i] = zs[:, i].mean()
    return xs, ys, zs


def build_part(k, plans, P, J, PV, U=None, shell=False, strip_w=None):
    """Part k as a height-field solid: top = part_top, bottom = the bed, a WALL-thick plate behind its rear joint
    plane. shell=True (the rows model): a WALL-thick slab everywhere except the joint strip |x| <= strip_w, which
    stays solid to the bed for the isopod's column."""
    name, mod, S, off = plans[k]
    kind = "head" if name == "head" else ("tail" if name == "tail" else "seg")
    nu, nv = GRID[kind]
    if shell and kind != "head": nv = max(nv, int(S["d"] / 0.7) | 1)        # long fused pieces keep ~0.7 mm rows
    xs, ys, zs = sample_plan(S, nu, nv)                                     # the generator's own smoothing (mesh.sample_grid, vectorised)
    y = ys + off; ga = J["gap_axial"]
    top = part_top(k, plans, J, PV, xs, y, zs, U if k == 1 else None)
    bot = np.zeros_like(top)
    if shell and kind != "head":
        yr = plans[k + 1][3] - 0.5 * ga if k < len(plans) - 1 else np.inf
        bot = np.where((np.abs(xs) > (strip_w or 0.0) + 1.0) | (y > yr), top - WALL, bot)   # the slab everywhere; the column strip
                                                                               # stays solid to the bed only ahead of the rear joint plane
    elif k < len(plans) - 1:
        yr = plans[k + 1][3] - 0.5 * ga
        bot = np.where(y > yr, top - WALL, bot)                             # the plate behind the rear joint plane
    bot = np.clip(np.minimum(bot, top - 0.6), 0.0, None)
    top = np.maximum(top, bot + 0.6)
    T = np.c_[xs.ravel(), y.ravel(), top.ravel()]; B = np.c_[xs.ravel(), y.ravel(), bot.ravel()]
    return M._closed(T, B, zs.shape[0], zs.shape[1])


def front_cutter(k, plans, J, PV, K, step=0.1, U=None):
    """What part k (k > 0) loses at its front: everything ahead of its column face (the isopod's, col_front behind the
    pivot) below the pivot, and ahead of the pivot below a lip where the plate would leave that lip thinner than
    LIP_MIN (a smooth, steep edge). What remains ahead of the column is the lip, above the pivot."""
    name, mod, S, off = plans[k]; pv = PV[k - 1]; yF = pv[0] + K["col_front"]; zp = pv[1]
    us = np.linspace(-1, 1, 401); W = max(abs(S["outline"](u, v)[0]) for u in us for v in (0.0, 0.5, 1.0)) + 2.0
    y0 = off - 2.0; y1 = yF
    if (2 * W / step) * ((y1 - y0) / step + 2) > CUTTER_MAX_POINTS:        # huge parts: a coarser step, not a 10^6-point grid
        step = math.sqrt(2 * W * (y1 - y0) / CUTTER_MAX_POINTS)
    X, Y = np.meshgrid(np.arange(-W, W + 1e-9, step), np.linspace(y0, y1, int((y1 - y0) / step) + 2))
    t = part_top(k, plans, J, PV, X, Y, None, U if k == 1 else None) - zp
    Z = zp + np.clip((LIP_MIN - t) * 20.0, 0.0, 60.0)
    T = np.c_[X.ravel(), Y.ravel(), Z.ravel()]; B = np.c_[X.ravel(), Y.ravel(), np.full(X.size, -1.0)]
    return M._closed(T, B, X.shape[0], X.shape[1])


def paste_middle(parts, plans, PV, K, skip=()):
    """The isopod's middle-bottom into every part: our strip |x| <= w below zc (ahead of the plate) goes, the isopod's
    goes in: the socket half at each part's rear joint, the ball half at its front joint. skip: repeats (copied after)."""
    n = len(plans)
    for k in range(n):
        if k in skip: continue
        m = M.to_manifold(parts[k])
        y_from = PV[k - 1][0] + K["ball_start"] if k > 0 else None
        y_to = PV[k][0] + K["col_rear"] if k < n - 1 else None             # = the part's rear face: the plate stays
        if y_from is None: y_from = PV[k][0] + K["sock_start"]              # head: only its socket half
        if y_to is None: y_to = PV[k - 1][0] + K["ball_end"]                # tail: only its ball half
        # the strip is cut to STRIP_PAST beyond the rear face: the height field's step from solid to plate at the joint
        # plane is one grid row wide (a sloped face), and that sliver sat in the slot the next neck runs through
        # (11 Oct 2026: a 0.04 mm skin, 0.25 mm3, on a fused piece whose grid phase put the slope inside the slot)
        y_cut = y_to + (STRIP_PAST if k < n - 1 else 0.0)
        m = m - M.to_manifold(M.box(2 * K["w"], y_cut - y_from, K["zc"] + 1.0, at=(0, y_from, -1.0), align=("c", "min", "min")))
        if k > 0: m = m + M.to_manifold(K["ball"].copy().apply_translation((0, PV[k - 1][0], 0)))
        if k < n - 1: m = m + M.to_manifold(K["socket"].copy().apply_translation((0, PV[k][0], 0)))
        parts[k] = M.from_manifold(_force(m))


def head_ornaments(P, S, head):
    """The head's own ornaments (eyes, genal arms, occipital spine, prongs), as assemble unions them."""
    for name, solids, how in HEAD.ornaments(P, S):
        for s in solids:
            head = M.union(head, s, M.mirror_x(s)) if how == "mirror" else M.union(head, s)
    return head


HEAD_FILL = (16.0, 20.0, 30.0)   # isopod-head rear middle rebuilt before the socket half goes in: half-width, length ahead
                                 # of the rear face, height (print mm)
HEAD_HEIGHT = 0.85               # head top / the tallest thorax piece's top, times the preset's headRelief (the isopod's
                                 # own head: 29.3 / 36.8 = 0.80)
HEAD_RIM = 0.15                  # the rim's height / the head's crown height (below it the dome curve does not reach)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source", "cache")


def face_of(P, extra=None):
    """The crescent head's face from the generator's own parameters (head_crescent.FACE was set from the default
    preset's values, so the default preset gives the approved face exactly)."""
    f = dict(eye_size=float(P["eyeSize"]), eye_pos=float(P["eyePos"]), eye_height=float(P["eyeHeight"]),
                eye_lat=float(P["eyeLat"]), glab_inflate=float(P["glabInflate"]), glab_rise=float(P["glabRise"]),
                glab_front=float(P["glabFront"]), glab_lobes=int(P["glabLobes"]))   # nose shape + furrow pairs: were defaulting, now live on the print model (head_crescent._z reads them)
    if float(P.get("sutureDepth", 0.0)) > 1e-9:                             # 2 Oct 2026: the facial suture and the eye ridge, on the crescent as on
        f.update(suture_end=float(P["sutureEnd"]), suture_depth=float(P["sutureDepth"]))   # the classic head. Added only when set, so a face
    if float(P.get("eyeRidge", 0.0)) > 1e-9: f.update(eye_ridge=float(P["eyeRidge"]))      # without them keeps its key in source/cache.
    if extra: f.update(extra)                                               # a head style's sculpt keys (head_styles.py)
    return f

BASE_FACE_KEYS = ("eye_size", "eye_pos", "eye_height", "eye_lat", "glab_inflate", "glab_rise", "glab_front", "glab_lobes")


def head_piece(face):
    """crescent_head.head_piece for this face, cached on disk by the face's values."""
    import hashlib, json as _json
    key = hashlib.md5(_json.dumps(face, sort_keys=True).encode()).hexdigest()[:12]
    f_npz, f_js = os.path.join(CACHE, f"head_{key}.npz"), os.path.join(CACHE, f"head_{key}.json")
    if os.path.exists(f_npz):                                               # exact vertices (an STL round trip is not watertight)
        d = np.load(f_npz); return trimesh.Trimesh(d["v"], d["f"], process=False), _json.load(open(f_js))
    import crescent_head as CH
    head, info = CH.head_piece(face); os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(f_npz, v=head.vertices, f=head.faces)
    info = dict(face=face, k=float(info["k"]), y_w=float(info["y_w"]), y_face=float(info["y_face"]), eye=[float(v) for v in info["eye"]])
    _json.dump(info, open(f_js, "w"), indent=1); return head, info


def _surface_z(mesh, x, y, search=0.0):
    """Top surface height at (x, y); with search > 0, the nearest hit within that radius if (x, y) itself misses."""
    for r in ([0.0] + (list(np.linspace(0.25, search, 8)) if search > 0 else [])):
        for a in ([0.0] if r == 0 else np.linspace(0, 2 * np.pi, 12, endpoint=False)):
            loc, _, _ = mesh.ray.intersects_location([(x + r * np.cos(a), y + r * np.sin(a), 1e3)], [(0, 0, -1.0)])
            loc = np.asarray(loc).reshape(-1, 3)
            if len(loc): return float(loc[:, 2].max())
    return None


def isopod_head(P, plans, J, PV, K, step=0.25, face=None, _grid=False):
    """The crescent head made on the isopod's head piece, set by the preset:
      * face: eye size / place / height and the glabella from the preset (face_of), as the crescent's detail skin;
      * size: scaled across (x, y) so it stands to our first segment as it stood to the isopod's (first segment's
        half-width over the isopod piece 1's); its height set to HEAD_HEIGHT x the tallest thorax piece x headRelief;
      * its column's rear face on the joint; its middle-bottom then becomes the same isopod socket half as every other
        part's (the scaled one would not fit the ball): the old socket and fork are filled / cleared, paste_middle
        puts the socket half in;
      * the preset's eye solid (eyeSolid: the generator's eye, its lens lattice, stalk, brim), at full size (never
        squashed), on the crescent's eye; the occipital spine and head prongs where the preset has them.
    Returns the head (model mm) and its underside for the lip under it."""
    fc = face_of(P, face)
    head, info = head_piece(fc)                                             # work frame, isopod size
    # the head's height is set from its crown. A sculpted face (boss, tubercles, bullae ...) can stand above the crown,
    # and scaling by ITS top would squash the whole head under one bump: scale by the same face without the sculpt keys.
    sculpt = [k for k in fc if k not in BASE_FACE_KEYS and k not in ("suture_end", "suture_depth", "eye_ridge")]
    z_ref = head_piece({k: fc[k] for k in BASE_FACE_KEYS})[0].bounds[1][2] if sculpt else head.bounds[1][2]
    iso = _iso_load(); z0 = iso.bounds[0][2]; xc = 0.5 * (iso.bounds[0][0] + iso.bounds[1][0])
    iso.apply_translation((-xc, 0, -z0))
    pieces = sorted(iso.split(only_watertight=False), key=lambda p: p.bounds[0][1])
    loc, _, _ = pieces[0].ray.intersects_location([(8.0, 0.0, 3.0)], [(0, 1.0, 0)])
    y_rear = np.asarray(loc).reshape(-1, 3)[:, 1].max()                    # the head column's rear face, 8 mm off axis
    s = K["s"]; S1 = plans[1][2]
    w_seg = max(abs(S1["outline"](u, v)[0]) for u in np.linspace(-1, 1, 401) for v in (0.0, 0.5, 1.0)) / s
    f = w_seg / pieces[1].bounds[1][0]; fxy = f * s
    z_thorax = max(float(M.sample_grid(Sk["outline"], Sk["zfun"], 61, 31)[2].max()) for _, _, Sk, _ in (plans[1:-1] or plans[1:]))
    fz = HEAD_HEIGHT * z_thorax * float(P.get("headRelief", 1.0)) / z_ref
    yr = PV[0][0] + K["col_rear"]                                           # our head's rear face (model)
    ty = yr - fxy * y_rear
    head.apply_transform(np.diag([fxy, fxy, fz, 1.0])); head.apply_translation((0, ty, 0))
    # the dome's curve from the preset (headDomeExp, the generator's dome exponent; 1.5 = the approved head): heights
    # above the rim are remapped rim + (top - rim) t^(1.5 / exp), t = (z - rim) / (top - rim). A larger exponent fills
    # the dome out (flatter crown, steeper flanks), a smaller one peaks it. The rim and everything at rim level (the
    # horns) and the crown height are unchanged.
    g = 1.5 / float(P.get("headDomeExp", 1.5))
    if abs(g - 1.0) > 1e-3:
        V = head.vertices.copy(); top = z_ref * fz                         # the crown (a sculpted bump above it keeps going up)
        rim = HEAD_RIM * top
        t = np.clip((V[:, 2] - rim) / (top - rim), 0, None); up = V[:, 2] > rim
        V[up, 2] = rim + (top - rim) * t[up] ** g
        head = trimesh.Trimesh(V, head.faces, process=False)
    # rear middle: fill the old (scaled) socket, clear its fork behind the rear face; paste_middle does the rest
    w, L, Z = (v * s for v in HEAD_FILL)
    # the fill never rises above the head's own top (the head is lowered, the box is not)
    FX, FY = np.meshgrid(np.linspace(-w, w, 65), np.linspace(yr - L, yr, 41))
    o = np.c_[FX.ravel(), FY.ravel(), np.full(FX.size, 1e3)]
    locs, ri, _ = head.ray.intersects_location(o, np.tile((0, 0, -1.0), (len(o), 1)), multiple_hits=True)
    ht = np.full(len(o), 0.0); np.maximum.at(ht, ri, np.asarray(locs)[:, 2])
    fz_top = np.clip(np.minimum(ht.reshape(FX.shape), Z), 0.3, None)       # up to the head's own top, or Z
    fill = M._closed(np.c_[FX.ravel(), FY.ravel(), fz_top.ravel()], np.c_[FX.ravel(), FY.ravel(), np.zeros(FX.size)], FX.shape[0], FX.shape[1])
    H = M.to_manifold(head)
    H = H - M.to_manifold(M.box(2 * w, L + 60 * s, Z + 1, at=(0, yr - L, -1.0), align=("c", "min", "min")))   # up to Z
    H = _force(H + M.to_manifold(fill))
    head = max(M.from_manifold(H).split(only_watertight=False), key=lambda b: b.volume)
    # behind its rear face the head is a plate, as every other part is: the isopod head piece's cheek walls reach down
    # to the bed behind its column (they wrap the isopod's first piece), which is where our first segment's column
    # stands. Cut everything behind the rear face lower than WALL under the head's own top; the horns (thinner than
    # WALL at rim level) stay.
    step = _grid_step(step, head.bounds[1][0] - head.bounds[0][0] + 2 * step, head.bounds[1][1] - yr + step)
    X = np.arange(head.bounds[0][0] - step, head.bounds[1][0] + step, step); Y = np.arange(yr + 0.05, head.bounds[1][1] + step, step)
    GX, GY = np.meshgrid(X, Y); o = np.c_[GX.ravel(), GY.ravel(), np.full(GX.size, 1e3)]
    locs, ri, _ = head.ray.intersects_location(o, np.tile((0, 0, -1.0), (len(o), 1)), multiple_hits=True)
    locs = np.asarray(locs).reshape(-1, 3); ht = np.full(len(o), -np.inf)
    if len(locs): np.maximum.at(ht, ri, locs[:, 2])                       # no hits (a body narrower than the head's rear): nothing to cut
    ht = ht.reshape(GX.shape)
    cz = np.where(np.isfinite(ht), ht - WALL, -0.9)
    cut = M._closed(np.c_[GX.ravel(), GY.ravel(), np.maximum(cz, -0.9).ravel()], np.c_[GX.ravel(), GY.ravel(), np.full(GX.size, -1.0)], GX.shape[0], GX.shape[1])
    head = M.from_manifold(_force(M.to_manifold(head) - M.to_manifold(cut)))
    head = max(head.split(only_watertight=False), key=lambda b: b.volume)
    # the preset's ornaments on this head
    SC = HEAD.plan(P); extra = []; notes = []
    Lc = float(SC["Lc"]); margin = float(SC["margin"])
    if P.get("eyeSolid", 0) > 0.5 and P["eyeSize"] > 0.01:
        xe, ye, Re = info["eye"]; x, y, R = xe * fxy, ye * fxy + ty, Re * fxy
        EP = HEAD.eye_params(P, R); eye, n_lens = HEAD.eye_solid(**EP); notes.append(f"{n_lens} lenses/eye")
        zc = _surface_z(head, x, y, search=R)
        if zc is None: raise ValueError(f"eye at ({x:.1f}, {y:.1f}) is off the head")
        if EP["stalk"] > 0.05: z = zc - 0.3 + EP["stalk"]
        else:                                                               # sessile: base at the lowest point under it
            ring = [_surface_z(head, x + R * np.cos(a), y + R * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
            z = min([zc] + [r for r in ring if r is not None]) - 0.3
        e = eye.copy(); e.apply_translation((x, y, z)); extra += [e, M.mirror_x(e)]
    if P["occipitalSpine"] > 0.02:
        y = yr - 0.07 * Lc; z = _surface_z(head, 0.0, y, search=0.1 * Lc)
        extra.append(spine_solid(0.6 * margin, 0.5, P["occipitalSpine"] * Lc, (0, y, z - 1.0), 0, pitch_deg=55))
    if int(P.get("headProngs", 0)) > 0:
        zp = 0.6 * margin + 0.5
        loc, _, _ = head.ray.intersects_location([(0.0, head.bounds[0][1] - 5.0, zp)], [(0, 1.0, 0)])
        yf = float(np.asarray(loc).reshape(-1, 3)[:, 1].min())
        extra.append(prong(int(P["headProngs"]), P["headProngLen"] * Lc, P["headProngSplay"], P.get("headProngWidth", 0.45) * margin + 0.6,
                           0.45, (0, yf + 1.5, zp), yaw_deg=180.0, pitch_deg=8.0, stem_frac=P.get("headProngStem", 0.0),
                           center_bias=P.get("headProngCenter", 1.0), curl_deg=P.get("headProngCurl", 0.0)))
    if extra: head = M.union(head, *extra)
    head.metadata["notes"] = notes
    # underside behind the rear face: lowest head surface over each point (+inf where the head is not overhead)
    step = _grid_step(step, 2 * head.bounds[1][0] + 2 * step, head.bounds[1][1] - yr + 0.5 + step)
    X = np.arange(-head.bounds[1][0] - step, head.bounds[1][0] + step, step); Y = np.arange(yr - 0.5, head.bounds[1][1] + step, step)
    GX, GY = np.meshgrid(X, Y); o = np.c_[GX.ravel(), GY.ravel(), np.full(GX.size, -1.0)]
    locs, ri, _ = head.ray.intersects_location(o, np.tile((0, 0, 1.0), (len(o), 1)), multiple_hits=True)
    locs = np.asarray(locs).reshape(-1, 3); hb = np.full(len(o), np.inf)
    if len(locs): np.minimum.at(hb, ri, locs[:, 2])
    hb = hb.reshape(GX.shape)
    U = _underside_fn(float(X[0]), float(Y[0]), float(step), hb)
    return (head, U, (float(X[0]), float(Y[0]), float(step), hb)) if _grid else (head, U)


SKIPPED = []


def body_ornaments(P, plans, parts, skip=()):
    """The preset's spines on the body, as the generator puts them: each thorax ring's axial spine, the tail's
    terminal spine and prongs (plan frame, moved to the part's rest placement). An axial spine stands where the ring's
    top is open to the sky: where the generator's spot lies under the plate of the part ahead (the lip), it moves back
    along the axis to the first open point and stands on the part's actual top there."""
    for k, (name, mod, S, off) in enumerate(plans):
        if name == "head" or k in skip: continue
        solids = []
        for entry in mod.ornaments(P, S):
            nm, ss, how = (entry if len(entry) == 3 else (entry[0], [entry[1]], "union"))
            for x in ss:
                x = x.copy(); x.apply_translation((0, off, 0))
                if nm == "axialSpine":
                    y0 = off + S["ovl"] + 0.45 * (S["d"] - S["ovl"]); z0 = S["h"] + S["rise"] - 1.0
                    placed = False
                    for y in np.arange(y0, off + S["d"] + 4.0, 0.1):
                        mine = _surface_z(parts[k], 0.0, y)
                        if mine is None: continue
                        # covered = the part ahead's plate lies right over this point (within 3 mm)
                        loc, _, _ = parts[k - 1].ray.intersects_location([(0.0, y, mine + 0.05)], [(0, 0, 1.0)])
                        loc = np.asarray(loc).reshape(-1, 3)
                        if not len(loc) or loc[:, 2].min() - mine > 3.0:
                            x.apply_translation((0, y - y0, mine - 1.0 - z0)); placed = True; break
                    if not placed:                                          # the whole ring is under the plate ahead
                        SKIPPED.append(f"{name}: axial spine (ring covered by the part ahead)"); continue
                solids.append(x)
        if solids: parts[k] = M.union(parts[k], *solids)


def shaped_head(P, plans, PV, K, J, outline, face):
    """The head drawn whole on a parametric outline (shaped_head.py), as a height-field solid like every other part:
    solid to the bed, except where it lies over the body behind its rear joint plane, where it is a WALL-thick plate.
    A genal spine that runs BESIDE the body (outside every segment's reach there) stays solid to the bed, so it
    prints without supports."""
    import shaped_head as SH
    fc = face_of(P, face); S = SH.plan(P, plans, K, outline, fc); off = plans[0][3]
    plans[0] = ("head", HEAD, S, off)
    nu, nv = 401, 201                                                       # sampled directly: the generator's smoothing
    sh = S["shape"]; k = S["scale"]                                         # (mesh.sample_grid's blur) would wipe the sculpt
    Ug, Vg = np.meshgrid(np.linspace(-1, 1, nu), np.linspace(0, 1, nv))
    xa = Ug * sh.Xmax; yf_, yr_ = sh.edges(np.abs(xa))
    xs = k * xa; ys = S["flap"] + k * (yr_ - Vg * (yr_ - yf_)); zs = S["zfun"](xs, ys)
    y = ys + off; ga = J["gap_axial"]
    yr = plans[1][3] - 0.5 * ga
    reach = np.zeros_like(y)                                                # the body's half-width at each y behind the head
    for name, mod, Sk, offk in plans[1:]:
        e = np.array([Sk["outline"](u, v) for u in np.linspace(-1, 1, 81) for v in (0.0, 1.0)])
        y0, y1 = e[:, 1].min() + offk - 4.0, e[:, 1].max() + offk + 4.0      # + the curl's fore-aft travel
        reach = np.where((y >= y0) & (y <= y1), np.maximum(reach, np.abs(e[:, 0]).max()), reach)
    beside = np.abs(xs) > reach + 0.8 * ga + 0.6
    bot = np.where((y > yr) & ~beside, zs - WALL, 0.0)
    bot = np.clip(np.minimum(bot, zs - 0.6), 0.0, None); top = np.maximum(zs, bot + 0.6)
    head = M._closed(np.c_[xs.ravel(), y.ravel(), top.ravel()], np.c_[xs.ravel(), y.ravel(), bot.ravel()], zs.shape[0], zs.shape[1])
    # the rear block (10 Oct 2026): the socket half is pasted in the strip |x| <= K.w from sock_start to the rear face; a
    # head whose rear margin ends ahead of that (phacops on a plain body) left the socket as a loose body. As the
    # isopod head's HEAD_FILL does, a block of the strip's width runs from 20 print mm ahead of the rear face to the
    # rear face, up to the head's own top where the head is overhead and to just over the strip's top where it is not.
    sK = K["s"]; w_f, L_f, Z_f = 16.0 * sK, 20.0 * sK, 30.0 * sK; Z_lo = 23.0 * sK
    FX, FY = np.meshgrid(np.linspace(-w_f, w_f, 41), np.linspace(yr - L_f, yr, 41))
    o = np.c_[FX.ravel(), FY.ravel(), np.full(FX.size, 1e3)]
    locs, ri, _ = head.ray.intersects_location(o, np.tile((0, 0, -1.0), (len(o), 1)), multiple_hits=True)
    ht = np.full(len(o), -1.0); np.maximum.at(ht, ri, np.asarray(locs).reshape(-1, 3)[:, 2] if len(locs) else np.zeros(0))
    fz = np.where(ht > 0.3, np.minimum(ht, Z_f), Z_lo).reshape(FX.shape)
    fill = M._closed(np.c_[FX.ravel(), FY.ravel(), fz.ravel()], np.c_[FX.ravel(), FY.ravel(), np.zeros(FX.size)], FX.shape[0], FX.shape[1])
    head = M.from_manifold(_force(M.to_manifold(head) + M.to_manifold(fill)))
    head = max(head.split(only_watertight=False), key=lambda b: b.volume)
    # the preset's ornaments, as isopod_head places them: on this head's own surface
    zt = lambda x_, y_: float(S["zfun"](np.array([abs(x_)]), np.array([y_ - off]))[0])
    extra = []; Lc = float(S["Lc"]); margin = float(S["margin"]); E = S["eye"]
    if P.get("eyeSolid", 0) > 0.5 and P["eyeSize"] > 0.01 and not E["blind"]:
        x, ye, R = E["xe"], E["ye"] + off, E["eR"]
        EP = HEAD.eye_params(P, R); eye, n_lens = HEAD.eye_solid(**EP)
        if EP["stalk"] > 0.05: z = zt(x, ye) - 0.3 + EP["stalk"]
        else: z = min([zt(x, ye)] + [zt(x + R * np.cos(a), ye + R * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]) - 0.3
        e = eye.copy(); e.apply_translation((x, ye, z)); extra += [e, M.mirror_x(e)]
    y_rear = S["flap"] + off; y_front = float(S["outline"](0.0, 1.0)[1]) + off
    if P["occipitalSpine"] > 0.02:
        yo = y_rear - 0.07 * Lc
        extra.append(spine_solid(0.6 * margin + 0.4, 0.5, P["occipitalSpine"] * Lc, (0, yo, zt(0.0, yo) - 1.0), 0, pitch_deg=55))
    if int(P.get("headProngs", 0)) > 0:
        extra.append(prong(int(P["headProngs"]), P["headProngLen"] * Lc, P["headProngSplay"], P.get("headProngWidth", 0.45) * margin + 0.6,
                           0.45, (0, y_front + 1.5, 0.6 * margin + 0.5), yaw_deg=180.0, pitch_deg=8.0, stem_frac=P.get("headProngStem", 0.0),
                           center_bias=P.get("headProngCenter", 1.0), curl_deg=P.get("headProngCurl", 0.0)))
    return M.union(head, *extra) if extra else head


ROWS_INFO = {}


def build(P, head="isopod", face=None, outline=None, rows=None):
    """head: 'isopod' (the crescent head made on the isopod's head piece, crescent_head.py; the default), 'shaped'
    (a parametric outline, shaped_head.py, with `outline` and `face`), or 'generator' (the classic anatomy head).
    rows (10 Oct 2026): a body spec for the rows model (isopod_model/rows.py: per-row length / width / trajectory,
    fused groups, its own print scale). None = the preset's thorax and tail as before. The spec's "model" picks the
    surface: "thorax" (default) = the generator's own segment and tail plans with the rows' numbers, built exactly
    as the preset is (solid to the bed, plate over lip, the isopod's middle) -- an unedited spec is the preset's
    build; "slab" = the tergite slab, a WALL-thick shell on the joint strip."""
    if rows is not None:
        import rows as RW
        R = RW.plans(P, rows); s = 1.0 / R["scale"]
        T0 = BALL.pose(P, [0.0] * (int(P["segCount"]) + 1))                     # the head and row 0 where placed_plans puts them
        plans = [("head", HEAD, HEAD.plan(P), T0[0][1, 3])] + [(n, m, S, T0[1][1, 3] + off) for (n, m, S, off) in R["parts"]]
        ga = BALL.GAP_AXIAL_FRAC * BALL.ISO_PITCH * s                          # the joint gaps: constant in print mm
        J = dict(gap_axial=ga, gap_vertical=BALL.GAP_VERT_FRAC * BALL.ISO_PITCH * s, gap_lateral=BALL.GAP_LAT_FRAC * BALL.ISO_PITCH * s)
        K = iso_kit(P, s=s); PV = pivots(plans, J, K)
        ROWS_INFO.clear(); ROWS_INFO.update(groups=R["groups"], scale=R["scale"], term_spine=R["term_spine"], rows=R["rows"], model=R["model"])
        return _build(P, head, face, outline, plans, J, K, PV, shell=(R["model"] == "slab"), term_spine=R["term_spine"])
    ROWS_INFO.clear()
    plans = placed_plans(P)
    J, d, zj, y_piv = BALL.geometry(P); K = iso_kit(P); PV = pivots(plans, J, K)
    return _build(P, head, face, outline, plans, J, K, PV)


def _plan_sig(S):
    """What decides a thoracic part's geometry besides its neighbours: its rows (the rows model) or its plan scalars
    (the preset's thorax: width, spine, pitch, heights -- everything else in thorax.plan is P, shared by all)."""
    if S.get("rows") is not None and ROWS_INFO.get("rows"):
        return json.dumps([ROWS_INFO["rows"][i] for i in S["rows"]], sort_keys=True)
    return json.dumps([round(float(S.get(k, 0.0)), 9) for k in ("w", "X_tip", "d", "h", "a", "margin", "rise", "t", "ovl")])


def repeats(plans, parts, head_reach):
    """{k: j}: part k is the same as part j (< k) moved along the axis. Same rows (or plan) as its own, as the part
    ahead (its lip is cut under that plate) and as the part behind (its plate lies over that lip), and neither
    within the head's reach (the horns are cleared against those). The head and the tail are never repeats.
    11 Oct 2026: the editor's "+ row" copies rows, so a body of copies builds its segment once."""
    n = len(plans); sig = ["head"] + [_plan_sig(plans[k][2]) for k in range(1, n - 1)] + ["tail"]
    near = [parts[k].bounds[0][1] <= head_reach for k in range(n)]
    key = {k: (sig[k - 1], sig[k], sig[k + 1], near[k]) for k in range(1, n - 1)}
    seen = {}; out = {}
    for k in range(1, n - 1):
        if key[k] in seen: out[k] = seen[key[k]]
        else: seen[key[k]] = k
    return out


STAGE_SECONDS = {}   # of the last build: {stage: seconds} (for profiling; printed by build.py --time)


HEAD_CACHE = os.environ.get("TRILO_HEAD_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "heads_cache"))   # not under web/cache: app._evict sweeps that
HEAD_CACHE_MAX = 40
_HEAD_SIG = None
def _head_sig():
    """A hash of the head builders' source: a code change never serves a stale head."""
    global _HEAD_SIG
    if _HEAD_SIG is None:
        import hashlib; h = hashlib.sha1(); here = os.path.dirname(os.path.abspath(__file__))
        for f in ("body.py", "shaped_head.py", "crescent_head.py", "head_styles.py", "../anatomy/head_crescent.py", "../anatomy/head.py", "../mesh.py", "../fields.py"):
            try: h.update(open(os.path.join(here, f), "rb").read())
            except OSError: pass
        _HEAD_SIG = h.hexdigest()[:10]
    return _HEAD_SIG


def _head_key(P, head, face, outline, plans, J, K, PV):
    """What the head's mesh depends on: the preset, the head kind, its face and outline, the body's rows (the head is
    sized to the thorax and the shaped head's spines keep clear of the body's reach), the first pivot, the kit scale,
    the gaps, and the builders' source."""
    import hashlib
    if head == "isopod":
        # the crescent head reads the body through two numbers: the first segment's reach across (its width with
        # spines) and the tallest thorax piece; a body edit that changes neither reuses the head
        S1 = plans[1][2]; w_seg = max(abs(S1["outline"](u, v)[0]) for u in np.linspace(-1, 1, 401) for v in (0.0, 0.5, 1.0))
        z_thorax = max(float(M.sample_grid(Sk["outline"], Sk["zfun"], 61, 31)[2].max()) for _, _, Sk, _ in (plans[1:-1] or plans[1:]))
        body = dict(w_seg=round(w_seg, 6), z_thorax=round(z_thorax, 6), head_off=round(float(plans[0][3]), 6))
    else:                                                                   # the shaped head keeps its spines clear of every segment's reach
        rows = ROWS_INFO.get("rows") if ROWS_INFO else None
        body = dict(rows=rows, groups=ROWS_INFO.get("groups") if ROWS_INFO else None, n=len(plans), offsets=[round(float(o), 6) for _, _, _, o in plans])
    blob = json.dumps(dict(P=schema.param_hash(P), head=head, face=face, outline=outline, body=body, pv0=[round(float(v), 6) for v in PV[0]],
                           s=round(float(K["s"]), 9), J={k: round(float(v), 9) for k, v in J.items() if isinstance(v, (int, float))}, sig=_head_sig()), sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _head_cached(P, head, face, outline, plans, J, K, PV):
    """The head and its underside U from the disk cache, or built and stored. 11 Oct 2026: the head is 9 s of a build
    and the editor loop changes only the rows; the same head need not be built twice. For the shaped head the plan
    (functions; cheap) is recomputed and the mesh (the booleans) cached."""
    key = _head_key(P, head, face, outline, plans, J, K, PV); f = os.path.join(HEAD_CACHE, key + ".npz")
    try:
        if os.path.exists(f):
            d = np.load(f, allow_pickle=False)
            H = trimesh.Trimesh(d["v"], d["f"], process=False); H.metadata["notes"] = json.loads(str(d["notes"]))
            if head == "shaped":
                import shaped_head as SH
                plans[0] = ("head", HEAD, SH.plan(P, plans, K, outline, face_of(P, face)), plans[0][3])   # the plan is not cached: it is functions
                return H, None
            X0, Y0, step, hb = float(d["X0"]), float(d["Y0"]), float(d["step"]), d["hb"]
            return H, _underside_fn(X0, Y0, step, hb)
    except Exception as ex:
        print("head cache: ignoring", f, ex, file=sys.stderr)
    if head == "shaped": H = shaped_head(P, plans, PV, K, J, outline, face); U = None; extra = {}
    else: H, U, (X0, Y0, step, hb) = isopod_head(P, plans, J, PV, K, face=face, _grid=True); extra = dict(X0=X0, Y0=Y0, step=step, hb=hb)
    try:
        os.makedirs(HEAD_CACHE, exist_ok=True); tmp = f + ".part.npz"
        np.savez(tmp, v=np.asarray(H.vertices, np.float64), f=np.asarray(H.faces, np.int64), notes=json.dumps(H.metadata.get("notes", [])), **extra); os.replace(tmp, f)
        old = sorted((os.path.getmtime(os.path.join(HEAD_CACHE, g)), g) for g in os.listdir(HEAD_CACHE) if g.endswith(".npz"))
        for _, g in old[:-HEAD_CACHE_MAX]: os.remove(os.path.join(HEAD_CACHE, g))
    except Exception as ex:
        print("head cache: could not store", f, ex, file=sys.stderr)
    return H, U


def _underside_fn(X0, Y0, step, hb):
    """The isopod head's underside as a function (x, y) -> z, +inf where the head is not overhead (bilinear on the ray
    grid; at the head's edge the lowest corner)."""
    nx, ny = hb.shape[1], hb.shape[0]
    def U(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        fx = np.clip((x - X0) / step, 0, nx - 1.001); fy = (y - Y0) / step
        out = np.full(np.broadcast(x, y).shape, np.inf)
        ok = (fy >= 0) & (fy <= ny - 1.001)
        i = fx.astype(int); j = np.clip(fy, 0, ny - 1.001).astype(int); tx = fx - i; ty_ = np.clip(fy, 0, None) - j
        c = [hb[j, i], hb[j, i + 1], hb[j + 1, i], hb[j + 1, i + 1]]
        allf = np.isfinite(c[0]) & np.isfinite(c[1]) & np.isfinite(c[2]) & np.isfinite(c[3])
        with np.errstate(invalid="ignore"):
            bil = (c[0] * (1 - tx) + c[1] * tx) * (1 - ty_) + (c[2] * (1 - tx) + c[3] * tx) * ty_
        low = np.minimum(np.minimum(c[0], c[1]), np.minimum(c[2], c[3]))  # at the head's edge: the lowest corner
        return np.where(ok, np.where(allf, bil, low), out)
    return U


# ---- forked workers (Linux): the plans are closures, so the work is handed over by fork, not by pickling
_POOL_CTX = {}
def _cutter_job(k):
    c = front_cutter(k, _POOL_CTX["plans"], _POOL_CTX["J"], _POOL_CTX["PV"], _POOL_CTX["K"], U=_POOL_CTX["U"] if k == 1 else None)
    return np.asarray(c.vertices, np.float64), np.asarray(c.faces, np.int64)   # (the cutter is never empty: a closed grid solid)

def _man(v, f):
    """a Manifold from vertex / face arrays; an empty one for no faces (a crop that caught nothing)"""
    v = np.asarray(v, np.float32).reshape(-1, 3); f = np.asarray(f, np.uint32).reshape(-1, 3)
    return Manifold(_MMesh(vert_properties=v, tri_verts=f)) if len(f) else Manifold()


def _arrays(man):
    mm = man.to_mesh(); return np.asarray(mm.vert_properties, np.float64).reshape(-1, 3)[:, :3] if len(mm.tri_verts) else np.zeros((0, 3)), np.asarray(mm.tri_verts, np.int64).reshape(-1, 3)


def _clear_job(args):
    """part k (arrays) less its neighbour ahead (arrays, already cropped) dilated by the gaps; the largest body kept."""
    pv, pf, nv, nf, gx, gy, gz = args
    part = _man(pv, pf); near = _man(nv, nf)
    out = _force(part - dilate(near, gx, gy, gz)) if near.num_tri() else part
    out, dropped = _largest(out); v, f = _arrays(out); return v, f, dropped


def _sweep_job(args):
    """part k (arrays) less the dilated head (arrays) at every pose (matrices, part frame), one at a time; largest body kept."""
    pv, pf, hv, hf, mats = args
    m = _man(pv, pf); HM = _man(hv, hf)
    if HM.num_tri():
        for T in mats: m = _force(m - HM.transform(np.asarray(T, float)))
    m, dropped = _largest(m); v, f = _arrays(m); return v, f, dropped


def _largest(man):
    """(the largest body of a manifold, the volume of the others): the flake rule, on a Manifold"""
    bits = man.decompose()
    if len(bits) <= 1: return man, 0.0
    bits = sorted(bits, key=lambda b: -abs(b.volume())); return bits[0], float(sum(abs(b.volume()) for b in bits[1:]))


def _man_bounds(man):
    b = man.bounding_box(); return np.array([[b[0], b[1], b[2]], [b[3], b[4], b[5]]], float)


def _pool(n_jobs):
    """A fork pool for the CUTTER jobs when it helps (several jobs, several cores), else None (inline). Only numpy
    work goes to forked children. The Manifold jobs (the head sweep, the neighbour clearances) run inline: a forked
    child inherits the parent's TBB thread-pool state without its threads, and a large boolean then waits forever
    for workers that do not exist (11 Oct 2026: the giant stress case hung; small booleans run serially and passed);
    a spawned child re-imports the main module (the site), and Manifold does not release the GIL for threads. The
    job functions keep the stages self-contained all the same."""
    if WORKERS < 2 or n_jobs < 2 or os.name != "posix": return None
    import multiprocessing as mp
    try: return mp.get_context("fork").Pool(min(WORKERS, n_jobs))
    except Exception: return None


def _build(P, head, face, outline, plans, J, K, PV, shell=False, term_spine=0.0):
    """The chain. Parts live as Manifold objects between the stages (11 Oct 2026: the trimesh round trips were a
    fifth of the build) and come back as trimesh for the ornaments and the kit."""
    import time as _t; t0 = _t.time(); STAGE_SECONDS.clear()
    def lap(name): STAGE_SECONDS[name] = round(_t.time() - t0 - sum(STAGE_SECONDS.values()), 1)
    U = None; H = None; n = len(plans); _progress(0, 0, n)
    if head in ("isopod", "shaped"): H, U = _head_cached(P, head, face, outline, plans, J, K, PV)
    lap("head")
    parts = []
    for k in range(n):
        parts.append(build_part(k, plans, P, J, PV, U, shell=shell, strip_w=K["w"])); _progress(0, k + 1, n)
    lap("parts")
    rep = repeats(plans, parts, (H.bounds[1][1] + 1.0) if H is not None else -np.inf)
    global LAST_REPEATS; LAST_REPEATS = dict(rep)
    W = [M.to_manifold(m) for m in parts]                                   # the parts, as manifolds, through the stages
    def copy(k):
        """part k as a moved copy of its repeat"""
        j = rep[k]; W[k] = W[j].translate((0, plans[k][3] - plans[j][3], 0))
    # the front cutters: the lip's shape is the expensive part, and it needs only the plans -- forked workers compute
    # the cutter meshes in parallel; the subtractions run here
    todo = [k for k in range(1, n) if k not in rep]
    _POOL_CTX.update(plans=plans, J=J, PV=PV, K=K, U=U); pool = _pool(len(todo))
    try:
        cutters = dict(zip(todo, pool.map(_cutter_job, todo) if pool else map(_cutter_job, todo)))
    finally:
        if pool: pool.close(); pool.join()
    for k in range(1, n):
        if k in rep: copy(k)
        else:
            v, f = cutters.pop(k); W[k] = _force(W[k] - _man(v, f))
        _progress(1, k, n)
    lap("front cutters")
    J["_dropped"] = {}
    def flakes(k):
        W[k], dropped = _largest(W[k])
        if dropped > 0: J["_dropped"][k] = J["_dropped"].get(k, 0.0) + dropped
    ga = J["gap_axial"]
    if head in ("isopod", "shaped"):
        W[0] = M.to_manifold(H)
        # the horns ride back over the first segments: each keeps clear of the head (dilated by the clearances) through
        # the whole curl
        HM = dilate(W[0], 0.45 * ga, 0.45 * ga, GAP_Z)
        # the horns' reach: the posed bounding boxes of every part the loop will clear, at every angle; the dilated
        # head is cropped to that box (what lies beyond it cannot meet a part, so the cut is the same)
        reach_lo = np.full(3, np.inf); reach_hi = np.full(3, -np.inf); in_loop = []
        for k in range(1, n):
            b = _man_bounds(W[k])
            if b[0][1] > H.bounds[1][1] + 1.0: break
            in_loop.append(k); amax = CURL if head == "isopod" else CURL / k
            corners = np.array([[x_, y_, z_, 1.0] for x_ in b[:, 0] for y_ in b[:, 1] for z_ in b[:, 2]])
            for a in np.arange(0.0, amax + 1e-9, CURL_CLEAR_STEP if head == "isopod" else CURL_CLEAR_STEP / k):
                c = (pose(PV, [a] * len(PV))[k] @ corners.T).T[:, :3]
                reach_lo = np.minimum(reach_lo, c.min(0)); reach_hi = np.maximum(reach_hi, c.max(0))
        if in_loop: HM = crop(HM, reach_lo - 2.0, reach_hi + 2.0)
        # A shaped head's spines can run the whole length of the body (harpetid prolongations, trinucleid needles),
        # so this loop reaches the rear segments and the tail. Curled 30 deg at every joint the tail has swung right
        # round to the head, and subtracting the head there cut the tail off (3 Oct 2026: harpes, cryptolithus, ampyx
        # built with a stump for a tail). The tail meeting the head is what STOPS the curl; it is not a clearance to
        # cut. For a shaped head each part is cleared only while it has turned less than CURL in total (k joints x a).
        # The crescent's horns reach three segments and its builds are approved: its angles are unchanged.
        # Each part's sweep is independent of the others' (the head is the same for all): the parts run in parallel.
        hv, hf = _arrays(HM); jobs = []
        for k in in_loop:
            amax = CURL if head == "isopod" else CURL / k
            mats = [np.linalg.inv(pose(PV, [a] * len(PV))[k])[:3, :] for a in np.arange(0.0, amax + 1e-9, CURL_CLEAR_STEP if head == "isopod" else CURL_CLEAR_STEP / k)]
            jobs.append((k, (*_arrays(W[k]), hv, hf, mats)))
        del HM; gc.collect()                                               # the dilated head is large; free it before the sweep
        results = [_sweep_job(a) for _, a in jobs]
        for (k, _), (v, f, dropped) in zip(jobs, results):
            W[k] = _man(v, f)
            if dropped > 0: J["_dropped"][k] = J["_dropped"].get(k, 0.0) + dropped
        gc.collect()
        lap("head clearance")
    else:
        W[0] = M.to_manifold(head_ornaments(P, plans[0][2], parts[0]))
    # every part keeps clear of the part ahead at rest: GAP_Z under it, and a little under half the axial gap beside it
    # (the designed gaps are untouched; this only removes material where the two would otherwise meet: the head's
    # ornaments, or anatomy the plate/lip rules do not describe). Each part is cleared against the part ahead as it
    # stands BEFORE its own clearance (that one removes material at its front, a pitch away from this part), so the
    # parts clear in parallel; a part whose front overlaps the part behind it is the forward-sweep case the manifest
    # flags, and the only one where the order would have mattered.
    g = np.array([0.45 * ga, 0.45 * ga, GAP_Z]) + 1.0; jobs = []
    for k in range(1, n):
        if k in rep: continue
        b = _man_bounds(W[k]); near = crop(W[k - 1], b[0] - g, b[1] + g)   # only what the part ahead has within this part's reach
        jobs.append((k, (*_arrays(W[k]), *_arrays(near), 0.45 * ga, 0.45 * ga, GAP_Z)))
    results = [_clear_job(a) for _, a in jobs]
    for (k, _), (v, f, dropped) in zip(jobs, results):
        W[k] = _man(v, f)
        if dropped > 0: J["_dropped"][k] = J["_dropped"].get(k, 0.0) + dropped
    for k in range(1, n):
        if k in rep: copy(k)
        _progress(2, k, n)
    lap("neighbour clearance")
    parts = [M.from_manifold(w) for w in W]; del W                          # back to trimesh for the ornaments and the kit
    def copy(k):
        j = rep[k]; parts[k] = parts[j].copy().apply_translation((0, plans[k][3] - plans[j][3], 0))
    # the spines go on last: the clearances above shape the plates and lips, they never cut a spine. A spine that meets
    # another part through the curl is what stops the curl (the enrollment test finds it), not something to trim.
    SKIPPED.clear(); body_ornaments(P, plans, parts, skip=set(rep))
    for k in rep: copy(k)                                                  # the ornaments of a repeat are its source's
    lap("ornaments")
    if term_spine > 0.05:                                                 # the rows model's terminal spine, from the tail's tip, along the axis
        name, mod, S, off = plans[-1]; y_tip = off + float(S["outline"](0.0, 1.0)[1]); zt = _surface_z(parts[-1], 0.0, y_tip - 2.0, search=3.0) or S["margin"]
        parts[-1] = M.union(parts[-1], spine_solid(0.6 * S["margin"] + 0.6, 0.5, term_spine, (0, y_tip - 1.5, max(zt - 1.0, 1.2)), 0, pitch_deg=0))
    gc.collect()                                                          # release the per-part clearance intermediates before the middle paste
    paste_middle(parts, plans, PV, K, skip=set(rep))
    for k in rep: copy(k)
    relieve_head_joint(parts, PV)
    _progress(3, 0, n)                                                     # = every part built
    lap("middle paste")
    global LAST_DROPPED; LAST_DROPPED = dict(J["_dropped"])                # flakes cut off by the clearances (model mm3)
    for k, v in clean_parts(parts, plans).items(): LAST_DROPPED[k] = LAST_DROPPED.get(k, 0.0) + v   # debris after the paste (11 Oct 2026)
    validate_parts(parts, plans)                                          # raises MeshError: nothing invalid leaves here
    lap("clean + validate"); gc.collect()
    return parts, plans, PV, K


def relieve_head_joint(parts, PV, to_deg=None, step_deg=1.0):
    """Let the cephalon joint J1 turn to HEAD_JOINT_DEG: cut from the HEAD only the material seg0 (kit included) sweeps through
    when J1 turns from CURL to to_deg with every other joint at rest. Nothing is cut at or below CURL, so the clearances that
    hold for every joint still hold; seg0 is untouched. Returns the volume removed (model mm3)."""
    to_deg = HEAD_JOINT_DEG if to_deg is None else to_deg
    if len(parts) < 2 or to_deg <= CURL + 1e-9: return 0.0
    Wh = M.to_manifold(parts[0]); W1 = M.to_manifold(parts[1]); swept = None
    for a in np.arange(CURL, to_deg + 1e-9, step_deg):
        T = pose(PV, [float(a)] + [0.0] * (len(PV) - 1))                  # head frame is the rest frame: seg0 moves
        inter = Wh ^ W1.transform(T[1][:3, :])
        if inter.volume() > 1e-6: swept = inter if swept is None else (swept + inter)
    if swept is None: return 0.0
    v = float(swept.volume()); parts[0] = M.from_manifold(Wh - swept)
    return v


class MeshError(ValueError):
    """A part came out of the build invalid (not watertight, several bodies, non-finite). The site refuses the build."""


DEBRIS_VOL = M.SLIVER_MM3   # model mm3 (2.0): a body smaller than this is boolean debris, not anatomy -- mesh.py's own threshold, so the
                            # end-of-build clean drops exactly what the per-step _force used to
DEBRIS_EXT = 0.5     # model mm: or thinner than this on its shortest axis (a two-face sliver)


def clean_parts(parts, plans, min_vol=DEBRIS_VOL, min_ext=DEBRIS_EXT):
    """Drop debris bodies from every part (the booleans after the clearance pass -- ornaments, the middle paste --
    can leave two-face slivers and specks: trilobite_bc19137c56's export carried two). Returns {part index:
    volume dropped}. A part keeps every body that is not debris; the validator then decides whether that is one."""
    dropped = {}
    for k, m in enumerate(parts):
        bits = m.split(only_watertight=False)
        if len(bits) <= 1: continue
        keep = [b for b in bits if abs(b.volume) >= min_vol and b.extents.min() >= min_ext]
        if not keep: keep = [max(bits, key=lambda b: abs(b.volume))]
        if len(keep) < len(bits):
            dropped[k] = float(sum(abs(b.volume) for b in bits) - sum(abs(b.volume) for b in keep))
            parts[k] = keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)
    return dropped


def validate_parts(parts, plans):
    """Every part: finite, watertight, consistently wound, positive volume, ONE body. Else MeshError naming the part."""
    for (name, *_), m in zip(plans, parts):
        if not np.isfinite(m.vertices).all(): raise MeshError(f"{name}: non-finite vertices")
        n = len(m.split(only_watertight=False))
        if n != 1: raise MeshError(f"{name}: {n} bodies after cleaning (should be one)")
        if not m.is_watertight: raise MeshError(f"{name}: not watertight")
        if not m.is_winding_consistent: raise MeshError(f"{name}: inconsistent winding")
        if m.volume <= 0: raise MeshError(f"{name}: volume {m.volume:.2f} mm3")
        if len(m.faces) > 2_000_000: raise MeshError(f"{name}: {len(m.faces):,} faces (runaway build)")


LAST_DROPPED = {}
LAST_REPEATS = {}    # {part index: the part it was copied from} in the last build

# progress (11 Oct 2026): the site's loading count. _build reports (done, total) in PARTS, total = the number of
# pieces (head + the printed body pieces + tail); a part counts as built in thirds: drawn (build_part), cleared
# (the front cutter and the neighbour clearances), finished (ornaments and the middle paste). Set to a callable.
PROGRESS_HOOK = None
def _progress(stage, k, n):
    """stage 0..2 complete for parts 0..k-1 of n -> done parts"""
    if PROGRESS_HOOK is None: return
    try: PROGRESS_HOOK(int((stage * n + k) / 3), n)
    except Exception: pass


def dilate(m, gx, gy, gz):
    """m (trimesh or Manifold) grown by gx, gy sideways and gz downward (the union of shifted copies): what a
    neighbour keeps clear of."""
    sh = [(0, 0, 0), (gx, 0, 0), (-gx, 0, 0), (0, gy, 0), (0, -gy, 0), (0, 0, -gz)]
    man = m if isinstance(m, Manifold) else M.to_manifold(m)
    return Manifold.batch_boolean([man.translate(t) for t in sh], OpType.Add)


def crop(m, lo, hi):
    """m (trimesh or Manifold) cut to the box [lo, hi] (model mm): the part of a neighbour that can reach another
    part. 11 Oct 2026: the clearances subtracted the WHOLE neighbour (the head: 194k faces, dilated six-fold) from
    every part; cropped to the other part's reach the result is identical and the booleans a fraction of the size."""
    man = m if isinstance(m, Manifold) else M.to_manifold(m)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return _force(man ^ M.to_manifold(M.box(*(hi - lo), at=tuple(lo), align=("min", "min", "min"))))


def pose(PV, angles):
    """Rest -> posed transform of every part, each joint curled by its angle about its pivot (the part behind down)."""
    T = [np.eye(4)]
    for (yc, zc), a in zip(PV, angles):
        T.append(T[-1] @ trimesh.transformations.rotation_matrix(math.radians(-a), (1, 0, 0), (0, yc, zc)))
    return T


def curl_check(parts, PV, angles):
    """Largest overlap between any part and its two neighbours behind, every joint curled by each angle, mm^3."""
    out = []
    W = [M.to_manifold(p) for p in parts]
    for a in angles:
        T = pose(PV, [a] * len(PV)); worst = 0.0
        for k in range(len(parts)):
            for j in range(k + 1, len(parts) if k == 0 else min(len(parts), k + 3)):
                R = np.linalg.inv(T[k]) @ T[j]                              # part j in part k's rest frame
                worst = max(worst, float((W[k] ^ M.to_manifold(parts[j].copy().apply_transform(R))).volume()))
        out.append((a, round(worst, 3)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default="out"); ap.add_argument("--preset", default=None)
    ap.add_argument("--head", default="isopod", choices=("generator", "isopod"))
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    P = schema.preset(A.preset) if A.preset else schema.defaults()
    parts, plans, PV, K = build(P, A.head)
    for (name, *_), p in zip(plans, parts):
        nb = len(p.split(only_watertight=False))
        print(f"{name}: {len(p.faces)} faces, watertight {p.is_watertight}, pieces {nb}")
    print("curl check (angle, worst overlap mm3):", curl_check(parts, PV, [0.0, 10.0, 20.0, CURL]))
    whole = trimesh.util.concatenate(parts); whole.apply_scale(1.0 / K["s"])
    name = "iso_body.stl" if A.head == "generator" else f"iso_body_{A.head}_head.stl"
    whole.export(os.path.join(A.out, name)); print(os.path.join(A.out, name), whole.bounds.round(1).tolist())
