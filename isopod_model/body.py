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
import os, sys, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, trimesh
import schema, mesh as M
from anatomy import head as HEAD, thorax as THORAX, tail as TAIL
from anatomy.common import spine_solid, prong
from joints import ball as BALL
from manifold3d import Manifold, OpType

WALL = 2.0          # plate thickness behind a part's rear joint plane (model mm)
GAP_Z = 0.5         # least air gap between a plate and the lip under it, through the curl (model mm)
CURL = 30.0         # the chain curls this far per joint (deg) without touching
LIP_MIN = 0.8       # thinnest lip kept (model mm)
GRID = {"head": (241, 121), "seg": (161, 61), "tail": (241, 121)}

# the isopod's middle-bottom (asset mm, the asset's work frame: midline x = 0, bed z = 0)
ISO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "isopod", "giant_isopod_main_itbefred.stl")
ISO_PIECE = 3       # the segment piece copied (ball at its front, socket at its rear)
ISO_W = 15.0        # the strip's half-width: the middle block (|x| <= 12.4) and the fork, cut in the empty slot beside
ISO_ZC = 20.0       # the strip's top: ball, neck, socket and fork are all below 13.5
ISO_YM = 91.0       # where piece 3's strip is split into its ball half and socket half (its column is prismatic here)
ISO_OVER = 0.4      # the halves overlap by this much past the split (the column is prismatic there)
ISO_PITCH = 18.0    # joints/ball.py's ISO_PITCH: the build is exported at 18 / pitch, so s = pitch / 18 below


def _iso_frame():
    """The copied isopod pieces (piece ISO_PIECE and the next) and the two ball centres, in the asset's work frame."""
    m = trimesh.load(ISO); z0 = m.bounds[0][2]; xc = 0.5 * (m.bounds[0][0] + m.bounds[1][0])
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


def iso_kit(P):
    """The isopod strip halves, scaled to the model (s = pitch / 18), each in its own joint frame (the ball centre at
    y = 0, z at scale): 'ball' = piece's front half (ball, neck, column front), 'socket' = its rear half (column rear,
    socket, fork). Plus the joint numbers in model mm."""
    A, cf, cr, rear, front = _iso_frame()
    s = 1.0 / BALL.model_scale(P)                                          # = pitch / 18
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
        U = lambda x_, y_: np.where(y_ <= y_end, zp_top(x_, y_) - WALL, np.inf)
    under = np.isfinite(U(x, y))
    def ok(z):
        good = np.ones_like(z, bool)
        for a in np.linspace(0.0, CURL, 11):
            ya, za = _turn(y, z, pv, a)
            good &= za <= U(x, ya) - GAP_Z                                  # +inf: not under the plate
        return good
    hi = top.copy(); lo = np.full_like(top, pv[1] - 5.0)
    fine = ok(hi); lim = np.where(fine, hi, lo)
    for _ in range(24):
        mid = 0.5 * (lo + hi); g = ok(mid)
        lo = np.where(g, mid, lo); hi = np.where(g, hi, mid)
    lim = np.where(fine, top, lo)
    return np.where(under, np.minimum(top, lim), top)


def build_part(k, plans, P, J, PV, U=None):
    """Part k as a height-field solid: top = part_top, bottom = the bed, a WALL-thick plate behind its rear joint
    plane."""
    name, mod, S, off = plans[k]
    kind = "head" if name == "head" else ("tail" if name == "tail" else "seg")
    nu, nv = GRID[kind]
    xs, ys, zs = M.sample_grid(S["outline"], S["zfun"], nu, nv)          # the generator's own smoothing
    y = ys + off; ga = J["gap_axial"]
    top = part_top(k, plans, J, PV, xs, y, zs, U if k == 1 else None)
    bot = np.zeros_like(top)
    if k < len(plans) - 1:
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
    X, Y = np.meshgrid(np.arange(-W, W + 1e-9, step), np.linspace(y0, y1, int((y1 - y0) / step) + 2))
    t = part_top(k, plans, J, PV, X, Y, None, U if k == 1 else None) - zp
    Z = zp + np.clip((LIP_MIN - t) * 20.0, 0.0, 60.0)
    T = np.c_[X.ravel(), Y.ravel(), Z.ravel()]; B = np.c_[X.ravel(), Y.ravel(), np.full(X.size, -1.0)]
    return M._closed(T, B, X.shape[0], X.shape[1])


def paste_middle(parts, plans, PV, K):
    """The isopod's middle-bottom into every part: our strip |x| <= w below zc (ahead of the plate) goes, the isopod's
    goes in: the socket half at each part's rear joint, the ball half at its front joint."""
    n = len(plans)
    for k in range(n):
        m = M.to_manifold(parts[k])
        y_from = PV[k - 1][0] + K["ball_start"] if k > 0 else None
        y_to = PV[k][0] + K["col_rear"] if k < n - 1 else None             # = the part's rear face: the plate stays
        if y_from is None: y_from = PV[k][0] + K["sock_start"]              # head: only its socket half
        if y_to is None: y_to = PV[k - 1][0] + K["ball_end"]                # tail: only its ball half
        m = m - M.to_manifold(M.box(2 * K["w"], y_to - y_from, K["zc"] + 1.0, at=(0, y_from, -1.0), align=("c", "min", "min")))
        if k > 0: m = m + M.to_manifold(K["ball"].copy().apply_translation((0, PV[k - 1][0], 0)))
        if k < n - 1: m = m + M.to_manifold(K["socket"].copy().apply_translation((0, PV[k][0], 0)))
        parts[k] = M.from_manifold(BALL._force(m))


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


def face_of(P):
    """The crescent head's face from the generator's own parameters (head_crescent.FACE was set from the default
    preset's values, so the default preset gives the approved face exactly)."""
    return dict(eye_size=float(P["eyeSize"]), eye_pos=float(P["eyePos"]), eye_height=float(P["eyeHeight"]),
                eye_lat=float(P["eyeLat"]), glab_inflate=float(P["glabInflate"]), glab_rise=float(P["glabRise"]))


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


def isopod_head(P, plans, J, PV, K, step=0.25):
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
    head, info = head_piece(face_of(P))                                     # work frame, isopod size
    iso = trimesh.load(ISO); z0 = iso.bounds[0][2]; xc = 0.5 * (iso.bounds[0][0] + iso.bounds[1][0])
    iso.apply_translation((-xc, 0, -z0))
    pieces = sorted(iso.split(only_watertight=False), key=lambda p: p.bounds[0][1])
    loc, _, _ = pieces[0].ray.intersects_location([(8.0, 0.0, 3.0)], [(0, 1.0, 0)])
    y_rear = np.asarray(loc).reshape(-1, 3)[:, 1].max()                    # the head column's rear face, 8 mm off axis
    s = K["s"]; S1 = plans[1][2]
    w_seg = max(abs(S1["outline"](u, v)[0]) for u in np.linspace(-1, 1, 401) for v in (0.0, 0.5, 1.0)) / s
    f = w_seg / pieces[1].bounds[1][0]; fxy = f * s
    z_thorax = max(float(M.sample_grid(Sk["outline"], Sk["zfun"], 61, 31)[2].max()) for _, _, Sk, _ in plans[1:-1])
    fz = HEAD_HEIGHT * z_thorax * float(P.get("headRelief", 1.0)) / head.bounds[1][2]
    yr = PV[0][0] + K["col_rear"]                                           # our head's rear face (model)
    ty = yr - fxy * y_rear
    head.apply_transform(np.diag([fxy, fxy, fz, 1.0])); head.apply_translation((0, ty, 0))
    # the dome's curve from the preset (headDomeExp, the generator's dome exponent; 1.5 = the approved head): heights
    # above the rim are remapped rim + (top - rim) t^(1.5 / exp), t = (z - rim) / (top - rim). A larger exponent fills
    # the dome out (flatter crown, steeper flanks), a smaller one peaks it. The rim and everything at rim level (the
    # horns) and the crown height are unchanged.
    g = 1.5 / float(P.get("headDomeExp", 1.5))
    if abs(g - 1.0) > 1e-3:
        V = head.vertices.copy(); top = V[:, 2].max()
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
    H = BALL._force(H + M.to_manifold(fill))
    head = max(M.from_manifold(H).split(only_watertight=False), key=lambda b: b.volume)
    # behind its rear face the head is a plate, as every other part is: the isopod head piece's cheek walls reach down
    # to the bed behind its column (they wrap the isopod's first piece), which is where our first segment's column
    # stands. Cut everything behind the rear face lower than WALL under the head's own top; the horns (thinner than
    # WALL at rim level) stay.
    X = np.arange(head.bounds[0][0] - step, head.bounds[1][0] + step, step); Y = np.arange(yr + 0.05, head.bounds[1][1] + step, step)
    GX, GY = np.meshgrid(X, Y); o = np.c_[GX.ravel(), GY.ravel(), np.full(GX.size, 1e3)]
    locs, ri, _ = head.ray.intersects_location(o, np.tile((0, 0, -1.0), (len(o), 1)), multiple_hits=True)
    ht = np.full(len(o), -np.inf); np.maximum.at(ht, ri, np.asarray(locs)[:, 2]); ht = ht.reshape(GX.shape)
    cz = np.where(np.isfinite(ht), ht - WALL, -0.9)
    cut = M._closed(np.c_[GX.ravel(), GY.ravel(), np.maximum(cz, -0.9).ravel()], np.c_[GX.ravel(), GY.ravel(), np.full(GX.size, -1.0)], GX.shape[0], GX.shape[1])
    head = M.from_manifold(BALL._force(M.to_manifold(head) - M.to_manifold(cut)))
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
    X = np.arange(-head.bounds[1][0] - step, head.bounds[1][0] + step, step); Y = np.arange(yr - 0.5, head.bounds[1][1] + step, step)
    GX, GY = np.meshgrid(X, Y); o = np.c_[GX.ravel(), GY.ravel(), np.full(GX.size, -1.0)]
    locs, ri, _ = head.ray.intersects_location(o, np.tile((0, 0, 1.0), (len(o), 1)), multiple_hits=True)
    hb = np.full(len(o), np.inf); np.minimum.at(hb, ri, np.asarray(locs)[:, 2]); hb = hb.reshape(GX.shape)
    def U(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        fx = np.clip((x - X[0]) / step, 0, len(X) - 1.001); fy = (y - Y[0]) / step
        out = np.full(np.broadcast(x, y).shape, np.inf)
        ok = (fy >= 0) & (fy <= len(Y) - 1.001)
        i = fx.astype(int); j = np.clip(fy, 0, len(Y) - 1.001).astype(int); tx = fx - i; ty_ = np.clip(fy, 0, None) - j
        c = [hb[j, i], hb[j, i + 1], hb[j + 1, i], hb[j + 1, i + 1]]
        allf = np.isfinite(c[0]) & np.isfinite(c[1]) & np.isfinite(c[2]) & np.isfinite(c[3])
        with np.errstate(invalid="ignore"):
            bil = (c[0] * (1 - tx) + c[1] * tx) * (1 - ty_) + (c[2] * (1 - tx) + c[3] * tx) * ty_
        low = np.minimum(np.minimum(c[0], c[1]), np.minimum(c[2], c[3]))  # at the head's edge: the lowest corner
        return np.where(ok, np.where(allf, bil, low), out)
    return head, U


SKIPPED = []


def body_ornaments(P, plans, parts):
    """The preset's spines on the body, as the generator puts them: each thorax ring's axial spine, the tail's
    terminal spine and prongs (plan frame, moved to the part's rest placement). An axial spine stands where the ring's
    top is open to the sky: where the generator's spot lies under the plate of the part ahead (the lip), it moves back
    along the axis to the first open point and stands on the part's actual top there."""
    for k, (name, mod, S, off) in enumerate(plans):
        if name == "head": continue
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


def build(P, head="isopod"):
    """head: 'isopod' (the crescent head made on the isopod's head piece, crescent_head.py; the default) or
    'generator' (the anatomy head, drawn like the rest)."""
    plans = placed_plans(P)
    J, d, zj, y_piv = BALL.geometry(P); K = iso_kit(P); PV = pivots(plans, J, K)
    U = None
    if head == "isopod": H, U = isopod_head(P, plans, J, PV, K)
    parts = [build_part(k, plans, P, J, PV, U) for k in range(len(plans))]
    for k in range(1, len(plans)):
        parts[k] = M.from_manifold(BALL._force(M.to_manifold(parts[k]) - M.to_manifold(front_cutter(k, plans, J, PV, K, U=U))))
    J["_dropped"] = {}
    def flakes(k):
        bits = sorted(parts[k].split(only_watertight=False), key=lambda m: -m.volume); parts[k] = bits[0]   # no flakes
        if len(bits) > 1: J["_dropped"][k] = J["_dropped"].get(k, 0.0) + float(sum(b.volume for b in bits[1:]))
    ga = J["gap_axial"]
    if head == "isopod":
        parts[0] = H
        # the horns ride back over the first segments: each keeps clear of the head (dilated by the clearances) through
        # the whole curl
        HM = dilate(H, 0.45 * ga, 0.45 * ga, GAP_Z)
        for k in range(1, len(plans)):
            if parts[k].bounds[0][1] > H.bounds[1][1] + 1.0: break
            m = M.to_manifold(parts[k])
            for a in np.arange(0.0, CURL + 1e-9, 1.5):                      # one pose at a time (memory)
                m = BALL._force(m - HM.transform(np.linalg.inv(pose(PV, [a] * len(PV))[k])[:3, :]))
            parts[k] = M.from_manifold(m); flakes(k)
    else:
        parts[0] = head_ornaments(P, plans[0][2], parts[0])
    # every part keeps clear of the part ahead at rest: GAP_Z under it, and a little under half the axial gap beside it
    # (the designed gaps are untouched; this only removes material where the two would otherwise meet: the head's
    # ornaments, or anatomy the plate/lip rules do not describe)
    for k in range(1, len(plans)):
        parts[k] = M.from_manifold(BALL._force(M.to_manifold(parts[k]) - dilate(parts[k - 1], 0.45 * ga, 0.45 * ga, GAP_Z))); flakes(k)
    # the spines go on last: the clearances above shape the plates and lips, they never cut a spine. A spine that meets
    # another part through the curl is what stops the curl (the enrollment test finds it), not something to trim.
    SKIPPED.clear(); body_ornaments(P, plans, parts)
    paste_middle(parts, plans, PV, K)
    global LAST_DROPPED; LAST_DROPPED = dict(J["_dropped"])                # flakes cut off by the clearances (model mm3)
    return parts, plans, PV, K


LAST_DROPPED = {}


def dilate(m, gx, gy, gz):
    """m grown by gx, gy sideways and gz downward (the union of shifted copies): what a neighbour keeps clear of."""
    sh = [(0, 0, 0), (gx, 0, 0), (-gx, 0, 0), (0, gy, 0), (0, -gy, 0), (0, 0, -gz)]
    return Manifold.batch_boolean([M.to_manifold(m.copy().apply_translation(t)) for t in sh], OpType.Add)


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
