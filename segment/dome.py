"""
dome.py -- ONE body piece of the isopod-style print: a solid dome-shaped shingle with the print-in-place
ball-and-socket joint built in. Defaults are the generator's isopod-model piece (trilobite_bc19137c56, piece 5):
18 mm pitch, 28 mm plate, 140 mm wide, 37 mm crest, the isopod's 11.5 mm ball on a neck at the front and its
socket at the rear, solid to the bed except the rear plate that overhangs the next piece's lip.

    python dome.py                                   -> out/dome.stl + out/dome.png (+ dome.json: the numbers used)
    python dome.py --copies 5 --taper 0.955          -> a five-piece chain, balls printed inside the sockets (print in place)
    python dome.py --crest 30 --width 60             -> smaller animal, same joint
    python dome.py --list                            -> every parameter with its meaning and default

Frame: x across (0 on the midline, mirror-symmetric), y along (0 = the ball's centre = this piece's pivot, + rearward),
z up from the bed. Millimetres, print size. The chain pivots at every ball centre; copy k sits k * pitch behind.

The piece, front to back:
  ball       the isopod's ball (r ball_r, centre ball_z above the bed) at y = 0, with the `slit` down the midline that
             lets it give. It prints inside the socket of the piece ahead with `clearance` all round.
  neck       a bar from the ball back into the body, a 45-deg wedge under it to the bed
  lip        the front `overlap` mm of the body: its top is dropped wall + `air` (+ the rise it makes when the joint
             extends) so it tucks under the plate of the piece ahead. The lip is solid to the bed, except that over the
             midline strip it arches over the socket lobe of the piece ahead.
  body       solid, to the bed, up to the dome top, back to the V-gap in front of the next pivot.
  plate      the rear `overlap` mm of the dome, `wall` thick, overhanging the next piece's lip.
  socket     at the rear of the body on the midline, a lobe (a sphere R_sock about the next ball's centre) holding
             the spherical cavity, open to the bed and rearward through a slot the next piece's neck swings in.
  V-gap      body rear and the next lip's front face a gap of `gap`, V-shaped about the pivot height: leaning extend/2
             above the pivot and enrol/2 below it, so neighbours turn without touching.
  top        `crest` high on the midline, a bell across (`profile`) down to `edge_h` at the rim, arched along (`arch`),
             the whole plan swept back `sweep` deg toward the tips. Behind the lip the top rises to the dome over `step`.
Swing: enrolment (tail down) swings the lip down and away, so it is limited by the V below the pivot and the neck in
its slot; extension (tail up) lifts the lip into the plate ahead and shifts the step forward under its rear edge, so it
is budgeted separately and small: overlap + gap + step + 2 (crest - ball_z) sin(extend) <= pitch.
Every build reports: watertight, one body, bed contact, overhang, and (for copies) the boolean intersection between
neighbours at rest, the swing each way before neighbours touch, and the ball-to-socket clearance actually achieved.
"""
import argparse, json, math, os, sys
import numpy as np
import trimesh

PARAMS = [  # (name, default, unit, meaning)
    ("pitch",      18.0, "mm",   "joint to joint along the axis (the isopod's 18)"),
    ("overlap",    10.0, "mm",   "how far the plate reaches past one pitch (= the lip the next piece tucks under it). Budget: overlap + gap + step + 2 (crest - ball_z) sin(extend) <= pitch"),
    ("width",      70.0, "mm",   "half-width at the rim"),
    ("sweep",      14.0, "deg",  "the plan sweeps back toward the tips by this angle"),
    ("curve",       6.0, "deg",  "extra backward bend over the outer half"),
    ("crest",      37.0, "mm",   "dome height on the midline"),
    ("edge_h",     10.0, "mm",   "dome height at the rim"),
    ("profile",     1.5, "-",    "bell exponent across: z = edge + (crest - edge) (1 - (|x|/width)^profile); 1 cone, 2 parabola"),
    ("arch",        0.06, "frac", "the dome also arches along y: this fraction lower at the plate's front and rear"),
    ("wall",        3.0, "mm",   "thickness of the rear plate (the part overhanging the next lip)"),
    ("air",         1.4, "mm",   "air between the lip and the plate above it, at rest"),
    ("step",        2.5, "mm",   "the lip rises to the full dome over this length"),
    ("col_w",       8.0, "mm",   "half-width of the midline strip that arches over the socket lobe ahead"),
    ("ball_r",      5.75, "mm",  "ball radius (the isopod's)"),
    ("ball_z",      7.5, "mm",   "ball centre above the bed (the isopod's)"),
    ("neck",        5.0, "mm",   "neck bar: width and height"),
    ("clearance",   0.25, "mm",  "ball to socket, neck to slot (the isopod's 0.244)"),
    ("slit",        1.0, "mm",   "slit down the midline of the ball and neck (0 none)"),
    ("gap",         1.0, "mm",   "the V-gap between a body's rear and the next lip's front, at the pivot height"),
    ("enrol",      30.0, "deg",  "ventral swing wanted at each joint (tail down: the enrolment direction)"),
    ("extend",      4.0, "deg",  "dorsal swing wanted at each joint (tail up); costs overlap (see overlap), keep it small"),
    ("copies",      1,   "n",    "how many pieces, one pitch apart, balls inside sockets"),
    ("taper",       0.955, "frac", "each copy's width x this"),
    ("grid",      161,   "n",    "grid columns across (odd)"),
]
DEFAULTS = {n: d for n, d, _, _ in PARAMS}


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------------------------------------- plan
def plan(p):
    """outline(u, v) -> (x, y) and the scalars the surfaces need. The plan is a band L = pitch + overlap long, swept
    back; its front edge is the lip's front face (gap/2 + the upper V lean behind the pivot plane; the lower V's
    foot at the bed is handled by the underside)."""
    w, L = p["width"], p["pitch"] + p["overlap"]; sw, cv = math.radians(p["sweep"]), math.radians(p["curve"])
    R_sock = p["ball_r"] + p["clearance"] + p["wall"]
    tan_u, tan_d = math.tan(math.radians(0.5 * p["extend"])), math.tan(math.radians(0.5 * p["enrol"]))
    front = 0.5 * p["gap"] + (p["crest"] - p["ball_z"]) * tan_u            # the lip's front face (its top corner is the limit)
    def back(ax):
        q = np.clip(np.abs(ax) / w, 0, 1); return np.tan(sw) * q * w + np.tan(cv) * np.clip(q - 0.5, 0, 1) ** 2 * w
    def edges(ax):
        ax = np.asarray(ax, float); b = back(ax); yf = b + front; yr = b + L
        r = np.clip((ax / w - 0.9) / 0.1, 0, 1); shrink = 0.5 * (yr - yf) * (1 - np.sqrt(np.clip(1 - r * r, 0, 1)))
        return yf + shrink, np.maximum(yr - shrink, yf + shrink + 0.3)
    def outline(u, v):
        x = u * w; yf, yr = edges(np.array([abs(x)])); return x, float(yf[0] + v * (yr[0] - yf[0]))
    return dict(outline=outline, edges=edges, back=back, w=w, L=L, front=front, R_sock=R_sock, tan_u=tan_u, tan_d=tan_d,
                y_lip_end=p["overlap"] + p["gap"] + (p["crest"] - p["ball_z"]) * math.sin(math.radians(p["extend"])))


def surface(p, S):
    """z(x, y): the top. The dome, dropped on the lip (front), rising over `step` behind it."""
    w, L = S["w"], S["L"]
    def dome(ax, y):
        q = np.clip(ax / w, 0, 1); b = S["back"](ax); t = np.clip((y - b) / L, 0, 1)
        z = p["edge_h"] + (p["crest"] - p["edge_h"]) * (1 - q ** p["profile"])
        return z * (1 - p["arch"] * (2 * t - 1) ** 2)
    def z(x, y):
        ax = np.abs(np.asarray(x, float)); y = np.asarray(y, float); b = S["back"](ax)
        full = dome(ax, y)
        # the lip sits under the plate AHEAD, whose top here is the dome one pitch further along its own length
        above = dome(ax, y + p["pitch"])
        lip = np.maximum(np.minimum(full, above) - p["wall"] - p["air"] - np.maximum(y, 0) * math.sin(math.radians(p["extend"])), 1.5)
        s = smoothstep(b + S["y_lip_end"], b + S["y_lip_end"] + p["step"], y)
        return lip * (1 - s) + full * s
    return z, dome


def underside(p, S, xs, ys, zs):
    """b(x, y): the bed (solid), except: the rear plate (`wall` under the top, measured along the normal, capped,
    from the V in front of the next pivot to the rear edge); the lower-V undercuts at the lip's front and the body's
    rear foot (leaning enrol/2 below the pivot height); the arch over the socket lobe ahead, on the midline strip."""
    ax = np.abs(xs); b = S["back"](ax); zc = p["ball_z"]
    dx = float(np.abs(np.diff(xs, axis=1)).mean()); dy = float(np.abs(np.diff(ys, axis=0)).mean())
    gy, gx = np.gradient(zs, dy, dx); k = np.minimum(np.sqrt(1 + gx ** 2 + gy ** 2), 2.5)
    plate = np.maximum(zs - np.maximum(p["wall"] * k, 1.0), 0.0)
    yl = ys - b                                                             # distance behind the pivot plane, in this piece's sweep frame
    bot = np.zeros_like(zs)
    # the rear plate: from the V-gap's apex line (gap/2 ahead of the next pivot, plus the upper lean up to the plate's underside)
    y_V = p["pitch"] - 0.5 * p["gap"] - np.maximum(plate - zc, 0) * S["tan_u"]
    bot = np.where(yl > y_V, plate, bot)
    # the lower V: the body's rear foot is set back zc tan(enrol/2) from the apex; the lip's front foot is set back the same
    foot = zc * S["tan_d"]
    rear_apex = p["pitch"] - 0.5 * p["gap"]
    under_rear = np.clip((yl - (rear_apex - foot)) / max(foot, 1e-6), 0, 1) * zc
    bot = np.where((yl > rear_apex - foot) & (yl <= y_V), np.maximum(bot, under_rear), bot)
    front_apex = 0.5 * p["gap"]
    under_front = np.clip(1 - (yl - front_apex) / max(foot, 1e-6), 0, 1) * zc
    bot = np.where(yl < front_apex + foot, np.maximum(bot, under_front), bot)
    # the arch over the socket lobe of the piece ahead (midline strip): outside a sphere R_sock + gap about this ball
    R = S["R_sock"] + p["gap"]
    arch = zc + np.sqrt(np.clip(R ** 2 - yl ** 2, 0, None))
    on_strip = (ax < p["col_w"] + 1.0) & (yl < R)
    bot = np.where(on_strip, np.maximum(bot, arch), bot)
    return np.minimum(bot, np.maximum(zs - 1.0, 0.0))


def closed_plate(p, S, z, nu):
    if nu % 2 == 0: nu += 1
    nv = max(81, int(S["L"] / 0.3) | 1); mid = nu // 2; xs = np.zeros((nv, nu)); ys = np.zeros((nv, nu))
    for j in range(nv):
        v = j / (nv - 1)
        for i in range(mid, nu):
            u = -1 + 2 * i / (nu - 1); x, y = S["outline"](u, v); xs[j, i] = x; ys[j, i] = y
    xs[:, :mid] = -xs[:, nu - 1:mid:-1]; ys[:, :mid] = ys[:, nu - 1:mid:-1]; xs[:, mid] = 0.0
    zs = z(np.abs(xs), ys); bs = underside(p, S, xs, ys, zs)
    T = np.c_[xs.ravel(), ys.ravel(), zs.ravel()]; B = np.c_[xs.ravel(), ys.ravel(), bs.ravel()]
    n = xs.size; V = np.vstack([T, B]); F = []; idx = lambda j, i: j * nu + i
    for j in range(nv - 1):
        for i in range(nu - 1):
            A_, B_, C_, D_ = idx(j, i), idx(j, i + 1), idx(j + 1, i), idx(j + 1, i + 1)
            F += [[A_, B_, D_], [A_, D_, C_], [n + A_, n + D_, n + B_], [n + A_, n + C_, n + D_]]
    loop = [idx(0, i) for i in range(nu)] + [idx(j, nu - 1) for j in range(1, nv)] + [idx(nv - 1, i) for i in range(nu - 2, -1, -1)] + [idx(j, 0) for j in range(nv - 2, 0, -1)]
    for k in range(len(loop)):
        a_, b_ = loop[k], loop[(k + 1) % len(loop)]; F += [[a_, n + a_, b_], [b_, n + a_, n + b_]]
    m = trimesh.Trimesh(V, np.array(F), process=True); m.merge_vertices(); m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m); m.metadata["grid"] = dict(xs=xs, ys=ys, zs=zs, bs=bs)
    return m


# ----------------------------------------------------------------------------------------------------------- joint
def box(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box((x1 - x0, y1 - y0, z1 - z0)); b.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)); return b


def prism_yz(poly, hw):
    """a prism |x| <= hw over a CONVEX polygon given in the (y, z) plane"""
    poly = np.asarray(poly, float); n = len(poly)
    V = np.r_[np.c_[np.full(n, -hw), poly], np.c_[np.full(n, hw), poly]]; F = []
    for k in range(1, n - 1): F += [[0, k + 1, k], [n, n + k, n + k + 1]]
    for k in range(n):
        a, b = k, (k + 1) % n; F += [[a, b, n + b], [a, n + b, n + a]]
    m = trimesh.Trimesh(V, np.array(F), process=True); trimesh.repair.fix_normals(m); return m


def union(ms): return trimesh.boolean.union(ms, engine="manifold")
def diff(a, b): return trimesh.boolean.difference([a, b], engine="manifold")


def joint_parts(p, S):
    """(add, cut): solids to union into the body and to subtract from it, in this piece's frame."""
    r, zc, nk, c = p["ball_r"], p["ball_z"], p["neck"], p["clearance"]; R = S["R_sock"] + p["gap"]
    ball = trimesh.creation.icosphere(4, r); ball.apply_translation((0, 0, zc))
    y_in = math.sqrt(max(R ** 2 - (nk / 2) ** 2, 1.0)) + 1.5                 # the bar reaches into the body behind the arch
    bar = box(-nk / 2, nk / 2, -1.0, y_in, zc - nk / 2, zc + nk / 2)
    zb = zc - nk / 2; yw = math.sqrt(max(R ** 2 - zc ** 2, 1.0)) + 0.5        # the arch's foot at the bed
    wedge = prism_yz([(yw - zb, 0.0), (yw + 0.5, 0.0), (yw + 0.5, zb + 1.0), (yw - 0.5, zb + 1.0)], nk / 2)   # overlaps the bar
    lobe = trimesh.creation.icosphere(4, S["R_sock"]); lobe.apply_translation((0, p["pitch"], zc))
    lobe = diff(lobe, box(-20, 20, -20, 40, -20, 0.0))                       # nothing below the bed
    add = [ball, bar, wedge, lobe]
    cut = []
    if p["slit"] > 0.01: cut.append(box(-p["slit"] / 2, p["slit"] / 2, -r - 1, yw - 0.5, 0.0, zc + r + 1))
    yp = p["pitch"]
    cav = trimesh.creation.icosphere(4, r + c); cav.apply_translation((0, yp, zc))
    slot_top = zc + nk / 2 + c + S["R_sock"] * math.sin(math.radians(p["extend"])) + 0.3
    slot = box(-(nk / 2 + c), nk / 2 + c, yp - 2.5, yp + S["R_sock"] + 2.0, 0.0, slot_top)
    cut += [cav, slot]
    return add, cut


def build_one(p):
    S = plan(p); z, dome = surface(p, S); body = closed_plate(p, S, z, int(p["grid"]))
    add, cut = joint_parts(p, S)
    m = diff(union([body] + add), union(cut)); m.merge_vertices()
    bodies = [b for b in m.split(only_watertight=False) if b.volume > 1.0]   # drop boolean slivers (< 1 mm3)
    if bodies: m = trimesh.util.concatenate(bodies) if len(bodies) > 1 else bodies[0]
    m.metadata["grid"] = body.metadata["grid"]
    return m, S, z


# ----------------------------------------------------------------------------------------------------------- checks
def overhang(m):
    n = m.face_normals; a = m.area_faces; zc = m.triangles_center[:, 2]
    bad = (n[:, 2] < -0.707) & (zc > 0.6); return float(a[bad].sum() / a.sum())


def bed_contact(m):
    n = m.face_normals; a = m.area_faces; zc = m.triangles_center[:, 2]
    return float(a[(n[:, 2] < -0.99) & (zc < 0.05)].sum())


def turned(m, piv_y, piv_z, deg):
    """the mesh turned `deg` about the x axis through (y, z) = pivot (+ = tail up)"""
    T = trimesh.transformations.rotation_matrix(math.radians(deg), (1, 0, 0), (0, piv_y, piv_z)); return m.copy().apply_transform(T)


def intersect_vol(a, b):
    try:
        I = trimesh.boolean.intersection([a, b], engine="manifold"); return abs(I.volume) if len(I.faces) else 0.0
    except BaseException as e:
        print("  intersection check failed:", e, file=sys.stderr); return float("inf")


def ball_clearance(piece_ahead, p, piv_y):
    r = p["ball_r"]; th = np.linspace(0, math.pi / 2, 12); ph = np.linspace(0, 2 * math.pi, 36, endpoint=False)
    TH, PH = np.meshgrid(th, ph); pts = np.c_[(r * np.sin(TH) * np.cos(PH)).ravel(), piv_y + (r * np.sin(TH) * np.sin(PH)).ravel(), p["ball_z"] + (r * np.cos(TH)).ravel()]
    d = trimesh.proximity.ProximityQuery(piece_ahead).signed_distance(pts)
    return float(-d.max())


def build(p):
    out = []; q = dict(p); notes = []
    for k in range(max(int(p["copies"]), 1)):
        m, S, z = build_one(q); dy = k * p["pitch"]; m.apply_translation((0, dy, 0)); out.append((m, S, z, dy)); q = dict(q, width=q["width"] * p["taper"])
    m0 = out[0][0]
    room = p["pitch"] - p["overlap"] - p["gap"] - p["step"]; h = p["crest"] - p["ball_z"]
    ext_max = math.degrees(math.asin(max(min(room / (2 * h), 1.0), 0.0)))
    notes.append(f"extension the overlap budget allows: {ext_max:.1f} deg (pitch - overlap - gap - step = {room:.1f} mm over two {h:.0f} mm levers)" + ("  <-- `extend` exceeds it" if p["extend"] > ext_max + 0.05 else ""))
    notes.append(f"bed contact {bed_contact(m0) / 100:.1f} cm2; overhang (needs support) {overhang(m0) * 100:.1f} % of the surface")
    if len(out) > 1:
        a, Sa, _, ya = out[0]; b, Sb, _, yb = out[1]
        rest = intersect_vol(a, b); notes.append(f"neighbours intersect at rest: {rest:.2f} mm3" + ("  <-- collide" if rest > 0.05 else " (clear)"))
        def free(sign):
            lo, hi = 0.0, 45.0
            if intersect_vol(a, turned(b, yb, p["ball_z"], sign * hi)) <= 0.05: return hi
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                if intersect_vol(a, turned(b, yb, p["ball_z"], sign * mid)) <= 0.05: lo = mid
                else: hi = mid
            return lo
        up, down = free(+1), free(-1)
        notes.append(f"swing before neighbours touch: enrols {down:.0f} deg per joint (wanted {p['enrol']:.0f}), extends {up:.0f} deg (wanted {p['extend']:.0f})" + ("  <-- short" if down < p["enrol"] or up < p["extend"] else ""))
        gaps = [ball_clearance(a_, p, yb_) for (a_, *_), (b_, Sb_, _, yb_) in zip(out, out[1:])]
        notes.append(f"ball to socket: least gap {min(gaps):.2f} mm (wanted {p['clearance']:.2f})")
    return out, notes


# ----------------------------------------------------------------------------------------------------------- sheet
def sheet(pieces, p, path, notes=()):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 10), facecolor="k")
    ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2); ax3 = fig.add_subplot(2, 1, 2)
    allm = trimesh.util.concatenate([pc[0] for pc in pieces])
    cuts = [((0, 1, 0), (0, pieces[0][1]["L"] * 0.55, 0), "section across, y = 0.55 L (through the socket)", 0, 2),
            ((1, 0, 0), (0.75, 0, 0), "section along, 0.75 mm off the midline (beside the slit): ball, neck, lip, body, socket, plate", 1, 2)]
    for ax, (n, o, t, i, j) in zip((ax1, ax3), cuts):
        sec = allm.section(plane_origin=o, plane_normal=n)
        if sec is not None:
            for e in sec.discrete: ax.fill(e[:, i], e[:, j], color="#8a8f99", alpha=0.9); ax.plot(e[:, i], e[:, j], "w", lw=0.6)
        ax.set_aspect("equal"); ax.set_title(t, color="w", fontsize=9)
    for (mk, Sk, zk, dy) in pieces:
        xs = np.linspace(0, Sk["w"], 200); yf, yr = Sk["edges"](xs)
        for sgn in (1, -1): ax2.plot(sgn * xs, -(yf + dy), "w", lw=0.8); ax2.plot(sgn * xs, -(yr + dy), "w", lw=0.8)
        ax2.plot([-Sk["w"], -Sk["w"]], [-(yf[-1] + dy), -(yr[-1] + dy)], "w", lw=0.8); ax2.plot([Sk["w"], Sk["w"]], [-(yf[-1] + dy), -(yr[-1] + dy)], "w", lw=0.8)
        ax2.add_patch(plt.Circle((0, -dy), p["ball_r"], fill=False, ec="#e8a33d", lw=0.8)); ax2.add_patch(plt.Circle((0, -(dy + p["pitch"])), Sk["R_sock"], fill=False, ec="#6fb3ff", lw=0.6, ls="--"))
    ax2.set_aspect("equal"); ax2.set_title("plan (front up) · orange: ball · dashed blue: socket lobe", color="w", fontsize=9)
    for a in (ax1, ax2, ax3): a.set_facecolor("k"); a.tick_params(colors="#777", labelsize=7); [s.set_color("#333") for s in a.spines.values()]
    fig.suptitle(f"{allm.extents[0]:.0f} × {allm.extents[1]:.0f} × {allm.extents[2]:.0f} mm · {len(pieces)} piece(s) · " + " · ".join(notes), color="w", fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=85, facecolor="k"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for n, d, u, doc in PARAMS: ap.add_argument("--" + n.replace("_", "-"), type=float, default=None, help=f"{doc} [{d} {u}]")
    ap.add_argument("--out", default="out/dome"); ap.add_argument("--params", default=None); ap.add_argument("--list", action="store_true"); ap.add_argument("--no-sheet", action="store_true")
    A = ap.parse_args()
    if A.list:
        for n, d, u, doc in PARAMS: print(f"{n:10s} {d:>7} {u:5s} {doc}")
        return
    p = dict(DEFAULTS)
    if A.params: p.update(json.load(open(A.params)))
    for n, _, _, _ in PARAMS:
        v = getattr(A, n); p[n] = v if v is not None else p[n]
    pieces, notes = build(p)
    os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
    whole = trimesh.util.concatenate([pc[0] for pc in pieces]); whole.export(A.out + ".stl"); json.dump(p, open(A.out + ".json", "w"), indent=1)
    if len(pieces) > 1:
        for k, (m, *_) in enumerate(pieces): m.export(f"{A.out}_{k:02d}.stl")
    m = pieces[0][0]
    print(f"{A.out}.stl  {whole.extents[0]:.1f} x {whole.extents[1]:.1f} x {whole.extents[2]:.1f} mm  {len(pieces)} piece(s)  one piece: {m.volume / 1000:.1f} cm3  faces {len(m.faces):,}  watertight {m.is_watertight}  one body {len(m.split(only_watertight=False)) == 1}")
    for n_ in notes: print("  " + n_)
    if not A.no_sheet: sheet(pieces, p, A.out + ".png", notes); print(f"{A.out}.png")


if __name__ == "__main__":
    main()
