"""eyes2 — a richer trilobite eye, standalone. NOT wired into parts.cephalon or the instrument (fixed-ruler law).

Frame as eye_solid: axis z up, outward +x, visual surface base at z = -embed.
Adds over eye_solid():
  plan   : elliptical / kidney plan (ky, kidney) instead of a circle
  crescent: band height tapers to the ends of the arc (taper)  -> lunate visual surface
  lenses : 'hex' | 'grid' lattice, diameter gradient toward the margins (grad)
  style  : 'raised' (cap), 'sunken' (socket + sclera rim), 'flush' (dimpled cornea)
  stalk  : pedunculate eye (stalk_len, stalk_r)
"""
import math, numpy as np, trimesh
from manifold3d import Manifold, OpType
import mesh as M


def _plan_r(theta, R, ky, kidney):
    """radius of the plan curve in direction theta (ellipse a=R, b=ky*R, pulled in at the back by 'kidney')."""
    a, b = R, ky * R
    r = 1.0 / math.sqrt((math.cos(theta) / a) ** 2 + (math.sin(theta) / b) ** 2)
    return r * (1 - kidney * max(0.0, -math.cos(theta)) ** 2)


def eye2(R=5.0, H=5.0, arc_deg=150, slope_deg=15, ky=1.0, kidney=0.0, taper=0.0, lobe_h=0.35,
         lensD=0.16, lensGap=0.3, lensRise=0.35, grad=0.0, lattice="hex", style="raised", rim=0.18,
         stalk_len=0.0, stalk_r=0.45, lean_out_deg=0.0, lean_fwd_deg=0.0, shade=0.0, lens_cap=True,
         clip_x=None, embed=1.0, max_lenses=600, nth=160, nr=14, min_D=0.5):
    arc = math.radians(arc_deg); sl = math.radians(slope_deg); tans = math.tan(sl)
    half = arc / 2

    def Hband(th):
        if abs(th) > half: return 0.0
        return H * (1 - (th / half) ** 2) ** taper if taper > 0 else H

    def rtop(th):   return _plan_r(th, R, ky, kidney)
    def rbase(th):  return rtop(th) + Hband(th) * tans

    # ---- body: star-shaped polar heightfield (top) + wall + flat base -------------------------------------
    ths = np.linspace(-math.pi, math.pi, nth, endpoint=False)
    top, rim_pts = [], []
    Hlobe = H * lobe_h
    def plateau(th):                                   # top height in direction th: H over the band, easing to the lobe behind
        if abs(th) <= half: return Hband(th) if taper > 0 else H
        u = (abs(th) - half) / (math.pi - half)
        return H * (1 - u ** 1.6) + Hlobe * u ** 1.6
    pavg = float(np.mean([plateau(t) for t in ths]))
    for th in ths:
        inband = abs(th) <= half
        Rb = rbase(th) if inband else 0.92 * rtop(th)
        Pt = plateau(th)
        for r in np.linspace(0, Rb, nr + 1):
            if inband:
                z = min(Pt, (Rb - r) / tans) if tans > 1e-6 else (Pt if r < Rb - 1e-9 else 0.0)
            else:
                z = Pt * math.sqrt(max(0.0, 1 - (r / Rb) ** 2.4))
            u = min(1.0, r / (0.5 * Rb)); w = u * u * (3 - 2 * u)
            z = pavg + (z - pavg) * w                            # smooth blend to a common apex (no spike)
            top.append((r * math.cos(th), r * math.sin(th), z))
        rim_pts.append((Rb * math.cos(th), Rb * math.sin(th)))
    top = np.array(top).reshape(nth, nr + 1, 3)
    top[:, 0, :2] = 0.0; top[:, 0, 2] = pavg                                 # single apex
    V = [top.reshape(-1, 3)]; F = []
    idx = lambda i, j: i * (nr + 1) + j
    for i in range(nth):
        i2 = (i + 1) % nth
        for j in range(nr):
            a, b, c, d = idx(i, j), idx(i2, j), idx(i2, j + 1), idx(i, j + 1)
            if j == 0: F.append((a, c, d))
            else: F += [(a, b, c), (a, c, d)]
    base_z = -embed - 1.0
    n0 = nth * (nr + 1)
    V.append(np.array([(x, y, base_z) for x, y in rim_pts])); V.append(np.array([[0, 0, base_z]]))
    for i in range(nth):
        i2 = (i + 1) % nth; t1, t2 = idx(i, nr), idx(i2, nr); b1, b2 = n0 + i, n0 + i2
        F += [(t1, b1, t2), (t2, b1, b2), (b1, n0 + nth, b2)]
    tm = trimesh.Trimesh(np.vstack(V), np.array(F), process=True); tm.merge_vertices(); trimesh.repair.fix_normals(tm)
    assert tm.is_watertight, 'eye body not watertight'
    body = M.to_manifold(tm)
    # lift so band base sits at z = -embed (top array has band base at z=0)
    body = body.translate((0, 0, -embed))

    # ---- lenses --------------------------------------------------------------------------------------
    D0 = lensD * R
    if D0 < min_D: return M.from_manifold(body), 0
    pitch = D0 * (1 + lensGap); cs = []
    rows = max(1, int(H / (0.87 * pitch)))
    for i in range(rows):
        s = (i + 0.5) * 0.87 * pitch
        if s > H - 0.5 * D0: break
        n = max(1, int(arc * R / pitch)); off = 0.5 * (i % 2) if lattice == "hex" else 0.0
        for k in range(n):
            th = -half + (k + off + 0.5) * arc / (n + 0.5)
            if abs(th) > half: continue
            hb = Hband(th)
            if s > hb - 0.5 * D0 * (1 - grad * abs(th) / half): continue
            D = D0 * (1 - grad * abs(th) / half)
            r = rtop(th) + (hb - s) * tans
            cs.append((th, s - embed, r, D))
    cs = cs[:max_lenses]
    nx, nz = math.cos(sl), math.sin(sl)
    adds, cuts, caps = [], [], []
    sph = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    for th, s, r, D in cs:
        rise = lensRise * D / 2; rho = max((rise ** 2 + (D / 2) ** 2) / (2 * rise), 0.3)
        nvec = np.array([nx * math.cos(th), nx * math.sin(th), nz])
        surf = np.array([r * math.cos(th), r * math.sin(th), s])
        if style == "raised":
            c = surf - (rho - rise) * nvec
            adds.append(M.to_manifold(sph.copy().apply_scale(rho).apply_translation(c)))
        else:
            depth = rise if style == "sunken" else 0.35 * rise
            c = surf + (rho - depth) * nvec                                 # sphere centre outside, sinks 'depth' in
            cuts.append(M.to_manifold(sph.copy().apply_scale(rho).apply_translation(c)))
            if style == "sunken" and lens_cap:                          # convex lens sitting inside the socket
                rise2 = 0.55 * rise; D2 = 0.85 * D; rho2 = max((rise2 ** 2 + (D2 / 2) ** 2) / (2 * rise2), 0.25)
                c2 = surf - (rho2 - rise2 + 0.35 * rise) * nvec           # crown 0.35·rise below the sclera surface
                caps.append(M.to_manifold(sph.copy().apply_scale(rho2).apply_translation(c2)))
            if style == "sunken" and rim > 0:
                cyl = trimesh.creation.cylinder(radius=D / 2 * (1 + rim), height=0.45 * rise + 0.15, sections=24)
                T = trimesh.geometry.align_vectors([0, 0, 1], nvec); T[:3, 3] = surf + 0.1 * rise * nvec
                adds.append(M.to_manifold(cyl.apply_transform(T)))
    if adds: body = body + Manifold.batch_boolean(adds, OpType.Add)
    if cuts: body = body - Manifold.batch_boolean(cuts, OpType.Add)
    if caps: body = body + Manifold.batch_boolean(caps, OpType.Add)
    if shade > 0.005:                                                    # eyeshade brim at the top of the visual band
        brim = M.to_manifold(M.cylinder(R * (1 + shade), 0.6 * D0 + 0.8, at=(0, 0, H - embed - 0.3 * D0 - 0.4)))
        body = body + brim
    if clip_x is not None:                                               # never overhang the cheek outline
        big = 8 * R + 4 * H
        body = body ^ M.to_manifold(M.box(2 * big, 4 * big, 4 * big, at=(clip_x - big, 0, 0)))

    # ---- stalk ---------------------------------------------------------------------------------------
    if stalk_len > 0:
        r0, r1 = stalk_r * R, 0.7 * stalk_r * R
        st = M.to_manifold(M.frustum(r1, r0, stalk_len)).translate((0, 0, -embed - stalk_len))
        body = body + st
    out = M.from_manifold(body)
    if stalk_len > 0 and (lean_out_deg or lean_fwd_deg):                # pivot at the stalk root; +out = +x, +fwd = -y
        piv = (0, 0, -embed - stalk_len)
        out.apply_transform(trimesh.transformations.rotation_matrix(math.radians(lean_out_deg), (0, 1, 0), piv))
        out.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-lean_fwd_deg), (1, 0, 0), piv))
    return out, len(cs)
