"""
anatomy/head.py: the cephalon, joint-free. THE ONE head: the pin build and every print joint build from this file.

  plan(P, notes)        outline, zfun, xmax, the crescent-arm path, the eye geometry, scalars
  shell / solid         the plate (rear hinge zone at y = 0, head toward -y)
  ornaments(P, S)       [(name, solid, how)] in build order: genal arms, eyes, occipital spine, anterior prong
                        how = "union" or "mirror" (eyes: union the RIGHT one, cut at x = 0, mirror, weld)
  ports(P, S)           the rear port
  cells(P)              b1 / a1 / c1
  eye_geometry, fov, eye_solid, lens_centres, eye_params  (the eye, as before)

Swap this module for another head as long as it returns the same things.
"""
import math
import numpy as np
import mesh as M
from fields import seg_halfwidth, head_halfwidth, furrow_amp
from anatomy.common import pitch, ring_top, smoothstep, plateau, trough, vault, spine_solid, prong, safe_expr, GRID_HEAD
from anatomy.port import Port

def eye_geometry(P):
    """Where the eye sits and how big it is, head frame (rear hinge y = 0, head toward -y). eyes.eye_geometry, verbatim."""
    Lc = P["cephFrac"] * P["length"]
    wh = head_halfwidth(P); a = P["axisFrac"] * wh
    eR = P["eyeSize"] * wh; ye = -P["eyePos"] * Lc
    f = min(max(-ye / Lc, 0.0), 1.0)
    glab = a * (1 + (P["glabInflate"] - 1) * f)
    lat = P.get("eyeLat", 0.0)
    xe = lat * wh if lat > 0.01 else glab + eR + 1.0
    xe = max(xe, glab + 0.85 * eR)                    # 27 Sep 2026: a very large (pelagic) eye may overlap the glabella by 15 % at most
    return dict(xe=float(xe), ye=float(ye), eR=float(eR), eH=float(P["eyeHeight"] * eR), arc_deg=float(P["eyeArc"]),
                exponent=float(P.get("eyeProfile", 4.0)), glab_half=float(glab), head_halfwidth=float(wh), head_length=float(Lc),
                blind=bool(P["eyeSize"] <= 0.01))

FOV_VERSION = "0.1"
def fov(P):
    """Field-of-view summary from the eye geometry (eyes.fov, verbatim; the blueprint's eye panels read it)."""
    g = eye_geometry(P)
    if g["blind"]:
        return dict(fov_version=FOV_VERSION, blind=True, azimuth_deg=0.0, front_blind_deg=180.0, rear_blind_deg=180.0, binocular_deg=0.0, elevation_half_deg=0.0)
    arc = g["arc_deg"]; front_blind = max(0.0, 180.0 - arc); rear_blind = front_blind; binoc = max(0.0, arc - 180.0)
    cov = 360.0 - front_blind - rear_blind; elev = math.degrees(math.atan(P["eyeHeight"]))
    return dict(fov_version=FOV_VERSION, blind=False, azimuth_deg=round(cov, 1), front_blind_deg=round(front_blind, 1), rear_blind_deg=round(rear_blind, 1),
                binocular_deg=round(binoc, 1), elevation_half_deg=round(elev, 1), eye_xy=(round(g["xe"], 2), round(g["ye"], 2)), eye_radius_mm=round(g["eR"], 2))

def plan(P, notes=None):
    """outline(u, v), zfun(x, y), the crescent-arm path and scalars. Transcribed from trilobite.build_cephalon, minus
    tubercles and the skin blend (ornament and scan-fit are out of the core; no preset used either)."""
    t, c, h = P["wall"] * P["headWall"], P["clearance"], P["relief"] * P["headRelief"]
    Lc = P["cephFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P); flap = max(ovl - 2.0, 1.0)
    wh = head_halfwidth(P)
    par = P["cephParallel"] * Lc
    a = P["axisFrac"] * wh
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    Le = Lc - par
    nO = P["headOutlineExp"]
    def xmax(y):
        yy = np.asarray(y, float)
        front = np.clip((-yy - par) / Le, 0, 1)
        return np.maximum(wh * np.clip(1 - front ** nO, 0, 1) ** (1 / nO), 1.5)
    gs = P["genalSweep"] * pitch(P)
    Lg = P["genalSpine"] * Lc
    gc = math.radians(P["genalCurve"])
    arc = P["headRearArc"] * Lc; nR = P["headRearExp"]
    def cheek(u):
        q = np.clip((np.abs(u) * wh - a) / (wh - a), 0, 1)
        return flap + gs * q ** 2.2 + arc * q ** nR
    def y_front(ax):
        r = np.clip(ax / wh, 0, 1)
        return -par - Le * np.clip(1 - r ** nO, 0, 1) ** (1 / nO)
    W_arm = P["genalWidthMM"] if P["genalWidthMM"] > 0.05 else P["genalWidth"] * wh
    W_arm = max(W_arm, 1.5)
    kT = P["genalTaper"]
    S_ = np.linspace(0, 1, 241)
    path = safe_expr(P["genalPath"], S_, notes)
    path = path - path[0]
    if np.max(np.abs(path)) > 1e-9: path = path / np.max(np.abs(path))
    x_off = math.tan(gc) * Lg * path
    xc_s = (wh - 0.5 * W_arm) + x_off
    W_s = W_arm * np.clip(1 - S_ ** (2.0 / max(kT, 0.2)), 0, 1) ** (0.5 * kT) + 0.7
    X_tip = float(np.max(xc_s + 0.5 * W_s))
    def head_edges(x):
        ax = np.abs(np.asarray(x, float))
        return y_front(np.minimum(ax, wh)), cheek(np.minimum(ax, wh) / wh)
    def outline(u, v):
        x = u * X_tip
        yf, yr = head_edges(np.array([abs(x)]))
        return x, float(yr[0] - v * (yr[0] - yf[0]))
    def glab_half(y):
        """Half-width of the glabella along the head. 11 Sep 2026: ovoid, not a bar — the sides bulge, the widest point
        moves forward with inflate, and the front closes as a rounded nose (superellipse over the last 30 %, exponent
        glabFront; 2.5 = ovoid, 6 = blunt). Rear corners soften into the occipital ring."""
        f = np.clip(-np.asarray(y, float) / Lc, 0, 1)
        w = a * (1 + (P["glabInflate"] - 1) * f)
        w = w * (1 + 0.10 * np.sin(np.pi * f))                                   # bulged sides
        nF = P.get("glabFront", 2.5); fn = np.clip((f - 0.70) / 0.30, 0, 1)
        w = w * np.clip(1 - fn ** nF, 0, 1) ** (1 / nF)                          # rounded nose
        w = w * (0.75 + 0.25 * smoothstep(0.0, 0.10, f))                          # soft rear corners
        return np.maximum(w, 0.35 * a)
    w0 = seg_halfwidth(P, 0)
    G = eye_geometry(P) if P["eyeSize"] > 0.01 else None
    _plan_fns = dict(y_front=y_front, cheek=cheek, xmax=xmax, wh=wh, a=a, Lc=Lc, glab_half=glab_half, eye=G)
    suture = suture_path(P, _plan_fns); s_depth = float(P.get("sutureDepth", 0.0))
    ridge = eye_ridge_path(P, _plan_fns); r_height = float(P.get("eyeRidge", 0.0))
    def zfun_base(x, y):
        ax = np.abs(x)
        xm = xmax(y)
        front = np.clip((-y - par) / Le, 0, 1)
        tz = np.sqrt(np.clip(1 - front ** 2, 0, 1)) * 0.9 + 0.1
        z_v4 = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
        mD, fill = P["headDomeExp"], P["headDomeFill"]
        rim = margin * 0.6
        yc = -0.19 * Lc; aD = fill * wh; bD = 0.88 * fill * Lc
        rD = (ax / aD) ** mD + (np.abs(y - yc) / bD) ** mD
        z_dome = rim + (h - rim) * np.clip(1 - rD, 0, 1) ** (1 / mD)
        z = (1 - P["tent"]) * z_v4 + P["tent"] * z_dome
        g = glab_half(y)
        gl = plateau(x, g) * (1 - smoothstep(0.80 * Lc, 0.92 * Lc, -y))
        z += (rise + P["glabRise"] * h * np.clip(-y / Lc, 0, 1)) * gl
        z -= F * trough(ax - (g + 0.7), 0.9) * (-y < 0.9 * Lc)
        z -= 0.8 * F * trough(y + 0.13 * Lc, 0.8) * plateau(x, g + 1.5, 1.0)
        for k in range(int(P["glabLobes"])):
            yk = -Lc * (0.28 + 0.16 * k)
            z -= 0.7 * F * trough(y - yk, 0.9) * trough(ax - (g - 1.2), 2.2) * (ax > 0.3 * g)
        if G is not None:
            eR, ye, xe, eH = G["eR"], G["ye"], G["xe"], G["eH"]
            r = np.hypot(ax - xe, (y - ye) / P.get("eyeAxial", 1.0))
            ang = np.degrees(np.arctan2(y - ye, ax - xe))
            mask = plateau(ang, P["eyeArc"] / 2, 12)
            if P.get("eyeSolid", 0) < 0.5: z += eH * np.exp(-(r / (0.95 * eR)) ** G["exponent"])
            else:
                z += 0.15 * eH * np.exp(-(r / (1.3 * eR)) ** 2)
                collar = P.get("eyeCollar", 0.0)                           # 27 Sep 2026: the cheek rises to meet a sunk eye (a fillet)
                if collar > 1e-9: z += collar * eR * np.exp(-((r - eR) / (0.55 * eR)) ** 2)
            z += 0.35 * eH * trough(r - eR, 1.0) * mask
        if P["borderWidth"] > 0.005:
            bw = P["borderWidth"] * wh
            dist = np.minimum(xm - ax, np.where(-y > par, (1 - front) * Le, 1e9))
            z -= 0.8 * F * trough(dist - bw, 0.8 + 0.2 * bw)
            z += 0.35 * F * plateau(dist, 0.45 * bw, 0.4 * bw)
        z_th = margin + (h - margin) * vault(np.clip(ax / w0, 0, 1), P)
        rear_band = 1 - smoothstep(flap + 1.0, flap + 4.0, -y)
        z = np.maximum(z, z_th * rear_band)
        if arc > 0.5:
            hood = smoothstep(flap - 0.5, flap + 2.0, y)
            z = np.maximum(z, hood * (z_th + P["bladeCamber"] * h + t + c + 0.3))
        occ = ring_top(P) * plateau(x, a, 1.0) * (1 - smoothstep(0.10 * Lc, 0.16 * Lc, -y))
        return np.maximum(z, occ)
    # fine surface detail (27 Sep 2026): the eye ridge (raised) and the facial suture (a groove), both narrower than
    # the shell builder's smoothing length, so the plan exposes them separately and the builder adds them after its
    # blur (mesh.sample_grid detail=). zfun = zfun_base + detail is what the plan views and the blueprint see.
    def detail(x, y):
        ax = np.abs(x); d = np.zeros(np.broadcast(ax, y).shape)
        if r_height > 1e-9: d = d + r_height * np.exp(-0.5 * (_polyline_dist(ax, y, ridge) / RIDGE_SIGMA_MM) ** 2)
        if s_depth > 1e-9:  d = d - s_depth * np.exp(-0.5 * (_polyline_dist(ax, y, suture) / SUTURE_SIGMA_MM) ** 2)
        return d
    has_detail = r_height > 1e-9 or s_depth > 1e-9
    zfun = (lambda x, y: zfun_base(x, y) + detail(x, y)) if has_detail else zfun_base
    # crescent arms (genal horns / harpetid prolongations): swept bands of width W_s along the user's path
    yr_root = float(cheek(np.array([1.0]))[0]); S0 = -0.12
    def arm_outline(u, v):
        s = S0 + (1 - S0) * 0.5 * (u + 1); sc = np.clip(s, 0, 1)
        xc = float(np.interp(sc, S_, xc_s)); w = float(np.interp(sc, S_, W_s))
        ds = 1e-3; xa = float(np.interp(np.clip(sc + ds, 0, 1), S_, xc_s)); xb = float(np.interp(np.clip(sc - ds, 0, 1), S_, xc_s))
        tx, ty = (xa - xb), Lg * 2 * ds; nrm = math.hypot(tx, ty); tx, ty = tx / nrm, ty / nrm
        nx, ny = ty, -tx
        off = (v - 0.5) * w
        return xc + nx * off, yr_root + Lg * s + ny * off
    def arm_z(x, y):
        ax = np.abs(x)
        z_h = max(P["marginHeight"] * h, 0.0) + t + 0.6 + c
        z_in = margin + (h - margin) * vault(np.clip(ax / w0, 0, 1), P) + P["bladeCamber"] * h + t + c + 0.3
        inboard = ax < w0 + 1.0
        z = np.where(inboard & (y > yr_root - 0.5), np.maximum(z_h, z_in), z_h)
        body = y < yr_root + 0.5
        zb = zfun(np.asarray(x, float), np.asarray(y, float)) - 0.3
        blend = smoothstep(yr_root - 0.5, yr_root + 2.5, y)
        return np.where(body, zb, (1 - blend) * zb + blend * z)
    return dict(outline=outline, zfun=zfun, xmax=xmax, arm_outline=arm_outline, arm_z=arm_z, Lg=Lg, t=t, c=c, h=h, Lc=Lc,
                wh=wh, a=a, margin=margin, u_a=a / X_tip, eye=G, suture=suture, eye_ridge=ridge,
                zfun_base=zfun_base, detail=detail if has_detail else None)

# ---------------------------------------------------------------- the facial suture (27 Sep 2026)
# The line the cephalon splits along at the moult (Gon 2009 p. 21, p. 26): the anterior branch runs from the front margin
# beside the glabella back to the eye, wraps the eye on its axial side (the visual surface stays on the free cheek), and
# the posterior branch runs from the rear of the eye to the margin. Where it meets the margin is the classifying character
# and the one parameter, sutureEnd: -1 lateral margin forward of the genal angle (proparian), 0 the genal angle
# (gonatoparian), +1 the posterior margin inboard of it (opisthoparian). Drawn as a groove of sutureDepth mm; the blueprint
# draws the same path as a line. A blind head (no eye) has no palpebral loop: front to rear straight, as in Trimerocephalus.
SUTURE_SIGMA_MM = 0.35        # groove half-width (Gaussian sigma); a 0.2 mm groove prints as a visible line, not a slot
SUTURE_EYE_CLEAR = 1.15       # the loop runs this many eye radii from the eye centre (outside the eye's own rim trough)

def suture_path(P, S):
    """Right-side polyline (x >= 0), head frame, (N, 2) array. Uses only the plan's own functions."""
    end = float(np.clip(P.get("sutureEnd", 0.0), -1.0, 1.0))
    wh, Lc, G = S["wh"], S["Lc"], S["eye"]
    y_gen = float(S["cheek"](np.array([1.0]))[0])                             # the genal angle (wh, y_gen)
    if G is not None:
        xe, ye, r = G["xe"], G["ye"], SUTURE_EYE_CLEAR * G["eR"]
        xi = min(xe, wh - 1.0)                                                # the loop's x, kept on the plate
        loop_r = min(r, xi - 0.6 * float(S["glab_half"](np.array([ye]))[0]))  # and clear of the glabella
        loop_r = max(loop_r, 0.8)
        y_ef, y_er = ye - loop_r, ye + loop_r                                 # front / rear of the loop
    else:
        xi = 0.5 * (float(S["glab_half"](np.array([-0.5 * Lc]))[0]) + wh); loop_r = 0.0
        y_ef = y_er = -0.55 * Lc
    x0 = min(xi, wh - 1.0)
    pts = [(x0, float(S["y_front"](np.array([x0]))[0]) + 0.3)]              # anterior margin
    if G is not None:                                                         # the palpebral loop, axial side: front -> inner -> rear
        for ang in np.linspace(-90.0, 90.0, 19):
            th = math.radians(ang)
            pts.append((xi - loop_r * math.cos(th), ye - loop_r * math.sin(th)))   # ang=-90: (xi, y_ef); 0: (xi-loop_r, ye); 90: (xi, y_er)
    else:
        pts.append((xi, y_ef))
    if end <= 0.0:                                                            # lateral margin: from the eye's rear (-1) to the genal angle (0)
        yE = y_er + (1.0 + end) * (y_gen - y_er)
        xE = float(S["xmax"](np.array([yE]))[0]) - 0.3
    else:                                                                     # posterior margin: from the genal angle (0) inboard to 0.6 wh (+1)
        xE = wh - end * 0.4 * wh
        yE = float(S["cheek"](np.array([xE / wh]))[0]) - 0.3
    pts.append((xE, yE))
    return np.array(pts, float)

# ---------------------------------------------------------------- the eye ridge (27 Sep 2026)
# Gon p. 41: the primitive Redlichia morphotype has "well developed eye ridges": a raised ridge leaving the axial furrow
# beside the glabella's frontal lobe and running back and outward to the front of the palpebral lobe. It is lost in most
# derived groups, so eyeRidge = 0 is the derived state and the default; a preset switches it on. Drawn as a raised
# Gaussian ridge along a straight line (the ridge is broader and softer than the suture, sigma 0.6 mm). On a blind
# head it runs to where the eye would be (ridges outlast eyes in some atheloptic forms).
RIDGE_SIGMA_MM = 0.6
RIDGE_ROOT_FRAC = 0.74        # where the ridge leaves the glabella, as a fraction of head length from the rear (the frontal lobe)

def eye_ridge_path(P, S):
    """Right-side polyline (2, 2): axial furrow at the frontal lobe -> front of the eye. Head frame."""
    wh, Lc, G = S["wh"], S["Lc"], S["eye"]
    y0 = -RIDGE_ROOT_FRAC * Lc
    x0 = float(S["glab_half"](np.array([y0]))[0]) + 0.9                        # just outside the axial furrow
    if G is not None:
        x1, y1 = G["xe"] - 0.55 * G["eR"], G["ye"] - 0.95 * G["eR"]            # the anterior end of the palpebral lobe
    else:
        x1, y1 = 0.5 * (x0 + wh), -0.55 * Lc
    y0 = min(y0, y1 - 0.5)                                                       # the ridge always runs backward from the lobe
    return np.array([(x0, y0), (x1, y1)], float)

def _polyline_dist(x, y, pts):
    """Distance from every (x, y) to the polyline pts (N, 2). Vectorised over the grid; pts is short."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    d2 = np.full(np.broadcast(x, y).shape, np.inf)
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        vx, vy = x2 - x1, y2 - y1; L2 = vx * vx + vy * vy
        tt = np.clip(((x - x1) * vx + (y - y1) * vy) / L2, 0.0, 1.0) if L2 > 1e-12 else 0.0
        d2 = np.minimum(d2, (x - (x1 + tt * vx)) ** 2 + (y - (y1 + tt * vy)) ** 2)
    return np.sqrt(d2)

def eye_solid(R, H, slope_deg, shade, arc_deg, lensD, lensGap, lensRise, stalk=0.0, stalk_r=0.45, lean=0.0, embed=1.0, max_lenses=400, clip_x=None,
              sphere=False, bend_deg=30.0, axial=1.0, tall=1.0, sink_frac=0.55):
    """The eye as its own solid (eye_solid.py on Manifold): revolved drum over the visual arc, a palpebral lobe inward,
    spherical-cap lenses on a hex lattice within +-arc/2 of the outward (+x) direction. Frame: axis z, outward +x,
    base at z = -embed. Returns (mesh, n_lenses).
    27 Sep 2026: a stalked eye is always a SPHERE on a CURVED stalk (sphere_eye); sphere=True asks for a sphere on a
    sessile eye too. The drum below is unchanged for the sessile default (the frozen phacopid head)."""
    if stalk > 0.05 or sphere:
        return sphere_eye(R, arc_deg, lensD, lensGap, lensRise, stalk=stalk, stalk_r=stalk_r, lean=lean, embed=embed,
                          max_lenses=max_lenses, bend_deg=bend_deg, axial=axial, tall=tall, sink_frac=sink_frac)
    from manifold3d import Manifold, CrossSection
    slope = math.radians(slope_deg); arc = math.radians(arc_deg); rb = R + H * math.tan(slope)
    drum = Manifold.revolve(CrossSection([[(0, -embed), (rb + embed * math.tan(slope), -embed), (R, H), (0, H)]]), 64)
    if shade > 0.005:
        drum = drum + M.to_manifold(M.cylinder(R + shade * R, 0.6 * lensD * R + 0.8, at=(0, 0, H - 0.5 * (0.6 * lensD * R + 0.8))))
    a_keep = math.degrees(arc) + 24; big = 6 * rb + 6 * H
    ak = math.radians(a_keep / 2)
    sector = Manifold.extrude(CrossSection([[(0, 0), (big * math.cos(ak), -big * math.sin(ak)), (big, -big * 0.2), (big, big * 0.2),
                                             (big * math.cos(ak), big * math.sin(ak))]]), 2 * H + 4).translate((0, 0, -embed - 1))
    lobe_pts = [(0, -embed), (1.8 * R, -embed)] + [(R + 0.8 * R * math.sin(tt), -embed + (H + embed) * math.cos(tt))
                                                     for tt in [math.radians(aa) for aa in (80, 65, 50, 35, 20, 8, 0)]] + [(0, H)]
    lobe = Manifold.revolve(CrossSection([lobe_pts]), 64) - sector
    if clip_x is not None:
        lobe = lobe ^ M.to_manifold(M.box(2 * big, 4 * big, 4 * big, at=(clip_x - big, 0, 0)))
    body = (drum ^ sector) + lobe
    D = lensD * R; rise = lensRise * D / 2; rho = max((rise ** 2 + (D / 2) ** 2) / (2 * rise), 0.35)
    if D < 0.5: return M.from_manifold(body), 0                       # holochroal facets below FDM relief: smooth band
    nx, nz = math.cos(slope), math.sin(slope)
    pitch_ = D * (1 + lensGap); rows = max(1, int(H / (0.87 * pitch_))); cs = []
    for i in range(rows):
        s = (i + 0.5) * 0.87 * pitch_
        if s > H - 0.5 * D: break
        r = R + (H - s) * math.tan(slope); n = max(1, int(arc * r / pitch_)); off = 0.5 * (i % 2)
        for k in range(n):
            th = -arc / 2 + (k + off + 0.5) * arc / (n + 0.5)
            if abs(th) <= arc / 2: cs.append((th, s, r))
    cs = cs[:max_lenses]
    import trimesh
    sph = trimesh.creation.icosphere(subdivisions=2, radius=rho)
    spheres = []
    for th, s, r in cs:
        depth = rho - rise
        c = ((r - depth * nx) * math.cos(th), (r - depth * nx) * math.sin(th), s + depth * nz)
        spheres.append(M.to_manifold(sph.copy().apply_translation(c)))
    if spheres: body = body + Manifold.batch_boolean(spheres, __import__("manifold3d").OpType.Add)
    if stalk > 0.05:                                                  # pedunculate eye: a tapered stalk under the drum,
        r0 = stalk_r * R                                              # planted where the sessile eye's base would sit
        body = body + M.to_manifold(M.frustum(r0, 0.7 * r0, stalk + embed)).translate((0, 0, -embed - stalk))
    out = M.from_manifold(body)
    if abs(lean) > 0.5:                       # tip the eye forward about its root — the foot of the stalk if there is
        import trimesh                        # one, else the base on the cheek. Head frame is front = -y, so +x turns
        piv = (0, 0, -embed - stalk)          # it forward; at 90 deg it looks straight out over the front margin.
        out.apply_transform(trimesh.transformations.rotation_matrix(math.radians(lean), (1, 0, 0), piv))
    return out, len(cs)

# ---------------------------------------------------------------- the sphere eye on a curved stalk (27 Sep 2026)
STALK_SEGMENTS = 14           # frustums along the arc (spheres at the joints keep it smooth)
STALK_TAPER = 0.7             # tip radius / root radius
LENS_ELEV_DEG = 42.0          # lenses within +- this elevation of the equator (the visual band of a spherical eye)
HANG_LENS_ELEV = (-78.0, 22.0)  # a hanging eye (sink_frac > 1): lenses from under the palpebral shelf down round the bottom

def stalk_arc(stalk, bend_deg, root_z):
    """Centreline of the stalk: an arc of length `stalk` leaving the root (0, 0, root_z) vertically and bending
    outward (+x) by bend_deg. Returns (points (n+1, 3), end tangent (3,))."""
    n = STALK_SEGMENTS; bend = math.radians(max(bend_deg, 0.0))
    if bend < 1e-4:
        pts = np.array([(0.0, 0.0, root_z + stalk * i / n) for i in range(n + 1)]); return pts, np.array([0.0, 0.0, 1.0])
    rho = stalk / bend
    pts = np.array([(rho * (1 - math.cos(bend * i / n)), 0.0, root_z + rho * math.sin(bend * i / n)) for i in range(n + 1)])
    return pts, np.array([math.sin(bend), 0.0, math.cos(bend)])

def sphere_eye(R, arc_deg, lensD, lensGap, lensRise, stalk=0.0, stalk_r=0.45, lean=0.0, embed=1.0, max_lenses=400, bend_deg=30.0,
               axial=1.0, tall=1.0, sink_frac=0.55):
    """A spherical (or, with axial / tall, ellipsoidal) eye. Sessile: sunk into the cheek by sink_frac of its height. Stalked: a
    tapered stalk of length `stalk` curves outward by bend_deg from the root at z = -embed - stalk, the sphere on its
    end. Lenses (if lensD R >= 0.5 mm) are spherical caps on rows of a lat-long lattice within +-arc/2 of outward and
    +-LENS_ELEV_DEG of the equator. Same frame and return as eye_solid: (mesh, n_lenses)."""
    from manifold3d import Manifold, OpType
    import trimesh
    parts = []
    if stalk > 0.05:
        r0 = stalk_r * R; root_z = -embed - stalk
        pts, tang = stalk_arc(stalk, bend_deg, root_z)
        n = len(pts) - 1
        for i in range(n):
            a, b = pts[i], pts[i + 1]; d = b - a; L = float(np.linalg.norm(d))
            ra = r0 * (1 - (1 - STALK_TAPER) * i / n); rb = r0 * (1 - (1 - STALK_TAPER) * (i + 1) / n)
            seg = M.frustum(ra, rb, L)
            seg.apply_transform(trimesh.geometry.align_vectors((0, 0, 1), d / L)); seg.apply_translation(a)
            parts.append(M.to_manifold(seg))
            if i < n - 1: parts.append(M.to_manifold(trimesh.creation.icosphere(subdivisions=2, radius=rb).apply_translation(b)))
        centre = pts[-1] + tang * (0.75 * R)                     # the sphere sits on the stalk's end, overlapping it
    else:
        centre = np.array([0.0, 0.0, tall * R * (1 - sink_frac) - embed])   # sink_frac 0.55: half sunk (the small-eye default);
                                                                             # ~0.9: centre at cheek level, the eye bulges out and under (pelagic)
    scale = np.array([1.0, axial, tall])                                     # the bean: longer fore-aft (axial > 1) than it is wide
    eye = trimesh.creation.icosphere(subdivisions=4, radius=R); eye.vertices = eye.vertices * scale; eye.apply_translation(centre)
    parts.append(M.to_manifold(eye))
    body = Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]
    D = lensD * R; cs = []
    if D >= 0.5:
        rise = lensRise * D / 2; rho = max((rise ** 2 + (D / 2) ** 2) / (2 * rise), 0.35)
        pitch_ = D * (1 + lensGap); arc = math.radians(arc_deg)
        el_lo, el_hi = (math.radians(HANG_LENS_ELEV[0]), math.radians(HANG_LENS_ELEV[1])) if (stalk <= 0.05 and sink_frac > 1.0) \
                       else (-math.radians(LENS_ELEV_DEG), math.radians(LENS_ELEV_DEG))
        rows = max(1, int((el_hi - el_lo) * R / (0.87 * pitch_)))
        for i in range(rows):
            el = el_lo + (i + 0.5) * (el_hi - el_lo) / rows; r_row = R * math.cos(el)
            nn = max(1, int(arc * r_row / pitch_)); off = 0.5 * (i % 2)
            for k in range(nn):
                th = -arc / 2 + (k + off + 0.5) * arc / (nn + 0.5)
                if abs(th) <= arc / 2: cs.append((th, el))
        cs = cs[:max_lenses]
        sph = trimesh.creation.icosphere(subdivisions=2, radius=rho); spheres = []
        for th, el in cs:
            u = np.array([math.cos(el) * math.cos(th), math.cos(el) * math.sin(th), math.sin(el)])
            p = u * R * scale; nrm = u / scale; nrm /= np.linalg.norm(nrm)          # on the ellipsoid, pushed in along its normal
            c = centre + p - nrm * (rho - rise)
            spheres.append(M.to_manifold(sph.copy().apply_translation(c)))
        if spheres: body = body + Manifold.batch_boolean(spheres, OpType.Add)
    out = M.from_manifold(body)
    if abs(lean) > 0.5:                                          # tip forward about the root, as the drum does
        piv = (0, 0, -embed - stalk)
        out.apply_transform(trimesh.transformations.rotation_matrix(math.radians(lean), (1, 0, 0), piv))
    return out, len(cs)

def lens_centres(R, H, slope, arc, lensD, lensGap):
    """(theta, height, radius) of every lens on the visual band — the same lattice eye_solid() builds (eye_solid.lens_centres)."""
    pitch_ = lensD * (1 + lensGap); rows = max(1, int(H / (0.87 * pitch_))); out = []
    for i in range(rows):
        s = (i + 0.5) * 0.87 * pitch_
        if s > H - 0.5 * lensD: break
        r = R + (H - s) * math.tan(slope); n = max(1, int(arc * r / pitch_)); off = 0.5 * (i % 2)
        for k in range(n):
            th = -arc / 2 + (k + off + 0.5) * arc / (n + 0.5)
            if abs(th) <= arc / 2: out.append((th, s, r))
    return out

def eye_params(P, eR):
    return dict(R=eR, H=P.get("eyeHeight", 1.7) * eR, slope_deg=P.get("eyeSlope", 15.0), shade=P.get("eyeShade", 0.0),
                arc_deg=P.get("eyeArc", 110.0), lensD=P.get("lensD", 0.16), lensGap=P.get("lensGap", 0.3), lensRise=P.get("lensRise", 0.35),
                stalk=P.get("eyeStalk", 0.0) * eR, stalk_r=P.get("eyeStalkR", 0.45), lean=P.get("eyeLean", 0.0),
                sphere=P.get("eyeSphere", 0.0) > 0.5, bend_deg=P.get("eyeStalkBend", 30.0),
                axial=P.get("eyeAxial", 1.0), tall=P.get("eyeTall", 1.0), sink_frac=P.get("eyeSink", 0.55))

def _base(S): return (S.get("zfun_base", S["zfun"]), S.get("detail"))   # a replacement head may give zfun alone

def shell(P, S, grid=GRID_HEAD):
    zb, det = _base(S)
    head = M.heightfield_shell(S["outline"], zb, S["t"], nu=grid[0], nv=grid[1], detail=det)
    env = M.under_envelope(S["outline"], zb, nu=grid[0], nv=grid[1], detail=det)
    return head, env

def solid(P, S, grid=GRID_HEAD):
    zb, det = _base(S)
    return M.under_envelope(S["outline"], zb, nu=grid[0], nv=grid[1], floor=0.0, detail=det)

def ornaments(P, S, notes=None):
    """In build order. Each entry is (name, solid, how): how = "union" (with its mirror image where symmetric) or
    "mirror" (the eye: union the right one only, cut the part at x = 0, mirror, weld; see the note on the eye)."""
    out = []
    if S["Lg"] > 0.5:
        arm = M.heightfield_shell(S["arm_outline"], S["arm_z"], S["t"], nu=81, nv=13, symmetric=False)
        out.append(("genalArms", [arm, M.mirror_x(arm)], "union"))
    if P.get("eyeSolid", 0) > 0.5 and P["eyeSize"] > 0.01:
        G = S["eye"]; EP = eye_params(P, G["eR"])
        xm = float(S["xmax"](np.array([G["ye"]]))[0]) - 0.4
        eye, n_lens = eye_solid(**EP, clip_x=None if (EP["stalk"] > 0.05 or EP["sphere"]) else xm - G["xe"])   # a stalked or spherical eye may overhang
        zb = float(S["zfun"](np.array([G["xe"]]), np.array([G["ye"]]))[0]) - 0.3 + EP["stalk"]
        out.append(("eyes", [eye.copy().apply_translation((G["xe"], G["ye"], zb))], "mirror"))
        if notes is not None: notes.append(("head", "eye solid", f"{n_lens} lenses/eye"))
    if P["occipitalSpine"] > 0.02:
        out.append(("occipitalSpine", [spine_solid(0.6 * S["margin"], 0.5, P["occipitalSpine"] * S["Lc"], (0, -0.07 * S["Lc"], ring_top(P) - 1.0), 0, pitch_deg=55)], "union"))
    if int(P.get("headProngs", 0)) > 0:
        yf = float(S["outline"](0.0, 1.0)[1])                 # front margin at the axis (u = 0, v = 1)
        out.append(("headProngs", [prong(int(P["headProngs"]), P["headProngLen"] * S["Lc"], P["headProngSplay"],
                                         P.get("headProngWidth", 0.45) * S["margin"] + 0.6, 0.45, (0, yf + 1.5, 0.6 * S["margin"] + 0.5),
                                         yaw_deg=180.0, pitch_deg=8.0, stem_frac=P.get("headProngStem", 0.0), center_bias=P.get("headProngCenter", 1.0),
                                         curl_deg=P.get("headProngCurl", 0.0))], "union"))
    return out

def ports(P, S):
    """One port: the rear joint plane at y = 0, wide (the head meets segment 0's full plate)."""
    return dict(rear=Port(y=0.0, rear=True, wide=True, halfwidth=S["wh"], ring_half=S["a"], kind="head"))

def cells(P, lap=0.06, grid=GRID_HEAD):
    """b1 (glabella + occipital ring, |u| <= a/X_tip) / a1 (cheek, eye, genal angle) / c1 = mirror a1."""
    S = plan(P); ua = S["u_a"]
    ring_out = lambda u, v: S["outline"](u * ua, v)
    pl_out = lambda u, v: S["outline"]((ua - lap) + 0.5 * (u + 1) * (1 - (ua - lap)), v)
    zb, det = _base(S)
    axis = M.heightfield_shell(ring_out, zb, S["t"], nu=grid[0] // 3 | 1, nv=grid[1], detail=det)
    cheek = M.heightfield_shell(pl_out, zb, S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False, detail=det)
    return dict(axis=axis, cheek_right=cheek, cheek_left=M.mirror_x(cheek))
