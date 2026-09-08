"""
blueprint.py — draw a built animal as a technical sheet: white ink on black.
Plan (dorsal) as contour lines, lateral silhouette, midline section, dimension lines, enrollment dial, title block.
    sheet(flat_mesh, P, measure, path)   -> writes a PNG (matplotlib) and returns the figure
"""
import math, numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse

INK, BG, DIM = "#ffffff", "#000000", "#9a9a9a"

def _contours(m, levels):
    """Horizontal sections (z = const): the topographic lines of the dorsal plan."""
    out = []
    for z in levels:
        try:
            s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        except Exception:
            s = None
        if s is None: continue
        for e in s.entities:
            out.append(s.vertices[e.points][:, :2])
    return out

def _silhouette(m, axes):
    hull = trimesh.path.polygons.projected(m, normal=[0, 0, 1]) if axes == (0, 1) else None
    return hull

def _dim(ax, x0, y0, x1, y1, text, off=0.0, side=1):
    """A dimension line with ticks and the value."""
    dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy); nx, ny = -dy / L * off * side, dx / L * off * side
    a = (x0 + nx, y0 + ny); b = (x1 + nx, y1 + ny)
    ax.plot([x0, a[0]], [y0, a[1]], color=DIM, lw=0.5); ax.plot([x1, b[0]], [y1, b[1]], color=DIM, lw=0.5)
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, text, color=INK, fontsize=7, ha="center", va="bottom" if abs(dy) < abs(dx) else "center",
            rotation=0 if abs(dy) < abs(dx) else 90, family="monospace")

def _eye_panels(fig, gs, m, P):
    """The eye blueprint on the page: plan and transverse section from the BUILT head (no drawn circles — the
    geometry is the drawing), the outward view of the band with its lens lattice, and the band unrolled with
    the lattice formulas. All four read the same parameters the builder used."""
    import math as _m
    from eyes import eye_geometry, fov
    G = eye_geometry(P)
    axes = [fig.add_subplot(gs[4:6, 4:5]), fig.add_subplot(gs[4:6, 5:6]), fig.add_subplot(gs[4:6, 6:7]), fig.add_subplot(gs[4:6, 7:9])]
    for ax in axes: ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    if G["blind"]:
        axes[0].set_title("EYE · blind (eyeSize 0)", color=INK, fontsize=8, family="monospace", loc="left"); return
    xe, ye, eR = G["xe"], G["ye"], G["eR"]; W = 2.6 * eR
    # the head alone: the component of the concatenated mesh that contains the eye (no seams from other parts)
    head = m
    try:
        comps = m.split(only_watertight=False)
        head = max(comps, key=lambda c: -np.linalg.norm(c.bounds.mean(0)[:2] - np.array([0, ye])) if len(c.faces) > 200 else -1e9)
    except Exception: pass
    # ---- plan
    ax = axes[0]
    # contour interval set by the eye, not the animal: ~12 levels over the eye's own height. Contours on the visual
    # band (outboard of the cap radius, inside the arc) in ink; the lobe and cheek in half tone.
    zc = head.bounds[1][2]
    try:
        sec0 = head.section(plane_origin=[0, ye, 0], plane_normal=[0, 1, 0])
        pts0 = np.vstack([sec0.vertices[e.points] for e in sec0.entities]); near = np.abs(pts0[:, 0] - xe) < 2.2 * eR
        zc, zf = pts0[near, 2].max(), pts0[near, 2].min()
    except Exception:
        zf = head.bounds[0][2]
    for z in np.linspace(zf + 0.3, zc - 0.15, 12):
        try:
            sec = head.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if sec is None: continue
            for e in sec.entities:
                p = sec.vertices[e.points]
                inside = (np.abs(p[:, 0] - xe) < W) & (np.abs(p[:, 1] - ye) < W)
                if not np.any(inside): continue
                dx, dy = p[:, 0] - xe, p[:, 1] - ye
                on_band = (np.hypot(dx, dy) > 0.9 * eR) & (dx > 0.15 * eR)              # outboard, facing outward: the lensed band
                ax.plot(np.where(on_band, p[:, 0], np.nan), np.where(on_band, p[:, 1], np.nan), color=INK, lw=0.55)
                ax.plot(np.where(~on_band, p[:, 0], np.nan), np.where(~on_band, p[:, 1], np.nan), color=DIM, lw=0.35, alpha=0.6)
        except Exception: pass
    _dim(ax, xe - eR, ye + W + 2.5, xe + eR, ye + W + 2.5, f"R {eR:.1f}", off=0)
    ax.set_xlim(xe - W, xe + W); ax.set_ylim(ye - W, ye + W + 7)
    ax.set_title(f"EYE PLAN · lat {xe / G['head_halfwidth']:.2f} wh · pos {-ye / G['head_length']:.2f} Lc", color=INK, fontsize=7, family="monospace", loc="left")
    # ---- transverse section through the eye centre
    ax = axes[1]
    try:
        sec = head.section(plane_origin=[0, ye, 0], plane_normal=[0, 1, 0]); zmax = 0
        for e in sec.entities:
            p = sec.vertices[e.points]; ax.plot(p[:, 0], p[:, 2], color=INK, lw=0.7); zmax = max(zmax, p[:, 2].max())
        pts = np.vstack([sec.vertices[e.points] for e in sec.entities]); crest = pts[(np.abs(pts[:, 0] - xe) < eR), 2].max()
        foot = pts[np.abs(pts[:, 0] - (xe + 1.9 * eR)) < 0.6, 2]; foot = float(np.median(foot)) if len(foot) else float(pts[:, 2].min())
        _dim(ax, xe + W + 1.5, foot, xe + W + 1.5, crest, f"h {crest - foot:.1f}", off=0)
        ax.plot([xe - W, xe + W], [foot, foot], color=DIM, lw=0.4, ls=(0, (4, 3)))
        ax.set_xlim(xe - W, xe + W + 7); ax.set_ylim(min(foot, pts[:, 2].min()) - 3, crest + 4)
    except Exception as ex:
        ax.text(0.05, 0.5, f"section: {str(ex)[:24]}", color=DIM, fontsize=6, family="monospace", transform=ax.transAxes)
    ax.set_title(f"SECTION B–B · lean {P.get('eyeSlope', 0):.0f}°", color=INK, fontsize=7, family="monospace", loc="left")
    # ---- outward view and unrolled band: the lattice as built
    try:
        from eye_solid import lens_centres, eye_params
        EP = eye_params(P, eR); slope = _m.radians(EP["slope_deg"]); arc = _m.radians(EP["arc_deg"]); H = EP["H"]
        D = EP["lensD"] * eR; cs = lens_centres(eR, H, slope, arc, D, EP["lensGap"]); rb = eR + H * _m.tan(slope)
        ax = axes[2]; yt, yb = eR * _m.sin(arc / 2), rb * _m.sin(arc / 2)
        ax.plot([-yt, yt, yb, -yb, -yt], [H, H, 0, 0, H], color=INK, lw=0.8)
        if EP["shade"] > 0.005: ax.plot([-yt - EP["shade"] * eR, yt + EP["shade"] * eR], [H + 0.1, H + 0.1], color=INK, lw=1.8)
        ax.plot([-yb - 2, yb + 2], [0, 0], color=DIM, lw=0.4, ls=(0, (4, 3)))
        for th, sz, r in cs:
            if _m.cos(th) > 0.05: ax.add_patch(Ellipse((r * _m.sin(th), sz), D * max(0.3, _m.cos(th)), D, fill=False, color=INK, lw=0.4))
        _dim(ax, yb + 3, 0, yb + 3, H, f"{H:.1f}", off=0)
        ax.set_xlim(-yb - 3, yb + 9); ax.set_ylim(-2, H + 3)
        ax.set_title(f"OUTWARD · {len(cs)} lenses · D {D:.2f}", color=INK, fontsize=7, family="monospace", loc="left")
        ax = axes[3]; arcl = arc * rb; pitch = D * (1 + EP["lensGap"]); rows = max(1, int(H / (0.87 * pitch)))
        ax.plot([-arcl / 2, arcl / 2, arcl / 2 * eR / rb, -arcl / 2 * eR / rb, -arcl / 2], [0, 0, H, H, 0], color=INK, lw=0.7)
        for th, sz, r in cs: ax.add_patch(Circle((th * r, sz), D / 2, fill=False, color=INK, lw=0.35))
        ax.set_xlim(-arcl / 2 - 1, arcl / 2 + 1); ax.set_ylim(-H * 1.15 - 2, H + 1.5)
        F = fov(P)
        ax.text(-arcl / 2, -1.2, "\n".join([f"arc {arcl:.1f} mm across · band {H:.1f} mm up",
                                             f"pitch {pitch:.2f} = D(1+{EP['lensGap']:.1f}) · rows {rows}",
                                             f"packing {D / pitch:.2f} = D/pitch (1 = holochroal)",
                                             f"files per row = arc·r / pitch",
                                             f"arc {EP['arc_deg']:.0f}° · lean {EP['slope_deg']:.0f}° · brim {EP['shade']:.2f} R",
                                             f"rise {EP['lensRise']:.2f} D/2 · axis = normal (v1)",
                                             f"FOV v{F['fov_version']} · az {F['azimuth_deg']:.0f}° · el ±{F['elevation_half_deg']:.0f}°",
                                             f"blind {F['front_blind_deg']:.0f}° fore / {F['rear_blind_deg']:.0f}° aft"]),
                color=DIM, fontsize=6.4, family="monospace", va="top", linespacing=1.55)
        ax.set_title("BAND UNROLLED · (θ, s)", color=INK, fontsize=7.5, family="monospace", loc="left")
    except Exception as ex:
        axes[2].text(0.05, 0.5, f"lattice: {str(ex)[:30]}", color=DIM, fontsize=6, family="monospace", transform=axes[2].transAxes)

def _eye_detail(ax, m, P):
    """The eye as a drawing: plan contours around the right eye from the built mesh, the eye circle and visual
    arc from eyes.eye_geometry (the same numbers the builder used), a transverse section through the eye centre
    below it, and the dimensions the primitive will be fitted to: R, band height over the cheek, arc."""
    try:
        from eyes import eye_geometry, fov
        G = eye_geometry(P)
    except Exception as ex:
        ax.text(0, 0, f"eye: {str(ex)[:30]}", color=DIM, fontsize=7, family="monospace"); return
    if G["blind"]:
        ax.set_title("EYE · blind (eyeSize 0)", color=INK, fontsize=8, family="monospace", loc="left"); return
    xe, ye, eR = G["xe"], G["ye"], G["eR"]; W = 3.0 * eR
    sub = m.slice_plane([xe - W, 0, 0], [1, 0, 0]).slice_plane([xe + W, 0, 0], [-1, 0, 0]) \
           .slice_plane([0, ye - W, 0], [0, 1, 0]).slice_plane([0, ye + W, 0], [0, -1, 0])
    (x0, y0, z0), (x1, y1, z1) = sub.bounds
    for z in np.linspace(z0 + 0.3, z1 - 0.15, 16):
        try:
            sec = sub.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if sec is None: continue
            for e in sec.entities: p = sec.vertices[e.points]; ax.plot(p[:, 0], p[:, 1], color=INK, lw=0.4, alpha=0.85)
        except Exception: pass
    ax.add_patch(Circle((xe, ye), eR, fill=False, color=INK, lw=1.0, ls=(0, (2, 2))))
    ax.add_patch(Arc((xe, ye), 2.5 * eR, 2.5 * eR, angle=0, theta1=-G["arc_deg"] / 2, theta2=G["arc_deg"] / 2, color=INK, lw=0.8))
    ax.plot([xe], [ye], marker="+", color=INK, ms=6, mew=0.6)
    # transverse section (plane y = ye) drawn below the plan, same x scale; cheek reference = z at the eye's outer foot
    try:
        sec = sub.section(plane_origin=[0, ye, 0], plane_normal=[0, 1, 0]); yb = ye - W - 4
        zs = []
        for e in sec.entities:
            p = sec.vertices[e.points]; o = np.argsort(p[:, 0]); ax.plot(p[o, 0], yb - (z1 - p[o, 2]), color=INK, lw=0.7); zs.append(p)
        pts = np.vstack(zs); crest = pts[:, 2].max()
        foot = pts[np.abs(pts[:, 0] - (xe + 1.6 * eR)) < 0.5, 2]; foot = float(np.median(foot)) if len(foot) else float(pts[:, 2].min())
        ax.plot([xe - W, xe + W], [yb - (z1 - foot)] * 2, color=DIM, lw=0.4, ls=(0, (4, 3)))
        _dim(ax, xe + W + 2, yb - (z1 - foot), xe + W + 2, yb - (z1 - crest), f"h {crest - foot:.1f}", off=0)
        ax.text(xe - W, yb - (z1 - z0) - 5, "SECTION B-B · transverse through the eye centre · cheek dashed", color=INK, fontsize=6.5, family="monospace")
        ax.set_ylim(yb - (z1 - z0) - 9, ye + W + 8)
    except Exception:
        ax.set_ylim(ye - W - 4, ye + W + 8)
    _dim(ax, xe - eR, ye + W + 2, xe + eR, ye + W + 2, f"R {eR:.1f}", off=0)
    if P.get("eyeSolid", 0) > 0.5:                                                   # lattice as built (eye_solid.lens_centres), outward view
        try:
            from eye_solid import lens_centres, eye_params
            EP = eye_params(P, eR); import math as _m
            cs = lens_centres(EP["R"], EP["H"], _m.radians(EP["slope_deg"]), _m.radians(EP["arc_deg"]), EP["lensD"] * eR, EP["lensGap"])
            ox = xe + W + 4; oz = ye - W - 4 - (z1 - z0) - 14 - EP["H"]
            rb = EP["R"] + EP["H"] * _m.tan(_m.radians(EP["slope_deg"]))
            yt, yb2 = EP["R"] * _m.sin(_m.radians(EP["arc_deg"]) / 2), rb * _m.sin(_m.radians(EP["arc_deg"]) / 2)
            ax.plot([ox - yt, ox + yt, ox + yb2, ox - yb2, ox - yt], [oz + EP["H"], oz + EP["H"], oz, oz, oz + EP["H"]], color=INK, lw=0.7)
            for th, sz, r in cs: ax.add_patch(Circle((ox + r * _m.sin(th), oz + sz), 0.5 * EP["lensD"] * eR, fill=False, color=INK, lw=0.35))
            ax.text(ox - yb2, oz - 3, f"OUTWARD VIEW · {len(cs)} lenses · D {EP['lensD'] * eR:.1f} mm", color=INK, fontsize=6.5, family="monospace")
            ax.set_xlim(xe - W - 4, ox + yb2 + 6)
        except Exception as ex:
            ax.text(xe + W + 4, ye, f"lattice: {str(ex)[:24]}", color=DIM, fontsize=6, family="monospace")
    ax.set_xlim(xe - W - 4, xe + W + 14)
    F = fov(P)
    ax.set_title(f"EYE · right · lat {xe / G['head_halfwidth']:.2f} wh · pos {-ye / G['head_length']:.2f} Lc · arc {G['arc_deg']:.0f}° · profile exp {G['exponent']:.0f}"
                 f"\nFOV v{F['fov_version']} · azimuth {F['azimuth_deg']:.0f}° · elev ±{F['elevation_half_deg']:.0f}° · blind {F['front_blind_deg']:.0f}° fore",
                 color=INK, fontsize=7.5, family="monospace", loc="left")

def _fov_lines(P):
    try:
        from eyes import fov
        f = fov(P)
        if f["blind"]: return ["FOV v%s   blind" % f["fov_version"]]
        return [f"FOV v{f['fov_version']}", f"  azimuth  {f['azimuth_deg']:.0f}°", f"  blind    {f['front_blind_deg']:.0f}° fore · {f['rear_blind_deg']:.0f}° aft",
                f"  binoc    {f['binocular_deg']:.0f}°", f"  elev     ±{f['elevation_half_deg']:.0f}°"]
    except Exception as ex:
        return [f"FOV      — ({str(ex)[:20]})"]

def sheet(m, P, meas, path, title="TRILOBITE MORPHOSPACE", enrolled=None):
    m = m.copy(); m.apply_translation([-m.centroid[0], 0, 0])
    (x0, y0, z0), (x1, y1, z1) = m.bounds
    fig = plt.figure(figsize=(16.5, 11.7), facecolor=BG)                      # A3 landscape
    gs = fig.add_gridspec(6, 9, left=0.03, right=0.98, top=0.95, bottom=0.05, wspace=0.25, hspace=0.35)
    # ---- fine grid over the whole sheet
    bgax = fig.add_axes([0, 0, 1, 1], zorder=-1); bgax.set_facecolor(BG); bgax.set_xticks([]); bgax.set_yticks([])
    for s in bgax.spines.values(): s.set_visible(False)
    for gx in np.linspace(0, 1, 41): bgax.axvline(gx, color="#1c1c1c", lw=0.4)
    for gy in np.linspace(0, 1, 29): bgax.axhline(gy, color="#1c1c1c", lw=0.4)
    # ---- plan view: contour lines
    ax = fig.add_subplot(gs[0:5, 0:4]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    for c in _contours(m, np.linspace(z0 + 0.4, z1 - 0.2, 14)):
        ax.plot(c[:, 0], c[:, 1], color=INK, lw=0.45, alpha=0.9)
    try:
        poly = trimesh.path.polygons.projected(m, normal=[0, 0, 1])
        for ring in ([poly.exterior] + list(poly.interiors)) if poly is not None else []:
            xy = np.array(ring.coords); ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=1.0)
    except Exception: pass
    ax.axhline(0, color=DIM, lw=0.4, ls=(0, (6, 4))); ax.axvline(0, color=DIM, lw=0.4, ls=(0, (6, 4)))
    try:
        from eyes import eye_geometry
        G = eye_geometry(P)
    except Exception: pass
    W, L = x1 - x0, y1 - y0
    _dim(ax, x0, y0, x1, y0, f"{W:.1f}", off=-8)
    _dim(ax, x1, y0, x1, y1, f"{L:.1f}", off=10)
    pitch = meas.get("pitch", 0)
    if pitch: _dim(ax, x0, y0 + P["cephFrac"] * P["length"] * 0 + 0, x0, y0 + pitch, f"p {pitch:.1f}", off=-18)
    ax.text(x0, y1 + 6, "PLAN · dorsal · contours every %.1f mm" % ((z1 - z0) / 14), color=INK, fontsize=8, family="monospace")
    # axis triad, 10 mm long, at the front-left corner: x across, y along, z toward the viewer (dot)
    tx, ty = x0 - 20, y0 - 14
    ax.annotate("", xy=(tx + 10, ty), xytext=(tx, ty), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8)); ax.text(tx + 11.5, ty, "x", color=INK, fontsize=7, va="center", family="monospace")
    ax.annotate("", xy=(tx, ty + 10), xytext=(tx, ty), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8)); ax.text(tx, ty + 11.5, "y", color=INK, fontsize=7, ha="center", family="monospace")
    ax.add_patch(Circle((tx, ty), 0.9, fill=False, color=INK, lw=0.8)); ax.add_patch(Circle((tx, ty), 0.25, color=INK)); ax.text(tx - 3.5, ty - 3.5, "z", color=INK, fontsize=7, family="monospace")
    ax.text(tx + 5, ty - 4.5, "10 mm", color=DIM, fontsize=6, ha="center", family="monospace")
    ax.set_xlim(x0 - 26, x1 + 26); ax.set_ylim(y0 - 22, y1 + 12)
    # ---- lateral: silhouette + midline section
    ax2 = fig.add_subplot(gs[5:6, 0:4]); ax2.set_facecolor(BG); ax2.set_aspect("equal"); ax2.axis("off")
    try:
        poly = trimesh.path.polygons.projected(m, normal=[1, 0, 0])
        for ring in ([poly.exterior] + list(poly.interiors)) if poly is not None else []:
            xy = np.array(ring.coords); ax2.plot(xy[:, 0], xy[:, 1], color=DIM, lw=0.6)
    except Exception: pass
    try:
        s = m.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0])
        for e in s.entities: p = s.vertices[e.points]; ax2.plot(p[:, 1], p[:, 2], color=INK, lw=0.7)
    except Exception: pass
    # section C–C: longitudinal through the eye centre (x = xe), so the eye's profile sits on the lateral view
    try:
        from eyes import eye_geometry
        G = eye_geometry(P)
        if not G["blind"]:
            s = m.section(plane_origin=[G["xe"], 0, 0], plane_normal=[1, 0, 0])
            for e in s.entities: p = s.vertices[e.points]; ax2.plot(p[:, 1], p[:, 2], color=INK, lw=0.55, alpha=0.55)
    except Exception: pass
    hz = meas.get("hinge_z")
    if hz: ax2.axhline(hz, color=DIM, lw=0.5, ls=(0, (4, 3))); ax2.text(y0 - 2, hz + 1.5, f"hinge z {hz:.1f}", color=INK, fontsize=7, va="bottom", ha="left", family="monospace")
    _dim(ax2, y1 + 14, z0, y1 + 14, z1, f"{z1 - z0:.1f}", off=0)
    ax2.text(y0, z1 + 6, "SECTION A–A · midline (ink) · C–C through the eye (half tone) · lateral silhouette dimmed", color=INK, fontsize=8, family="monospace")
    ty2, tz2 = y0 - 6, z0 - 2
    ax2.annotate("", xy=(ty2 + 10, tz2), xytext=(ty2, tz2), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8)); ax2.text(ty2 + 11.5, tz2, "y", color=INK, fontsize=7, va="center", family="monospace")
    ax2.annotate("", xy=(ty2, tz2 + 10), xytext=(ty2, tz2), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8)); ax2.text(ty2, tz2 + 11.5, "z", color=INK, fontsize=7, ha="center", family="monospace")
    ax2.set_xlim(y0 - 12, y1 + 34); ax2.set_ylim(z0 - 6, z1 + 12)
    # ---- enrollment: the animal itself, enrolled to e_max, superimposed on the flat section
    ax3 = fig.add_subplot(gs[0:3, 4:7]); ax3.set_facecolor(BG); ax3.set_aspect("equal"); ax3.axis("off")
    tot = meas.get("total_curl_deg", 0) or 0; free = meas.get("free_curl_deg", 0) or 0
    try:
        s = m.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0])
        for e in s.entities: p = s.vertices[e.points]; ax3.plot(p[:, 1], p[:, 2], color=DIM, lw=0.5, alpha=0.7)
    except Exception: pass
    if enrolled is not None:
        en = enrolled.copy(); en.apply_translation([-en.centroid[0], 0, 0])
        try:                                                                      # faint silhouette, then the section in ink
            poly = trimesh.path.polygons.projected(en, normal=[1, 0, 0])
            for ring in ([poly.exterior] + list(poly.interiors)) if poly is not None else []:
                xy = np.array(ring.coords); ax3.plot(xy[:, 0], xy[:, 1], color=INK, lw=0.5, alpha=0.35)
        except Exception: pass
        try:
            s = en.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0])
            for e in s.entities: p = s.vertices[e.points]; ax3.plot(p[:, 1], p[:, 2], color=INK, lw=0.8)
        except Exception: pass
        # the closure gap, dimensioned between the nearest head and tail points of the section (from the part split)
        gap = meas.get("closure_gap_mm")
        if gap not in (None, "—") and float(gap) > 0:
            (ey0, ez0), (ey1, ez1) = en.bounds[0][1:], en.bounds[1][1:]
            ax3.text(ey0, ez0 - 6, f"closure gap {float(gap):.1f} mm", color=INK, fontsize=7, family="monospace")
        elif gap not in (None, "—"):
            ax3.text(en.bounds[0][1], en.bounds[0][2] - 6, "closed", color=INK, fontsize=7, family="monospace")
    ax3.set_title(f"ENROLLED · e_max {meas.get('e_max', '—')} · {free:.0f}° of {tot:.0f}° · flat dimmed", color=INK, fontsize=8, family="monospace", loc="left")
    # ---- the six dials, drawn as they are on the site: a line and a dot. This animal is one point in the space they span.
    ax4 = fig.add_subplot(gs[3:4, 4:7]); ax4.set_facecolor(BG); ax4.axis("off")
    dials = [("SCULPT", P.get("furrowDepth", 0) / 1.2), ("HEAD", (P["cephFrac"] - 0.22) / 0.2), ("TAIL", (P["pygFrac"] - 0.05) / 0.33),
             ("ELONGATION", (P["segCount"] - 4) / 10), ("SPINES", min(1, P.get("spineBase", 0) / 0.9 + P.get("genalSpine", 0) / 1.8)), ("EYES", (P.get("eyeSize", 0) - 0.04) / 0.26)]
    for i, (n, v) in enumerate(dials):
        v = float(min(1, max(0, v))); y = 1 - i * 0.19
        ax4.plot([0, 1], [y, y], color=INK, lw=0.6)
        ax4.plot([v], [y], marker="o", ms=6, color=INK)
        ax4.text(-0.03, y, n, color=INK, fontsize=7, ha="right", va="center", family="monospace")
        ax4.text(1.04, y, f"{v:.2f}", color=DIM, fontsize=7, va="center", family="monospace")
    ax4.set_xlim(-0.34, 1.16); ax4.set_ylim(-0.12, 1.1)
    ax4.set_title("DIALS · morphospace", color=INK, fontsize=8, family="monospace", loc="left")
    # ---- eye detail: enlarged plan of the right eye with contours, and a transverse section through its centre
    _eye_panels(fig, gs, m, P)
    # ---- title block
    ax5 = fig.add_subplot(gs[0:3, 7:9]); ax5.set_facecolor(BG); ax5.axis("off")
    for s in ["    " + title, "", f"DRAWING  {meas.get('params', '—')}", f"SCHEMA   {P.get('_schema', '5.0')}", "",
              f"LENGTH   {L:.1f} mm", f"WIDTH    {W:.1f} mm", f"RELIEF   {z1 - z0:.1f} mm", f"SEGMENTS {int(P['segCount'])}", f"PITCH    {pitch:.2f} mm", "",
              f"HINGE    z {meas.get('hinge_z', '—')}   Ø {P.get('boreDia', '—')} bore", f"KNUCKLE  {meas.get('knuckle', '—')} mm × {int(P.get('nKnuckles', 3))}", f"STOP     {P['maxAngle']}° / joint", "",
              f"E_MAX    {meas.get('e_max', '—')}", f"GAP      {meas.get('closure_gap_mm', '—')} mm", f"CLASS    {str(meas.get('enroll_class', '—')).upper()}", f"PRINT    {'VALID' if meas.get('print_valid') else 'CHECK'}", "",
              "GENAL PATH", f"  {P.get('genalPath', '—')}", f"  {P.get('genalCurve', 0)}°  {P.get('genalWidthMM', 0)} mm", "", "SHEET 1 / 1     REV A"]:
        pass
    lines = ["    " + title, "", f"DRAWING  {meas.get('params', '—')}", "",
             f"LENGTH   {L:.1f} mm", f"WIDTH    {W:.1f} mm", f"RELIEF   {z1 - z0:.1f} mm", f"SEGMENTS {int(P['segCount'])}", f"PITCH    {pitch:.2f} mm", "",
             f"HINGE z  {meas.get('hinge_z', '—')} mm", f"KNUCKLE  {meas.get('knuckle', '—')} mm × {int(P.get('nKnuckles', 3))}", f"STOP     {P['maxAngle']}° / joint", "",
             f"E_MAX    {meas.get('e_max', '—')}", f"GAP      {meas.get('closure_gap_mm', '—')} mm", f"CLASS    {str(meas.get('enroll_class', '—')).upper()}", f"PRINT    {'VALID' if meas.get('print_valid') else 'CHECK'}", "",
             "GENAL PATH", f"  {P.get('genalPath', '—')}", f"  {P.get('genalCurve', 0)}° · {P.get('genalWidthMM', 0)} mm", "", "SHEET 1 / 1   REV A"]
    ax5.text(0.02, 0.98, "\n".join(lines), color=INK, fontsize=8, family="monospace", va="top", ha="left", linespacing=1.55)
    for s in ax5.spines.values(): s.set_visible(True); s.set_color(INK); s.set_linewidth(0.8)
    ax5.set_xticks([]); ax5.set_yticks([]); ax5.axis("on")
    fig.text(0.98, 0.018, "Claire Choi · Cornell", color=INK, fontsize=8, ha="right", va="bottom", family="monospace")
    fig.savefig(path, dpi=110, facecolor=BG); plt.close(fig); return path

if __name__ == "__main__":
    import sys, json
    m = trimesh.load(sys.argv[1]); P = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    meas = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
    en = trimesh.load(sys.argv[5]) if len(sys.argv) > 5 else None
    sheet(m, P, meas, sys.argv[4] if len(sys.argv) > 4 else "sheet.png", enrolled=en)
