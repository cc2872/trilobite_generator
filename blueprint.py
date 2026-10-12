"""
blueprint.py — draw a built animal as a technical sheet: white ink on black.
Plan (dorsal) as contour lines, lateral silhouette, midline section, dimension lines, enrollment dial, title block.
    sheet(flat_mesh, P, measure, path)   -> writes a PNG (matplotlib) and returns the figure
"""
import math, numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, Rectangle

INK, BG, DIM = "#ffffff", "#000000", "#9a9a9a"
CMAPS = ("magma", "viridis", "twilight", "inferno", "cividis", "gray", "rainbow")   # scientific field colormaps (wave-optics look) + rainbow (full spectrum, vivid)
RELIEF_ALPHA = 0.5   # the relief field sits semi-transparent under the blueprint lines — muted, not vibrant
BED_MM = 256         # target print bed (square), mm — common desktop FDM; the sheet flags a specimen that won't fit

def _fits_bed(w, h, bed=BED_MM):
    """Does a w x h footprint fit on a square bed, flat or rotated? (print-in-place must lie flat)."""
    if max(w, h) <= bed: return True
    for th in np.radians(np.arange(1, 46, 1.0)):                               # try rotations up to 45 deg
        c, s = np.cos(th), np.sin(th)
        if w * c + h * s <= bed and w * s + h * c <= bed: return True
    return False

def _eye_lenses(ax, head, P, color=INK):
    """Fill each eye on the dorsal plan with its lens packing (the packing factor, as circles): the eye outline and a
    hex lattice of lenses sized by lensD and spaced by lensGap. The two eye domes are detected off-axis on the head."""
    try:
        if P.get("eyeSolid", 1) < 0.5 or P.get("eyeSize", 0) <= 0.01: return
        V = head.vertices; hw = max(abs(V[:, 0].min()), V[:, 0].max())
        eR = P["eyeSize"] * hw
        sel = (V[:, 0] > 0.3 * hw) & (V[:, 0] < 0.97 * hw)                   # the right eye dome: off-axis, highest point
        if sel.sum() < 20 or eR < 1: return
        Vs = V[sel]; ex, ey = Vs[np.argmax(Vs[:, 2]), 0], Vs[np.argmax(Vs[:, 2]), 1]
        D = max(P["lensD"] * eR, 0.4); pitch = D * (1 + P["lensGap"]); row = 0.87 * pitch
        lens = []
        for iy in range(-int(eR / row) - 1, int(eR / row) + 2):
            yy = iy * row; offx = 0.5 * pitch * (iy & 1)
            for ix in range(-int(eR / pitch) - 1, int(eR / pitch) + 2):
                xx = ix * pitch + offx
                if xx * xx + yy * yy <= (eR - 0.45 * D) ** 2: lens.append((xx, yy))
        from matplotlib.patches import Circle
        for sx in (1, -1):
            ax.add_patch(Circle((sx * ex, ey), eR, fill=False, color=color, lw=0.9))
            for lx, ly in lens: ax.add_patch(Circle((sx * ex + lx, ey + ly), 0.5 * D, fill=False, color=color, lw=0.35))
    except Exception: pass

def _field(ax, GA, GB, Z, cmap):
    """Draw a relief scalar field as a colormapped image. Scientific maps render soft and semi-transparent (the
    wave-optics look); 'rainbow' renders the full hue spectrum, vivid and opaque, to match the 3D Relief rainbow."""
    if Z is None or not np.isfinite(Z).any(): return
    from matplotlib.colors import PowerNorm
    vlo, vhi = np.nanmin(Z), np.nanmax(Z)
    rainbow = (cmap == "rainbow")
    mcmap = "hsv" if rainbow else cmap                                 # matplotlib's cyclic full-spectrum = the 3D's height->hue ramp
    alpha = 0.92 if rainbow else RELIEF_ALPHA                          # rainbow runs vivid; the scientific fields stay muted
    ax.imshow(np.ma.masked_invalid(Z), extent=[GA.min(), GA.max(), GB.min(), GB.max()], origin="lower", cmap=mcmap,
              norm=PowerNorm(0.6, vmin=vlo - 0.04 * (vhi - vlo), vmax=vhi), interpolation="bilinear", aspect="equal", zorder=0, alpha=alpha)

def _strong_outline(ax, GX, GY, Mask, color=INK, lw=1.9, draw=False):
    """A silhouette traced from the footprint mask's boundary (robust; no shapely). Disabled by default — the bright
    white line was too harsh over the soft relief field; the field + contour lines carry the shape. Set draw=True to show it."""
    if not draw or Mask is None or not Mask.any(): return
    try:
        ax.contour(GX, GY, Mask.astype(float), levels=[0.5], colors=[color], linewidths=lw, antialiased=True)
    except Exception: pass

def _relief_grid(m, n=300, view="top", field=True):
    """A relief scalar field on a regular grid, masked to the footprint (NaN outside), for a colormapped rendering.
    view='top': dorsal height z over (x, y). view='side': flank depth |x| (half-width) over (y, z) — the lateral relief.
    field=False skips the (expensive) interpolation and returns only the mask (for the blueprint silhouette).
    Returns (GA, GB, field-or-None, mask); the mask boundary is the silhouette. Robust (KDTree mask, no shapely)."""
    try:
        from scipy.spatial import cKDTree
        V = m.vertices
        if view == "side":                                         # the right flank, depth = |x| over (y, z)
            sel = V[:, 0] > 0.5
            if sel.sum() < 50: sel = np.ones(len(V), bool)
            Pxy = V[sel][:, [1, 2]]; val = np.abs(V[sel][:, 0])
        else:                                                      # the dorsal (upward-facing) surface, height z over (x, y)
            up = m.vertex_normals[:, 2] > 0.15; sel = up if up.sum() > 50 else np.ones(len(V), bool)
            Pxy = V[sel][:, [0, 1]]; val = V[sel][:, 2]
        a0, b0, a1, b1 = Pxy[:, 0].min(), Pxy[:, 1].min(), Pxy[:, 0].max(), Pxy[:, 1].max()
        ga = np.linspace(a0, a1, n); gb = np.linspace(b0, b1, max(2, int(n * (b1 - b0) / max(a1 - a0, 1e-6))))
        GA, GB = np.meshgrid(ga, gb); pts = np.c_[GA.ravel(), GB.ravel()]
        # mask to the actual footprint: a grid cell is "on the animal" if a surface point sits within ~2 cells of it.
        d, _ = cKDTree(Pxy).query(pts)
        cell = 2.5 * max(ga[1] - ga[0], gb[1] - gb[0])
        M = (d < cell).reshape(GA.shape)
        Z = None
        if field:
            from scipy.interpolate import griddata
            Z = np.where(M, griddata(Pxy, val, (GA, GB), method="linear"), np.nan)
        return GA, GB, Z, M
    except Exception:
        return None, None, None, None

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
    from anatomy.head import eye_geometry, fov
    G = eye_geometry(P)
    axes = [fig.add_subplot(gs[4:6, 4:5]), fig.add_subplot(gs[4:6, 5:6]), fig.add_subplot(gs[4:6, 6:7]), fig.add_subplot(gs[4:6, 7:9])]
    for ax in axes: ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    if G["blind"]:
        axes[0].set_title("EYE · blind", color=INK, fontsize=8, family="monospace", loc="left"); return
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
        from anatomy.head import lens_centres, eye_params
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
        ax.text(-arcl / 2, -1.2, "\n".join([f"arc {arcl:.1f} mm · band {H:.1f} mm",
                                             f"pitch {pitch:.2f} · rows {rows}",
                                             f"packing {D / pitch:.2f}",
                                             f"arc {EP['arc_deg']:.0f}° · lean {EP['slope_deg']:.0f}° · brim {EP['shade']:.2f} R",
                                             f"rise {EP['lensRise']:.2f} D/2",
                                             f"az {F['azimuth_deg']:.0f}° · el ±{F['elevation_half_deg']:.0f}°",
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
        from anatomy.head import eye_geometry, fov
        G = eye_geometry(P)
    except Exception as ex:
        ax.text(0, 0, f"eye: {str(ex)[:30]}", color=DIM, fontsize=7, family="monospace"); return
    if G["blind"]:
        ax.set_title("EYE · blind", color=INK, fontsize=8, family="monospace", loc="left"); return
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
        ax.text(xe - W, yb - (z1 - z0) - 5, "SECTION B–B", color=INK, fontsize=6.5, family="monospace")
        ax.set_ylim(yb - (z1 - z0) - 9, ye + W + 8)
    except Exception:
        ax.set_ylim(ye - W - 4, ye + W + 8)
    _dim(ax, xe - eR, ye + W + 2, xe + eR, ye + W + 2, f"R {eR:.1f}", off=0)
    if P.get("eyeSolid", 0) > 0.5:                                                   # lattice as built (eye_solid.lens_centres), outward view
        try:
            from anatomy.head import lens_centres, eye_params
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
        from parts import fov
        f = fov(P)
        if f["blind"]: return ["FOV v%s   blind" % f["fov_version"]]
        return [f"FOV v{f['fov_version']}", f"  azimuth  {f['azimuth_deg']:.0f}°", f"  blind    {f['front_blind_deg']:.0f}° fore · {f['rear_blind_deg']:.0f}° aft",
                f"  binoc    {f['binocular_deg']:.0f}°", f"  elev     ±{f['elevation_half_deg']:.0f}°"]
    except Exception as ex:
        return [f"FOV      — ({str(ex)[:20]})"]

def sheet(m, P, meas, path, title="TRILOBITE MORPHOSPACE", enrolled=None, cmap="none"):
    bw = cmap not in CMAPS                                                     # 'none'/invalid -> black & white blueprint (default)
    m = m.copy(); m.apply_translation([-m.centroid[0], 0, 0])
    (x0, y0, z0), (x1, y1, z1) = m.bounds
    fig = plt.figure(figsize=(16.5, 11.7), facecolor=BG)                      # A3 landscape
    gs = fig.add_gridspec(6, 9, left=0.03, right=0.98, top=0.95, bottom=0.05, wspace=0.25, hspace=0.35)
    # ---- fine grid over the whole sheet
    bgax = fig.add_axes([0, 0, 1, 1], zorder=-1); bgax.set_facecolor(BG); bgax.set_xticks([]); bgax.set_yticks([])
    for s in bgax.spines.values(): s.set_visible(False)
    for gx in np.linspace(0, 1, 41): bgax.axvline(gx, color="#1c1c1c", lw=0.4)
    for gy in np.linspace(0, 1, 29): bgax.axhline(gy, color="#1c1c1c", lw=0.4)
    # ---- plan view: relief as a colormapped field + fine contour lines
    ax = fig.add_subplot(gs[0:5, 0:4]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    GX, GY, Z, Mask = _relief_grid(m, field=not bw)
    if not bw: _field(ax, GX, GY, Z, cmap)                                                  # the dorsal relief field (heatmap mode)
    for c in _contours(m, np.linspace(z0 + 0.4, z1 - 0.2, 14)):
        ax.plot(c[:, 0], c[:, 1], color=INK, lw=0.45 if bw else 0.4, alpha=0.9 if bw else 0.22)
    _strong_outline(ax, GX, GY, Mask, draw=bw)                                             # the strong silhouette in blueprint mode
    try:
        poly = None
        for ring in ([poly.exterior] + list(poly.interiors)) if poly is not None else []:
            xy = np.array(ring.coords); ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=1.0)
    except Exception: pass
    ax.axhline(0, color=DIM, lw=0.4, ls=(0, (6, 4))); ax.axvline(0, color=DIM, lw=0.4, ls=(0, (6, 4)))
    try:
        from parts import eye_geometry
        G = eye_geometry(P)
    except Exception: pass
    W, L = x1 - x0, y1 - y0
    _dim(ax, x0, y0, x1, y0, f"{W:.1f}", off=-8)
    _dim(ax, x1, y0, x1, y1, f"{L:.1f}", off=10)
    pitch = meas.get("pitch", 0)
    if pitch: _dim(ax, x0, y0 + P["cephFrac"] * P["length"] * 0 + 0, x0, y0 + pitch, f"p {pitch:.1f}", off=-18)
    ax.text(x0, y1 + 6, "PLAN · blueprint" if bw else "PLAN · relief field (%s)" % cmap, color=INK, fontsize=8, family="monospace")
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
        from parts import eye_geometry
        G = eye_geometry(P)
        if not G["blind"]:
            s = m.section(plane_origin=[G["xe"], 0, 0], plane_normal=[1, 0, 0])
            for e in s.entities: p = s.vertices[e.points]; ax2.plot(p[:, 1], p[:, 2], color=INK, lw=0.55, alpha=0.55)
    except Exception: pass
    hz = meas.get("hinge_z")
    if hz: ax2.axhline(hz, color=DIM, lw=0.5, ls=(0, (4, 3))); ax2.text(y0 - 2, hz + 1.5, f"hinge z {hz:.1f}", color=INK, fontsize=7, va="bottom", ha="left", family="monospace")
    _dim(ax2, y1 + 14, z0, y1 + 14, z1, f"{z1 - z0:.1f}", off=0)
    ax2.text(y0, z1 + 6, "SECTION A–A · C–C at the eye", color=INK, fontsize=8, family="monospace")
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
    ax3.set_title(f"ENROLLED · e@stop {meas.get('e_max', '—')} · {free:.0f}° of {tot:.0f}°", color=INK, fontsize=8, family="monospace", loc="left")
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
    ax4.set_title("DIALS", color=INK, fontsize=8, family="monospace", loc="left")
    # ---- eye detail: enlarged plan of the right eye with contours, and a transverse section through its centre
    _eye_panels(fig, gs, m, P)
    # ---- title block
    ax5 = fig.add_subplot(gs[0:3, 7:9]); ax5.set_facecolor(BG); ax5.axis("off")
    lines = ["    " + title, "", f"DRAWING  {meas.get('params', '—')}", "",
             f"LENGTH   {L:.1f} mm", f"WIDTH    {W:.1f} mm", f"RELIEF   {z1 - z0:.1f} mm", f"SEGMENTS {int(P['segCount'])}", f"PITCH    {pitch:.2f} mm", "",
             f"HINGE z  {meas.get('hinge_z', '—')} mm", f"KNUCKLE  {meas.get('knuckle', '—')} mm × {int(P.get('nKnuckles', 3))}", f"STOP     {P['maxAngle']}° / joint", "",
             f"E@STOP   {meas.get('e_max', '—')}", f"GAP      {meas.get('closure_gap_mm', '—')} mm", f"CLASS    {str(meas.get('enroll_class', '—')).upper()}", f"PRINT    {'VALID' if meas.get('print_valid') else 'CHECK'}", "",
             "GENAL PATH", f"  {P.get('genalPath', '—')}", f"  {P.get('genalCurve', 0)}° · {P.get('genalWidthMM', 0)} mm", "",
             "SPEC     panel below +", "         PNG metadata", "", "SHEET 1 / 1   REV A"]
    ax5.text(0.02, 0.98, "\n".join(lines), color=INK, fontsize=8, family="monospace", va="top", ha="left", linespacing=1.55)
    for s in ax5.spines.values(): s.set_visible(True); s.set_color(INK); s.set_linewidth(0.8)
    ax5.set_xticks([]); ax5.set_yticks([]); ax5.axis("on")
    # ---- SPEC panel: the parameters moved off the 6.1 defaults, printed on the page. The FULL 98-param spec is in the
    #      PNG metadata (embedded at save, read_sheet_spec.py) — this visible summary also survives a screenshot.
    ax6 = fig.add_subplot(gs[3:4, 7:9]); ax6.set_facecolor(BG); ax6.axis("off")
    try:
        import schema as _sch; _def = _sch.table_defaults()
        def _diff(a, b): return abs(a - b) > 1e-9 if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a != b
        _fmt = lambda v: (f"{v:g}" if isinstance(v, (int, float)) else str(v))
        _chg = [f"{k}={_fmt(P[k])}" for k in sorted(P) if k in _def and _diff(P[k], _def[k])]
    except Exception:
        _chg = []
    _MAXP = 24; _extra = len(_chg) - _MAXP
    _show = _chg[:_MAXP] + ([f"...+{_extra} more (in metadata)"] if _extra > 0 else [])
    _rows = ["  ".join(_show[i:i + 3]) for i in range(0, len(_show), 3)] or ["(all parameters at 6.1 defaults)"]
    ax6.set_title("SPEC · changed from 6.1 defaults", color=INK, fontsize=7, family="monospace", loc="left")
    ax6.text(0.0, 0.98, "full JSON (98 params) in PNG metadata · read_sheet_spec.py\n\n" + "\n".join(_rows),
             color=INK, fontsize=5.5, family="monospace", va="top", ha="left", linespacing=1.7, transform=ax6.transAxes)
    fig.text(0.98, 0.018, "Claire Choi · Cornell", color=INK, fontsize=8, ha="right", va="bottom", family="monospace")
    # embed the full specification in the PNG so the sheet is self-describing: the exact params (and hash / schema /
    # instrument) travel with the image and can be read back with scripts/read_sheet_spec.py — no cache lookup needed.
    import json as _json
    try:
        import schema as _schema; _sv = _schema.SCHEMA_VERSION
    except Exception:
        _sv = str(P.get("schema", ""))
    spec = _json.dumps({"hash": meas.get("params", ""), "schema": _sv, "instrument": meas.get("instrument", ""),
                        "params": P}, sort_keys=True, default=str)
    fig.savefig(path, dpi=110, facecolor=BG, metadata={"trilobite": spec}); plt.close(fig); return path

if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] != "gallery":
    import sys, json
    m = trimesh.load(sys.argv[1]); P = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    meas = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
    en = trimesh.load(sys.argv[5]) if len(sys.argv) > 5 else None
    sheet(m, P, meas, sys.argv[4] if len(sys.argv) > 4 else "sheet.png", enrolled=en)


# ---------------------------------------------------------------- the isopod model's sheet (30 Sep 2026)
# The isopod model is a different animal from the pin build: ball joints, a crescent head on the isopod's head piece,
# print-in-place at the isopod's own size. The pin sheet above (hinge z, knuckles, the enrollment instrument, the eye
# lattice) does not describe it, so it gets its own sheet: dorsal plan, lateral + midline section, the chain curled on
# its ball joints, and a title block with the print size, the pieces, and the per-joint pivots.
def _curl_mats(pivots, curl_deg):
    """Rest -> posed transform per piece: each joint curled by curl_deg about its pivot (y, z), the chain behind down.
    The isopod kinematics (isopod_model/body.pose), in the exported print-mm frame."""
    T = [np.eye(4)]
    for (yc, zc) in pivots:
        T.append(T[-1] @ trimesh.transformations.rotation_matrix(math.radians(-curl_deg), (1, 0, 0), (0, yc, zc)))
    return T

PLA_G_CM3 = 1.24       # PLA density, for the schedule's mass column
PRINT_CM3_PER_H = 15.0 # a nominal FDM deposition rate (0.4 nozzle, 0.2 mm layers, ~20 % infill) for the schedule's time estimate
OVERHANG_DEG = 45.0    # faces steeper than this, facing down, count as unsupported overhang

def _sections_x0(parts):
    """The midline (x = 0) section of every part as a list of (y, z) polylines, in the parts' own frame. A rotation
    about x keeps that plane, so posed sections are these polylines rotated in 2D (no re-sectioning)."""
    out = []
    for p in parts:
        polys = []
        try:
            s = p.section(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0])
            if s is not None:
                for e in s.entities: polys.append(s.vertices[e.points][:, 1:3].copy())
        except Exception: pass
        out.append(polys)
    return out

def _rot2(pts, yc, zc, deg):
    """Rotate (y, z) points about (yc, zc) by deg (positive = the chain behind curls down, as body.pose does)."""
    c, s = math.cos(math.radians(-deg)), math.sin(math.radians(-deg))
    d = pts - [yc, zc]
    return np.c_[d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c] + [yc, zc]

def _posed_sections(secs, pivots, curl_deg):
    """Every part's midline polylines carried through the chain curled curl_deg per joint (2D kinematics)."""
    out = [list(secs[0])]
    for k in range(1, len(secs)):
        polys = secs[k]
        for j in range(k - 1, -1, -1):                      # joints ahead of this part, innermost first (same as the matrix chain)
            yc, zc = pivots[j]; polys = [_rot2(q, yc, zc, curl_deg) for q in polys]
        out.append(polys)
    return out

def _overhang_fraction(m, z_floor, deg=OVERHANG_DEG):
    """Area fraction of the mesh facing down steeper than deg, excluding the flat bottom on the bed (a slicer's
    support estimate)."""
    try:
        nz = m.face_normals[:, 2]; cz = m.triangles_center[:, 2]; A = m.area_faces
        over = (nz < -math.cos(math.radians(deg))) & (cz > z_floor + 0.6)
        return float(A[over].sum() / max(A.sum(), 1e-9))
    except Exception: return float("nan")

def _pack_bed(sizes, bed=BED_MM, margin=4.0):
    """Shelf-pack (name, w, l) footprints onto bed x bed plates, each piece turned long side along x. Returns
    [plate, ...], plate = [(name, x, y, w, h)], plus the names that fit on no plate."""
    items = []
    for nm, w, l in sizes:
        a, b = (w, l) if w >= l else (l, w)
        items.append((nm, a + margin, b + margin))
    items.sort(key=lambda t: -t[2])
    plates, nofit = [], []
    for nm, w, h in items:
        if w > bed + margin: nofit.append(nm); continue
        placed = False
        for pl in plates:
            x, y, shelf_h, boxes = pl["x"], pl["y"], pl["h"], pl["boxes"]
            if x + w <= bed: pass
            else: x, y, shelf_h = 0.0, y + shelf_h, 0.0
            if y + h <= bed:
                boxes.append((nm, x, y, w - margin, h - margin)); pl.update(x=x + w, y=y, h=max(shelf_h, h)); placed = True; break
        if not placed:
            plates.append(dict(x=w, y=0.0, h=h, boxes=[(nm, 0.0, 0.0, w - margin, h - margin)]))
    return [pl["boxes"] for pl in plates], nofit

def _spark(vals, n=8):
    """A one-line sparkline of values (block glyphs), for the editor's row-width profile."""
    if not vals: return ""
    lo, hi = min(vals), max(vals); blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(n - 1, int((v - lo) / (hi - lo + 1e-9) * n))] for v in vals)

def _box(ax, color="#555555"):
    for s in ax.spines.values(): s.set_visible(True); s.set_color(color); s.set_linewidth(0.7)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")

def isopod_sheet(parts, P, info, path, title="ISOPOD TRILOBITE", cmap="none"):
    """parts: the pieces in their rest frame (print mm, head first). info: {names, pivots, curl_deg, params, schema,
    build_seconds, volumes, meta (per-piece bodies/watertight), checks (the manifest's print block), head, body,
    swing (free swing per joint, deg)}. cmap: a relief colormap, or 'none' for the black & white blueprint (default).
    The sheet (12 Oct 2026): dorsal and ventral plans with the joints numbered, lateral section, the coil at its true
    size, a sectioned joint at 2:1 with the curl drawn, the enrolment in five frames, a piece schedule (size, volume,
    mass, checks, pivot), the bed layout, print checks, morphospace + editor coordinates, and a title strip."""
    import json as _json
    bw = cmap not in CMAPS
    n = len(parts)
    names = info.get("names") or [f"piece {i}" for i in range(n)]
    pivots = [tuple(map(float, pv)) for pv in (info.get("pivots") or [])]
    curl = float(info.get("curl_deg", 30.0)); nj = len(pivots)
    meta = info.get("meta") or [{} for _ in parts]
    checks = info.get("checks") or {}
    flat = trimesh.util.concatenate([p.copy() for p in parts]); cx = float(flat.centroid[0]); flat.apply_translation([-cx, 0, 0])
    parts = [p.copy().apply_translation([-cx, 0, 0]) for p in parts]
    (x0, y0, z0), (x1, y1, z1) = flat.bounds; W, L, H = x1 - x0, y1 - y0, z1 - z0
    secs = _sections_x0(parts)
    T = _curl_mats(pivots, curl)
    posed = trimesh.util.concatenate([parts[k].copy().apply_transform(T[k]) for k in range(n)])
    psec = _posed_sections(secs, pivots, curl)
    mono = dict(family="monospace", color=INK)

    fig = plt.figure(figsize=(16.5, 11.7), facecolor=BG)                       # A3 landscape
    gs = fig.add_gridspec(24, 24, left=0.03, right=0.98, top=0.95, bottom=0.075, wspace=0.3, hspace=0.9)   # the plan fills the upper left, the section sits low under it
    bgax = fig.add_axes([0, 0, 1, 1], zorder=-1); bgax.set_facecolor(BG); bgax.set_xticks([]); bgax.set_yticks([])
    for s in bgax.spines.values(): s.set_visible(False)
    for gx in np.linspace(0, 1, 41): bgax.axvline(gx, color="#1c1c1c", lw=0.4)
    for gy in np.linspace(0, 1, 29): bgax.axhline(gy, color="#1c1c1c", lw=0.4)

    def plan(ax, m, mirror, levels, label, strong=True, fs=8, joints=True, pad=(20.0, 9.0)):
        GX, GY, Z, Mask = _relief_grid(m, view="top", field=not bw)
        if not bw: _field(ax, GX, GY, Z, cmap)
        sx = -1.0 if mirror else 1.0
        for c in _contours(m, levels): ax.plot(sx * c[:, 0], c[:, 1], color=INK, lw=0.45 if bw else 0.4, alpha=0.9 if bw else 0.22)
        if GX is not None and bw: _strong_outline(ax, sx * GX, GY, Mask, draw=strong)
        for k, (yc, _) in enumerate(pivots):
            ax.axhline(yc, color=DIM, lw=0.35, ls=(0, (3, 4)), alpha=0.5)
            if joints: ax.text(x1 + 2, yc, f"J{k + 1}", fontsize=5.5, va="center", color=DIM, family="monospace")
        ax.axvline(0, color=DIM, lw=0.4, ls=(0, (6, 4)), alpha=0.6)
        ax.text(x0, y1 + 4, label, fontsize=fs, **mono)
        ax.set_xlim(x0 - pad[0], x1 + pad[1]); ax.set_ylim(y0 - 14, y1 + 9)

    # ---- PLAN · dorsal
    ax = fig.add_subplot(gs[0:19, 0:8]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    plan(ax, flat, False, np.linspace(z0 + 0.4, z1 - 0.2, 14), ("PLAN · dorsal · relief %.1f mm" % H) if bw else ("PLAN · relief field (%s) · %.1f mm" % (cmap, H)))
    _eye_lenses(ax, parts[0], P)
    _dim(ax, x0, y0, x1, y0, f"{W:.1f}", off=-8); _dim(ax, x0, y0, x0, y1, f"{L:.1f}", off=-13)      # length on the left, joints on the right
    ax.text(x0, y0 - 13, "A", fontsize=7, **mono); ax.text(x1, y0 - 13, "A", fontsize=7, ha="right", **mono)
    sb = 100.0 if W >= 120 else 50.0                                                          # a scale bar in the plan's own mm
    sx0 = x0 + 0.5 * (W - sb)                                                                # centred between the A marks
    for i in range(5): ax.add_patch(Rectangle((sx0 + i * sb / 5, y0 - 20), sb / 5, 2.5, facecolor=INK if i % 2 == 0 else BG, edgecolor=INK, lw=0.5))
    ax.text(sx0, y0 - 21.5, "0", fontsize=5, va="top", **mono); ax.text(sx0 + sb, y0 - 21.5, f"{sb:.0f} mm", fontsize=5, va="top", ha="right", **mono)
    ax.set_ylim(y0 - 27, y1 + 9)
    # ---- PLAN · ventral (the animal turned over about its long axis: x mirrored, z flipped; the joint kit shows)
    axv = fig.add_subplot(gs[1:13, 8:12]); axv.set_facecolor(BG); axv.set_aspect("equal"); axv.axis("off")
    under = flat.copy(); under.apply_transform(np.diag([-1.0, 1.0, -1.0, 1.0])); under.apply_translation([0, 0, z0 + z1])  # z0..z1 kept
    kit_top = z0 + min(14.5, 0.6 * H)
    plan(axv, under, False, np.linspace(z0 + 0.3, kit_top, 11), "VENTRAL · sections z 0–%.0f (kit)" % (kit_top - z0), fs=7, joints=False, pad=(10.0, 10.0))
    axv.set_ylim(y0 - 27, y1 + 9)

    # ---- ENROLLED · the coil at its own size
    ax3 = fig.add_subplot(gs[0:9, 12:18]); ax3.set_facecolor(BG); ax3.set_aspect("equal"); ax3.axis("off")
    GYs, GZs, _, Ms = _relief_grid(posed, n=220, view="side", field=False)
    if GYs is not None: _strong_outline(ax3, GYs, GZs, Ms, draw=True, lw=0.5)
    for polys in psec:
        for q in polys: ax3.plot(q[:, 0], q[:, 1], color=INK, lw=0.9)
    (py0, pz0), (py1, pz1) = posed.bounds[0][1:], posed.bounds[1][1:]
    coil = max(py1 - py0, pz1 - pz0)
    ax3.set_title(f"ENROLLED · {curl:.0f}°/joint · {curl * nj:.0f}° over {nj} joints · Ø {coil:.0f} mm", fontsize=8, loc="left", **mono)
    _dim(ax3, py0, pz0 - 6, py1, pz0 - 6, f"{py1 - py0:.1f}", off=0)
    ax3.set_xlim(py0 - 12, py1 + 12); ax3.set_ylim(pz0 - 16, pz1 + 8)

    # ---- DETAIL B · one joint sectioned at the midline, 2:1, the piece behind drawn at rest and curled
    axb = fig.add_subplot(gs[0:9, 18:24]); axb.set_facecolor(BG); axb.set_aspect("equal"); axb.axis("off")
    if nj:
        j = nj // 2; yc, zc = pivots[j]; win = 30.0; ztop = min(z1, zc + 26)
        for q in secs[j]: axb.plot(q[:, 0], q[:, 1], color=INK, lw=1.0)                                   # the piece ahead (socket)
        for q in secs[j + 1]: axb.plot(q[:, 0], q[:, 1], color=INK, lw=1.0, ls=(0, (4, 2)))              # the piece behind (ball)
        for q in secs[j + 1]: r = _rot2(q, yc, zc, curl); axb.plot(r[:, 0], r[:, 1], color=DIM, lw=0.45, alpha=0.55)   # ... curled
        axb.plot([yc - 2.5, yc + 2.5], [zc, zc], color=INK, lw=0.5); axb.plot([yc, yc], [zc - 2.5, zc + 2.5], color=INK, lw=0.5)
        try:
            import joints.ball as _B; ballD = 2.0 * float(_B._ASSET_R)               # the kit is printed 1:1 (18 mm pitch)
        except Exception: ballD = None
        lo_z = z0 - 1.0
        if ballD: _dim(axb, yc - ballD / 2, lo_z - 2.0, yc + ballD / 2, lo_z - 2.0, f"Ø {ballD:.1f} ball", off=0)
        _dim(axb, yc + win - 5, z0, yc + win - 5, zc, f"{zc - z0:.1f}", off=0)
        for a, lab in ((0.0, "rest"), (curl, f"enrol {curl:.0f}°")):
            axb.plot([yc, yc + (win - 2) * math.cos(math.radians(-a))], [zc, zc + (win - 2) * math.sin(math.radians(-a))], color=DIM, lw=0.45, ls=(0, (5, 3)))
            axb.text(yc + (win - 1) * math.cos(math.radians(-a)), zc + (win - 1) * math.sin(math.radians(-a)), lab, fontsize=5.5, color=DIM, family="monospace", va="center")
        gv = lambda k, d=0.0: float(P.get(k, d))
        axb.text(yc - win + 1, lo_z - 9.5,
                 f"clearance {gv('clearance'):.2f} mm · pivot z {zc:.1f} · solid = ahead, dashed = behind, grey = behind curled", fontsize=5.5, color=DIM, family="monospace")
        axb.set_title(f"DETAIL B · joint J{j + 1} ({names[j]}–{names[j + 1]}) · section y–z · 2:1", fontsize=8, loc="left", **mono)
        axb.set_xlim(yc - win, yc + win); axb.set_ylim(lo_z - 12, ztop + 3)
    else: axb.set_title("DETAIL B · no joints", fontsize=8, loc="left", **mono)

    # ---- ENROLMENT · five frames, lateral midline, the pieces' sections posed in 2D
    axs = fig.add_subplot(gs[9:14, 12:24]); axs.set_facecolor(BG); axs.set_aspect("equal"); axs.axis("off")
    fr = [0.0, 0.25, 0.5, 0.75, 1.0]; xoff = 0.0; extents = []
    for f in fr:
        ps = _posed_sections(secs, pivots, curl * f)
        allp = np.vstack([q for polys in ps for q in polys]) if any(ps) else np.zeros((1, 2))
        fy0, fy1, fz0, fz1 = allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max()
        for polys in ps:
            for q in polys: axs.plot(q[:, 0] - fy0 + xoff, q[:, 1], color=INK, lw=0.6)
        axs.plot([xoff, xoff + fy1 - fy0], [fz0 - 1, fz0 - 1], color=DIM, lw=0.3)
        extents.append((fz0, fz1, xoff + 0.5 * (fy1 - fy0), f)); xoff += (fy1 - fy0) + 16
    lo = min(e[0] for e in extents); hi = max(e[1] for e in extents)
    for _, _, xm, f in extents: axs.text(xm, lo - 7, f"{curl * f * nj:.0f}°", fontsize=6.5, ha="center", va="top", **mono)
    axs.set_title(f"ENROLMENT · 0 → {curl * nj:.0f}° · lateral midline", fontsize=8, loc="left", **mono)
    axs.set_xlim(-6, xoff - 8); axs.set_ylim(lo - 14, hi + 6)

    # ---- SECTION A–A · lateral
    ax2 = fig.add_subplot(gs[20:24, 0:9]); ax2.set_facecolor(BG); ax2.set_aspect("equal"); ax2.axis("off")
    GYs, GZs, Zs, Ms = _relief_grid(flat, view="side", field=not bw)
    if not bw: _field(ax2, GYs, GZs, Zs, cmap)
    for k, polys in enumerate(secs):
        for q in polys: ax2.plot(q[:, 0], q[:, 1], color=INK, lw=0.6 if bw else 0.5, alpha=0.9 if bw else 0.5)
    _strong_outline(ax2, GYs, GZs, Ms, draw=bw, lw=1.2)
    for k, (yc, zc) in enumerate(pivots):
        ax2.add_patch(Circle((yc, zc), 1.0, fill=False, ec=INK, lw=0.5)); ax2.text(yc, z0 - 4.5, f"J{k + 1}", fontsize=5, ha="center", color=DIM, family="monospace")
    _dim(ax2, y1 + 10, z0, y1 + 10, z1, f"{H:.1f}", off=0)
    ax2.text(y0, z1 + 5, ("SECTION A–A · lateral · midline cut, pivots ○" if bw else "SECTION A–A · lateral relief (%s)" % cmap), fontsize=8, **mono)
    ax2.set_xlim(y0 - 10, y1 + 26); ax2.set_ylim(z0 - 9, z1 + 10)

    # ---- PIECE SCHEDULE
    ax5 = fig.add_subplot(gs[15:24, 9:15]); ax5.set_facecolor(BG); ax5.axis("off")
    vols = info.get("volumes") or [None] * n
    rows = ["PIECE     W      L      H    cm3     g  b wt  PIVOT y,z"]
    tot_v = 0.0; sizes = []
    for k, (nm, p) in enumerate(zip(names, parts)):
        (bx0, by0, bz0), (bx1, by1, bz1) = p.bounds; pw, pl, ph = bx1 - bx0, by1 - by0, bz1 - bz0
        v = float(vols[k]) if vols[k] is not None else float(abs(p.volume)); tot_v += v; sizes.append((nm, pw, pl))
        mk = meta[k] if k < len(meta) else {}
        b = mk.get("bodies", p.body_count); wt = mk.get("watertight", p.is_watertight)
        piv = f"{pivots[k - 1][0]:6.1f},{pivots[k - 1][1]:4.1f}" if 0 < k <= nj else "     —     "
        rows.append(f"{nm:<7} {pw:6.1f} {pl:6.1f} {ph:5.1f} {v / 1000:6.1f} {v / 1000 * PLA_G_CM3:5.1f}  {b} {'✓' if wt else '✗'}  {piv}")
    if len(rows) > 24: rows = rows[:23] + [f"… {len(rows) - 23} more pieces"]
    hours = tot_v / 1000 / PRINT_CM3_PER_H
    rows += ["─" * 56, f"TOTAL  {n} pieces · {tot_v / 1000:.0f} cm3 · {tot_v / 1000 * PLA_G_CM3:.0f} g PLA · ≈ {int(hours)} h {int(60 * (hours % 1)):02d} m @ {PRINT_CM3_PER_H:.0f} cm3/h"]
    ax5.set_title("PIECE SCHEDULE · mm · cm3 · g PLA · bodies · watertight · pivot", fontsize=7, loc="left", **mono)
    ax5.text(0.0, 0.97, "\n".join(rows), fontsize=5.9, va="top", ha="left", linespacing=1.45, transform=ax5.transAxes, **mono)
    _box(ax5)

    # ---- BED LAYOUT
    ax6 = fig.add_subplot(gs[15:24, 15:19]); ax6.set_facecolor(BG); ax6.set_aspect("equal"); ax6.axis("off")
    plates, nofit = _pack_bed(sizes)
    ax6.add_patch(Rectangle((0, 0), BED_MM, BED_MM, fill=False, ec=INK, lw=0.9))
    for nm, bx, by, bwid, bh in (plates[0] if plates else []):
        ax6.add_patch(Rectangle((bx, by), bwid, bh, fill=False, ec=INK, lw=0.5))
        ax6.text(bx + 2.5, by + bh / 2, nm, fontsize=5.3, va="center", **mono)
    others = " · ".join(f"plate {i + 2}: {', '.join(b[0] for b in pl)}" for i, pl in enumerate(plates[1:]))
    foot = (others if others else "one plate") + (f" · NO FIT (> {BED_MM} mm): {', '.join(nofit)} — scale" if nofit else "")
    ax6.text(0, -12, foot[:90], fontsize=5.2, va="top", **mono)
    ax6.set_title(f"BED LAYOUT · {BED_MM} × {BED_MM} mm · {len(plates)} plate{'s' if len(plates) != 1 else ''}", fontsize=7.5, loc="left", **mono)
    ax6.set_xlim(-6, BED_MM + 6); ax6.set_ylim(-34, BED_MM + 6)

    # ---- INFO · print checks, morphospace, editor coordinates
    ax4 = fig.add_subplot(gs[15:24, 19:24]); ax4.set_facecolor(BG); ax4.axis("off")
    gv = lambda k, d=0.0: float(P.get(k, d))
    oh = _overhang_fraction(flat, z0)
    touching = checks.get("touching") or []; debris = checks.get("debris_mm3") or {}
    cuts = [c for c in (checks.get("notes") or []) if "cut away" in c]
    valid = checks.get("print_valid")
    head = info.get("head"); body = info.get("body")
    if isinstance(head, dict): head_s = f"{head.get('style', 'custom')} (editor {head.get('id', '')})" + (f" · {len(head.get('outline') or {})} outline keys" if head.get("outline") else "")
    else: head_s = str(head or "crescent")
    if isinstance(body, dict) and body.get("rows"):
        rw = [float(r.get("width", 0)) for r in body["rows"]]
        body_s = [f"BODY     {body.get('model', 'thorax')} · {len(rw)} rows · width {min(rw):.0f}–{max(rw):.0f} mm", f"         {_spark(rw)}"]
    elif isinstance(body, dict): body_s = ["BODY     " + ", ".join(f"{k} {v}" for k, v in body.items() if k != "rows")[:46]]
    else: body_s = ["BODY     preset rows (segCount %d)" % int(gv("segCount", 0))]
    lines = [f"LENGTH {L:6.1f}  WIDTH {W:6.1f}  RELIEF {H:5.1f}", f"CLEAR  {gv('clearance'):5.2f}  PIECES {n:2d}  JOINTS {nj} ball",
             f"CURL   {curl:.0f}°/joint · {curl * nj:.0f}°  BED {BED_MM} · {len(plates)} plate{'s' if len(plates) != 1 else ''}", "",
             "PRINT CHECKS",
             f" valid     {'✓' if valid else ('✗' if valid is not None else '— (not checked)')}",
             f" contacts  {('none at rest' if not touching else ', '.join('%s–%s %.2f mm3' % (t['a'], t['b'], t['mm3']) for t in touching))[:44]}",
             f" cuts      {'none' if not cuts else str(len(cuts)) + ' — see notes'}   debris {sum(debris.values()):.1f} mm3",
             f" overhang  {100 * oh:4.1f} % of area > {OVERHANG_DEG:.0f}°" if oh == oh else " overhang  —", "",
             "MORPHOSPACE", f" ceph {gv('cephFrac'):.2f}  pyg {gv('pygFrac'):.2f}  axis {gv('axisFrac'):.2f}  glab {gv('glabInflate'):.2f}",
             f" eye  {gv('eyeSize'):.2f}/{gv('eyePos'):.2f}  genalSp {gv('genalSpine'):.2f}  pygSp {gv('pygSpine'):.2f}", "",
             f"HEAD     {head_s}"[:50], *body_s, "",
             f"SCHEMA {info.get('schema', '—')} · REV A · BUILD {info.get('build_seconds', '—')} s"]
    ax4.set_title("    " + title, fontsize=8, loc="left", **mono)
    ax4.text(0.03, 0.97, "\n".join(lines), fontsize=5.9, va="top", ha="left", linespacing=1.42, transform=ax4.transAxes, **mono)
    _box(ax4, INK)

    # ---- TITLE STRIP
    st = fig.add_axes([0.03, 0.012, 0.95, 0.042]); st.set_facecolor(BG); _box(st, INK)
    st.text(0.008, 0.5, f"{title} · {info.get('params', '—')}", fontsize=8, va="center", **mono)
    st.text(0.50, 0.5, "SHEET 1/1 · mm · 3rd-angle · A3", fontsize=7.5, va="center", ha="center", **mono)
    st.text(0.992, 0.5, f"REV A · {info.get('date', '')} · Claire Choi · Cornell".replace("·  ·", "·"), fontsize=8, va="center", ha="right", **mono)
    spec = _json.dumps({"hash": info.get("params", ""), "schema": info.get("schema", ""), "model": "isopod", "params": P}, sort_keys=True, default=str)
    fig.savefig(path, dpi=110, facecolor=BG, metadata={"trilobite": spec}); plt.close(fig); return path


# ---------------------------------------------------------------- gallery mode (prompt 8: absorbs legacy/sheet10.py)
def gallery(out_dir="out", path=None, title="TEN ORDERS — instrument 2.1"):
    """One sheet for every order in out_dir/<name>/ (written by `python sweep.py presets`): dorsal silhouette of the
    flat animal, the reading (theta/joint, total, class, limiter) and the closure curve when the reading exists.
    Returns the PNG path."""
    import glob as _glob, json as _json, os as _os
    names = sorted(_os.path.basename(d) for d in _glob.glob(_os.path.join(out_dir, "*")) if _os.path.exists(_os.path.join(d, "measure.json")))
    if not names: raise FileNotFoundError(f"no out_dir/<name>/measure.json under {out_dir}")
    n = len(names); cols = min(5, n); rows = -(-n // cols)
    fig = plt.figure(figsize=(3.4 * cols, 4.2 * rows), facecolor="#0b0b0b")
    gs = fig.add_gridspec(rows * 2, cols, height_ratios=[3, 1.2] * rows, hspace=0.35, wspace=0.15)
    for k, name in enumerate(names):
        r, c = divmod(k, cols); meas = _json.load(open(_os.path.join(out_dir, name, "measure.json")))
        ax = fig.add_subplot(gs[2 * r, c]); ax.set_facecolor("#0b0b0b"); ax.set_aspect("equal"); ax.axis("off")
        fp = _os.path.join(out_dir, name, "flat.stl")
        if _os.path.exists(fp):
            m = trimesh.load(fp)
            v = m.vertices; ax.scatter(v[::7, 0], -v[::7, 1], s=0.15, c="#a09d92", linewidths=0)
        th, cls, lim = meas.get("theta_joint_deg"), meas.get("enroll_class", "—"), meas.get("limited_by", "—")
        sb = meas.get("stopped_by") or []; limpair = f"{sb[0][0]}–{sb[0][1]}" if sb else ""
        head = f"{name}\n{th if th is not None else '—'}°/joint · {meas.get('total_deg') or '—'}° · {cls}"
        ax.set_title(head + (f"\n{lim} {limpair}" if th is not None else f"\n{lim}: {meas.get('reason','')}"), color="#e8e6df", fontsize=9, loc="left")
        ax2 = fig.add_subplot(gs[2 * r + 1, c]); ax2.set_facecolor("#0b0b0b")
        for sp in ax2.spines.values(): sp.set_color("#2c2c2a")
        ax2.tick_params(colors="#8a897f", labelsize=7)
        gc = meas.get("gap_curve")
        if gc:
            xs = [g[0] for g in gc]; ys = [g[1] for g in gc]; ax2.plot(xs, ys, color="#f0b040", lw=1.2); ax2.axvline(th, color="#5a5952", lw=0.8, ls="--")
            ax2.set_xlabel("deg / joint", color="#8a897f", fontsize=7); ax2.set_ylabel("head–tail gap mm", color="#8a897f", fontsize=7)
        else:
            ax2.text(0.5, 0.5, "censored", color="#5a5952", ha="center", va="center", transform=ax2.transAxes); ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle(title, color="#8a897f", fontsize=11, x=0.02, ha="left")
    path = path or _os.path.join(out_dir, "gallery.png"); fig.savefig(path, dpi=110, facecolor="#0b0b0b", bbox_inches="tight"); plt.close(fig)
    return path

if __name__ == "__main__" and (len(__import__("sys").argv) < 2 or __import__("sys").argv[1] == "gallery"):
    import sys as _sys                                   # python blueprint.py gallery [out_dir]
    print(gallery(_sys.argv[2] if len(_sys.argv) > 2 else "out"))
