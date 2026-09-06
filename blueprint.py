"""
blueprint.py — draw a built animal as a technical sheet: white ink on black.
Plan (dorsal) as contour lines, lateral silhouette, midline section, dimension lines, enrollment dial, title block.
    sheet(flat_mesh, P, measure, path)   -> writes a PNG (matplotlib) and returns the figure
"""
import math, numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

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
    hz = meas.get("hinge_z")
    if hz: ax2.axhline(hz, color=DIM, lw=0.5, ls=(0, (4, 3))); ax2.text(y1 + 3, hz, f"hinge {hz:.1f}", color=INK, fontsize=7, va="center", family="monospace")
    _dim(ax2, y1 + 14, z0, y1 + 14, z1, f"{z1 - z0:.1f}", off=0)
    ax2.text(y0, z1 + 6, "SECTION A–A · midline · lateral silhouette dimmed", color=INK, fontsize=8, family="monospace")
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
    ax4 = fig.add_subplot(gs[3:6, 4:7]); ax4.set_facecolor(BG); ax4.axis("off")
    dials = [("SCULPT", P.get("furrowDepth", 0) / 1.2), ("HEAD", (P["cephFrac"] - 0.22) / 0.2), ("TAIL", (P["pygFrac"] - 0.05) / 0.33),
             ("ELONGATION", (P["segCount"] - 4) / 10), ("SPINES", min(1, P.get("spineBase", 0) / 0.9 + P.get("genalSpine", 0) / 1.8)), ("EYES", (P.get("eyeSize", 0) - 0.04) / 0.26)]
    for i, (n, v) in enumerate(dials):
        v = float(min(1, max(0, v))); y = 1 - i * 0.135
        ax4.plot([0, 1], [y, y], color=INK, lw=0.6)
        ax4.plot([v], [y], marker="o", ms=6, color=INK)
        ax4.text(-0.03, y, n, color=INK, fontsize=7, ha="right", va="center", family="monospace")
        ax4.text(1.04, y, f"{v:.2f}", color=DIM, fontsize=7, va="center", family="monospace")
    ax4.set_xlim(-0.34, 1.16); ax4.set_ylim(-0.12, 1.1)
    ax4.set_title("DIALS · morphospace", color=INK, fontsize=8, family="monospace", loc="left")
    ax4.text(-0.34, 0.16, "A morphospace is a coordinate system for shape. Each dial is one axis of variation the body plan\n"
                          "can take (Raup 1966; six axes after Gon's survey of fifty genera).",
             color=DIM, fontsize=6.4, family="monospace", va="top", linespacing=1.5)
    # ---- title block
    ax5 = fig.add_subplot(gs[0:6, 7:9]); ax5.set_facecolor(BG); ax5.axis("off")
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
