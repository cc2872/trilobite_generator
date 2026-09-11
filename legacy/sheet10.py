"""Ten-order sheet (plan contours + posed lateral section + reading per cell) and a closure-curve figure."""
import json, glob, os, numpy as np, trimesh
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, BG, DIM, WARN = "#ffffff", "#000000", "#9a9a9a", "#ffb347"
ORDER = ["agnostida","redlichiida","corynexochida","lichida","odontopleurida","phacopida","proetida","asaphida","harpetida","ptychopariida"]
def sections(m, normal, origins):
    out = []
    for o in origins:
        try:
            s = m.section(plane_origin=o, plane_normal=normal)
            if s is None: continue
            for e in s.entities: out.append(s.vertices[e.points])
        except Exception: pass
    return out
fig = plt.figure(figsize=(16.5, 11.7), facecolor=BG)
gs = fig.add_gridspec(2, 5, left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.08, hspace=0.18)
fig.text(0.02, 0.965, "TRILOBITE MORPHOSPACE · GON'S TEN ORDERS · INSTRUMENT 2.0", color=INK, fontsize=11, family="monospace")
fig.text(0.02, 0.945, "angle sweep per joint to first collision or closure · fixed wide bevel on the measurement build · closure = tail to head BODY (arms off)\n"
                      "θ = largest collision-free uniform flexion per joint · total = θ × joints · limited_by: CLOSED | ANATOMY | BOUND (censored) · orange = a weakness, stated",
         color=DIM, fontsize=7.5, family="monospace", va="top", linespacing=1.4)
for k, name in enumerate(ORDER):
    r, c = divmod(k, 5)
    ax = fig.add_subplot(gs[r, c]); ax.set_facecolor(BG); ax.axis("off"); ax.set_aspect("equal")
    d = f"out10/{name}"; mj = f"{d}/measure_v2.json"; pj = f"presets/{name}.json"
    blurb = json.load(open(pj))["blurb"] if os.path.exists(pj) else ""
    ax.set_title(name.upper(), color=INK, fontsize=9, family="monospace", loc="left")
    if not os.path.exists(mj):
        ax.text(0.5, 0.5, "not built yet", color=WARN, ha="center", va="center", family="monospace", fontsize=8, transform=ax.transAxes)
        ax.text(0.0, 0.02, blurb[:60], color=DIM, fontsize=6, family="monospace", transform=ax.transAxes); continue
    m = json.load(open(mj)); flat = trimesh.load(f"{d}/flat.stl"); posed = trimesh.load(f"{d}/posed.stl")
    (x0, y0, z0), (x1, y1, z1) = flat.bounds; L = y1 - y0
    # plan: horizontal contours (the silhouette is the lowest one)
    for i, p in enumerate(sections(flat, [0, 0, 1], [[0, 0, z] for z in np.linspace(z0 + 0.6, z1 - 0.3, 7)])):
        ax.plot(p[:, 0], p[:, 1], color=INK, lw=0.7 if i == 0 else 0.35, alpha=1 if i == 0 else 0.6)
    # lateral: midline section of the posed animal, laid below the plan, same scale
    pb = posed.bounds; yoff = y0 - 8 - (pb[1][2] - pb[0][2]) ; xoff = -0.5 * (pb[0][1] + pb[1][1])
    for p in sections(posed, [1, 0, 0], [[0, 0, 0]]):
        ax.plot(p[:, 1] + xoff, p[:, 2] + yoff, color=INK if m["limited_by"] == "closed" else DIM, lw=0.7)
    for p in sections(flat, [1, 0, 0], [[0, 0, 0]]):
        ax.plot(p[:, 1] - 0.5 * (y0 + y1), p[:, 2] + yoff, color=DIM, lw=0.4, alpha=0.5)
    ax.text(0.5 * (x0 + x1), yoff - 3, f"posed at θ = {m['theta_joint_deg']}°  (flat dimmed)", color=DIM, fontsize=6, family="monospace", ha="center", va="top")
    ax.set_xlim(min(x0, xoff + pb[0][1]) - 4, max(x1, xoff + pb[1][1]) + 4); ax.set_ylim(yoff - 8, y1 + 3)
    lim = m["limited_by"]; ok = lim in ("closed", "anatomy") and not m.get("unsane_parts") and m["print_valid"]
    lines = [f"n {m['joints']-1}  ·  bevel {int(m['bound_deg'])}°  ·  L {L:.0f} mm",
             f"θ {m['theta_joint_deg']}°/joint  ·  total {m['total_deg']}°",
             f"gap {m['closure_gap_mm']} mm  ({m['gap_over_L']} L)",
             f"{lim.upper()}" + (f"  by {m['stopped_by'][0][0]}–{m['stopped_by'][0][1]} ({m['stopped_by'][0][2]} mm³)" if m["stopped_by"] else "")
             + (f"  →  printed stop {m['stop_recommended_deg']}°" if m.get("stop_recommended_deg") is not None else "")]
    if m.get("measure_timed_out"): lines.append("MEASUREMENT TIMED OUT — not verified")
    if m.get("unsane_parts"): lines.append("UNSANE BUILD: " + ", ".join(m["unsane_parts"]))
    lines.append("print-valid" if m["print_valid"] else "NOT PRINT-VALID: " + "; ".join(v[:34] for v in m["violations"][:2]))
    ax.text(0.0, -0.01, "\n".join(lines), color=INK if ok else WARN, fontsize=6.4, family="monospace", va="top", transform=ax.transAxes, linespacing=1.45)
fig.text(0.98, 0.012, "Claire Choi · Cornell · 8 Sep 2026 · instrument 2.0 · sheet 1/2", color=INK, fontsize=8, ha="right", family="monospace")
fig.savefig("out10/ten_orders_sheet.png", dpi=110, facecolor=BG); plt.close(fig)
# ---- closure curves
fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG); ax.set_facecolor(BG)
for name in ORDER:
    mj = f"out10/{name}/measure_v2.json"
    if not os.path.exists(mj): continue
    m = json.load(open(mj)); gc = m.get("gap_curve")
    if not gc: continue
    L = json.load(open(f"presets/{name}.json"))["params"]["length"]
    ax.plot([a for a, _ in gc], [b / L for _, b in gc], marker="o", ms=3, lw=1, color=INK if m["limited_by"] == "closed" else DIM,
            ls="-" if not m.get("unsane_parts") else "--", label=f"{name} ({m['joints']} joints, {m['limited_by']})")
ax.axhline(3.0 / 150, color=WARN, lw=0.7, ls=":"); ax.text(0.5, 0.024, "closed (3 mm at 150 mm)", color=WARN, fontsize=7, family="monospace")
ax.set_xlabel("uniform flexion per joint (°)", color=INK, fontsize=8, family="monospace"); ax.set_ylabel("tail → head-body gap / body length", color=INK, fontsize=8, family="monospace")
ax.tick_params(colors=INK, labelsize=7); [s.set_color(DIM) for s in ax.spines.values()]
ax.legend(fontsize=7, facecolor=BG, edgecolor=DIM, labelcolor=INK)
ax.set_title("CLOSURE CURVES · white = closes within the bevel · dashed = severed segments in the build · sheet 2/2", color=INK, fontsize=8, family="monospace", loc="left")
fig.savefig("out10/closure_curves.png", dpi=110, facecolor=BG, bbox_inches="tight"); print("sheets written")
