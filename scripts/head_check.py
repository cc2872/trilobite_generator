"""head_check.py: is every head silhouette the generator makes a natural one? For every head style (and the plain /
crescent outlines) take the plan silhouette the shaped head builds from (shaped_head.diagram: one front and one rear edge
per column, spine included), walk it, and find what a trilobite never has:
  * corners  -- a tangent turn of more than CORNER degrees within one step along the margin, anywhere but the genal angle,
                the spine tip and the rear corners;
  * notches  -- inward (concave) runs along the front / side margin (between the axis and the genal angle) turning more
                than NOTCH degrees in total; the rear margin and the spine's inner edge may be concave (the posterior
                border meets the spine);
  * walls    -- the rear edge jumping between neighbouring columns by more than WALL mm (the column representation's
                vertical face: the overnight genal bug).
Writes a sheet of every outline with the findings marked and a markdown table.   python head_check.py OUTDIR"""
import os, sys, warnings, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); os.chdir(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/isopod_model")
warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import schema, head_styles as HS, shaped_head as SH, body as BODY

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CORNER, NOTCH, WALL, DS = 22.0, 10.0, 1.5, 0.01      # degrees, degrees, mm, resample step as a fraction of the head length

def outline_of(style):
    P = schema.coerce(schema.defaults())
    if style in HS.STYLES: P, face = HS.apply(P, style); P = schema.coerce(P)
    else: face = None
    o = HS.outline(style) if style in HS.OUTLINES else None
    Dg = SH.diagram(o, BODY.face_of(P, face))
    sz = Dg["size"]; mm = float(P["length"]) * float(P["cephFrac"]) / max(sz["length"], 1e-6)   # diagram units -> model mm
    return Dg, mm, P

def resample(pts, ds):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1], ds); return np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], t / s[-1]

def analyse(style):
    Dg, mm, P = outline_of(style)
    plan = np.asarray(Dg["outline"]) * mm                               # right half: front edge (axis -> tip), rear edge (tip -> axis), mm
    H = {k: np.asarray(v) * mm for k, v in Dg["handles"].items()}
    L = float(Dg["size"]["length"]) * mm
    n = len(plan) // 2; front, rear = plan[:n], plan[n:]
    # the wall test on the raw columns (rear edge jumps between neighbouring columns)
    rear_sorted = rear[np.argsort(rear[:, 0])]; dy = np.abs(np.diff(rear_sorted[:, 1])); dx = np.diff(rear_sorted[:, 0])
    x_in = float(H["root"][0]) - 0.5 * abs(float(H["spine"][0]) - float(H["root"][0])) * 0 if "root" in H else np.inf   # (the spine's own inner edge is a vertical in the columns, not a wall)
    walls = [(float(rear_sorted[i, 0]), float(dy[i])) for i in np.where((dy > WALL) & (dx < 0.6))[0] if "root" not in H or abs(rear_sorted[i, 0] - H["root"][0]) > 0.5 * abs(H["angle"][0] - H["root"][0]) + 1.0]
    fs = np.abs(np.diff(front[:, 1])); fdx = np.diff(front[:, 0])
    walls += [(float(front[i, 0]), float(fs[i])) for i in np.where((fs > WALL) & (fdx < 0.6))[0]]
    # the full closed outline, CCW: left half mirrored. Walk: right front edge (axis -> genal), right rear (-> axis), left rear, left front
    right = np.r_[front, rear]; left = np.c_[-right[::-1, 0], right[::-1, 1]]
    poly = np.r_[right, left]; poly = poly[np.r_[True, np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-9]]
    # orientation: make it CCW (positive signed area)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0: poly = poly[::-1]
    ds = DS * L; pts, frac = resample(np.r_[poly, poly[:1]], ds)
    d = np.diff(np.r_[pts, pts[:1]], axis=0); ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.degrees((np.diff(np.r_[ang, ang[:1]]) + np.pi) % (2 * np.pi) - np.pi)   # turn at vertex i+1, signed (+ = left = convex for CCW)
    turn = np.roll(turn, 1)
    xg, yg = H["angle"]; xs_tip = H.get("spine", None)
    yfront_side = float(np.interp(0.2 * L, front[:, 0], front[:, 1])) - 0.5 if "process" in H else -np.inf   # a frontal point: the margin ahead of its base
    def zone(p):
        x, y = abs(p[0]), p[1]
        if np.hypot(x - xg, y - yg) < 0.06 * L: return "genal angle"
        if xs_tip is not None and np.hypot(x - abs(xs_tip[0]), y - xs_tip[1]) < 0.06 * L: return "spine tip"
        if xs_tip is not None and x > xg - 0.02 * L and y > yg - 0.5: return "spine"
        if y > yg - 0.03 * L and x < xg: return "rear margin"
        fp = H.get("process")
        if fp is not None and x < 0.2 * L and y < yfront_side: return "front point"
        if x < 0.02 * L: return "axis"
        return "front/side margin"
    corners = [(i, float(turn[i]), zone(pts[i])) for i in np.where(np.abs(turn) > CORNER)[0]]
    # concave runs
    runs = []; i = 0; N = len(turn)
    while i < N:
        if turn[i] < -0.5:
            j = i
            while j < N and turn[j] < -0.5: j += 1
            tot = float(turn[i:j].sum()); mid = (i + j) // 2
            if tot < -NOTCH: runs.append((i, j, tot, zone(pts[mid])))
            i = j
        else: i += 1
    bad_corners = [c for c in corners if c[2] in ("front/side margin", "axis")]
    bad_runs = [r for r in runs if r[3] in ("front/side margin", "axis")]
    return dict(style=style, pts=pts, turn=turn, corners=corners, runs=runs, walls=walls, bad_corners=bad_corners, bad_runs=bad_runs,
                L=L, handles=H, notes=Dg.get("notes", []))

styles = ["plain"] + sorted(HS.STYLES) + ["crescent"]
R = [analyse(s) for s in styles]
cols = 6; rows = (len(R) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), facecolor="k")
for ax, r in zip(axs.ravel(), R):
    ax.set_facecolor("k"); p = r["pts"]; t = r["turn"]
    ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]], color="#cfcfcf", lw=1.0)
    for i0, i1, tot, z in r["runs"]:
        ax.plot(p[i0:min(i1 + 1, len(p)), 0], p[i0:min(i1 + 1, len(p)), 1], color=("#f6704f" if z in ("front/side margin", "axis") else "#8aa0ff"), lw=2.2)
    for i, tv, z in r["corners"]:
        ax.plot(p[i, 0], p[i, 1], "o", ms=5, mfc=("#f6704f" if z in ("front/side margin", "axis") else "#8aa0ff"), mec="w", mew=0.5)
    for x, dy in r["walls"]: ax.plot([x, -x], [r["handles"]["angle"][1]] * 2, "s", ms=5, color="#ffd34f")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    flag = ("%d corner(s) " % len(r["bad_corners"]) if r["bad_corners"] else "") + ("%d notch(es) " % len(r["bad_runs"]) if r["bad_runs"] else "") + ("%d wall(s)" % len(r["walls"]) if r["walls"] else "")
    ax.set_title(r["style"] + (": " + flag if flag else ": natural"), color=("#f6704f" if flag else "#9ad17a"), fontsize=9)
for ax in axs.ravel()[len(R):]: ax.axis("off"); ax.set_facecolor("k")
fig.suptitle("every head silhouette the generator makes -- red: a corner or an inward notch on the front/side margin; blue: at the genal angle, spine or rear (allowed); yellow: a wall in the rear edge",
             color="#cfcfcf", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "head_outlines.png"), dpi=110, facecolor="k")
with open(os.path.join(OUT, "head_check.md"), "w") as f:
    f.write("| style | corners (front/side) | notches (front/side) | walls (rear edge) | allowed (genal / spine / rear) |\n|---|---|---|---|---|\n")
    for r in R:
        f.write(f"| {r['style']} | " + ("; ".join(f"{tv:+.0f} deg at ({r['pts'][i,0]:.1f},{r['pts'][i,1]:.1f})" for i, tv, z in r["bad_corners"]) or "-") + " | "
                + ("; ".join(f"{tot:.0f} deg over {r['pts'][i0,0]:.1f}..{r['pts'][min(i1,len(r['pts'])-1),0]:.1f} x" for i0, i1, tot, z in r["bad_runs"]) or "-") + " | "
                + ("; ".join(f"{dy:.1f} mm at x={x:.1f}" for x, dy in r["walls"]) or "-") + " | "
                + (f"{len(r['corners']) - len(r['bad_corners'])} corners, {len(r['runs']) - len(r['bad_runs'])} concave runs") + " |\n")
for r in R: print(f"{r['style']:16s} corners {len(r['bad_corners'])} notches {len(r['bad_runs'])} walls {len(r['walls'])} | allowed corners {len(r['corners'])-len(r['bad_corners'])} runs {len(r['runs'])-len(r['bad_runs'])}" + (" NOTES " + "; ".join(r["notes"]) if r["notes"] else ""))

# ---- second pass: what the eye sees -------------------------------------------------------------------------------
# (a) waviness of the margin: deviation from a smoothed copy (a 5%-of-length moving average along the front/side margin)
# (b) the outer corner at the genal angle: the side margin's arrival tangent vs the spine's outer edge (a real genal
#     spine continues the lateral border, so this turn is small)
# (c) proportion: head width / length
from scipy.ndimage import uniform_filter1d
print("\nstyle            waviness(mm)  outer genal turn(deg)   W/L   spread")
for r in R:
    p = r["pts"]; L = r["L"]; H = r["handles"]; xg, yg = H["angle"]
    front = p[(p[:, 1] < yg - 0.02 * L) & (np.abs(p[:, 0]) < xg + 0.5)]           # the front/side margin, both sides
    # smooth along the (closed, ordered) outline then measure only on the front/side part
    k = max(3, int(0.05 * len(p))); sm = np.c_[uniform_filter1d(p[:, 0], k, mode="wrap"), uniform_filter1d(p[:, 1], k, mode="wrap")]
    dev = np.linalg.norm(p - sm, axis=1); m = (p[:, 1] < yg - 0.03 * L) & (np.abs(p[:, 0]) < xg - 0.03 * L)
    wav = float(np.percentile(dev[m], 98)) if m.any() else 0.0
    # outer genal turn: tangent of the margin just before the genal angle (right side) vs the spine's outer edge direction
    right = p[p[:, 0] > 0]; i_g = int(np.argmin(np.hypot(right[:, 0] - xg, right[:, 1] - yg)))
    before = right[max(0, i_g - 6)]; after = right[min(len(right) - 1, i_g + 6)]
    t_in = np.degrees(np.arctan2(yg - before[1], xg - before[0])); t_out = np.degrees(np.arctan2(after[1] - yg, after[0] - xg))
    turn = (t_out - t_in + 180) % 360 - 180
    spine = "spine" in H
    o = HS.outline(r["style"]) if r["style"] in HS.OUTLINES else {}
    r["waviness"] = wav; r["genal_turn"] = float(turn) if spine else float("nan"); r["WL"] = 2 * float(np.abs(p[:, 0]).max()) / L
    print(f"{r['style']:16s} {wav:6.2f}        {r['genal_turn']:7.1f}            {r['WL']:4.2f}   {o.get('genal_spread', '-')}")
json.dump([dict(style=r["style"], waviness=r["waviness"], genal_turn=r["genal_turn"], WL=r["WL"], corners=len(r["bad_corners"]), notches=len(r["bad_runs"])) for r in R], open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
