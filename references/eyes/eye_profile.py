"""eye_profile.py — read an eye's profile numbers off a scan.
    python eye_profile.py scan.stl X Y Z [tag]        (X Y Z = a point ON the visual surface, scan units)
Frame: sphere fit to the visual surface near the landmark -> centre, R. outward = mean normal of the band;
up = mean normal of the surrounding shell, orthogonalised. Outputs band height/R, slope from vertical, arc, palpebral
overhang/R, and a radial profile PNG. Landmark must be on the EYE: a smooth glabella fits a bigger sphere and fools it —
check R against the eye's plan radius before believing the numbers."""
import sys, trimesh, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
m = trimesh.load(sys.argv[1], force="mesh"); e0 = np.array([float(v) for v in sys.argv[2:5]]); tag = sys.argv[5] if len(sys.argv) > 5 else "eye"
V, N = m.vertices, m.vertex_normals
best = None
for rad_sel in (0.4, 0.6, 0.8, 1.0, 1.3):
    scale = np.linalg.norm(m.extents) * 0.05 * rad_sel                       # try neighbourhoods of a few % of the model size
    sel = np.linalg.norm(V - e0, axis=1) < scale; P = V[sel]
    if sel.sum() < 30: continue
    A = np.c_[2*P, np.ones(len(P))]; b = (P**2).sum(1); c = np.linalg.lstsq(A, b, rcond=None)[0]
    ctr = c[:3]; R = np.sqrt(max(c[3] + (ctr**2).sum(), 1e-9)); res = np.abs(np.linalg.norm(P - ctr, axis=1) - R).std()
    print(f"neighbourhood {scale:.1f}: n {sel.sum()} centre {ctr.round(2)} R {R:.2f} resid {res:.3f}")
    if best is None or res / R < best[0]: best = (res / R, ctr, R, scale)
_, ctr, R, scale = best
d = V - ctr; r = np.linalg.norm(d, axis=1); rad = d / np.maximum(r, 1e-9)[:, None]
band = (np.abs(r - R) < 0.1 * R) & ((rad * N).sum(1) > 0.7) & (np.linalg.norm(V - e0, axis=1) < 2.2 * R)
B = V[band]; outward = N[band].mean(0); outward /= np.linalg.norm(outward)
nb = (np.linalg.norm(V - e0, axis=1) < 2.2 * R) & ~band
up = N[nb].mean(0); up -= outward * (up @ outward); up /= np.linalg.norm(up); side = np.cross(up, outward)
bp = B - ctr; bu, bo, bs = bp @ up, bp @ outward, bp @ side
top, bot = np.percentile(bu, 97), np.percentile(bu, 3)
o_top = np.median(bo[bu > np.percentile(bu, 80)]); o_bot = np.median(bo[bu < np.percentile(bu, 20)])
slope = np.degrees(np.arctan2(o_top - o_bot, top - bot)); ang = np.degrees(np.arctan2(bs, bo)); arc = np.percentile(ang, 98) - np.percentile(ang, 2)
nbp = V[nb] - ctr; nu, no = nbp @ up, nbp @ outward
cap = (nu > top) & (nu < top + 0.3 * R) & (np.abs(nbp @ side) < 0.3 * R)
shade = (np.percentile(no[cap], 95) - o_top) if cap.sum() > 5 else float("nan")
print(f"{tag}: R {R:.2f} | height {(top-bot)/R:.2f} R | slope {slope:.1f} deg | arc {arc:.0f} deg | overhang {shade/R:.2f} R | band n {len(B)}")
s = m.section(plane_origin=ctr, plane_normal=side)
fig, axs = plt.subplots(1, 3, figsize=(16, 5.5)); ax = axs[0]
if s is not None:
    for e in s.entities: p = s.vertices[e.points] - ctr; ax.plot(p @ outward, p @ up, "k-", lw=0.8)
ax.plot(bo, bu, ".", ms=2, color="r", alpha=0.4); ax.add_patch(plt.Circle((0, 0), R, fill=False, color="b", lw=0.5, ls="--"))
ax.set_xlim(-1.5*R, 1.5*R); ax.set_ylim(-1.5*R, 1.5*R); ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.set_title(f"{tag}: radial profile, R {R:.2f}")
axs[1].plot(bo, bu, ".", ms=2, color="r"); axs[1].set_aspect("equal"); axs[1].set_title("band: outward vs up")
axs[2].plot(bs, bo, ".", ms=2, color="r"); axs[2].set_aspect("equal"); axs[2].set_title(f"band in plan, arc ≈ {arc:.0f}°")
plt.tight_layout(); plt.savefig(f"eye_profile_{tag}.png", dpi=80); print("wrote", f"eye_profile_{tag}.png")
