import json, numpy as np, matplotlib, time, trimesh
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import parts, eyes2, mesh as M

# (preset, label, head overrides, eye kwargs).  H and stalk_len are in units of eye radius R.
cases = [
 ("phacopida",      "PHACOPID · schizochroal", dict(eyeSize=0.20, eyePos=0.48, eyeLat=0.52),
    dict(H=1.4, arc_deg=125, slope_deg=12, ky=0.75, kidney=0.4, taper=0.2, lensD=0.075, lensGap=0.5, lensRise=0.5, style="sunken", rim=0.28, grad=0.25)),
 ("ptychopariida",  "HOLOCHROAL · lunate",     dict(eyeSize=0.18, eyePos=0.45, eyeLat=0.55),
    dict(H=0.55, arc_deg=175, slope_deg=28, ky=0.7, kidney=0.35, taper=0.55, lensD=0.06, lensGap=0.05, lensRise=0.3, style="flush", grad=0.4, min_D=0.25)),
 ("asaphida",       "ASAPHUS · stalked",       dict(eyeSize=0.12, eyePos=0.45),
    dict(H=0.8, arc_deg=200, slope_deg=20, ky=1.0, taper=0.0, lensD=0.13, lensGap=0.1, style="flush", stalk_len=4.5, stalk_r=0.5, lobe_h=0.6, lean_out_deg=18, lean_fwd_deg=12, min_D=0.3)),
 ("odontopleurida", "ERBENOCHILE · tower",     dict(eyeSize=0.14, eyePos=0.5),
    dict(H=2.4, arc_deg=210, slope_deg=0, ky=1.0, taper=0.0, lensD=0.12, lensGap=0.15, lensRise=0.4, style="sunken", rim=0.15, lattice="grid", lobe_h=0.9, shade=0.22, min_D=0.3)),
]
views = [("dorsal", 88, -90), ("oblique", 32, -50), ("lateral", 8, 0), ("eye close-up", 22, -25)]
light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)

def draw(ax, m, elev, azim, eye_mask=None, focus=None):
    V, F = m.vertices, m.faces; tri = V[F]
    nrm = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]); nrm /= (np.linalg.norm(nrm,axis=1)[:,None]+1e-12)
    e, a = np.radians(elev), np.radians(azim); cam = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    r = max(V.max(0)-V.min(0))/2; c = (V.max(0)+V.min(0))/2
    if focus is not None: c, r = focus
    cen = tri.mean(1)
    keep = (nrm @ cam > -0.05) & np.all(np.abs(cen - c) < 1.8 * r, 1)     # cull back faces and out-of-frame faces
    tri, nrm = tri[keep], nrm[keep]; em = eye_mask[keep] if eye_mask is not None else None
    sh = 0.22 + 0.78*np.clip(nrm@light, 0, 1)
    col = np.stack([sh*.93, sh*.89, sh*.78, np.ones_like(sh)],1)
    if em is not None: col[em] = np.stack([sh[em]*.85, sh[em]*.75, sh[em]*.35, np.ones(em.sum())],1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors="none"))
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_box_aspect((1,1,1))

for ci, (preset, label, ov, kw) in enumerate(cases):
    t = time.time()
    P = json.load(open(f"presets/{preset}.json"))["params"]; P["eyeSolid"] = 0; P.update(ov)
    head = parts.cephalon(P)
    S = parts.cephalon_plan(P); G = S["eye"]; eR = G["eR"]
    kw2 = dict(kw); kw2["R"] = eR; kw2["H"] = kw["H"] * eR
    if "stalk_len" in kw2: kw2["stalk_len"] *= eR
    xm = float(S["xmax"](np.array([G["ye"]]))[0]) - 0.4
    if kw2.get("stalk_len", 0) == 0: kw2["clip_x"] = xm - G["xe"]
    eye, nl = eyes2.eye2(**kw2)
    zb = float(S["zfun"](np.array([G["xe"]]), np.array([G["ye"]]))[0]) - 0.3
    if kw2.get("stalk_len", 0) > 0: zb += kw2["stalk_len"]        # stalk plants in the cheek, eye rides above it
    eR_ = eye.copy().apply_translation((G["xe"], G["ye"], zb))
    eL_ = M.mirror_x(eR_)
    combo = M.union(head, eR_, eL_)
    nh = len(head.faces)
    # face mask for tinting eyes: faces whose centroid is within the eye bounding boxes
    cen = combo.triangles_center
    mask = np.zeros(len(cen), bool)
    for e in (eR_, eL_):
        lo, hi = e.bounds; mask |= np.all((cen > lo - 0.05) & (cen < hi + 0.05), 1)
    print(f"{label:28s} eR={eR:.1f}mm lenses={nl} faces={len(combo.faces)} bodies={len(combo.split())} {time.time()-t:.0f}s")
    combo.export(f"/home/claude/head_{preset}.stl")
    fig = plt.figure(figsize=(20, 5), facecolor="black")
    for vi, (vname, el, az) in enumerate(views):
        ax = fig.add_subplot(1, 4, vi + 1, projection="3d", facecolor="black")
        foc = ((eR_.bounds.mean(0)), 0.95*max(eR_.bounds[1]-eR_.bounds[0])) if vname.startswith('eye') else None
        draw(ax, combo, el, az, mask, foc)
        ax.set_title(f"{label} · {vname}" if vi == 0 else vname, color="#ddd", family="monospace", fontsize=10, pad=0, loc="left")
    plt.subplots_adjust(left=0.01,right=0.99,top=0.92,bottom=0.01,wspace=0)
    plt.savefig(f"/home/claude/row_{ci}.png", dpi=110, facecolor="black"); plt.close(fig)
from PIL import Image
rows = [Image.open(f"/home/claude/row_{i}.png") for i in range(len(cases))]
W = max(r.width for r in rows); Hh = sum(r.height for r in rows)
sheet = Image.new("RGB", (W, Hh), "black"); y = 0
for r in rows: sheet.paste(r, (0, y)); y += r.height
sheet.save("/mnt/user-data/outputs/heads_with_eyes.png")
