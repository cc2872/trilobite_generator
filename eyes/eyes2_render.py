import numpy as np, matplotlib, time
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import eyes2

specs = [
 ("PHACOPID · schizochroal", dict(R=6, H=8, arc_deg=120, slope_deg=12, ky=0.8, kidney=0.35, taper=0.25, lensD=0.11, lensGap=0.45, lensRise=0.5, style="sunken", rim=0.25, grad=0.2)),
 ("HOLOCHROAL · fine, lunate", dict(R=6, H=4, arc_deg=170, slope_deg=25, ky=0.75, kidney=0.3, taper=0.5, lensD=0.09, lensGap=0.05, lensRise=0.3, style="flush", grad=0.4)),
 ("ASAPHUS · stalked", dict(R=3, H=2.4, arc_deg=200, slope_deg=20, ky=1.0, taper=0.0, lensD=0.17, lensGap=0.15, style="flush", stalk_len=14, stalk_r=0.5, lobe_h=0.6)),
 ("ERBENOCHILE · tower + shade", dict(R=4, H=9, arc_deg=200, slope_deg=0, ky=1.0, taper=0.0, lensD=0.16, lensGap=0.15, lensRise=0.4, style="sunken", rim=0.15, lattice="grid", lobe_h=0.9)),
 ("OLENELLID · thin crescent", dict(R=7, H=2.4, arc_deg=150, slope_deg=35, ky=0.6, kidney=0.5, taper=0.6, lensD=0.085, lensGap=0.02, style="flush", grad=0.5)),
 ("DEFAULT eye_solid-like", dict(R=5.5, H=4.7, arc_deg=150, slope_deg=15, style="raised")),
]
eyes = []
for name, kw in specs:
    t = time.time(); m, n = eyes2.eye2(**kw); eyes.append((name, m, n))
    print(f"{name:30s} lenses={n:4d} faces={len(m.faces):6d} watertight={m.is_watertight} {time.time()-t:.1f}s")
    m.export("/home/claude/eye2_" + name.split()[0].lower() + ".stl")

fig = plt.figure(figsize=(15, 10.5), facecolor="black")
light = np.array([0.5, -0.5, 0.7]); light /= np.linalg.norm(light)
for i, (n, m, nl) in enumerate(eyes):
    ax = fig.add_subplot(2, 3, i+1, projection="3d", facecolor="black")
    V, F = m.vertices, m.faces; tri = V[F]
    nrm = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]); nrm /= (np.linalg.norm(nrm,axis=1)[:,None]+1e-12)
    sh = 0.22 + 0.78*np.clip(nrm@light, 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.stack([sh*.93, sh*.89, sh*.78, np.ones_like(sh)],1), edgecolors="none"))
    r = max(V.max(0)-V.min(0))/2; c = (V.max(0)+V.min(0))/2
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.view_init(elev=24, azim=-32); ax.set_axis_off(); ax.set_box_aspect((1,1,1))
    ax.set_title(f"{n}\n{nl} lenses", color="#ddd", family="monospace", fontsize=10, pad=0)
plt.subplots_adjust(left=0.01,right=0.99,top=0.94,bottom=0.02,wspace=0,hspace=0.08)
plt.savefig("/mnt/user-data/outputs/trilobite_eyes_v2.png", dpi=130, facecolor="black")
