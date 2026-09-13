import json, glob, math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import parts

names = ["proetida","phacopida","asaphida","odontopleurida","lichida","harpetida","redlichiida","ptychopariida","corynexochida"]
eyes = []
for n in names:
    P = json.load(open(f"presets/{n}.json"))["params"]
    G = parts.eye_geometry(P); EP = parts.eye_params(P, G["eR"])
    m, nl = parts.eye_solid(**EP)
    eyes.append((n, m, nl, EP))
    print(f"{n:15s} R={EP['R']:.1f}mm H={EP['H']:.1f}mm arc={EP['arc_deg']:.0f} lenses={nl} faces={len(m.faces)}")
    m.export(f"/home/claude/eye_{n}.stl")

fig = plt.figure(figsize=(15, 15), facecolor="black")
light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
for i, (n, m, nl, EP) in enumerate(eyes):
    ax = fig.add_subplot(3, 3, i+1, projection="3d", facecolor="black")
    V, F = m.vertices, m.faces
    tri = V[F]; nrm = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]); nrm /= (np.linalg.norm(nrm,axis=1)[:,None]+1e-12)
    shade = 0.25 + 0.75*np.clip(nrm@light, 0, 1)
    cols = np.stack([shade*0.93, shade*0.89, shade*0.78, np.ones_like(shade)], 1)
    pc = Poly3DCollection(tri, facecolors=cols, edgecolors="none"); ax.add_collection3d(pc)
    r = max(V.max(0)-V.min(0))/2; c = (V.max(0)+V.min(0))/2
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.view_init(elev=28, azim=-35); ax.set_axis_off(); ax.set_box_aspect((1,1,1))
    ax.set_title(f"{n.upper()}\nR {EP['R']:.1f} mm · arc {EP['arc_deg']:.0f}° · {nl} lenses", color="#ddd", family="monospace", fontsize=10, pad=2)
fig.text(0.5, 0.02, "EYES ONLY · eye_solid() · outward = +x · scale per panel", color="#888", family="monospace", ha="center")
plt.subplots_adjust(left=0.02,right=0.98,top=0.95,bottom=0.05,wspace=0,hspace=0.12)
plt.savefig("/mnt/user-data/outputs/trilobite_eyes.png", dpi=130, facecolor="black")
