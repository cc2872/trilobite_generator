import pickle, numpy as np, matplotlib, trimesh
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import furrow_slicer as fs
M=pickle.load(open("meshes.pkl","rb"))
v,f=M["olenoides"]; info=fs.all_seams(v,n_segments=8,band=(36,102)); vv=info["v"].copy()
vv[:,2]=vv[:,2].max()-vv[:,2]                       # dorsal up
shell=trimesh.Trimesh(vv,f,process=False)
Ld=np.load("olen_legs.npz"); legs=trimesh.Trimesh(Ld["v"],Ld["f"],process=False)
whole=trimesh.util.concatenate([shell,legs])
whole.export("olenoides_assembled.stl"); print("exported STL faces",len(whole.faces))

def shade(m):
    vn=m.vertex_normals; light=np.array([-0.35,0.5,0.8]); light/=np.linalg.norm(light)
    return m.vertices, np.clip(0.42+0.5*np.clip(vn@light,0,1),0,1)
def draw(ax,el,az):
    for m,ss in [(shell,1.1),(legs,1.1)]:
        V,g=shade(m); ax.scatter(V[::2,0],V[::2,1],V[::2,2],s=ss,c=g[::2],cmap='gray',vmin=0,vmax=1,linewidths=0)
    b=whole.bounds; ax.set_box_aspect((b[1,0]-b[0,0],b[1,1]-b[0,1],max(b[1,2]-b[0,2],8)))
    ax.view_init(elev=el,azim=az); ax.set_axis_off()
fig=plt.figure(figsize=(15,7))
ax=fig.add_subplot(1,2,1,projection='3d'); draw(ax,30,-60); ax.set_title("assembled — real sculpted legs")
ax=fig.add_subplot(1,2,2,projection='3d'); draw(ax,7,-90);  ax.set_title("lateral")
plt.tight_layout(); plt.savefig("assembled.png",dpi=90,facecolor='white'); print("wrote assembled.png")
