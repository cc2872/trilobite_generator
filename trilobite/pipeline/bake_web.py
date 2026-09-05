"""Bake each sliceable sculpt to compact web data: dorsal-up mesh + per-vertex segment index + joints."""
import pickle, numpy as np, json, trimesh
import furrow_slicer as fs
M=pickle.load(open("meshes.pkl","rb"))
CFG={"olenoides":(8,(36,102)),"harpetid":(10,(52,122)),"proetida":(9,(24,110))}

def dorsal_up(v):
    # axial lobe should be the highest midline ridge; if it reads low, flip z
    W=np.abs(v[:,0]).max()
    mid=np.abs(v[:,0])<0.18*W; ple=(np.abs(v[:,0])>0.40*W)&(np.abs(v[:,0])<0.72*W)
    za=v[mid,2].mean() if mid.any() else 0; zp=v[ple,2].mean() if ple.any() else 0
    v=v.copy()
    if za<zp: v[:,2]=v[:,2].max()-v[:,2]
    return v

out={}
for name,(nseg,band) in CFG.items():
    v,f=M[name]; info=fs.all_seams(v,n_segments=nseg,band=band); vv=info["v"].copy()
    vv=dorsal_up(vv)
    m=trimesh.Trimesh(vv,f,process=False)
    m=trimesh.Trimesh(m.vertices,m.faces,process=True)
    m=m.simplify_quadric_decimation(face_count=8000)
    m=trimesh.Trimesh(m.vertices,m.faces,process=True)
    V=m.vertices; F=m.faces
    joints=sorted(info["mins"]); L=float(V[:,1].max())
    segIdx=np.array([int(np.sum(np.array(joints)<=y)) for y in V[:,1]],dtype=int)
    relief=float(V[:,2].max())
    out[name]=dict(verts=np.round(V,1).astype(float).ravel().tolist(),
                   faces=F.astype(int).ravel().tolist(),
                   seg=segIdx.tolist(), joints=[float(j) for j in joints],
                   L=L, relief=relief, nseg=nseg)
    print(f"{name:10s} verts {len(V):5d} faces {len(F):5d} joints {len(joints)} L {L:.0f} relief {relief:.0f}")

# phacops as a static (unsliced) shell, dorsal up
v,f=M["phacops"]; vp=v.copy(); vp[:,2]=vp[:,2].max()-vp[:,2]
mp=trimesh.Trimesh(vp,f,process=False); mp.merge_vertices(); mp=mp.simplify_quadric_decimation(face_count=8000); mp.merge_vertices()
mp.update_faces(mp.nondegenerate_faces()); mp.remove_unreferenced_vertices()
out["phacops"]=dict(verts=np.round(mp.vertices,1).astype(float).ravel().tolist(),
                    faces=mp.faces.astype(int).ravel().tolist(),
                    seg=[0]*len(mp.vertices), joints=[], L=float(mp.vertices[:,1].max()),
                    relief=float(mp.vertices[:,2].max()), nseg=0)
print("phacops (static)")

js="window.SPECIES = "+json.dumps(out)+";\n"
open("species_data.js","w").write(js)
print("wrote species_data.js  %.1f KB"%(len(js)/1024))
