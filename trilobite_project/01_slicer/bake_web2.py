"""Re-bake: weld raw STL/OBJ FIRST, then decimate, orient, cut into welded per-segment pieces."""
import trimesh, numpy as np, json
import load_any, furrow_slicer as fs, cutter as C
B="/home/claude/sculpts"

def load_welded(name, target=16000):
    if name=="olenoides":
        m=trimesh.load(f"{B}/llllline_202609284405_82eb7f81390a49cab3ce5a1cc39bf258/Trilobite/Olenoides_serratus_shell_pose_02.stl",process=False)
    elif name=="harpetid":
        m=trimesh.load(f"{B}/Trib-Harpedita-Simple_fixed_stls/obj_1_Trib-Harpedita-Simple_fixed.stl",process=False)
    elif name=="proetida":
        path=f"{B}/trilobite-proetida/source/extracted/export.obj"
        V=[];faces={};cur="d"
        for line in open(path):
            if line.startswith("v "): V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("g "): cur=line[2:].strip()
            elif line.startswith("f "): faces.setdefault(cur,[]).append([int(t.split("/")[0])-1 for t in line.split()[1:4]])
        V=np.array(V); keep=[n for n in faces if any(s in n for s in("Cephalon","Pygidium","Segment_","Thorax","thorax"))] or list(faces)
        F=np.array([f for n in keep for f in faces[n]]); idx=sorted(set(F.ravel())); rm={o:i for i,o in enumerate(idx)}
        m=trimesh.Trimesh(V[idx][:,[0,2,1]], np.vectorize(rm.get)(F), process=False)
    m.merge_vertices(digits_vertex=4)
    if len(m.faces)>target: m=m.simplify_quadric_decimation(face_count=target)
    m=trimesh.Trimesh(m.vertices,m.faces,process=False); m.merge_vertices(digits_vertex=4)
    m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices()
    return m

def dorsal_up(v):
    W=np.abs(v[:,0]).max(); mid=np.abs(v[:,0])<0.18*W; ple=(np.abs(v[:,0])>0.4*W)&(np.abs(v[:,0])<0.72*W)
    v=v.copy()
    if v[mid,2].mean()<v[ple,2].mean(): v[:,2]=v[:,2].max()-v[:,2]
    return v

CFG={"olenoides":(8,(36,102)),"harpetid":(10,(52,122)),"proetida":(9,(24,110))}
out={}
for name,(nseg,band) in CFG.items():
    m=load_welded(name)
    v0=load_any.orient_scale(m.vertices)
    info=fs.all_seams(v0, n_segments=nseg, band=band)
    vv=dorsal_up(info["v"]); F=m.faces
    idx=C.assign_faces(vv, F, info["seams"])
    joints=[float(j) for j in sorted(info["mins"])]; L=float(vv[:,1].max()); relief=float(vv[:,2].max())
    pieces=[]
    for k in range(idx.max()+1):
        fk=F[idx==k]
        if len(fk)==0: pieces.append(dict(v=[],f=[])); continue
        used=np.unique(fk); rm=-np.ones(len(vv),int); rm[used]=np.arange(len(used))
        pv=vv[used]; pf=rm[fk]
        pm=trimesh.Trimesh(pv,pf,process=False); comps=len(pm.split(only_watertight=False))
        pieces.append(dict(v=np.round(pv,1).ravel().tolist(), f=pf.ravel().astype(int).tolist(), comps=comps))
    tot=sum(len(p.get("v",[]))//3 for p in pieces)
    print(f"{name:10s} {len(pieces)} pieces, {tot} verts total, piece comps {[p.get('comps','-') for p in pieces]}")
    out[name]=dict(pieces=[{k:p[k] for k in ('v','f')} for p in pieces], joints=joints, L=L, relief=relief, nseg=nseg)

# phacops static, welded
mp=load_welded.__wrapped__ if hasattr(load_welded,'__wrapped__') else None
import pickle
Mp=pickle.load(open("meshes.pkl","rb"))["phacops"]
vp=Mp[0].copy(); vp[:,2]=vp[:,2].max()-vp[:,2]
pm=trimesh.Trimesh(vp,Mp[1],process=True); pm.merge_vertices(digits_vertex=3)
pm=pm.simplify_quadric_decimation(face_count=12000); pm=trimesh.Trimesh(pm.vertices,pm.faces,process=False); pm.merge_vertices(digits_vertex=3)
out["phacops"]=dict(pieces=[dict(v=np.round(pm.vertices,1).ravel().tolist(), f=pm.faces.ravel().astype(int).tolist())],
                    joints=[], L=float(pm.vertices[:,1].max()), relief=float(pm.vertices[:,2].max()), nseg=0)
print("phacops static")
js="window.SPECIES = "+json.dumps(out)+";\n"; open("species_data.js","w").write(js)
print("wrote species_data.js  %.0f KB"%(len(js)/1024))
