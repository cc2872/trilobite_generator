"""Load each species, orient to X-across / Y-along(head at -y) / Z-up, scale to 130mm length."""
import numpy as np, trimesh, atlas

def _decimate(m, target_faces):
    if len(m.faces) > target_faces:
        try: m = m.simplify_quadric_decimation(face_count=int(target_faces))
        except Exception: pass
    return m

def orient_scale(v, target_length=130.0, fossil=False, flip_z=None, flip_y=None):
    v = atlas.pca_align(v)
    if flip_z is None:
        v = atlas.orient(v)
    else:
        if flip_z: v[:,2] *= -1
        if flip_y: v[:,1] *= -1
        v = v - [0, v[:,1].min(), v[:,2].min()]
    L = np.ptp(v[:,1]); s = target_length / L
    v = v * s
    v[:,0] -= np.median(v[:,0])                 # midline to x=0
    v = v - [0, v[:,1].min(), v[:,2].min()]     # head front at y=0, base at z=0
    return v

def load_species(name):
    B = "/home/claude/sculpts"
    if name == "olenoides":
        m = trimesh.load(f"{B}/llllline_202609284405_82eb7f81390a49cab3ce5a1cc39bf258/Trilobite/Olenoides_serratus_shell_pose_02.stl", process=False)
        m = _decimate(m, 180000)
        v = orient_scale(m.vertices, flip_z=None)
        F = m.faces
    elif name == "harpetid":
        m = trimesh.load(f"{B}/Trib-Harpedita-Simple_fixed_stls/obj_1_Trib-Harpedita-Simple_fixed.stl", process=False)
        m = _decimate(m, 180000)
        v = orient_scale(m.vertices); F = m.faces
    elif name == "proetida":
        # OBJ with named groups; keep only shell (drop legs/antennae)
        path = f"{B}/trilobite-proetida/source/extracted/export.obj"
        V=[]; faces={}; cur="d"
        for line in open(path):
            if line.startswith("v "): V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("g "): cur=line[2:].strip()
            elif line.startswith("f "): faces.setdefault(cur,[]).append([int(t.split("/")[0])-1 for t in line.split()[1:4]])
        V=np.array(V)
        keep=[n for n in faces if ("Cephalon" in n or "Pygidium" in n or "Segment_" in n or "Thorax" in n or "thorax" in n)]
        if not keep: keep=list(faces)     # fallback: everything
        F=np.array([f for n in keep for f in faces[n]])
        idx=sorted(set(F.ravel())); remap={o:i for i,o in enumerate(idx)}
        F=np.vectorize(remap.get)(F); vb=V[idx][:,[0,2,1]]
        v=orient_scale(vb); 
    elif name == "phacops":
        path=f"{B}/trilobite-phacops-sp/source/extracted/trilobite-3.obj"
        m=trimesh.load(path, process=False, force='mesh')
        m=_decimate(m,180000)
        v=orient_scale(m.vertices, flip_z=None); F=m.faces
    mesh=trimesh.Trimesh(v, F, process=False)
    return mesh

if __name__ == "__main__":
    for n in ["olenoides","harpetid","proetida","phacops"]:
        m=load_species(n)
        e=m.extents
        print(f"{n:10s} verts {len(m.vertices):7d} faces {len(m.faces):7d}  L(y) {e[1]:6.1f}  W(x) {e[0]:6.1f}  H(z) {e[2]:5.1f}  watertight {m.is_watertight}")
