"""load_sculpt.py — decode the purchased glTF, merge, orient, scale. X across, Y along (head at -Y), Z up."""
import json, numpy as np, trimesh
import os
SRC=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
def read_prim(g, buf, prim):
    acc=g["accessors"]; bv=g["bufferViews"]
    def rd(ai, dtype, n):
        a=acc[ai]; v=bv[a["bufferView"]]; off=v.get("byteOffset",0)+a.get("byteOffset",0)
        return np.frombuffer(buf, dtype=dtype, count=a["count"]*n, offset=off).reshape(a["count"],n)
    V=rd(prim["attributes"]["POSITION"], np.float32, 3).astype(float)
    ai=acc[prim["indices"]]; dt={5123:np.uint16,5125:np.uint32}[ai["componentType"]]
    F=rd(prim["indices"], dt, 1).reshape(-1,3).astype(int)
    return V,F
def load(target_length=130.0):
    g=json.load(open(f"{SRC}/1472447.gltf")); buf=open(f"{SRC}/1472447.data.bin","rb").read()
    node_mesh={n["name"]:n for n in g["nodes"] if "mesh" in n}
    parts={}
    for name,n in node_mesh.items():
        V,F=read_prim(g,buf,g["meshes"][n["mesh"]]["primitives"][0])
        M=np.array(n.get("matrix",np.eye(4).ravel(order="F"))).reshape(4,4,order="F")
        V=(M@np.c_[V,np.ones(len(V))].T).T[:,:3]
        m=trimesh.Trimesh(V,F,process=False); m.merge_vertices(); parts[name]=m
    body=parts["Trilobite_spine"]
    s=target_length/(body.bounds[1][1]-body.bounds[0][1])
    ymin=body.bounds[0][1]; zmin=body.bounds[0][2]
    T=np.eye(4); T[:3,:3]*=s; T[:3,3]=(0,-ymin*s,-zmin*s)     # head front at y=0, base at z=0
    for m in parts.values(): m.apply_transform(T)
    return parts, s
if __name__=="__main__":
    parts,s=load()
    for k,m in parts.items(): print(f"{k:16s} verts {len(m.vertices):6d} watertight {m.is_watertight} volume {m.is_volume} extents {np.round(m.extents,1)} y[{m.bounds[0][1]:.1f},{m.bounds[1][1]:.1f}]")
    print("scale", round(s,3))
