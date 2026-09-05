"""cutter.py — turn furrow seams into vertical prism cutters and slice the body into pieces."""
import numpy as np, trimesh
from shapely.geometry import Polygon
import furrow_slicer as fs

def _clean_curve(xr,yr,x_edge):
    # resample the seam on a uniform x grid spanning the full cutter width, clamp ends flat
    xg=np.linspace(-x_edge,x_edge,120)
    yg=np.interp(xg,xr,yr,left=yr[0],right=yr[-1])
    return xg,yg

def prism(front,rear,x_edge,z_lo,z_hi):
    """Solid between two seam curves front(x)<rear(x); None=open end."""
    xg=np.linspace(-x_edge,x_edge,120)
    yf=np.interp(xg,front[0],front[1]) if front is not None else np.full(120,-1e4)
    yr=np.interp(xg,rear[0],rear[1])  if rear  is not None else np.full(120, 1e4)
    yr=np.maximum(yr,yf+0.5)
    poly=np.r_[np.c_[xg,yf],np.c_[xg[::-1],yr[::-1]]]
    return trimesh.creation.extrude_polygon(Polygon(poly).buffer(0),z_hi-z_lo).apply_translation((0,0,z_lo))

def cut(mesh, info):
    v=info["v"]; z_lo=v[:,2].min()-5; z_hi=v[:,2].max()+5
    x_edge=info["x_edge"]
    seams=[ _clean_curve(xr,yr,x_edge) for xr,yr in info["seams"] ]
    body=trimesh.Trimesh(v, mesh.faces, process=False)
    bounds=[None]+seams+[None]
    pieces=[]
    for i in range(len(bounds)-1):
        cutr=prism(bounds[i],bounds[i+1],x_edge,z_lo,z_hi)
        p=trimesh.boolean.intersection([body,cutr],engine="manifold")
        if p is not None and len(p.faces): pieces.append(p)
        else: pieces.append(trimesh.Trimesh())
    names=["head"]+[f"seg{i}" for i in range(len(seams)-1)]+["tail"]
    return pieces, names

def assign_faces(v, faces, seams):
    """Piece index per face = how many seam curves the face centroid lies BEHIND (higher y)."""
    c=v[faces].mean(axis=1)                     # (F,3) centroids
    idx=np.zeros(len(faces),int)
    for (xr,yr) in seams:
        yseam=np.interp(c[:,0], xr, yr, left=yr[0], right=yr[-1])
        idx += (c[:,1] > yseam).astype(int)
    return idx

def split_faces(v, faces, idx):
    pieces=[]
    for k in range(idx.max()+1 if len(idx) else 0):
        fk=faces[idx==k]
        if len(fk)==0: pieces.append(None); continue
        used=np.unique(fk); remap=-np.ones(v.shape[0],int); remap[used]=np.arange(len(used))
        pieces.append(trimesh.Trimesh(v[used], remap[fk], process=False))
    return pieces
