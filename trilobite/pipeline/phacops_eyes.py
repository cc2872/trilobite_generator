import pickle, numpy as np, trimesh, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
M=pickle.load(open("meshes.pkl","rb"))
v,f=np.array(M["phacops"][0],float), np.array(M["phacops"][1])
v[:,2]=v[:,2].max()-v[:,2]                         # dorsal convex up
# cephalon at low y (broad end already low y in this frame); confirm & set
L=np.ptp(v[:,1]); y0=v[:,1].min()
ceph_mask_v = v[:,1] < y0 + 0.36*L
# eyes: raised lobes on the cheeks, lateral to the glabella. Height residual above a smooth baseline.
cell=0.8
x0,x1=v[:,0].min(),v[:,0].max(); yy0,yy1=v[:,1].min(),v[:,1].max()
nx=int((x1-x0)/cell)+1; ny=int((yy1-yy0)/cell)+1
Z=np.full((ny,nx),np.nan)
ix=np.clip(((v[:,0]-x0)/cell).astype(int),0,nx-1); iy=np.clip(((v[:,1]-yy0)/cell).astype(int),0,ny-1)
o=np.argsort(v[:,2]); Z[iy[o],ix[o]]=v[o,2]
Zf=np.nan_to_num(Z,nan=np.nanmin(Z))
base=ndimage.uniform_filter(Zf,size=13)
resid=Zf-base
# candidate eye cells: within cephalon rows, off-axis, positive residual
Ceph=np.zeros_like(Zf,bool)
yrow=yy0+cell*np.arange(ny)
Ceph[(yrow<y0+0.40*L)&(yrow>y0+0.05*L),:]=True
xcol=x0+cell*np.arange(nx)
offaxis=(np.abs(xcol)>7)&(np.abs(xcol)<38)
cand=Ceph & offaxis[None,:] & (resid>0.30*resid[Ceph].max()) & (Z> -1e8)
lab,n=ndimage.label(cand)
sizes=ndimage.sum(cand,lab,range(1,n+1))
# keep the largest blob on each side of x=0
eyes=[]
for side in (+1,-1):
    best=None;bestsz=0
    for lid in range(1,n+1):
        ys_,xs_=np.where(lab==lid)
        if len(xs_)<8: continue
        mx=np.mean(xcol[xs_])
        if np.sign(mx)!=side: continue
        if sizes[lid-1]>bestsz: bestsz=sizes[lid-1]; best=lid
    if best is not None: eyes.append(best)
eyemask=np.isin(lab,eyes)
# map eye cells back to vertices, then faces whose centroid falls in an eye cell
def cell_of(pts):
    cix=np.clip(((pts[:,0]-x0)/cell).astype(int),0,nx-1); ciy=np.clip(((pts[:,1]-yy0)/cell).astype(int),0,ny-1)
    return ciy,cix
fc=v[f].mean(1); ciy,cix=cell_of(fc)
face_is_eye = eyemask[ciy,cix] & (fc[:,1]<y0+0.42*L)
print("eye faces:", face_is_eye.sum(), "of", len(f))
# render
fig=plt.figure(figsize=(12,7))
ax=fig.add_subplot(1,2,1); ax.scatter(v[:,0],v[:,1],s=1,c="lightgray")
ax.scatter(fc[face_is_eye,0],fc[face_is_eye,1],s=2,c="crimson"); ax.set_aspect("equal")
ax.set_title("phacops: detected eyes (red)"); ax.set_yticks(np.arange(0,131,10)); ax.grid(alpha=.3)
ax2=fig.add_subplot(1,2,2, projection="3d")
ee=f[face_is_eye]
ax2.plot_trisurf(v[:,0],v[:,1],v[:,2],triangles=ee,color="crimson",alpha=0.9,linewidth=0)
ax2.set_title("extracted eyes (iso)"); ax2.view_init(elev=35,azim=-60)
try:
    ax2.set_box_aspect((np.ptp(v[:,0]),np.ptp(v[:,1]),np.ptp(v[:,2])*3))
except Exception: pass
plt.tight_layout(); plt.savefig("phacops_eyes_extracted.png",dpi=80); print("wrote phacops_eyes_extracted.png")
# save eye meshes
used=np.unique(ee); remap=-np.ones(len(v),int); remap[used]=np.arange(len(used))
eye_mesh=trimesh.Trimesh(v[used],remap[ee],process=False)
eye_mesh.export("phacops_eyes.stl"); print("saved phacops_eyes.stl", len(eye_mesh.faces),"faces")
