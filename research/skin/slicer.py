"""slicer.py — trace the pleural furrows of a fused sculpt and cut it into head / segments / tail."""
import numpy as np, trimesh
from load_sculpt import load

def heightfield(body, cell=0.8, pad=2.0):
    v=body.vertices
    x0,x1=v[:,0].min()-pad, v[:,0].max()+pad; y0,y1=v[:,1].min()-pad, v[:,1].max()+pad
    nx=int((x1-x0)/cell)+1; ny=int((y1-y0)/cell)+1
    Z=np.full((ny,nx), np.nan)
    ix=((v[:,0]-x0)/cell).astype(int); iy=((v[:,1]-y0)/cell).astype(int)
    np.maximum.at(Z,(iy,ix),v[:,2]) if False else None
    # max per cell
    order=np.lexsort((v[:,2],ix,iy))
    for k in order: 
        if np.isnan(Z[iy[k],ix[k]]) or v[k,2]>Z[iy[k],ix[k]]: Z[iy[k],ix[k]]=v[k,2]
    # fill small gaps with neighbourhood max, then light blur
    from itertools import product
    Zf=Z.copy()
    for _ in range(2):
        Zp=np.pad(Zf,1,mode="edge"); stack=np.stack([Zp[1+dy:ny+1+dy,1+dx:nx+1+dx] for dy,dx in product((-1,0,1),repeat=2)])
        Zf=np.where(np.isnan(Zf), np.nanmax(stack,axis=0), Zf)
    Zb=Zf.copy(); Zb[np.isnan(Zb)]=0
    k=np.array([1,2,1])/4
    for _ in range(1):
        Zb=np.apply_along_axis(lambda r: np.convolve(r,k,mode="same"),1,Zb); Zb=np.apply_along_axis(lambda r: np.convolve(r,k,mode="same"),0,Zb)
    xs=x0+cell*np.arange(nx); ys=y0+cell*np.arange(ny)
    return xs,ys,Zb,~np.isnan(Zf)

def midline_furrows(xs,ys,Z,mask,y_lo,y_hi,halfw=2.0,min_gap=3.0):
    cols=np.abs(xs)<halfw; crest=np.nanmax(np.where(mask[:,cols],Z[:,cols],np.nan),axis=1)
    crest=np.nan_to_num(crest,nan=0.0)
    mins=[]
    for i in range(3,len(ys)-3):
        if ys[i]<y_lo or ys[i]>y_hi: continue
        if crest[i]<crest[i-1] and crest[i]<=crest[i+1] and crest[i]<crest[i-3]-0.08 and crest[i]<crest[i+3]-0.08:
            if not mins or ys[i]-mins[-1]>=min_gap: mins.append(ys[i])
    return np.array(mins), crest

def trace_boundary(xs,ys,Z,mask,y_mid,x_max,back=3.5,fwd=1.5):
    """Follow the furrow outward from the midline. Returns a callable y(x): smoothed inside the body,
    linear extrapolation with the margin slope beyond it (spines are straight, so the gap between them is too)."""
    pts=[]
    for side in (1,-1):
        y_prev=y_mid
        for x in np.arange(0,x_max,1.0):
            ci=np.argmin(np.abs(xs-side*x))
            sel=(ys>=y_prev-fwd)&(ys<=y_prev+back)&mask[:,ci]
            if sel.sum()<3: break
            col=Z[:,ci].copy(); col[~sel]=np.inf
            j=np.argmin(col); y_prev=ys[j]; pts.append((abs(x),y_prev))
    pts=np.array(pts) if len(pts) else np.zeros((0,2))
    if len(pts)<6:
        return (lambda x: np.full_like(np.asarray(x,float), y_mid)), pts
    xa=np.sort(np.unique(pts[:,0])); ya=np.array([pts[pts[:,0]==x,1].mean() for x in xa])
    k=np.array([1,2,3,2,1])/9.0; ys_s=np.convolve(np.pad(ya,2,mode="edge"),k,mode="valid")
    x_end=xa[-1]; tail=slice(max(0,len(xa)-8),len(xa)); slope=np.polyfit(xa[tail],ys_s[tail],1)[0]
    def f(x):
        x=np.abs(np.asarray(x,float))
        inside=np.interp(x,xa,ys_s)
        return np.where(x<=x_end, inside, ys_s[-1]+slope*(x-x_end))
    return f, pts

def curve(f,x): return f(x)

def prism(front_coef,rear_coef,x_max,z_lo=-5,z_hi=80,n=40):
    """Vertical prism between two boundary curves (None = open end)."""
    xs=np.linspace(-x_max,x_max,n)
    yf=front_coef(xs) if front_coef is not None else np.full(n,-50.0)
    yr=rear_coef(xs) if rear_coef is not None else np.full(n,400.0)
    yr=np.maximum(yr, yf+0.6)                      # consecutive cuts may cross far out; keep the strip well-formed
    poly=np.r_[np.c_[xs,yf], np.c_[xs[::-1],yr[::-1]]]
    from shapely.geometry import Polygon
    return trimesh.creation.extrude_polygon(Polygon(poly).buffer(0), z_hi-z_lo).apply_translation((0,0,z_lo))

def slice_body(body, boundaries, x_max=60):
    parts=[]
    seq=[None]+list(boundaries)+[None]
    for i in range(len(seq)-1):
        cutter=prism(seq[i],seq[i+1],x_max)
        piece=trimesh.boolean.intersection([body,cutter],engine="manifold")
        parts.append(piece)
    return parts
