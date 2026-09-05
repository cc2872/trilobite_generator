"""
furrow_slicer.py — cut a trilobite sculpt into head / thoracic segments / pygidium along its OWN
interpleural furrows, so each pleural (and genal) spine stays whole with the segment it belongs to.

Seam = furrow valley traced across the width, constrained to sweep backward (never forward into the
spine ahead), continued past the body margin along the furrow's own tangent into the notch between
spine bases, then held flat where there is no material. Cutter is a vertical prism along that seam.
"""
import numpy as np

# ---------- dorsal heightfield ----------
def heightfield(v, cell=0.7, blur=1):
    x0,x1=v[:,0].min(),v[:,0].max(); y0,y1=v[:,1].min(),v[:,1].max()
    nx=int((x1-x0)/cell)+1; ny=int((y1-y0)/cell)+1
    Z=np.full((ny,nx),np.nan)
    ix=np.clip(((v[:,0]-x0)/cell).astype(int),0,nx-1); iy=np.clip(((v[:,1]-y0)/cell).astype(int),0,ny-1)
    o=np.argsort(v[:,2]); Z[iy[o],ix[o]]=v[o,2]
    mask=~np.isnan(Z)
    Zf=Z.copy()
    from itertools import product
    for _ in range(2):
        Zp=np.pad(Zf,1,mode="edge")
        st=np.stack([Zp[1+dy:ny+1+dy,1+dx:nx+1+dx] for dy,dx in product((-1,0,1),repeat=2)])
        Zf=np.where(np.isnan(Zf), np.nanmax(np.where(np.isnan(st),-1e9,st),axis=0), Zf)
    Zf=np.where(Zf<-1e8,np.nan,Zf)
    Zb=np.nan_to_num(Zf,nan=np.nanmin(Zf))
    if blur:
        k=np.array([1,2,1])/4.
        for _ in range(blur):
            Zb=np.apply_along_axis(lambda r:np.convolve(r,k,mode="same"),1,Zb)
            Zb=np.apply_along_axis(lambda r:np.convolve(r,k,mode="same"),0,Zb)
    xs=x0+cell*np.arange(nx); ys=y0+cell*np.arange(ny)
    return xs,ys,Zb,mask,Zf

def halfwidth(xs,ys,mask):
    W=np.zeros(len(ys))
    for j in range(len(ys)):
        cols=np.where(mask[j])[0]
        if len(cols): W[j]=max(abs(xs[cols[0]]),abs(xs[cols[-1]]))
    k=np.array([1,2,3,2,1])/9.; return np.convolve(np.pad(W,2,mode="edge"),k,mode="valid")

def orient_head_low(v):
    ys=np.linspace(v[:,1].min(),v[:,1].max(),60); crest=[]
    for a,b in zip(ys[:-1],ys[1:]):
        sel=(v[:,1]>=a)&(v[:,1]<b)&(np.abs(v[:,0])<0.08*np.ptp(v[:,0]))
        crest.append(v[sel,2].max() if sel.any() else np.nan)
    c=np.nan_to_num(np.array(crest),nan=np.nanmin(crest))
    cs=np.convolve(np.pad(c,4,mode="edge"),np.ones(9)/9,mode="valid")
    if np.argmax(cs) > len(cs)//2:
        v=v.copy(); v[:,1]=v[:,1].max()-v[:,1]
    return v

# ---------- furrow detection: width-integrated transverse valleys, pitch from autocorrelation ----------
def furrow_signal(xs,ys,Zb,mask,W):
    """Combined furrow signal: z-normalised axial-ring crest (narrow midline band) + pleural-band mean.
    Both dip at interpleural furrows; the axial term carries the segment count for spiny forms where the
    pleural term is swamped by spines, the pleural term carries effaced forms."""
    def norm(a):
        a=np.array(a,float); good=np.isfinite(a)
        if good.sum()<8: return None
        a=np.interp(np.arange(len(a)),np.where(good)[0],a[good])
        r=np.nanmax(a)-np.nanmin(a); return (a-np.nanmin(a))/r if r>1e-6 else a*0
    axial=np.full(len(ys),np.nan); pleur=np.full(len(ys),np.nan)
    for j in range(len(ys)):
        if W[j]<=1: continue
        cen=mask[j] & (np.abs(xs)<max(2.0,0.12*W[j]))
        if cen.sum()>=2: axial[j]=np.max(Zb[j,cen])
        band=mask[j] & (np.abs(xs)>=0.15*W[j]) & (np.abs(xs)<=0.78*W[j])
        if band.sum()>=4: pleur[j]=np.mean(Zb[j,band])
    a=norm(axial); p=norm(pleur)
    if a is None and p is None: return None
    if a is None: sig=p
    elif p is None: sig=a
    else: sig=0.6*a+0.4*p
    k=np.array([1,2,3,2,1])/9.; return np.convolve(np.pad(sig,2,mode="edge"),k,mode="valid")

def estimate_pitch(sig, thorax_idx, cell):
    s=sig[thorax_idx]; s=s-np.convolve(np.pad(s,10,mode="edge"),np.ones(21)/21,mode="valid")  # detrend
    if len(s)<12: return None
    ac=np.correlate(s,s,mode="full")[len(s)-1:]; ac[:2]=-1e9
    lo=max(2,int(3.0/cell))
    peak=lo+int(np.argmax(ac[lo:min(len(ac),int(18/cell))]))
    return float(np.clip(peak*cell,5.0,14.0))

def _thorax_band(ys,sig):
    """Segmented region = where the detrended axial signal oscillates. Orientation-independent."""
    n=len(sig); trend=np.convolve(np.pad(sig,12,mode="edge"),np.ones(25)/25,mode="valid")[:n]
    d=sig-trend; w=8
    env=np.array([d[max(0,i-w):i+w+1].std() for i in range(n)])
    thr=0.35*env.max()
    on=env>thr
    # largest contiguous True run
    best=(0,0); i=0
    while i<n:
        if on[i]:
            j=i
            while j<n and on[j]: j+=1
            if j-i>best[1]-best[0]: best=(i,j)
            i=j
        else: i+=1
    a,b=best
    return max(0,a-2), min(n-1,b+2)

def find_furrows(xs,ys,Zb,mask,W, thorax=(0.14,0.90), n=None, n_segments=None, band=None):
    sig=furrow_signal(xs,ys,Zb,mask,W)
    if sig is None: return np.array([]), np.zeros(len(ys))
    if band is not None:
        ya,yb=band; ia=int(np.clip((ya-ys[0])/(ys[1]-ys[0]),0,len(ys)-1)); ib=int(np.clip((yb-ys[0])/(ys[1]-ys[0]),0,len(ys)-1))
    else:
        ia,ib=_thorax_band(ys,sig); ya,yb=ys[ia],ys[ib]
    if n_segments is None:
        # fall back to peak picking within the band
        cell=ys[1]-ys[0]; pitch=estimate_pitch(sig,np.arange(ia,ib),cell) or 8.0
        min_gap=max(2.5,0.5*pitch); w=max(2,int(round(min_gap/cell/2)))
        rng=np.nanmax(sig[ia:ib])-np.nanmin(sig[ia:ib]); prom=0.04*rng; cand=[]
        for i in range(ia,ib):
            lo=max(0,i-w); hi=min(len(ys),i+w+1)
            if sig[i]==sig[lo:hi].min() and sig[lo:hi].max()-sig[i]>prom: cand.append(i)
        kept=[]
        for i in sorted(cand):
            if kept and ys[i]-ys[kept[-1]]<min_gap:
                if sig[i]<sig[kept[-1]]: kept[-1]=i
            else: kept.append(i)
        if n is not None and len(kept)>n:
            depth={i:sig[max(0,i-w):i+w+1].max()-sig[i] for i in kept}
            kept=sorted(sorted(kept,key=lambda i:-depth[i])[:n])
        return (ys[np.array(kept)] if kept else np.array([])), sig
    # uniform placement of (n_segments+1) seams across the band, each snapped to the local furrow valley
    nseams=n_segments+1
    targets=np.linspace(ya,yb,nseams)
    cell=ys[1]-ys[0]; win=int(0.35*(yb-ya)/nseams/cell)+1
    placed=[]
    for t in targets:
        it=int(np.clip(round((t-ys[0])/cell),0,len(ys)-1))
        lo=max(0,it-win); hi=min(len(ys),it+win+1)
        j=lo+int(np.argmin(sig[lo:hi]))
        placed.append(ys[j])
    placed=sorted(set(placed)); dd=[placed[0]] if placed else []
    for y in placed[1:]:
        if y-dd[-1]>=4.0: dd.append(y)
    return np.array(dd), sig

# ---------- trace ONE furrow across the width ----------
def trace_furrow(xs,ys,Zb,mask,W, y_f, band=5.0, spine_reach=0.5, relief=1.0):
    """Blend a straight transverse cut (at y_f) with the valley trace, weighted by how deep the
    interpleural furrow actually is at each column. Deep furrow (spiny pleura) -> follow it fully;
    shallow/effaced -> stay straight. Extend into the spine notch along the local tangent, then hold."""
    cell=xs[1]-xs[0]
    col_at=lambda x:int(np.clip(round((x-xs[0])/cell),0,len(xs)-1))
    Wmax=np.nanmax(W); x_edge=1.9*Wmax
    seam={}
    for side in (+1,-1):
        prev_y=y_f; _disp=0.0; slope_hist=[]; in_body=True; ext=0.0; Wlocal=Wmax; x_in=0.0
        for x in np.arange(0, x_edge, cell):
            ci=col_at(side*x)
            sel=mask[:,ci] & (ys>=prev_y-band) & (ys<=prev_y+band)
            if in_body and sel.sum()>=3:
                zc=Zb[:,ci]; yv=ys[sel]; zv=zc[sel]
                jmin=np.argmin(zv); y_valley=yv[jmin]; z_valley=zv[jmin]
                z_top=zv.max()                                   # local ridge height in the band
                conf=np.clip((z_top-z_valley)/(0.28*relief+1e-6),0,1)   # furrow depth confidence
                y_valley=max(y_valley, prev_y-0.4)               # near-monotone backsweep
                y=(1-conf)*y_f + conf*y_valley                   # blend straight <-> valley
                # keep the blended seam from ever jumping forward of y_f by much
                y=max(y, y_f-0.6)
                seam[side*x]=y; slope_hist.append((y-prev_y)/cell); prev_y=0.5*prev_y+0.5*y; x_in=x
            else:
                if in_body: in_body=False; Wlocal=max(x_in,1.0)
                ext+=cell
                if ext<=spine_reach*Wlocal:
                    sl=np.clip(np.median(slope_hist[-6:]) if len(slope_hist)>=4 else 0.0,-0.05,0.8)
                    prev_y=prev_y+sl*cell
                seam[side*x]=prev_y
    xr=np.array(sorted(seam)); yr=np.array([seam[x] for x in xr])
    k=np.array([1,2,3,4,3,2,1])/16.; yr=np.convolve(np.pad(yr,3,mode="edge"),k,mode="valid")
    return xr,yr

def all_seams(v, cell=0.7, thorax=(0.14,0.90), n=None, n_segments=None, band=None, flip_y=False):
    v=orient_head_low(v)
    if flip_y:
        v=v.copy(); v[:,1]=v[:,1].max()-v[:,1]
    xs,ys,Zb,mask,Zf=heightfield(v,cell)
    W=halfwidth(xs,ys,mask)
    mins,sig=find_furrows(xs,ys,Zb,mask,W,thorax,n=n,n_segments=n_segments,band=band)
    relief=float(np.nanpercentile(Zb[mask],99)-np.nanpercentile(Zb[mask],2))
    seams=[trace_furrow(xs,ys,Zb,mask,W,m,relief=relief) for m in mins]
    return dict(v=v,xs=xs,ys=ys,Zb=Zb,mask=mask,W=W,mins=mins,sig=sig,seams=seams,x_edge=1.9*np.nanmax(W))

def old_seams(info):
    xr=np.linspace(-info["x_edge"],info["x_edge"],80)
    return [(xr,np.full_like(xr,m)) for m in info["mins"]]
