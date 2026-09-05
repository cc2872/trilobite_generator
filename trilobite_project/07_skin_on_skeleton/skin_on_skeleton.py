"""
skin_on_skeleton.py — the cut Olenoides laid on YOUR parametric skeleton.

Yours (imported, unmodified): schema.py, fields.py (skeleton), wrap.py (P0_from), instrument.py
(transforms, collisions, e_max, measure, print_validity) via a 3-line trilobite.py shim.
Mine (this file): tiling the sculpt's segments onto segCount slots and trimming neighbours — i.e.
your wrap.resample_segments adapted to open shells (vertex trim instead of a boolean difference).
"""
import json, numpy as np, trimesh
import schema, fields, wrap, instrument as I

def clean_piece(v, f, min_faces=8):
    """Drop debris components (slivers < min_faces). Real detached spine tips (hundreds of faces)
    are kept: they are separate shells in the purchased sculpt and belong to this piece."""
    if len(f)==0: return v,f
    m=trimesh.Trimesh(v,f,process=False); keep=[c for c in m.split(only_watertight=False) if len(c.faces)>=min_faces]
    if not keep: return v,f
    m=trimesh.util.concatenate(keep); return np.asarray(m.vertices), np.asarray(m.faces)

def load_sculpt(name="olenoides", path="species_data.js"):
    js=open(path).read(); d=json.loads(js[js.index("{"):js.rindex("}")+1])[name]
    pieces=[clean_piece(np.array(p["v"]).reshape(-1,3), np.array(p["f"]).reshape(-1,3)) for p in d["pieces"]]
    return pieces, d["joints"], d["L"], d["relief"]

def load_legs(Hs, path="olen_legs.npz"):
    d=np.load(path); v=d["v"]; f=d["f"]; fc=v[f].mean(1)
    seg=np.array([int(np.sum(np.array(Hs)<=y)) for y in fc[:,1]]); out={}
    for k in np.unique(seg):
        fk=f[seg==k]; used=np.unique(fk); rm=-np.ones(len(v),int); rm[used]=np.arange(len(used))
        out[int(k)]=(v[used].copy(), rm[fk])
    return out

def P0(pieces, Hs, Ls):
    """Your wrap.P0_from on the sculpt. One override: axisRise=0 — the sculpt's relief already
    includes its axial ring, so ring_top == max z and hinge_z lands inside the shell."""
    body=trimesh.util.concatenate([trimesh.Trimesh(v,f,process=False) for v,f in pieces if len(f)])
    return schema.coerce(dict(wrap.P0_from(body, Hs[0]/Ls, Hs[-1]/Ls, len(pieces)-2), axisRise=0.0))

def skeleton(P):
    L=fields.landmarks(P); d=fields.pitch(P); n=P["segCount"]
    J=[P["length"]*L["s_h"]+k*d for k in range(n+1)]
    return dict(P=P, sh=L["s_h"], st=L["s_t"], L=P["length"], n=n, pitch=d, joints=J, hinge_z=fields.hinge_z(P))

def skin(pieces, Hs, Ls, relief_s, P, P0_=None, legs=None, clearance=None):
    """Lay the sculpt on the skeleton slots (your wrap law: y'=slot, x'*=WP/WP0, z'*=relief ratio),
    resampling the sculpt's N segments onto segCount (your resample_segments), then trim neighbours."""
    P=schema.coerce(P); P0_=P0_ or P0(pieces,Hs,Ls); sk=skeleton(P); L=sk["L"]; J=sk["joints"]; n=sk["n"]
    Wc,Wc0=fields.width_curve(P),fields.width_curve(P0_); zr=P["relief"]/max(relief_s,1e-6); N=len(pieces)-2
    def remap(v,y0s,y1s,y0k,y1k):
        v=v.copy(); sc=(y1k-y0k)/max(y1s-y0s,1e-6); yn=y0k+(v[:,1]-y0s)*sc
        v[:,0]*=np.asarray(Wc(yn/L))/np.maximum(np.asarray(Wc0(yn/L)),1e-6); v[:,1]=yn; v[:,2]*=zr; return v
    def part(idx,y0s,y1s,y0k,y1k):
        v,f=pieces[idx]; v=remap(v,y0s,y1s,y0k,y1k)
        if legs and idx in legs: lv,lf=legs[idx]; v=np.vstack([v,remap(lv,y0s,y1s,y0k,y1k)]); f=np.vstack([f,lf+len(pieces[idx][0])])
        return v,f
    out=[part(0,0,Hs[0],0,L*sk["sh"])]
    for k in range(n):
        src=int(round(k*(N-1)/max(n-1,1))) if N>1 else 0
        out.append(part(1+src,Hs[src],Hs[src+1],J[k],J[k+1]))
    out.append(part(N+1,Hs[N],Ls,L*sk["st"],L))
    return trim_overlaps(out, P["clearance"] if clearance is None else clearance), sk

def trim_overlaps(parts, clearance, nb=60):
    """Your resample_segments rule ("neighbours must not interleave; the front part wins") for open
    shells: a face survives only if EVERY vertex is behind the predecessor's rear edge + clearance."""
    out=[parts[0]]
    for k in range(1,len(parts)):
        va,fa=out[-1]; vb,fb=parts[k]
        if len(fa)==0 or len(fb)==0: out.append((vb,fb)); continue
        xs=np.linspace(min(va[:,0].min(),vb[:,0].min()),max(va[:,0].max(),vb[:,0].max()),nb+1)
        rear=np.full(nb,-np.inf); ia=np.clip(np.searchsorted(xs,va[:,0])-1,0,nb-1); np.maximum.at(rear,ia,va[:,1])
        vv=vb[fb]; iv=np.clip(np.searchsorted(xs,vv[:,:,0])-1,0,nb-1)
        keep=np.all(vv[:,:,1]>rear[iv]+clearance,axis=1); fb2=fb[keep]
        used=np.unique(fb2) if len(fb2) else np.array([],int); rm=-np.ones(len(vb),int); rm[used]=np.arange(len(used))
        out.append((vb[used],rm[fb2]) if len(fb2) else (vb[:0],fb[:0]))
    return out

def localized(parts, sk):
    """Your part convention: each part's front hinge (head: rear hinge) at local y=0."""
    origins=[sk["L"]*sk["sh"]]+sk["joints"][:-1]+[sk["joints"][-1]]
    return [trimesh.Trimesh(np.c_[v[:,0],v[:,1]-o,v[:,2]],f,process=False) for (v,f),o in zip(parts,origins)]

def posed(meshes, P, enroll):
    """Your instrument.transforms + _posed."""
    return I._posed(meshes, I.transforms(schema.coerce(P), enroll))

if __name__=="__main__":
    pieces,Hs,Ls,rel=load_sculpt(); P0_=P0(pieces,Hs,Ls); legs=load_legs(Hs)
    res=[]
    for lg in (False,True):
        for cl in (0.3,0.8):
            parts,sk=skin(pieces,Hs,Ls,rel,P0_,P0_,legs=legs if lg else None,clearance=cl)
            r=I.measure(P0_, localized(parts,sk))
            row=dict(legs=lg,clearance=cl,**{k:r[k] for k in ("e_max","free_curl_deg","total_curl_deg","closure_gap_mm","enroll_class","stopped_by","touching_at_zero","print_valid")})
            print(row); res.append(row)
    json.dump(dict(skeleton="schema.py+fields.py", skin="Olenoides furrow-cut (01_slicer)", instrument="instrument.py v1.0 unmodified (trilobite.py shim)",
                   P0={k:round(P0_[k],3) for k in ("length","width","relief","segCount","cephFrac","pygFrac","widthMaxPos","axisRise")}, results=res),
              open("instrument_results.json","w"), indent=1)
