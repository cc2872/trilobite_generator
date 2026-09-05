"""assemble.py — the completed sculpt-mesh trilobite from one parameter dict."""
import numpy as np, trimesh, math
import furrow_slicer as fs, cutter as C, rig, schema, fields

def _sphere(sub=2): return trimesh.creation.icosphere(subdivisions=sub, radius=1.0)

def make_eye(cx, cy, cz, n, kind, along, wid, ht, side):
    parts=[]; schizo=(kind=='schizo')
    base=_sphere(2); base.apply_scale([wid, along*0.5, ht])
    base.apply_translation([cx+side*wid*0.3, cy, cz+ht*0.35])
    base.visual.vertex_colors=[150,110,70,255]; parts.append(base)
    if n>0:
        files=max(2,round(math.sqrt(n/2.0))) if schizo else max(3,round(math.sqrt(n)))
        rows=max(1,math.ceil(n/files)); lensR=float(np.clip((1.7 if schizo else 1.0)*math.sqrt(20/max(n,6)),0.55,2.8))
        c=0
        for f in range(files):
            for r in range(rows):
                if c>=n: break
                a=(f+0.5)/files-0.5; b=(r+0.5)/rows-0.5
                px=cx+side*(wid*a*2+wid*0.55); py=cy+along*b
                dome=ht*max(0,1-(2*a)**2)*max(0,1-(1.4*b)**2); pz=cz+ht*0.35+dome
                Lz=_sphere(1); Lz.apply_scale([lensR,lensR,lensR*(1.15 if schizo else 0.85)])
                Lz.apply_translation([px,py,pz]); Lz.visual.vertex_colors=[38,34,30,255]; parts.append(Lz); c+=1
    return trimesh.util.concatenate(parts)

def _cyl(p0,p1,r0,r1):
    p0=np.array(p0,float);p1=np.array(p1,float);d=p1-p0;L=np.linalg.norm(d)+1e-9
    c=trimesh.creation.cylinder(radius=(r0+r1)/2,height=L,sections=8)
    z=np.array([0,0,1.]);dn=d/L;ax=np.cross(z,dn);s=np.linalg.norm(ax)
    if s>1e-6:
        ax/=s;ang=math.acos(np.clip(np.dot(z,dn),-1,1))
        c.apply_transform(trimesh.transformations.rotation_matrix(ang,ax))
    c.apply_translation((p0+p1)/2);return c

def assemble(v, faces, seams, seam_ys, L0, P, eye_n=18, eye_kind='schizo', legs_on=True, eyePos=0.5):
    P=schema.coerce(P); P0=schema.coerce(dict(length=float(L0)))
    idx=C.assign_faces(v, faces, seams)
    pieces_v=[v]*(idx.max()+1); pieces_f=[faces[idx==k] for k in range(idx.max()+1)]
    res=rig.rig(pieces_v, pieces_f, seam_ys, L0, P, P0=P0, enroll=P["enroll"] if "enroll" in P else 0.0)
    # per-piece compacted body meshes, coloured by segment
    from matplotlib import cm
    body=[]
    for k,(vp,fp) in enumerate(zip(res["posed"],res["faces"])):
        if len(fp)==0: continue
        used=np.unique(fp); remap=-np.ones(len(vp),int); remap[used]=np.arange(len(used))
        m=trimesh.Trimesh(vp[used], remap[fp], process=False)
        col=(np.array(cm.tab20(k%20))*255).astype(int); m.visual.vertex_colors=col
        body.append(m)
    wrapped=res["posed"][0]                       # head frame == identity, so posed head == wrapped body
    relief_actual=float(wrapped[:,2].max())
    # wrapped half-width lookup
    yb=np.linspace(0,P["length"],140); Wb=np.zeros_like(yb)
    for i,(a,b) in enumerate(zip(yb[:-1],yb[1:])):
        s=(wrapped[:,1]>=a)&(wrapped[:,1]<b); Wb[i]=np.abs(wrapped[s,0]).max() if s.any() else 0
    halfW=lambda y: float(np.interp(y,yb,Wb))
    # eyes on the cephalon
    eyes=[]
    if eye_n is not None:
        headLen=P["cephFrac"]*P["length"]; yE=headLen*(1-eyePos*0.55); W=halfW(yE)
        along=0.26*headLen; wid=0.11*W; ht=relief_actual*(0.30 if eye_kind=='schizo' else 0.20)
        for side in (1,-1):
            cx=side*W*0.58
            sel=(np.abs(wrapped[:,0]-cx)<4)&(np.abs(wrapped[:,1]-yE)<5)
            cz=(float(wrapped[sel,2].max()) if sel.any() else float(wrapped[:,2].max())) - ht*0.35
            eyes.append(make_eye(cx,yE,cz,eye_n,eye_kind,along,wid,ht,side))
    # legs per thoracic segment
    legs=[]
    if legs_on:
        H=res["hinge_ys"]; mats=res["mats"]; relief=relief_actual
        for k in range(1,len(H)):
            y=0.5*(H[k-1]+H[k]); W=halfW(y); dp=H[k]-H[k-1]
            for side in (1,-1):
                hip=[side*W*0.80,y,relief*0.06]; knee=[side*W*0.98,y+dp*0.20,-relief*0.28]; foot=[side*W*0.86,y+dp*0.42,-relief*0.60]
                for c in (_cyl(hip,knee,1.0,0.7), _cyl(knee,foot,0.7,0.35)):
                    c.apply_transform(mats[k]); c.visual.vertex_colors=[95,65,39,255]; legs.append(c)
    scene=trimesh.Scene()
    for i,m in enumerate(body): scene.add_geometry(m,geom_name=f"seg{i}")
    for i,e in enumerate(eyes): scene.add_geometry(e,geom_name=f"eye{i}")
    if legs: scene.add_geometry(trimesh.util.concatenate(legs),geom_name="legs")
    return dict(scene=scene, body=body, eyes=eyes, legs=legs)
