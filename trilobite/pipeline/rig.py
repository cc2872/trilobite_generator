"""
rig.py — hang the cut sculpt assets on the parametric pin-hinge skeleton and make them elongate /
shorten / articulate when the skeleton parameters change.

Skeleton (from the existing code, schema.py + fields.py):
  - rest spacing  : joint_offsets(P) = [0] + [pitch(P)]*segCount,  pitch = L(1-ceph-pyg)/segCount
  - hinge height  : hinge_z(P)
  - pose          : each part rotates -angle about X at the hinge, accumulated head->tail (chain)

Binding:
  1. Each cut piece keeps its body coordinate s = y/L0 (L0 = the sculpt's own length).
  2. wrap: s -> s' (piecewise-linear through the head/thorax/tail landmarks of P), then
     y' = P.length * s',  x' *= width_curve(P)/width_curve(P0),  z' *= P.relief/relief0.
     => the asset stretches/compresses with length, pitch and width of the skeleton.
  3. hinge axes sit at the wrapped seam positions, z = hinge_z(P).
  4. curl: rotate each piece about its hinge by enroll*maxAngle, accumulated down the chain.
"""
import numpy as np, math, fields, schema

# ---- landmarks of the source cut (the sculpt's own proportions) ----
def source_landmarks(seam_ys, L0):
    s = np.asarray(seam_ys)/L0
    return dict(L0=L0, s_h0=float(s[0]), s_t0=float(s[-1]), seam_s=s)

def s_map(P, lm):
    """piecewise-linear s -> s' sending the cut's head/thorax/tail landmarks to P's fractions,
    with the thorax mapped linearly so the segments stay ~uniform as it stretches."""
    a0,b0 = lm["s_h0"], lm["s_t0"]; a1,b1 = P["cephFrac"], 1-P["pygFrac"]
    return lambda s: np.interp(np.clip(s,0,1), [0,a0,b0,1], [0,a1,b1,1])

# ---- wrap one piece's vertices into the skeleton proportions of P ----
def wrap_vertices(v, lm, P, P0, use_width=True, use_relief=True):
    v = np.asarray(v,float).copy()
    s = v[:,1]/lm["L0"]
    sm = s_map(P,lm)(s)
    if use_width:
        WP, WP0 = fields.width_curve(P), fields.width_curve(P0)
        ratio = np.asarray(WP(sm))/np.maximum(np.asarray(WP0(sm)),1e-6)
        v[:,0] = v[:,0]*ratio
    if use_relief:
        v[:,2] = v[:,2]*(P["relief"]/max(P0["relief"],1e-6))
    v[:,1] = P["length"]*sm
    return v

# ---- skeleton pose (curl), accumulated down the chain, about the wrapped hinge axes ----
def pose_mats(hinge_ys, zh, angles_deg):
    M=np.eye(4); mats=[np.eye(4)]
    for y,th in zip(hinge_ys, angles_deg):
        R=trimesh_rotation(-math.radians(th),[1,0,0],[0,y,zh])
        M=M@R; mats.append(M.copy())
    return mats

def trimesh_rotation(angle, direction, point):
    import numpy as np
    d=np.asarray(direction,float); d=d/np.linalg.norm(d)
    a=angle; c=math.cos(a); s=math.sin(a); C=1-c
    x,y,z=d
    R=np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])
    M=np.eye(4); M[:3,:3]=R
    p=np.asarray(point,float); M[:3,3]=p-R@p
    return M

# ---- full rig: cut pieces + params -> posed, wrapped pieces + hinge info ----
def rig(pieces_v, pieces_f, seam_ys, L0, P, P0=None, enroll=0.0, **wrap_kw):
    """pieces_v/f: lists (head, seg0..segN-1, tail). seam_ys: N+1 midline furrow y's (joints)."""
    P=schema.coerce(P); P0=schema.coerce(P0) if P0 else schema.coerce(dict(length=L0))
    lm=source_landmarks(seam_ys, L0)
    wv=[wrap_vertices(v, lm, P, P0, **wrap_kw) for v in pieces_v]
    hinge_ys=[P["length"]*float(s_map(P,lm)(sy/L0)) for sy in seam_ys]
    zh=fields.hinge_z(P)
    angles=[enroll*P["maxAngle"]]*len(hinge_ys)
    mats=pose_mats(hinge_ys, zh, angles)
    posed=[]
    for v,M in zip(wv,mats):
        vh=np.c_[v,np.ones(len(v))]@M.T
        posed.append(vh[:,:3])
    return dict(posed=posed, faces=pieces_f, hinge_ys=hinge_ys, zh=zh, wrapped=wv, mats=mats)

def posed_skeleton(res):
    """Posed hinge points and the bone polyline through them (for drawing the skeleton)."""
    pts=[]
    for j,y in enumerate(res["hinge_ys"]):
        p=res["mats"][j]@np.array([0,y,res["zh"],1.0]); pts.append(p[:3])
    return np.array(pts)
