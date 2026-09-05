"""Load the sculpt's real legs and put them through the SAME orientation as the shell so they align."""
import numpy as np, trimesh, atlas

def _top_var(vv):
    ys=np.linspace(vv[:,1].min(),vv[:,1].max(),40);prof=[]
    for a,b in zip(ys[:-1],ys[1:]):
        sel=(vv[:,1]>=a)&(vv[:,1]<b)&(np.abs(vv[:,0])<0.05*np.ptp(vv[:,0]))
        prof.append(vv[sel,2].max() if sel.any() else np.nan)
    p=np.array(prof);p=p[np.isfinite(p)];return np.var(np.diff(p)) if len(p)>3 else 0

def orient_pair(shell, legs, target=130.0):
    """Return (shell_oriented, legs_oriented) in the X-across/Y-along(head -y)/Z-up 130mm frame,
    all decisions taken from the shell, then dorsal-up flip. Matches the assembled shell frame."""
    # 1. PCA align (params from shell)
    c=shell.mean(0); cs=shell-c; w,U=np.linalg.eigh(np.cov(cs.T)); R=U[:,::-1]
    ap=lambda v:((v-c)@R)[:,[1,0,2]]
    S=ap(shell); Lg=ap(legs)
    # 2. orient flips (from shell)
    Sf=S.copy(); Sf[:,2]*=-1; zf=-1 if _top_var(Sf)>_top_var(S) else 1
    S[:,2]*=zf; Lg[:,2]*=zf
    y0,y1=S[:,1].min(),S[:,1].max(); L=y1-y0
    wf=np.ptp(S[S[:,1]<y0+0.3*L,0]); wb=np.ptp(S[S[:,1]>y1-0.3*L,0])
    yf=-1 if (wb>wf) else 1
    S[:,1]*=yf; Lg[:,1]*=yf
    off=np.array([0,S[:,1].min(),S[:,2].min()]); S-=off; Lg-=off
    # 3. scale + centre x + drop to origin (from shell)
    s=target/np.ptp(S[:,1]); S*=s; Lg*=s
    xm=np.median(S[:,0]); S[:,0]-=xm; Lg[:,0]-=xm
    off2=np.array([0,S[:,1].min(),S[:,2].min()]); S-=off2; Lg-=off2
    # 4. head-low (orient_head_low), decision from shell
    ys=np.linspace(S[:,1].min(),S[:,1].max(),60);cr=[]
    for a,b in zip(ys[:-1],ys[1:]):
        sel=(S[:,1]>=a)&(S[:,1]<b)&(np.abs(S[:,0])<0.08*np.ptp(S[:,0]))
        cr.append(S[sel,2].max() if sel.any() else np.nan)
    cc=np.nan_to_num(np.array(cr),nan=np.nanmin(cr));cs2=np.convolve(np.pad(cc,4,mode="edge"),np.ones(9)/9,mode="valid")
    if np.argmax(cs2)>len(cs2)//2:
        ymax=S[:,1].max(); S[:,1]=ymax-S[:,1]; Lg[:,1]=ymax-Lg[:,1]
    # 5. dorsal-up flip (z), from shell
    zmax=S[:,2].max(); S[:,2]=zmax-S[:,2]; Lg[:,2]=zmax-Lg[:,2]
    return S, Lg

if __name__=="__main__":
    B="/home/claude/sculpts/llllline_202609284405_82eb7f81390a49cab3ce5a1cc39bf258/Trilobite"
    sh=trimesh.load(f"{B}/Olenoides_serratus_shell_pose_02.stl",process=False)
    lg=trimesh.load(f"{B}/Olenoides_serratus_legs_pose_02_lvl3.stl",process=False)
    print("raw shell",len(sh.faces),"legs",len(lg.faces))
    sh=sh.simplify_quadric_decimation(face_count=150000)
    lg=lg.simplify_quadric_decimation(face_count=120000)
    S,Lg=orient_pair(sh.vertices, lg.vertices)
    print("shell z[%.1f,%.1f] legs z[%.1f,%.1f]  shell y[%.1f,%.1f] legs y[%.1f,%.1f]"%(
        S[:,2].min(),S[:,2].max(),Lg[:,2].min(),Lg[:,2].max(),S[:,1].min(),S[:,1].max(),Lg[:,1].min(),Lg[:,1].max()))
    np.savez("olen_legs.npz", v=Lg, f=lg.faces)
    trimesh.Trimesh(Lg,lg.faces,process=False).export("olenoides_real_legs.stl")
    print("saved olen_legs.npz + olenoides_real_legs.stl")
