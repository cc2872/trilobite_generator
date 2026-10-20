"""
scripts/trilomorph_pca.py — the generator's presets in TriloMorph's cephalic morphospace (20 Oct 2026).

    python scripts/trilomorph_pca.py <trilomorph checkout> <generated_landmarks.json> <out dir>

<generated_landmarks.json> is {name: {"lm": {"LM1": [x, y], ...}}} as the 20 Oct review bundle wrote it (landmarks.cephalon(P)["lm"]
per preset). GPA (2-D, rotation, no reflection) on the 16 fixed landmarks of every sighted, complete TriloMorph cephalon plus the
generated ones; PCA centred on the real cloud; nearest real genus per preset; two figures. pip install pyyaml numpy matplotlib.
"""
import sys, os
TM, GEN, OUT = (sys.argv + ['.', 'generated_landmarks.json', '.'])[1:4]
import json, glob, re, numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def read_txt(p):
    t=open(p,encoding='utf-8',errors='ignore').read(); m=re.search(r'<landmarks.scaled[^>]*>(.*?)</landmarks.scaled>',t,re.S)
    if not m: return None
    L={}
    for line in m.group(1).strip().splitlines():
        a=line.split()
        if len(a)==3: L[a[0]]=(float(a[1]),float(a[2]))
    return L
def read_tps(p):
    t=open(p,encoding='utf-8',errors='ignore').read().splitlines(); n=int(re.match(r'LM=(\d+)',t[0]).group(1))
    return {f'LM{i+1}':tuple(map(float,l.split())) for i,l in enumerate(t[1:1+n])}
docs=yaml.safe_load(open(os.path.join(TM,'trilomorph.yaml'),encoding='utf-8')); meta={d['ID']:d for d in docs}
real={}
for p in glob.glob(os.path.join(TM,'Cephala','landmarks','*')):
    L=read_txt(p) if p.endswith('.txt') else read_tps(p) if p.endswith('.tps') else None
    if not L or len(L)<16: continue
    X=np.array([L[f'LM{i}'] for i in range(1,17)],float)
    if not np.isfinite(X).all() or np.hypot(*(X[7]-X[8]))<1e-6: continue
    Xc=X-X.mean(0)
    if np.linalg.norm(Xc)<1e-9: continue
    if X[0,1] < X[3,1]: X[:,1]*=-1
    if X[14,0] < X[0,0]: X[:,0]*=-1
    real[os.path.basename(p).rsplit('_C.',1)[0]]=X
gen=json.load(open(GEN))
G={n:np.array([d['lm'][f'LM{i}'] for i in range(1,17)]) for n,d in gen.items() if np.hypot(*(np.array(d['lm']['LM8'])-np.array(d['lm']['LM9'])))>1e-6}
names=list(real)+list(G); X=np.array([real[n] for n in real]+[G[n] for n in G],float)
def cs(A): A=A-A.mean(0); return A/np.linalg.norm(A)
def align(A,B):
    U,s,Vt=np.linalg.svd(A.T@B); R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    return A@R
S=np.array([cs(a) for a in X]); ref=S[0]
for it in range(50):
    S=np.array([align(a,ref) for a in S]); new=cs(S.mean(0))
    if np.linalg.norm(new-ref)<1e-10: break
    ref=new
mean=ref; nr=len(real)
Z=S.reshape(len(S),-1); Zc=Z-Z[:nr].mean(0)          # PCA centred on the REAL cloud
U,s,Vt=np.linalg.svd(Zc[:nr],full_matrices=False); pcs=Zc@Vt.T; var=s**2/np.sum(s**2)
print("real cephala used:", nr, "| generated:", list(G)); print("PC variance (real):", np.round(var[:5],3))
D=lambda a,b: float(np.linalg.norm(a-b))
genus=lambda i: str(meta.get(names[i],{}).get('taxonomy',{}).get('genus'))
nnr=np.array([min(D(S[i],S[j]) for j in range(nr) if j!=i) for i in range(nr)])
print(f"real nearest-neighbour Procrustes distance: median {np.median(nnr):.3f}, 95th pct {np.percentile(nnr,95):.3f}")
rows=[]
for k,n in enumerate(G):
    i=nr+k; d=sorted((D(S[i],S[j]),j) for j in range(nr))
    rows.append(dict(preset=n, nearest=[(genus(j),round(dd,3)) for dd,j in d[:3]], d_nearest=round(d[0][0],3), inside_95=bool(d[0][0]<=np.percentile(nnr,95))))
    print(f"{n:15s} d_nearest={d[0][0]:.3f} {'inside' if rows[-1]['inside_95'] else 'OUTSIDE'} the real cloud | nearest: " + ", ".join(f"{g} ({dd})" for g,dd in rows[-1]['nearest']))
fig,ax=plt.subplots(1,2,figsize=(15,7))
for a,(px,py) in zip(ax,[(0,1),(0,2)]):
    a.scatter(pcs[:nr,px],pcs[:nr,py],s=14,color='0.72',label=f'TriloMorph cephala (n={nr}, sighted, 16 fixed landmarks)')
    a.scatter(pcs[nr:,px],pcs[nr:,py],s=70,color='crimson',edgecolor='k',zorder=5,label='generator presets (classic head, schema 6.2)')
    for k,n in enumerate(G): a.annotate(n,(pcs[nr+k,px],pcs[nr+k,py]),fontsize=8,xytext=(5,4),textcoords='offset points')
    for j in [int(np.argmin(pcs[:nr,px])),int(np.argmax(pcs[:nr,px])),int(np.argmin(pcs[:nr,py])),int(np.argmax(pcs[:nr,py]))]:
        a.annotate(genus(j),(pcs[j,px],pcs[j,py]),fontsize=7,color='0.35',xytext=(3,-9),textcoords='offset points')
    a.set_xlabel(f'PC{px+1} ({var[px]*100:.0f}% of real variance)'); a.set_ylabel(f'PC{py+1} ({var[py]*100:.0f}%)'); a.axhline(0,color='0.92'); a.axvline(0,color='0.92')
ax[0].set_title('Cephalic Procrustes morphospace: TriloMorph specimens and the generator presets'); ax[0].legend(fontsize=8,loc='best')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'morphospace_trilomorph_vs_generator.png'),dpi=120)
fig,a=plt.subplots(figsize=(6.5,6.5))
gm=align(cs(S[nr:].mean(0)),mean)
for i in range(16): a.annotate('',xy=gm[i],xytext=mean[i],arrowprops=dict(arrowstyle='->',color='crimson',lw=1.2))
a.scatter(mean[:,0],mean[:,1],color='0.3',s=36,zorder=3,label='TriloMorph mean shape'); a.scatter(gm[:,0],gm[:,1],color='crimson',s=20,zorder=3,label='mean of the generator presets')
for i in range(16): a.annotate(str(i+1),mean[i],fontsize=8,xytext=(4,3),textcoords='offset points')
a.set_aspect('equal'); a.legend(fontsize=8,loc='lower left'); a.set_title('Mean cephalic configuration (anterior up, right half)\narrows: where the generator sits relative to the real mean')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'mean_shapes.png'),dpi=120)
disp=gm-mean; order=np.argsort(-np.hypot(*disp.T))
print("generator mean vs real mean, largest landmark offsets (Procrustes units; +x lateral, +y anterior):")
for i in order[:8]: print(f"  LM{i+1}: dx {disp[i,0]:+.3f}  dy {disp[i,1]:+.3f}")
json.dump(dict(pc_var=var[:6].tolist(), rows=rows, n_real=nr, nn_median=float(np.median(nnr)), nn95=float(np.percentile(nnr,95)),
               mean_offsets={f"LM{i+1}":[round(float(disp[i,0]),4),round(float(disp[i,1]),4)] for i in range(16)}), open(os.path.join(OUT,'pca_summary.json'),'w'), indent=1)
