"""Build the 'diagram' preset, export the STL, and lay its top-view silhouette over the textbook diagram at the same length."""
import os, sys, warnings; ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); os.chdir(ROOT); sys.path[:0]=[ROOT, ROOT+"/isopod_model", ROOT+"/scripts"]; warnings.filterwarnings("ignore")
import numpy as np, trimesh, schema, body as BODY, head_styles as HS, mesh as M, render as RD
from PIL import Image, ImageDraw
P=schema.preset("diagram"); P,face=HS.apply(P,"diagram"); P=schema.coerce(dict(P, **schema.PRESETS["diagram"]))
parts,plans,PV,K=BODY.build(P,"shaped",face=face,outline=HS.outline("diagram")); s=K["s"]
pieces=[p.copy().apply_scale(1/s) for p in parts]; whole=trimesh.util.concatenate(pieces)
os.makedirs("out/stl", exist_ok=True)
whole.export("out/stl/diagram_trilobite.stl")
for k,p in enumerate(pieces): p.export(f"out/stl/{k:02d}_{plans[k][0]}.stl")
print("print size mm", np.round(whole.extents,1).tolist(), "pieces", len(pieces), "watertight", sum(p.is_watertight for p in pieces), "audit", BODY.LAST_AUDIT)
# silhouette overlay: the diagram's animal mask vs our top view mask, both scaled to the same body length
im=np.array(Image.open("pictures/diagram.png").convert("RGB")).astype(int)
r,g,b=im[...,0],im[...,1],im[...,2]; from scipy.ndimage import binary_dilation, binary_fill_holes
col=((r>200)&(g>150)&(g<200)&(b>150)&(b<210))|((r>200)&(g>200)&(b<170))|((g>150)&(r<190)&(b<190)&(g>r+10))
anim=binary_fill_holes(binary_dilation(col, iterations=2))
ys,xs=np.where(anim); y0,y1=ys.min(),ys.max(); x0,x1=xs.min(),xs.max()
top=RD.render(whole,90,-90,900,900); mask=(np.array(top).mean(2)>30)[::-1]   # head up, like the diagram
ys2,xs2=np.where(mask); Y0,Y1=ys2.min(),ys2.max(); X0,X1=xs2.min(),xs2.max()
# the render has the head at the top? (elev 90, azim -90): check: more mass at the head end -> align by length and centre
H=y1-y0; sc=H/(Y1-Y0); ours=Image.fromarray((mask*255).astype(np.uint8)).crop((X0,Y0,X1+1,Y1+1)).resize((int((X1-X0+1)*sc), H+1))
ours=np.array(ours)>127
canvas=np.zeros((H+1, max(x1-x0+1, ours.shape[1])+2, 3), np.uint8)
d=anim[y0:y1+1, x0:x1+1]; cx=canvas.shape[1]//2
def paste(mk, col):
    w=mk.shape[1]; xo=cx-w//2
    for c in range(3):
        if col[c]: canvas[:mk.shape[0], xo:xo+w, c]=np.maximum(canvas[:mk.shape[0], xo:xo+w, c], (mk*col[c]).astype(np.uint8))
paste(d,(0,160,255)); paste(ours,(255,120,0))
inter=np.zeros_like(canvas[...,0],bool)
Image.fromarray(canvas).resize((canvas.shape[1]*3, canvas.shape[0]*3), Image.NEAREST).save("out/142_overlay.png")
# widths along the body (fractions of length) for both
rows=np.linspace(0,1,11)
print("y/L : diagram width/L, ours width/L")
for f in rows:
    yd=int(y0+f*(y1-y0)); wd=(np.where(anim[yd])[0].max()-np.where(anim[yd])[0].min())/(y1-y0) if anim[yd].any() else 0
    yo=int(f*(ours.shape[0]-1)); wo=(np.where(ours[yo])[0].max()-np.where(ours[yo])[0].min())/(ours.shape[0]-1) if ours[yo].any() else 0
    print(f"{f:.1f} : {wd:.2f}  {wo:.2f}")
sheet=Image.new("RGB",(900*2+10,900)); sheet.paste(Image.fromarray(top),(0,0)); sheet.paste(Image.fromarray(RD.render(whole,35,-55,900,900)),(910,0)); sheet.save("out/143_diagram_stl.png")
