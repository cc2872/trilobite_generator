import sys, trimesh, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
def render(specs, out, dpi=110):
    fig=plt.figure(figsize=(5.2*len(specs[0][1]),4.4*len(specs))); k=1
    ls=LightSource(azdeg=315, altdeg=45)
    for f,views in specs:
        m=trimesh.load(f)
        for title,(el,az) in views:
            ax=fig.add_subplot(len(specs),len(views),k,projection='3d'); k+=1
            ax.plot_trisurf(m.vertices[:,0],m.vertices[:,1],m.vertices[:,2],triangles=m.faces,color=(0.82,0.78,0.7),edgecolor='none',shade=True,lightsource=ls)
            ax.view_init(el,az); ax.set_title(f"{f}: {title}",fontsize=9); ax.set_box_aspect(tuple(m.extents)); ax.set_axis_off()
    plt.tight_layout(); plt.savefig(out,dpi=dpi)
if __name__=="__main__":
    f=sys.argv[1]; out=sys.argv[2]
    render([(f,[("top",(88,-90)),("iso",(32,-55)),("front",(8,-90)),("rear",(12,90))])],out)
