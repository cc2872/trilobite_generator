"""Trace the reference drawing's silhouette and the build's plan silhouette, normalize, overlay.
Usage: python trace_overlay.py out.png edits.json [stl]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "isopod_model"))
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

REF = "/mnt/user-data/uploads/4dab61c3-image.png"

def ref_mask():
    g = np.asarray(Image.open(REF).convert("L"), float)
    lines = g < 128                                        # the ink
    filled = ndimage.binary_fill_holes(lines)              # the closed outer line -> the body
    return filled

def build_mask(pieces, res=900):
    """Rasterize the union of the parts' triangles projected to the plan (one mask, gaps kept)."""
    W = max(p.vertices[:, 0].max() - p.vertices[:, 0].min() for p in pieces)
    allv = np.vstack([p.vertices for p in pieces])
    x0, y0 = allv[:, 0].min(), allv[:, 1].min(); x1, y1 = allv[:, 0].max(), allv[:, 1].max()
    sx = (res - 4) / max(x1 - x0, y1 - y0)
    H = int((y1 - y0) * sx) + 4
    mask = np.zeros((H, res), bool)
    from matplotlib.path import Path
    yy, xx = np.mgrid[0:H, 0:res]
    pts = np.c_[xx.ravel(), yy.ravel()]
    for p in pieces:
        V = p.vertices; F = p.faces
        fn = p.face_normals; F = F[fn[:, 2] > 0.05]                        # dorsal-facing only: the plan you see
        P2 = np.c_[(V[:, 0] - x0) * sx + 2, (V[:, 1] - y0) * sx + 2]
        # silhouette via winding: rasterize each triangle with a coarse scanline (vectorized bbox fill)
        for tri in P2[F]:
            xmin, ymin = np.floor(tri.min(0)).astype(int); xmax, ymax = np.ceil(tri.max(0)).astype(int)
            xmin, ymin = max(xmin, 0), max(ymin, 0); xmax, ymax = min(xmax, res - 1), min(ymax, H - 1)
            if xmax <= xmin or ymax <= ymin: continue
            gy, gx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
            d = np.c_[gx.ravel(), gy.ravel()].astype(float)
            a, b, c = tri
            den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(den) < 1e-9: continue
            l1 = ((b[1] - c[1]) * (d[:, 0] - c[0]) + (c[0] - b[0]) * (d[:, 1] - c[1])) / den
            l2 = ((c[1] - a[1]) * (d[:, 0] - c[0]) + (a[0] - c[0]) * (d[:, 1] - c[1])) / den
            inside = (l1 >= -0.01) & (l2 >= -0.01) & (l1 + l2 <= 1.01)
            mask[gy.ravel()[inside], gx.ravel()[inside]] = True
    return mask

def contours(mask):
    """0.5-level paths of a boolean mask, as (x, y) arrays (y up)."""
    fig = plt.figure(); ax = fig.add_subplot(111)
    cs = ax.contour(mask.astype(float), [0.5])
    paths = [s.copy() for s in cs.allsegs[0] if len(s) > 60]               # one array per loop (no cross-links)
    plt.close(fig)
    return paths

def norm(paths, flip_y=False):
    """Scale to unit width, centre x, put the top of the shape at y = 0 (y grows downward)."""
    allv = np.vstack(paths)
    if flip_y:
        ymax = allv[:, 1].max(); paths = [np.c_[p[:, 0], ymax - p[:, 1]] for p in paths]; allv = np.vstack(paths)
    w = allv[:, 0].max() - allv[:, 0].min()
    cx = 0.5 * (allv[:, 0].max() + allv[:, 0].min()); ytop = allv[:, 1].min()
    return [np.c_[(p[:, 0] - cx) / w, (p[:, 1] - ytop) / w] for p in paths]

if __name__ == "__main__":
    out, edits_f = sys.argv[1], sys.argv[2]
    from iterate_agnostid import build
    E = json.load(open(edits_f))
    P, pieces, PV, K, plans = build(E)
    bm = build_mask(pieces)
    rm = ref_mask()
    # reference: y grows downward in image coords already (head at top); build: head toward -y in part frames ->
    # after the renders' pi rotation head is +y; find head side by mass: tail blob is biggest? safer: flip so the
    # widest piece (head) ends up on top, checked from the first piece's centroid
    bp = contours(bm); rp = contours(rm)
    headc = pieces[0].vertices[:, 1].mean(); allc = np.vstack([p.vertices for p in pieces])[:, 1].mean()
    flip_build = headc < allc                                 # head toward -y -> flip so it draws on top
    bpn = norm(bp, flip_y=not flip_build)
    rpn = norm(rp, flip_y=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), facecolor="black")
    for ax, title in zip(axes, ["REFERENCE", "BUILD", "OVERLAY"]):
        ax.set_facecolor("black"); ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_title(title, color="white", family="monospace"); ax.axis("off")
    for p in rpn: axes[0].plot(p[:, 0], p[:, 1], color="#888888", lw=1.2)
    for p in bpn: axes[1].plot(p[:, 0], p[:, 1], color="white", lw=1.2)
    for p in rpn: axes[2].plot(p[:, 0], p[:, 1], color="#888888", lw=1.8)
    for p in bpn: axes[2].plot(p[:, 0], p[:, 1], color="white", lw=1.0)
    ra = np.vstack(rpn); ba = np.vstack(bpn)
    txt = f"ref L/W {ra[:,1].max():.3f}   build L/W {ba[:,1].max():.3f}"
    fig.text(0.5, 0.03, txt, color="#9a9a9a", family="monospace", ha="center")
    fig.savefig(out, dpi=130, facecolor="black", bbox_inches="tight")
    print(txt)
