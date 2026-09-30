"""
isopod_model/render.py: the model's photos. A small z-buffer renderer (orthographic, smooth grey or tinted shading)
and the sheets build.py makes.

    python isopod_model/render.py            -> photos/*.png from stl/pieces/*.stl and the isopod asset

numba makes it fast (seconds); without it the same code runs in plain Python (minutes).
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
try:
    from numba import njit
except ImportError:                                                         # plain Python fallback
    def njit(*a, **k): return (lambda f: f) if not (a and callable(a[0])) else a[0]

WARM = np.array([0.95, 0.72, 0.60]); COOL = np.array([0.62, 0.78, 0.98])


@njit(cache=True)
def _raster(P, F, N, W, H, img, zb):
    for t in range(F.shape[0]):
        a, b, c = F[t, 0], F[t, 1], F[t, 2]
        x0, y0, z0 = P[a, 0], P[a, 1], P[a, 2]; x1, y1, z1 = P[b, 0], P[b, 1], P[b, 2]; x2, y2, z2 = P[c, 0], P[c, 1], P[c, 2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12: continue
        xmin = max(int(min(x0, x1, x2)), 0); xmax = min(int(max(x0, x1, x2)) + 1, W - 1)
        ymin = max(int(min(y0, y1, y2)), 0); ymax = min(int(max(y0, y1, y2)) + 1, H - 1)
        for py in range(ymin, ymax + 1):
            for px in range(xmin, xmax + 1):
                fx = px + 0.5; fy = py + 0.5
                w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) / den
                w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) / den
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0: continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z > zb[py, px]:
                    zb[py, px] = z
                    for k in range(3):
                        img[py, px, k] = w0 * N[a, k] + w1 * N[b, k] + w2 * N[c, k]


def render(m, elev, azim, W=900, H=600, pad=0.04, vcol=None, bounds=None):
    """Orthographic view from (elev, azim) degrees; vcol: per-vertex tint; bounds: frame these bounds (to show two
    models at one scale). Returns an RGB uint8 image."""
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])      # toward the viewer
    up0 = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    r = np.cross(up0, d); r /= np.linalg.norm(r); u = np.cross(d, r)
    bb = m.bounds if bounds is None else np.asarray(bounds, float)
    V = m.vertices - bb.mean(0)
    X, Y, Z = V @ r, V @ u, V @ d
    C8 = np.array([[bb[i][0], bb[j][1], bb[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) - bb.mean(0)
    s = (1 - 2 * pad) * min(W / np.ptp(C8 @ r if bounds is not None else X), H / np.ptp(C8 @ u if bounds is not None else Y))
    P = np.c_[W / 2 + s * X, H / 2 - s * Y, Z].astype(np.float64)
    vn = m.vertex_normals
    L1 = 0.55 * d + 0.6 * u - 0.45 * r; L1 /= np.linalg.norm(L1)
    L2 = 0.8 * d - 0.2 * u + 0.5 * r; L2 /= np.linalg.norm(L2)
    sh = 0.10 + 0.72 * np.clip(vn @ L1, 0, 1) + 0.22 * np.clip(vn @ L2, 0, 1)
    col = np.c_[sh * 0.93, sh * 0.90, sh * 0.84] if vcol is None else (0.25 + 0.75 * sh)[:, None] * vcol
    img = np.zeros((H, W, 3)); img[:] = (0.07, 0.07, 0.08); zb = np.full((H, W), -1e18)
    _raster(P, m.faces.astype(np.int64), col.astype(np.float64), W, H, img, zb)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def sheet(rows, path, W=900, H=600):
    """rows: lists of (mesh, elev, azim, title[, render kwargs])."""
    nr, nc = len(rows), max(len(r) for r in rows)
    out = Image.new("RGB", (nc * W, nr * H), (18, 18, 20)); dr = ImageDraw.Draw(out)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception: font = ImageFont.load_default()
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            m, el, az, title = cell[:4]; kw = cell[4] if len(cell) > 4 else {}
            out.paste(Image.fromarray(render(m, el, az, W, H, **kw)), (j * W, i * H))
            dr.text((j * W + 18, i * H + 14), title, fill=(235, 235, 235), font=font)
    os.makedirs(os.path.dirname(path), exist_ok=True); out.save(path); return path


def _pair(A, B, dy=0.0):
    A = A.copy(); B = B.copy(); B.apply_translation((0, dy, 0))
    m = trimesh.util.concatenate([A, B]); col = np.r_[np.tile(WARM, (len(A.vertices), 1)), np.tile(COOL, (len(B.vertices), 1))]
    return m, col


def _crop(m, c, lo, hi):
    import mesh as M
    q = m.copy(); q.apply_translation(-np.asarray(c))
    box = M.to_manifold(M.box(*(np.subtract(hi, lo)), at=tuple(lo), align=("min", "min", "min")))
    return M.from_manifold(M.to_manifold(q) ^ box)


def photos(pieces, pivots, out_dir, curl_deg=(0.0, 15.0, 22.0)):
    """pieces: the model's pieces in print mm (head first); pivots: [(y, z)] per joint in print mm."""
    import check as C
    whole = trimesh.util.concatenate(pieces)
    sheet([[(whole, 30, -55, "3/4"), (whole, 90, -90, "top")],
           [(whole, 5, 0, "side"), (whole, -40, -55, "underside")]], os.path.join(out_dir, "overview.png"), W=1000, H=620)
    def posed(a):
        T = C.pose(pivots, [a] * len(pivots))
        return trimesh.util.concatenate([p.copy().apply_transform(T[k]) for k, p in enumerate(pieces)])
    sheet([[(posed(a), 4, 0, f"curled {a:g} deg per joint") for a in curl_deg]], os.path.join(out_dir, "curl.png"), W=900, H=600)
    # the joint beside the isopod's, from the asset itself
    iso, bed, xmid = C.isopod_pieces()
    ci = C.fit_ball(iso[4], (xmid, 99.5, bed + 7.5), xmid)[0]
    k = 3; co = C.fit_ball(pieces[k + 1], (0, pivots[k][0], pivots[k][1]), 0.0)[0]
    src = {"ISOPOD (your file)": (iso[3], iso[4], ci), "OURS": (pieces[k], pieces[k + 1], co)}
    lo, hi = (-22, -15, -8), (22, 15, 40); lo2, hi2 = (-15, -8.5, -7.6), (15, 9.0, 12.5); lo3, hi3 = (0.25, -12, -7.6), (22, 14, 22)
    rows = [[], [], []]
    for tag, (a, b, c) in src.items():
        m, col = _pair(_crop(a, c, lo, hi), _crop(b, c, lo, hi)); rows[0].append((m, -28, -55, f"{tag}: joint, from below", dict(vcol=col, bounds=np.array([lo, hi], float))))
        m, col = _pair(_crop(a, c, lo2, hi2), _crop(b, c, lo2, hi2), dy=16.0)
        rows[1].append((m, -35, -40, f"{tag}: copied strip, pulled apart", dict(vcol=col, bounds=np.array([[-15, -8.5, -7.6], [15, 25.0, 12.5]]))))
        m, col = _pair(_crop(a, c, lo3, hi3), _crop(b, c, lo3, hi3)); rows[2].append((m, 0, 180, f"{tag}: cut down the middle", dict(vcol=col, bounds=np.array([lo3, hi3], float))))
    sheet(rows, os.path.join(out_dir, "joint_vs_isopod.png"), W=1000, H=640)
    lo4, hi4 = (-45, -22, -8), (45, 22, 40); rows = []
    for el, az, t in ((10, 0, "side"), (-45, -60, "below")):
        row = []
        for tag, (a, b, c) in {"ISOPOD pieces 3+4": src["ISOPOD (your file)"], f"OURS pieces {k}+{k + 1}": src["OURS"]}.items():
            m, col = _pair(_crop(a, c, lo4, hi4), _crop(b, c, lo4, hi4)); row.append((m, el, az, f"{tag}, {t}", dict(vcol=col, bounds=np.array([lo4, hi4], float))))
        rows.append(row)
    sheet(rows, os.path.join(out_dir, "joint_in_context.png"), W=1000, H=640)


if __name__ == "__main__":
    import json, glob
    pieces = [trimesh.load(p) for p in sorted(glob.glob(os.path.join(HERE, "stl", "pieces", "*.stl")))]
    pv = json.load(open(os.path.join(HERE, "checks.json")))["pivots_mm"]
    photos(pieces, [tuple(p) for p in pv], os.path.join(HERE, "photos")); print("photos ->", os.path.join(HERE, "photos"))
