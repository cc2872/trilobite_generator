"""
wrap.py — make the sculpted family parametric.

Body-coordinate remap: every vertex is (s along the body, x across, z up). The sculpt's own outline
is its identity; the knobs move it by ratio, so the reference preset reproduces the sculpt exactly:
    y' = L · s'(s)                  s' piecewise-linear through the head/thorax/tail landmarks
    x' = x · W_P(s') / W_P0(s')     width curve from fields.py, ratio to the sculpt's own fitted curve
    z' = z · relief / relief0
plus regional scaling of the pleural and genal spines (vertices beyond the pleural margin).
Segment count is handled after slicing by resampling the segment sequence.
"""
import numpy as np, trimesh
import fields

def profiles(body, nbins=130):
    """The sculpt's own half-width and crest height along s in [0,1]."""
    v = body.vertices; y0, y1 = v[:, 1].min(), v[:, 1].max(); L = y1 - y0
    edges = np.linspace(y0, y1, nbins + 1); W = np.zeros(nbins); H = np.zeros(nbins)
    idx = np.clip(((v[:, 1] - y0) / L * nbins).astype(int), 0, nbins - 1)
    np.maximum.at(W, idx, np.abs(v[:, 0])); np.maximum.at(H, idx, v[:, 2])
    k = np.array([1, 2, 3, 2, 1]) / 9.0
    Ws = np.convolve(np.pad(W, 2, mode="edge"), k, mode="valid"); Hs = np.convolve(np.pad(H, 2, mode="edge"), k, mode="valid")
    s = (edges[:-1] + edges[1:]) / 2 / L - y0 / L
    return dict(s=s, W=Ws, H=Hs, L=L, y0=y0, W_of=lambda q: np.interp(q, s, Ws), H_of=lambda q: np.interp(q, s, Hs))

def P0_from(body, s_head, s_tail, seg_count):
    """The parameter vector that reproduces the sculpt unchanged (its own landmarks and outline)."""
    pr = profiles(body); W = pr["W"].max(); smax = float(pr["s"][pr["W"].argmax()])
    pitch_s = (s_tail - s_head) / max(seg_count, 1); s0 = s_head + 0.5 * pitch_s
    return dict(length=pr["L"], width=2 * W, relief=float(body.vertices[:, 2].max()),
                cephFrac=float(s_head), pygFrac=float(1 - s_tail), segCount=int(seg_count),
                widthMaxPos=smax, widthHeadFront=float(pr["W_of"](min(0.10, smax - 0.02)) / W),
                widthThoraxFront=float(pr["W_of"](s0) / W), widthThoraxRear=float(pr["W_of"](s_tail) / W),
                widthTail=float(pr["W_of"](0.98) / W), spineBase=0.3, genalSpine=0.35, pygSpine=0.9)

def s_map(P, P0):
    """Piecewise-linear s -> s' sending the sculpt's landmarks to the target fractions."""
    a0, b0 = P0["cephFrac"], 1 - P0["pygFrac"]; a1, b1 = P["cephFrac"], 1 - P["pygFrac"]
    return lambda s: np.interp(s, [0, a0, b0, 1], [0, a1, b1, 1])

def wrap_mesh(mesh, P, P0, pr):
    """Return a copy of `mesh` remapped from the sculpt's proportions (P0) to P."""
    v = mesh.vertices.copy(); s = (v[:, 1] - pr["y0"]) / pr["L"]
    sm = s_map(P, P0)(np.clip(s, 0, 1))
    WP, WP0 = fields.width_curve(P), fields.width_curve(P0)
    ratio = np.asarray(WP(sm)) / np.maximum(np.asarray(WP0(sm)), 1e-6)
    ax = np.abs(v[:, 0]); margin = 0.72 * pr["W_of"](np.clip(s, 0, 1))
    # spine stretch beyond the pleural margin (thorax + tail) and the genal margin (head)
    f_pl = P["spineBase"] / max(P0["spineBase"], 1e-6); f_ge = P["genalSpine"] / max(P0["genalSpine"], 1e-6)
    in_head = s < P0["cephFrac"]
    f = np.where(in_head, f_ge, f_pl)
    ax2 = np.where(ax > margin, margin + (ax - margin) * f, ax)
    v[:, 0] = np.sign(v[:, 0]) * ax2 * ratio
    v[:, 1] = P["length"] * sm
    v[:, 2] = v[:, 2] * (P["relief"] / max(P0["relief"], 1e-6))
    out = mesh.copy(); out.vertices = v; return out

def resample_segments(pieces, funcs, N):
    """head + m segments + tail  ->  head + N segments + tail, by dropping/duplicating segments and
    re-spacing them over the same thorax length; overlaps between neighbours are trimmed."""
    m = len(pieces) - 2
    if N == m or m < 2: return pieces, funcs
    src = np.round(np.linspace(0, m - 1, N)).astype(int)
    fronts = [float(f(0.0)) for f in funcs]                       # boundary y at the midline: head|s0, s0|s1 ... s_{m-1}|tail
    y_h, y_t = fronts[0], fronts[-1]; pitch = (y_t - y_h) / N
    widths = np.array([pieces[1 + i].extents[0] for i in range(m)]); w_new = np.interp(np.linspace(0, m - 1, N), np.arange(m), widths)
    new_pieces, new_funcs = [pieces[0]], []
    for k, i in enumerate(src):
        seg = pieces[1 + i].copy(); dy = (y_h + k * pitch) - fronts[i]; sx = w_new[k] / max(widths[i], 1e-6)
        T = np.eye(4); T[0, 0] = sx; T[1, 3] = dy; seg.apply_transform(T)
        f_front = funcs[i]; f_rear = funcs[i + 1]
        new_funcs.append(lambda x, f=f_front, dy=dy: f(x) + dy)
        new_pieces.append(seg); last_rear = (f_rear, dy)
    tail = pieces[-1].copy(); dy_t = (y_h + N * pitch) - fronts[-1]; tail.apply_translation((0, dy_t, 0))
    new_funcs.append(lambda x, f=last_rear[0], dy=last_rear[1]: f(x) + dy)
    new_pieces.append(tail)
    for i in range(len(new_pieces) - 1):                            # trim any overlap: the front part wins
        d = trimesh.boolean.difference([new_pieces[i + 1], new_pieces[i]], engine="manifold")
        if d is not None and len(d.faces) and d.is_volume: new_pieces[i + 1] = d
    return new_pieces, new_funcs
