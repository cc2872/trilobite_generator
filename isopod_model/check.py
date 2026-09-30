"""
isopod_model/check.py: the model's checks, all in print mm (the STL's own units).

    python isopod_model/check.py             -> checks.json, from stl/pieces/*.stl and the isopod asset

  * pieces: each watertight and one body;
  * rest: no piece touches its neighbours (the head: any piece);
  * curl: every joint bent by the same angle, the first angle at which anything touches (neighbours, and the head
    against the whole chain, where the tail comes round under it);
  * joints vs the isopod: at every joint, a sphere fitted to our ball and to the isopod's (piece 4, the source joint
    3|4), centres lined up, then the least distance from points on each surface inside the copied strip to the other
    file's surface, both ways, for the socket side and the ball side. Independent of the code that built the joints.
"""
import os, sys, math, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import numpy as np, trimesh
import mesh as M

ASSET = os.path.join(HERE, "..", "assets", "isopod", "giant_isopod_main_itbefred.stl")
STRIP = (np.array([-15.0, -8.5, None]), np.array([15.0, 9.0, None]))   # the copied strip around a ball (y), z: bed..20


def isopod_pieces():
    """The asset's 8 pieces in the file's own frame, with its bed height and midline."""
    raw = trimesh.load(ASSET)
    return sorted(raw.split(only_watertight=False), key=lambda p: p.bounds[0][1]), raw.bounds[0][2], 0.5 * (raw.bounds[0][0] + raw.bounds[1][0])


def fit_ball(p, c0, xm):
    """Least-squares sphere through a ball's own surface: vertices 4.5-6.5 mm from a rough centre, off the slit."""
    c = np.asarray(c0, float)
    for _ in range(4):
        v = p.vertices; dd = np.linalg.norm(v - c, axis=1)
        v = v[(dd > 4.5) & (dd < 6.5) & (np.abs(v[:, 0] - xm) > 0.8) & (v[:, 2] > c[2] - 4.0)]
        A = np.c_[2 * v, np.ones(len(v))]; sol = np.linalg.lstsq(A, (v ** 2).sum(1), rcond=None)[0]
        c = sol[:3]; r = math.sqrt(sol[3] + c @ c)
    return c, r


def pose(pivots, angles):
    """Rest -> posed transform of every piece, each joint curled by its angle about its pivot (the piece behind down)."""
    T = [np.eye(4)]
    for (yc, zc), a in zip(pivots, angles):
        T.append(T[-1] @ trimesh.transformations.rotation_matrix(math.radians(-a), (1, 0, 0), (0, yc, zc)))
    return T


def _pairs(n): return [(k, j) for k in range(n) for j in range(k + 1, n if k == 0 else min(n, k + 3))]


def contacts(pieces, pivots, a, tol=0.01):
    T = pose(pivots, [a] * len(pivots)); W = [M.to_manifold(p) for p in pieces]; out = []
    for k, j in _pairs(len(pieces)):
        v = (W[k] ^ M.to_manifold(pieces[j].copy().apply_transform(np.linalg.inv(T[k]) @ T[j]))).volume()
        if v > tol: out.append((k, j, round(float(v), 3)))
    return out


def joint_deviation(pieces, pivots, bed=None):
    iso, bed_i, xm = isopod_pieces()
    ci, ri = fit_ball(iso[4], (xm, 99.5, bed_i + 7.5), xm)
    rows = []
    for k, (yc, zc) in enumerate(pivots):
        co, ro = fit_ball(pieces[k + 1], (0, yc, zc), 0.0); t = ci - co
        lo = np.array([-15.0, -8.5, bed_i - ci[2] + 0.3]); hi = np.array([15.0, 9.0, bed_i + 20.0 - ci[2] - 0.3])
        dev = []
        for ours, theirs in ((pieces[k], iso[3]), (pieces[k + 1], iso[4])):
            O = ours.copy(); O.apply_translation(t); d = 0.0
            for A, B in ((O, theirs), (theirs, O)):
                p, _ = trimesh.sample.sample_surface_even(A, 60000, seed=3); q = p - ci
                p = p[np.all((q > lo + 0.3) & (q < hi - 0.3), axis=1)]
                d = max(d, float(np.abs(trimesh.proximity.ProximityQuery(B).signed_distance(p)).max()))
            dev.append(round(d, 4))
        rows.append(dict(joint=k, ball_radius=round(ro, 4), ball_height=round(float(co[2]), 4), socket_side_max_dev_mm=dev[0], ball_side_max_dev_mm=dev[1]))
    return dict(isopod_ball_radius=round(ri, 4), isopod_ball_height=round(float(ci[2] - bed_i), 4), joints=rows)


def run(pieces, pivots, names=None):
    names = names or [f"piece{k}" for k in range(len(pieces))]
    res = dict(asset_md5=hashlib.md5(open(ASSET, "rb").read()).hexdigest(), pivots_mm=[list(map(float, p)) for p in pivots])
    res["pieces"] = [dict(name=n, watertight=bool(p.is_watertight), bodies=len(p.split(only_watertight=False)),
                          bounds=np.round(p.bounds, 2).tolist()) for n, p in zip(names, pieces)]
    res["rest_contacts"] = contacts(pieces, pivots, 0.0)
    clean = 0.0
    for a in np.arange(1.0, 36.0, 1.0):
        c = contacts(pieces, pivots, a)
        if c: res["first_contact"] = dict(angle_deg=float(a), pairs=[(names[k], names[j], v) for k, j, v in c]); break
        clean = float(a)
    res["curl_clean_to_deg"] = clean
    res["joints_vs_isopod"] = joint_deviation(pieces, pivots)
    json.dump(res, open(os.path.join(HERE, "checks.json"), "w"), indent=1)
    return res


def summary(res):
    j = res["joints_vs_isopod"]; w = max(max(r["socket_side_max_dev_mm"], r["ball_side_max_dev_mm"]) for r in j["joints"])
    lines = [f"pieces: {len(res['pieces'])}, all watertight single bodies: {all(p['watertight'] and p['bodies'] == 1 for p in res['pieces'])}",
             f"touching at rest: {res['rest_contacts'] or 'nothing'}",
             f"curl: clean to {res['curl_clean_to_deg']:g} deg per joint" + (f"; at {res['first_contact']['angle_deg']:g}: {res['first_contact']['pairs']}" if "first_contact" in res else ""),
             f"joints vs the isopod: {len(j['joints'])} joints, worst surface deviation {w:.4f} mm (ball r {j['joints'][0]['ball_radius']} vs {j['isopod_ball_radius']})"]
    return "\n".join(lines)


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob(os.path.join(HERE, "stl", "pieces", "*.stl")))
    pieces = [trimesh.load(p) for p in files]
    pv = [tuple(p) for p in json.load(open(os.path.join(HERE, "checks.json")))["pivots_mm"]]
    print(summary(run(pieces, pv, [os.path.basename(f)[3:-4] for f in files])))
