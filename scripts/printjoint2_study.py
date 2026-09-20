import sys, os, json, time, numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schema, parts, mesh as M, printjoint2 as J2
P = schema.coerce(dict(maxAngle=36))
SETS = {"tight": dict(gap_axial=0.15, gap_vertical=0.25, gap_lateral=0.20), "default": {}, "loose": dict(gap_axial=0.30, gap_vertical=0.40, gap_lateral=0.35)}
out = {}
for name in (sys.argv[1:] or SETS):
    Jg = SETS[name]; t = time.time(); J, d, zj, y_piv = J2.geometry(P, Jg); a, b = J2.chain(P, Jg, 0.0, n=2)
    ma = M.to_manifold(a)
    def pen(T):                                   # exact: Manifold overlap volume (mm3) minus a 0.005 tolerance
        return (ma ^ M.to_manifold(b.copy().apply_transform(T))).volume() - 0.005
    pa = a.sample(3000); rest = -trimesh.proximity.ProximityQuery(b).signed_distance(pa).max()
    rec = dict(gaps={k: J[k] for k in ("gap_axial", "gap_vertical", "gap_lateral")}, rest_gap=round(float(rest), 3)); piv = (0, y_piv, zj)
    for lab, axis, sgn in (("flex_ventral", (1,0,0), -1), ("flex_dorsal", (1,0,0), 1), ("lateral", (0,0,1), 1), ("twist", (0,1,0), 1)):
        ang = 0.0
        for a_ in np.arange(2, 50, 2.0):
            if pen(trimesh.transformations.rotation_matrix(np.radians(sgn*a_), axis, piv)) > 0.05: break
            ang = a_
        rec[lab + "_deg"] = ang
    for lab, v in (("pull", (0,1,0)), ("push", (0,-1,0)), ("shear", (1,0,0)), ("lift", (0,0,1))):
        play = 0.0
        for s_ in np.arange(0.05, 1.2, 0.05):
            if pen(trimesh.transformations.translation_matrix(np.array(v, float)*s_)) > 0.05: break
            play = s_
        rec[lab + "_play_mm"] = round(play, 2)
    rec["seconds"] = round(time.time() - t); out[name] = rec; print(name, json.dumps(rec), flush=True)
    os.makedirs("/mnt/user-data/outputs/coupons2", exist_ok=True)
    trimesh.util.concatenate([a, b]).export(f"/mnt/user-data/outputs/coupons2/flexi2_coupon_{name}.stl")
path = "docs/printjoint2_study.json"; prev = json.load(open(path)) if os.path.exists(path) else {}; prev.update(out); json.dump(prev, open(path, "w"), indent=1)
