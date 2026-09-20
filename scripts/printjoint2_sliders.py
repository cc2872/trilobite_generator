"""One-slider-at-a-time study of the print joint across the site's slider ranges. For each case: does the joint
fit, does the chain reach its stop, does anything non-adjacent collide, how wide is the base, what holds it."""
import sys, os, json, time, numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schema, parts, mesh as M, printjoint2 as J2
SLIDERS = {"length": [90, 120, 150, 200, 250], "width": [40, 55, 72, 100, 130], "relief": [8, 12, 17.1, 25, 35],
           "segCount": [3, 5, 7, 9, 12], "maxAngle": [12, 18, 24, 30, 36], "cephFrac": [0.2, 0.26, 0.33, 0.4, 0.45]}
which = sys.argv[1:] or list(SLIDERS)
rows = []
def case(name, val):
    P = schema.coerce({name: val}); J_, d, zj, y_piv = J2.geometry(P); J = J_
    S = parts.segment_plan(P, 1); ztop_axis = float(S["zfun"](np.array([0.0]), np.array([0.5 * d]))[0])
    fit_pitch = d - (2 * J["lip"] + 3 * J["gap_axial"] + J["knobH"])              # room left in a pitch after the joint
    fit_roof = ztop_axis - (zj + 0.5 * J["knobH"] + J["gap_vertical"] + J["lip"])    # material above the pocket
    base = d - 2 * zj * np.tan(np.radians(0.5 * P["maxAngle"]))                      # base length after the V cut
    t = time.time()
    try:
        n = min(4, int(P['segCount'])); segs = J2.chain(P, deg=0.0, n=n); ok = all(len(M.bodies(s)) == 1 for s in segs)
    except Exception as ex:
        return dict(slider=name, value=val, pitch=round(d, 2), built=False, err=str(ex)[:60])
    a, b = segs[0], segs[1]; ma = M.to_manifold(a); J = J_
    def pen(T): return (ma ^ M.to_manifold(b.copy().apply_transform(T))).volume() - 0.005
    stop = 0.0
    for ang in np.arange(2, 60, 2.0):
        if pen(trimesh.transformations.rotation_matrix(np.radians(-ang), (1, 0, 0), (0, y_piv, zj))) > 0: break
        stop = ang
    pull = 0.0
    for s_ in np.arange(0.05, 1.0, 0.05):
        if pen(trimesh.transformations.translation_matrix((0, s_, 0))) > 0: break
        pull = s_
    posed = J2.chain(P, deg=min(stop, P["maxAngle"]), n=n); mans = [M.to_manifold(s) for s in posed]
    nonadj = max([(mans[i] ^ mans[j]).volume() for i in range(n) for j in range(i + 2, n)] or [0.0])
    return dict(slider=name, value=val, pitch=round(d, 2), ring_top=round(parts.ring_top(P), 1), pivot_z=round(zj, 1), built=ok, joint_scale=J_.get("scaled", 1.0),
                fit_pitch_mm=round(fit_pitch, 2), roof_mm=round(fit_roof, 2), base_mm=round(base, 1), base_frac=round(base / d, 2),
                stop_deg=stop, target_deg=P["maxAngle"], pull_play_mm=round(pull, 2), nonadjacent_overlap_mm3=round(nonadj, 2), seconds=round(time.time() - t))
for name in which:
    for v in SLIDERS[name]:
        r = case(name, v); rows.append(r); print(json.dumps(r), flush=True)
path = "docs/printjoint2_sliders.json"; prev = json.load(open(path)) if os.path.exists(path) else []; json.dump(prev + rows, open(path, "w"), indent=1)
