"""
isopod_model/build.py: builds the model: the crescent head made on the isopod, on the isopod-style body with the
isopod's joints. One command, everything lands in this folder.

    python isopod_model/build.py                  -> stl/trilobite_isopod.stl, stl/pieces/*.stl, checks.json, photos/
    python isopod_model/build.py --rebuild-head   ... first remakes source/crescent_head_piece.stl (crescent_head.py)
    python isopod_model/build.py --no-photos --no-checks

The animal is the generator's default (schema.defaults()); --preset NAME builds a preset instead.
"""
import os, sys, argparse, json, glob, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import numpy as np, trimesh
import schema
import body as BODY, check as CHECK

PIECE_NAMES = lambda n: ["head"] + [f"thorax{i + 1}" for i in range(n - 2)] + ["tail"]


def rebuild_head():
    """Remake the head from scratch: the rim-level band (source/crescent_band.stl) is recomputed from the isopod's
    joint ranges, the per-face head cache is cleared, and the approved head piece (the default face) is written to
    source/crescent_head_piece.stl for reference."""
    import shutil, crescent_head as CH
    for f in (os.path.join(HERE, "source", "crescent_band.stl"),):
        if os.path.exists(f): os.remove(f)
    shutil.rmtree(BODY.CACHE, ignore_errors=True)
    head, _ = CH.head_piece({}, fit="widest"); head.export(os.path.join(HERE, "source", "crescent_head_piece.stl"))
    print("head rebuilt: band, cache, source/crescent_head_piece.stl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-head", action="store_true"); ap.add_argument("--preset", default=None)
    ap.add_argument("--no-checks", action="store_true"); ap.add_argument("--no-photos", action="store_true")
    A = ap.parse_args(); t0 = time.time()
    if A.rebuild_head: rebuild_head()
    P = schema.preset(A.preset) if A.preset else schema.defaults()
    parts, plans, PV, K = BODY.build(P, "isopod")
    s = K["s"]; pieces = [p.copy().apply_scale(1.0 / s) for p in parts]          # print mm: the isopod's own size
    names = PIECE_NAMES(len(pieces))
    # write: the whole print-in-place file and each piece
    out = os.path.join(HERE, "stl"); os.makedirs(os.path.join(out, "pieces"), exist_ok=True)
    for f in glob.glob(os.path.join(out, "pieces", "*.stl")): os.remove(f)
    for k, (n, p) in enumerate(zip(names, pieces)): p.export(os.path.join(out, "pieces", f"{k:02d}_{n}.stl"))
    whole = trimesh.util.concatenate(pieces); whole.export(os.path.join(out, "trilobite_isopod.stl"))
    print(f"stl/trilobite_isopod.stl: {len(pieces)} pieces, {np.round(whole.extents, 1).tolist()} mm, {len(whole.faces)} triangles")
    # pivots: the ball centres as built (sphere fits), print mm
    pv = [tuple(CHECK.fit_ball(pieces[k + 1], (0, y / s, z / s), 0.0)[0][1:]) for k, (y, z) in enumerate(PV)]
    json.dump(dict(pivots_mm=[list(map(float, p)) for p in pv]), open(os.path.join(HERE, "checks.json"), "w"), indent=1)
    if not A.no_checks: print(CHECK.summary(CHECK.run(pieces, pv, names)))
    if not A.no_photos:
        import render as R; R.photos(pieces, pv, os.path.join(HERE, "photos")); print("photos/ written")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
