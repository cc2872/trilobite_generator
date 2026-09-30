"""
isopod_model/heads_sheet.py: the head alone for several presets, side by side (3/4 and side profile), to compare
what the preset sets on it: face (eyes, glabella), eye solids, dome curve, height, spines.

    python isopod_model/heads_sheet.py [preset ...]   -> photos/heads.png
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import numpy as np
import body as BODY, presets as PR, render as R
from joints import ball as BALL


def head_of(name):
    P = PR.params(name); plans = BODY.placed_plans(P)
    J, *_ = BALL.geometry(P); K = BODY.iso_kit(P); PV = BODY.pivots(plans, J, K)
    H, _ = BODY.isopod_head(P, plans, J, PV, K); H.apply_scale(1.0 / K["s"]); return H, P


if __name__ == "__main__":
    names = sys.argv[1:] or ["textured", "agnostida", "asaphida", "phacopida", "odontopleurida", "harpetida"]
    heads = [(n,) + head_of(n) for n in names]
    lo = np.min([h.bounds[0] for _, h, _ in heads], axis=0); hi = np.max([h.bounds[1] for _, h, _ in heads], axis=0)
    rows = []
    for n, h, P in heads:
        h = h.copy(); h.apply_translation((0, -h.bounds[0][1], 0))
        b = np.array([[lo[0], 0, 0], [hi[0], hi[1] - lo[1], hi[2]]])
        tag = f"dome {P['headDomeExp']:g}, eyes {'solid' if P['eyeSolid'] > 0.5 and P['eyeSize'] > 0.01 else ('bump' if P['eyeSize'] > 0.01 else 'none')}"
        rows.append([(h, 28, -60, f"{n.upper()}  ({tag})", dict(bounds=b)), (h, 0, 0, f"side, {h.bounds[1][2]:.0f} mm tall", dict(bounds=b))])
    R.sheet(rows, os.path.join(HERE, "photos", "heads.png"), W=820, H=460); print("photos/heads.png")
