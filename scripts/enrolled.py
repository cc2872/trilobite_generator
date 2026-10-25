"""enrolled.py: build an animal and lay its pieces out ENROLLED (kinematics.angles: the Esteve profile, head joint most),
check that no two pieces overlap in that pose, and export the assembled enrolled animal as one STL (plus renders).
The printable pieces stay flat (as built); this is the assembly.

    python scripts/enrolled.py OUTDIR [preset] [head_deg]
"""
import os, sys, warnings, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); os.chdir(ROOT)
sys.path[:0] = [ROOT, ROOT + "/isopod_model", ROOT + "/scripts"]; warnings.filterwarnings("ignore")
import numpy as np, trimesh
import schema, mesh as M, body as BODY, head_styles as HS, render as RD, kinematics as KIN
from PIL import Image, ImageDraw

OUT = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else "diagram"; target = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0   # total turning: the tail tucked under the head
os.makedirs(OUT, exist_ok=True)
P = schema.preset(name); style = name if name in HS.STYLES else "plain"
P, face = HS.apply(P, style); P = schema.coerce(dict(P, **schema.PRESETS[name]))
parts, plans, PV, K = BODY.build(P, "shaped", face=face, outline=HS.outline(style)); s = K["s"]
n = len(PV); W0 = [M.to_manifold(p) for p in parts]
def overlaps(ang):
    T = BODY.pose(PV, list(ang)); W = [W0[k].transform(T[k][:3, :]) for k in range(len(parts))]; out = []
    for k in range(len(parts)):
        for j in range(k + 1, len(parts)):
            v = float((W[k] ^ W[j]).volume())
            if v > 0.5 * s ** 3: out.append((plans[k][0], plans[j][0], round(v / s ** 3, 1)))   # more than 0.5 mm3 at print size
    return out
# the Esteve profile scaled to `target` degrees in all; if pieces collide there, the most closed pose short of a collision
head_deg = KIN.HEAD_DEG * target / KIN.total(n, KIN.HEAD_DEG); ang = KIN.angles(n, head_deg); worst = overlaps(ang)
if worst:
    lo, hi = 0.0, 1.0
    for _ in range(10):
        mid = 0.5 * (lo + hi); (lo, hi) = (mid, hi) if not overlaps([mid * a for a in ang]) else (lo, mid)
    ang = [lo * a for a in ang]; worst_at_target = worst; worst = overlaps(ang)
    print(f"at {target:.0f} deg the pieces collide ({worst_at_target[:4]} ...); enrolled to {sum(ang):.0f} deg instead")
T = BODY.pose(PV, list(ang)); posed = [p.copy().apply_transform(T[k]) for k, p in enumerate(parts)]
whole = trimesh.util.concatenate([p.copy().apply_scale(1.0 / s) for p in posed])
whole.export(os.path.join(OUT, f"{name}_enrolled.stl"))
print(f"{name}: {n} joints, head joint {ang[0]:.1f} deg, per joint {[round(a,1) for a in ang]}, total {sum(ang):.0f} deg; overlaps in the enrolled pose: {worst or 'none'}; size {np.round(whole.extents, 1).tolist()} mm")
views = [(RD.render(whole, 0, 0, 700, 700), "SIDE"), (RD.render(whole, 25, -40, 700, 700), "OBLIQUE"), (RD.render(whole, 0, 90, 700, 700), "FRONT")]
im = Image.new("RGB", (704 * 3, 740)); d = ImageDraw.Draw(im)
for i, (a, l) in enumerate(views): im.paste(Image.fromarray(a), (2 + 704 * i, 30)); d.text((6 + 704 * i, 8), l, fill=(246, 112, 79))
d.text((6, 724), f"{name} enrolled: {n} joints, {sum(ang):.0f} deg in all; overlapping pieces: {len(worst)}", fill=(170, 170, 170))
im.save(os.path.join(OUT, f"{name}_enrolled.png"))
