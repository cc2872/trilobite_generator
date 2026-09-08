"""
eyes.py — one eye geometry for the builder, the sheet and the FOV ruler.

eye_geometry(P): where the eye sits and how big it is, in the cephalon's local frame (rear hinge at y=0, head
toward -y, x across). Used by trilobite.build_cephalon (the surface term), blueprint.sheet (the drawn eye) and
fov() below — so the drawn eye, the built eye and the measured eye can never disagree.

fov(P): FOV ruler v0.1. A declared, crude reading — stated assumptions:
  * each eye's visual surface is a crescent of angular extent eyeArc, centred on the outward lateral direction;
  * azimuth is read in the plan only; no occlusion by the glabella, the border or the other eye;
  * elevation half-angle is atan(eyeHeight) — the dome's slope, not a lens-by-lens reading.
Version it like the enrollment ruler: FOV_VERSION changes when any assumption above does.
"""
import math, numpy as np
FOV_VERSION = "0.1"

def eye_geometry(P):
    Lc = P["cephFrac"] * P["length"]
    from fields import head_halfwidth
    wh = head_halfwidth(P); a = P["axisFrac"] * wh
    eR = P["eyeSize"] * wh; ye = -P["eyePos"] * Lc
    f = min(max(-ye / Lc, 0.0), 1.0)
    glab = a * (1 + (P["glabInflate"] - 1) * f)                   # glabella half-width at the eye's y (same formula as build_cephalon)
    lat = P.get("eyeLat", 0.0)
    xe = lat * wh if lat > 0.01 else glab + eR + 1.0                 # eyeLat 0 = the v4 rule: hugging the glabella
    return dict(xe=float(xe), ye=float(ye), eR=float(eR), eH=float(P["eyeHeight"] * eR), arc_deg=float(P["eyeArc"]),
                exponent=float(P.get("eyeProfile", 4.0)), glab_half=float(glab), head_halfwidth=float(wh), head_length=float(Lc),
                blind=bool(P["eyeSize"] <= 0.01))

def fov(P):
    g = eye_geometry(P)
    if g["blind"]:
        return dict(fov_version=FOV_VERSION, blind=True, azimuth_deg=0.0, front_blind_deg=180.0, rear_blind_deg=180.0,
                    binocular_deg=0.0, elevation_half_deg=0.0)
    arc = g["arc_deg"]
    cov = min(360.0, 2.0 * min(arc, 180.0) + 0.0)                    # two mirrored crescents centred on ±90° (lateral)
    front_blind = max(0.0, 180.0 - arc); rear_blind = front_blind    # symmetric in v0.1
    binoc = max(0.0, arc - 180.0)                                    # front overlap when each arc passes the midline
    cov = 360.0 - front_blind - rear_blind
    elev = math.degrees(math.atan(P["eyeHeight"]))
    return dict(fov_version=FOV_VERSION, blind=False, azimuth_deg=round(cov, 1), front_blind_deg=round(front_blind, 1),
                rear_blind_deg=round(rear_blind, 1), binocular_deg=round(binoc, 1), elevation_half_deg=round(elev, 1),
                eye_xy=(round(g["xe"], 2), round(g["ye"], 2)), eye_radius_mm=round(g["eR"], 2))

if __name__ == "__main__":
    import schema, json
    for n in ("textured", "phacopid", "harpetid"):
        P = schema.preset(n); print(n, json.dumps(fov(P)))
