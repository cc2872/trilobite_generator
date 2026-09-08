"""
eye_solid.py — the eye as its own primitive (8 Sep 2026).

A revolved profile: flat palpebral cap of radius R at height H, a visual band leaning `eyeSlope` from vertical down
to the cheek (base radius R + H tan slope), an optional brim (`eyeShade`) past the cap edge. Lenses: spherical caps
on a hexagonal lattice in the band's own (theta, s) coordinates, only within ±eyeArc/2 of the outward direction,
fused in one boolean. Parameters (fossil table): eyeHeight (H/R), eyeSlope (deg), eyeShade (/R), eyeArc (deg),
lensD (/R), lensGap (/D), lensRise (/(D/2)). Declared simplifications: the band exists over the visual arc + 24°; inward the eye is a low
revolved palpebral lobe (v1.1); eyeElong is not applied in v1; lens axes = band normal.
"""
import math, numpy as np
from build123d import *
EYE_SOLID_VERSION = "1.2"
MIN_PRINT_LENS_MM = 0.5        # lenses smaller than this (at print scale) are not built: below FDM relief

def lens_centres(R, H, slope, arc, lensD, lensGap):
    pitch = lensD * (1 + lensGap); rows = max(1, int(H / (0.87 * pitch))); out = []
    for i in range(rows):
        s = (i + 0.5) * 0.87 * pitch
        if s > H - 0.5 * lensD: break
        r = R + (H - s) * math.tan(slope); n = max(1, int(arc * r / pitch)); off = 0.5 * (i % 2)
        for k in range(n):
            th = -arc / 2 + (k + off + 0.5) * arc / (n + 0.5)
            if abs(th) <= arc / 2: out.append((th, s, r))
    return out

def build_eye(R, H, slope_deg, shade, arc_deg, lensD, lensGap, lensRise, embed=1.0, max_lenses=400, clip_x=None):
    """Eye solid in its own frame: axis = z, outward = +x, base at z = 0 (sunk `embed` mm below the cheek)."""
    slope = math.radians(slope_deg); arc = math.radians(arc_deg); rb = R + H * math.tan(slope)
    # revolved body: a truncated cone from (rb, -embed) to (R, H), flat cap, plus brim if any
    pts = [(0, 0, -embed), (rb + embed * math.tan(slope), 0, -embed), (R, 0, H), (0, 0, H), (0, 0, -embed)]   # profile in the XZ plane
    drum = revolve(make_face(Polyline(*pts)), Axis.Z, 360)
    if shade > 0.005:
        drum += Cylinder(R + shade * R, 0.6 * lensD * R + 0.8, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((0, 0, H - 0.6 * lensD * R - 0.8)))
    # 8 Sep 2026: the drum wall exists only over the visual arc (+ a small margin). Inward of that the eye is a
    # palpebral LOBE: a low dome from the cap edge down to the cheek, so the eye reads as a lobe with a lensed
    # band on its outer face, not as a cylinder standing on the head (the "googly eye" of v1.0).
    a_keep = math.degrees(arc) + 24
    big = 6 * rb + 6 * H
    sector = extrude(make_face(Polyline((0, 0, 0), (big * math.cos(math.radians(a_keep / 2)), -big * math.sin(math.radians(a_keep / 2)), 0),
                                        (big, -big * 0.2, 0), (big, big * 0.2, 0),
                                        (big * math.cos(math.radians(a_keep / 2)), big * math.sin(math.radians(a_keep / 2)), 0), (0, 0, 0))),
                     amount=2 * H + 4).moved(Location((0, 0, -embed - 1)))
    lobe_pts = [(0, 0, -embed), (1.8 * R, 0, -embed)] + [(R + 0.8 * R * math.sin(t), 0, -embed + (H + embed) * math.cos(t) ** 1.0)
                                                          for t in [math.radians(a) for a in (80, 65, 50, 35, 20, 8, 0)]] + [(0, 0, H), (0, 0, -embed)]
    lobe = revolve(make_face(Polyline(*lobe_pts)), Axis.Z, 360)
    lobe = lobe - sector
    if clip_x is not None:                                                     # keep the lobe inside the head outline (outward = +x here)
        lobe = lobe & Box(clip_x, 4 * big, 4 * big, align=(Align.MIN, Align.CENTER, Align.CENTER)).moved(Location((-clip_x + clip_x, 0, 0))) if False else \
               lobe & Box(2 * big, 4 * big, 4 * big).moved(Location((clip_x - big, 0, 0)))
    body = (drum & sector) + lobe
    # lenses: spheres of radius rho sunk so a cap of height `rise` protrudes along the band normal
    D = lensD * R; rise = lensRise * D / 2; rho = max((rise ** 2 + (D / 2) ** 2) / (2 * rise), 0.35)
    if D < MIN_PRINT_LENS_MM:                                                   # holochroal facets below print relief: smooth band, count reported
        return body, 0
    nx, nz = math.cos(slope), math.sin(slope)                         # band normal, outward and slightly down
    cs = lens_centres(R, H, slope, arc, D, lensGap)[:max_lenses]
    spheres = []
    for th, s, r in cs:
        depth = rho - rise
        cx, cy, cz = (r - depth * nx) * math.cos(th), (r - depth * nx) * math.sin(th), s + depth * nz
        spheres.append(Sphere(rho).moved(Location((cx, cy, cz))))
    if spheres:
        body += Part(children=spheres)
    return body, len(cs)

def eye_params(P, eR):
    """Primitive parameters from the schema (fossil-table defaults when keys are missing)."""
    return dict(R=eR, H=P.get("eyeHeight", 1.7) * eR, slope_deg=P.get("eyeSlope", 15.0), shade=P.get("eyeShade", 0.0),
                arc_deg=P.get("eyeArc", 110.0), lensD=P.get("lensD", 0.16), lensGap=P.get("lensGap", 0.3), lensRise=P.get("lensRise", 0.35))

if __name__ == "__main__":
    import time
    t0 = time.time(); e, n = build_eye(6.0, 10.0, 15, 0.0, 110, 0.16, 0.3, 0.35)
    print("eye", n, "lenses", round(time.time() - t0, 1), "s solids", len(e.solids()), "vol", round(e.volume, 1))
