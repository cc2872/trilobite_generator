"""
anatomy/tail.py: the pygidium, joint-free. Same six functions as anatomy/thorax.py.
"""
import math
import numpy as np
from anatomy import organic as OR
import mesh as M
from fields import tail_halfwidth, furrow_amp
from anatomy.common import pitch, ring_top, smoothstep, plateau, trough, vault, spine_solid, curved_spine, prong, GRID_TAIL
from anatomy.port import Port

AUTO_LOBE_BAND = 0.35   # 9 Nov 2026: the inner shield's smooth share when the lobes are curved (see tail_plan)

def plan(P):
    """outline(u, v), zfun(x, y) and scalars for the tail. Transcribed from trilobite.build_pygidium, minus tubercles."""
    t, c, h = P["wall"] * P["tailWall"], P["clearance"], P["relief"] * P["tailRelief"]
    Lp = P["pygFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P)
    wp = tail_halfwidth(P)
    # 21 Oct 2026 (schema 6.4): four tail controls fitted to TriloMorph pygidia. Each default reproduces the tail built since v4.
    nO = float(P.get("tailOutlineExp", 2.0))                    # margin superellipse: 2 = the half-ellipse, < 2 pointed, > 2 square
    Lax = float(P.get("pygAxisLen", 0.85))                      # axis length / tail length (the v4 axis faded out at 0.85 Lp)
    ka = Lax / 0.85                                             # stretch of every axis-relative position against v4
    pa = float(P.get("pygAxisFrac", 0.0))
    a = (pa if pa > 0 else P["axisFrac"]) * wp                  # 0 = the thoracic axis fraction, as before
    pb = float(P.get("pygBorderWidth", -0.01))
    bwf = P["borderWidth"] if pb < 0 else pb                    # negative = the head's border fraction, as before
    se = lambda c: np.sign(c) * np.abs(c) ** (2.0 / nO)         # superellipse coordinate; se(c) = c when nO = 2
    nk = float(P.get("pygNeck", 0.0)); nkl = max(float(P.get("pygNeckLen", 0.30)), 1e-3)
    def neck_f(yy):                                             # 7 Nov 2026 (schema 6.9): the tail necks in at its front
        return 1.0 - nk * (1.0 - smoothstep(0.0, nkl, np.clip(yy, 0, 1)))   # and swells to full width by nkl of its length
    # 29 Oct 2026: the swept anterior margin. Real pygidia carry the anterolateral corner BEHIND the axial front
    # (TriloMorph: median 21 deg, 86 % of 111 pygidia); the tail built since v4 had a straight front. pygFrontSweep is
    # tan of that angle (corner set-back / lateral run), 0 = straight as before. The whole tail is sheared along y by
    # fy(x), growing from the axial furrow (0, so the hinge is untouched) to the corner with the thoracic blade's own
    # exponent, and fading to nothing at the tip, so the furrows, ribs, border and the front flap all follow the sweep.
    fsw = float(P.get("pygFrontSweep", 0.0)) * max(wp - a, 1e-6)
    def fy(x):
        q = np.clip((np.abs(np.asarray(x, float)) - a) / max(wp - a, 1e-6), 0, 1); return fsw * q ** 1.6
    def shear(x, y):                                            # plan (x, y) -> built (x, y'): y' = fy + y (1 - fy / Lp)
        f = fy(x); return x, f + np.asarray(y, float) * np.maximum(1 - f / Lp, 0.05)
    def unshear(x, y):
        f = fy(x); return x, (np.asarray(y, float) - f) / np.maximum(1 - f / Lp, 0.05)
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P) * float(P.get("pygFurrowScale", 1.0))
    rib = float(P.get("pygRibRelief", 0.0)) * F                   # 24 Oct 2026: convex ribs between the pleural furrows
    n = int(P["pygRings"])
    def xmax(y):
        yy = np.clip(np.asarray(y, float) / Lp, 0, 1)
        base = wp * np.clip(1 - yy ** nO, 0, 1) ** (1.0 / nO)
        if nk > 0.005: base = base * neck_f(yy)
        return np.maximum(base, 1.5)
    Lf_full = P["pygSpine"] * Lp
    n_m = int(P["pygMarginal"]); Lm_full = P["pygMarginalLen"] * Lp
    # 24 Oct 2026: with pygSpineRise / Bend / Grow the margin spines run on as curved solid spines (ornaments); the flat
    # bump in the plan then stops at 0.55 of its length and the solid spine takes over from there (the blade-into-spine
    # hand-over of the pleural horns), so no flat blade runs on under a rising spine.
    tube = abs(float(P.get("pygSpineRise", 0.0))) > 1e-6 or abs(float(P.get("pygSpineBend", 0.0))) > 1e-6 or float(P.get("pygSpineGrow", 0.0)) > 0.01
    cut = float(P.get("pygSpineBlade", 1.0)) if tube else 1.0  # the flat blade's share of each spine; 0 = none, the round spine leaves the margin itself
    Lf = Lf_full * cut; Lm = Lm_full * cut
    sp = math.radians(P["pygSplay"])
    phi_f = math.atan(wp / Lp * math.tan(math.pi / 2 - sp)) if Lf_full > 0.5 else None
    wk = float(P.get("pygSpineWidth", 1.0))                      # 25 Oct 2026: the lobes' width at the root, 1 = as built
    hphi_f = math.radians(13) * wk
    # 26 Oct 2026: where along the margin the marginals sit (deg, 0 = right front corner, 90 = rear, 180 = left front
    # corner; the defaults 50..130 are the band round the rear they always had) and how far each lobe leans back
    # (pygSpineSweep deg; 0 = straight out from the centre, as before). pygLobeBand: the outer share of the grid that
    # carries the lobes, so the dome inside is sampled on smooth rings rather than on scaled copies of the spiky margin.
    m_from, m_to = float(P.get("pygMarginalFrom", 50.0)), float(P.get("pygMarginalTo", 130.0))
    sw0 = math.radians(float(P.get("pygSpineSweep", 0.0)))
    band = float(P.get("pygLobeBand", 0.0))
    phi_m = [math.radians(m_from + (m_to - m_from) * (k + 0.5) / n_m) for k in range(n_m)]
    hphi_m = math.radians((m_to - m_from) / max(n_m, 1)) * 0.5 * 1.3 * wk
    # the margin spines as ONE list (24 Oct 2026): each dict(phi, Lk = the flat bump's length in the plan, L_full = the whole
    # spine, hh, e, rise, roll) -- the preset's pair + marginals, or the body's own `_tailSpines` list (the studio's dots:
    # phi deg, len mm, rise, roll, blade), right side only (phi <= 90), mirrored here. rise / roll None = P's.
    spines = []
    own = P.get("_tailSpines")
    if own:
        for sd in own:
            L_full = float(sd.get("len", 0.0)); bl = float(sd.get("blade", 0.0)); phi_k = math.radians(float(sd.get("phi", 90.0)))
            if L_full > 0.5: spines.append(dict(phi=phi_k, Lk=L_full * bl, L_full=L_full, hh=hphi_f * float(sd.get("width", 1.0)), e=1.5 if bl < 0.999 else 1.2,
                                                rise=float(sd.get("rise", 30.0)), roll=float(sd.get("roll", 0.0)), width=float(sd.get("width", 1.0)),
                                                sweep=math.radians(float(sd.get("sweep", math.degrees(sw0)))),
                                                curve=math.radians(float(sd.get("curve", 0.0))),     # 9 Nov 2026: the lobe bends as it goes out (+ toward the rear)
                                                aim=math.radians(float(sd.get("aim", 0.0)))))        # ... and leaves turned this far outward first (a pincer)
        # 9 Nov 2026: curved (or aimed) lobes are smoothed inside automatically -- with the whole grid following a bent margin,
        # each lobe was echoed in scaled copies across the shield as folds (undercut lips, small tunnels). The inner shield
        # is then sampled on smooth rings and the lobes keep the outer AUTO_LOBE_BAND of it. A band set in the studio
        # (pygLobeBand > 0) or deliberate creases (the fan ribs, pygRibFan) are left exactly as set.
        if band < 0.01 and float(P.get("pygRibFan", 0.0) or 0.0) < 0.01 and any(abs(s_.get("curve", 0.0)) > 1e-6 or abs(s_.get("aim", 0.0)) > 1e-6 for s_ in spines):
            band = AUTO_LOBE_BAND
        # 25 Oct 2026: a list of flat lobes (blade 1, no rise) stays a flat margin -- no solid spine runs on top of a lobe
        tube = any(abs(s_["rise"]) > 1e-6 or abs(s_["roll"]) > 1e-6 or s_["Lk"] < s_["L_full"] - 1e-6 for s_ in spines)
    else:
        if Lf_full > 0.5: spines.append(dict(phi=phi_f, Lk=Lf, L_full=Lf_full, hh=hphi_f, e=1.5, rise=None, roll=None, width=1.0, sweep=sw0))
        for pk in phi_m: spines.append(dict(phi=pk, Lk=Lm, L_full=Lm_full, hh=hphi_m, e=float(P.get("pygLobePoint", 1.2)), rise=None, roll=None, width=hphi_m / max(hphi_f, 1e-9), sweep=sw0))   # width in units of the pair's root (an own copy rebuilds the same lobe)
    # each lobe leans toward the rear (+y) in two ways, never past the midline: (1) its tip sits behind the middle of its
    # base (skew: the rear flank steeper than the front one -- a radial bump, so the outline cannot fold), and (2) the
    # bump's direction turns toward the rear by as much as the lobe's width allows (more would fold the polar grid)
    for sd in spines:
        bx, by = wp * se(math.cos(sd["phi"])), Lp * se(math.sin(sd["phi"]))
        if nk > 0.005: bx = bx * neck_f(by / Lp)
        th = math.atan2(by, bx); r0 = math.hypot(bx, by)
        sw_ = min(sd.get("sweep", 0.0), 0.45 * abs(math.pi / 2 - th)); Lk_ = max(sd["Lk"], 1e-6)
        sd["skew"] = min(0.6 * sd["hh"], Lk_ * math.tan(sw_) / max(r0, 1e-6))
        room = min(1.0, 0.8 * (sd["hh"] - sd["skew"]) * r0 / (sd["e"] * Lk_))   # the rear flank's steepest slope is e * Lk / (hh - skew), at the root
        sd["lean"] = min(sw_, math.asin(room))
        sd["lean_dir"] = min(sw_, sd["lean"] + math.atan(sd["skew"] * r0 / Lk_))    # the tip's direction from the base's middle
    k_or = OR.k_of(P)                                           # 1 Nov 2026, organic rule 2: a lobe ends round, not in a cusp
    def bump(phi, phi_k, hh, e, skew=0.0, s=0.0):
        d = phi - (phi_k + s * skew); w = np.where(s * d < 0, hh + skew, hh - skew) if s != 0.0 else hh
        ad = np.abs(d)
        if k_or > 0: ep = k_or * OR.LOBE_ROUND * w; ad = np.sqrt(d * d + ep * ep) - ep   # |d| eased at the apex
        return np.clip(1 - ad / w, 0, 1) ** e
    _LOBES = []                                                    # (pm, side, lobe) per candidate, for margin_pt's curved lobes
    def radial_extra(phi, lean_out=False):
        ex = np.zeros_like(phi); ln = np.zeros_like(phi); who = np.full(np.shape(phi), -1); _LOBES.clear()
        for sd in spines:
            if sd["Lk"] > 0.5:
                for pm in (sd["phi"], math.pi - sd["phi"]):
                    sgn = 0.0 if abs(pm - math.pi / 2) < 1e-6 else (1.0 if pm < math.pi / 2 else -1.0)   # right side leans +, left side -, both toward the rear
                    b = sd["Lk"] * bump(phi, pm, sd["hh"], sd["e"], sd.get("skew", 0.0), sgn); win = b > ex
                    ex = np.where(win, b, ex); ln = np.where(win, sgn * sd.get("lean", 0.0), ln)
                    if lean_out: who = np.where(win, len(_LOBES), who); _LOBES.append((pm, sgn, sd))
        return (ex, ln, who) if lean_out else ex
    def margin_pt(phi):
        x0, y0 = wp * se(np.cos(phi)), Lp * se(np.sin(phi))
        if nk > 0.005: x0 = x0 * neck_f(y0 / Lp)
        r0 = np.maximum(np.hypot(x0, y0), 1e-6); ex, ln, who = radial_extra(phi, lean_out=True)
        th = np.arctan2(y0, x0) + ln                                   # the radial direction, leaned toward the rear
        X_, Y_ = x0 + ex * np.cos(th), y0 + ex * np.sin(th)
        for i, (pm, sgn, sd) in enumerate(list(_LOBES)):               # 9 Nov 2026: a curved lobe (its `curve`, deg at the tip, + toward the
            cv = float(sd.get("curve", 0.0)); aim = float(sd.get("aim", 0.0)); m = (who == i) & (ex > 1e-9)   # rear): the lobe's section at height b
            if (abs(cv) < 1e-6 and abs(aim) < 1e-6) or not m.any(): continue   # out is turned by curve x b / Lk about the lobe's base centre and carried along the
            bx, by = wp * se(math.cos(pm)), Lp * se(math.sin(pm))      # arc its centre line follows -- a bent lobe, not a sheared one
            if nk > 0.005: bx = bx * neck_f(by / Lp)
            th0 = math.atan2(by, bx) + sgn * (sd.get("lean", 0.0) - aim); kap = sgn * cv / max(sd["Lk"], 1e-6)   # aim: the centre line leaves turned out
            b = ex[m]; al = kap * b
            ax_ = np.where(np.abs(al) > 1e-9, (np.sin(th0 + al) - math.sin(th0)) / np.where(np.abs(kap) > 1e-12, kap, 1.0), b * math.cos(th0))
            ay_ = np.where(np.abs(al) > 1e-9, (math.cos(th0) - np.cos(th0 + al)) / np.where(np.abs(kap) > 1e-12, kap, 1.0), b * math.sin(th0))
            dx, dy = x0[m] - bx, y0[m] - by; c_, s_ = np.cos(al), np.sin(al)
            X_[m] = bx + c_ * dx - s_ * dy + ax_; Y_[m] = by + s_ * dx + c_ * dy + ay_
        return X_, Y_
    PHI_MIN = math.radians(1.5)   # the margin never reaches the front line: at phi = 0 the outer grid column would lie
    def outline_plan(u, v):       # along y = 0 and its corner triangles would be collinear (the spline fit used to hide it)
        phi = PHI_MIN + (math.pi - 2 * PHI_MIN) * 0.5 * (1 - u)
        mx, my = margin_pt(np.array([phi])); fx = u * (wp - 1.0)
        if band > 0.01:                                            # inside: smooth rings (the base margin scaled about the axis
            bx, by = wp * se(math.cos(phi)), Lp * se(math.sin(phi))   # line) out to v = 1 - band; then the band, from that ring out
            if nk > 0.005: bx = bx * neck_f(by / Lp)
            vi = min(v, 1 - band); ix, iy = (1 - vi) * fx + vi * bx, vi * by   # to the margin with its lobes -- never of zero width,
            if v <= 1 - band: return float(ix), float(iy)          # as it would be where there is no lobe if it began on the margin
            w_ = (v - (1 - band)) / band
            return float(ix + w_ * (mx[0] - ix)), float(iy + w_ * (my[0] - iy))
        return float((1 - v) * fx + v * mx[0]), float(v * my[0])
    def outline(u, v):
        x, y = outline_plan(u, v)
        if fsw <= 1e-9: return x, y
        xs, ys = shear(x, y); return float(xs), float(ys)
    def outside(x, y):
        wx = wp * neck_f(np.asarray(y, float) / Lp) if nk > 0.005 else wp
        r = (np.abs(x / wx) ** nO + np.abs(y / Lp) ** nO) ** (1.0 / nO)
        Lmax = max([sd["Lk"] for sd in spines if sd["Lk"] > 0.5] + [1e-6])
        return np.clip((r - 1) * np.hypot(x, y) / np.maximum(r, 1e-6) / Lmax, 0, 1)
    def ov_pre(ax, y): return np.where(y > 0.35 * Lp, outside(ax, y), 0.0)   # the margin spines' region (no ribs there)
    def zfun(x, y):
        if fsw > 1e-9: x, y = unshear(np.asarray(x, float), y)       # back to the plan's frame: the relief rides the shear
        ax = np.abs(x); xm = xmax(y); yy = np.clip(y / Lp, 0, 1)
        nT = P["tailDomeExp"]
        tz = np.clip(1 - yy ** nT, 0, 1) ** (1 / nT) * 0.9 + 0.1
        z = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
        taper = float(P.get("pygAxisTaper", 0.85))                        # 9 Nov 2026 (schema 6.9): settable; 0.85 as built
        ayf = lambda yv: a * (1 - taper * np.clip(np.asarray(yv, float) / (Lp * ka), 0, 1))
        ay = ayf(y)
        ae_ = float(P.get("pygAxisEnd", 0.0))
        if ae_ < 0.005:                                                    # as built: the axis fades out, flank furrows only
            z += rise * plateau(x, ay) * (1 - smoothstep((Lax - 0.10) * Lp, (Lax + 0.10) * Lp, y))
            z -= F * trough(ax - (ay + 0.6), 0.9) * (y < (Lax + 0.05) * Lp)
        else:                                                              # 9 Nov 2026: the axis as a CLOSED capsule mirroring the
            kc = float(P.get("pygAxisCap", 1.0)); y0e = 0.9 * Lax * Lp      # head's glabella (head_crescent front_furrow): flanks to
            for _ in range(3): y0e = max(Lax * Lp - kc * float(ayf(y0e)), 0.3 * Lp)   # y0e, then an elliptical cap of half-width ay(y0e)
            g_e = float(ayf(y0e)); cl = max(kc * g_e, 0.5)                  # and length kc x that, ending at the axis' end Lax Lp.
            dy = np.clip(y - y0e, 0, None)                                  # Rise and furrow follow ONE normalised radius, so the
            u = np.where(y <= y0e, ax / np.maximum(ay, 1e-6), np.sqrt((ax / g_e) ** 2 + (dy / cl) ** 2))   # top is continuous and
            gE = np.where(y <= y0e, ay, g_e)                                # the furrow runs round the end without a break
            rnd = float(P.get("pygAxisRound", 0.0))
            top = (1 - rnd) * plateau(u * gE, gE) + rnd * np.sqrt(np.clip(1 - (u * gE / (gE + 0.6)) ** 2, 0, 1))
            z += rise * top
            z -= F * trough((u - 1) * gE - 0.6, 0.9) * np.where(y <= y0e, 1.0, ae_)
        nod = float(P.get("pygNode", 0.0)) * 0.04 * h                      # 7 Nov 2026 (schema 6.9): a median node on the
        if nod > 0.005:                                                    # axis -- a round boss, like the head face's `node`
            yn = float(P.get("pygNodePos", 0.30)) * Lax * Lp * ka
            rn = 1.4                                                       # a tubercle (like the head face's node), not a mound
            z += nod * np.exp(-(((x / rn) ** 2) + (((y - yn) / rn) ** 2)))
        dy_k = Lp * ka / (n + 1.5)                                 # the furrow spacing
        r_at = float(P.get("pygRingAt", -1.0))                     # 8 Nov 2026 (schema 6.9): where the first ring furrow
        fan = float(P.get("pygRibFan", 0.0))                       # 9 Nov 2026 (schema 6.9): the pleural furrows as a FAN
        for k in range(n):                                         # (the corynexochid tail) are surface DETAIL (fan_field, below,
            yk = Lp * ka * ((k + 1) / (n + 1.5) if r_at < 0 else (r_at + k / (n + 1.5)))   # added after the grid's blur);
            fade = 1 - 0.6 * k / max(n, 1)                         # the ring furrows stay in the base
            z -= 0.7 * F * fade * trough(y - yk, 0.8) * plateau(x, ay + 1.5, 1.0)
            if fan < 0.005:
                px = np.clip((ax - ay) / np.maximum(xm - ay, 1), 0, 1); ly = yk + px * 0.5 * Lp * ka / (n + 1.5)
                z -= 0.6 * F * fade * trough(y - ly, 0.9) * (ax > ay + 1.0)
                if rib > 0.01:                                     # the rib: a ridge with concave flanks and a crest, over the pleural field behind this furrow
                    lm = ly + 0.5 * dy_k; w_ = np.clip(1 - np.abs(y - lm) / (0.5 * dy_k), 0, 1)
                    z += rib * fade * (w_ ** 1.6) * smoothstep(ay + 0.5, ay + 2.5, ax) * (1 - 0.5 * px) * (1 - np.clip(ov_pre(ax, y), 0, 1))
        if bwf > 0.005:
            bw = bwf * wp
            dist = np.minimum(xm - ax, (1 - yy) * Lp)
            z -= 0.8 * F * trough(dist - bw, 0.8 + 0.2 * bw)
            z += 0.35 * F * plateau(dist, 0.45 * bw, 0.4 * bw)
        ov = np.where(y > 0.35 * Lp, outside(ax, y), 0.0)
        z = np.where(ov > 0, np.maximum(margin * 0.6 * (1 - 0.5 * ov), t + 0.6), z)
        if P["tailRelief"] < 0.999:
            z = np.maximum(z, ring_top(P) * plateau(x, a, 1.0) * (1 - smoothstep(0.5 * ovl, ovl + 1.0, y)))
        drop = np.minimum(t + c + 0.7 * F + 0.3, np.maximum(z - 1.2, 0))
        z -= drop * (1 - smoothstep(ovl - 1.5, ovl, y))
        return z
    def fan_field(x, y):
        """pygRibFan > 0: the pleural furrows radiate -- each leaves the axis flank at its ring and runs STRAIGHT to its own
        point on the margin, the set from (nearly) transverse at the front corner round to the rear (crowding rearward,
        as drawn), plus 0.4 n post-axial furrows leaving the axis' end so the fan closes behind it; the ribs
        (pygRibRelief) crest midway between neighbouring furrows whatever their direction. A detail field: added AFTER
        the grid's smoothing blur (mesh.sample_grid detail=), as the head's sutures are, so 1 mm furrows survive."""
        fan = float(P.get("pygRibFan", 0.0)); n_ = int(P["pygRings"])
        if fan < 0.005 or n_ < 1: return np.zeros_like(np.asarray(x, float))
        if fsw > 1e-9: x, y = unshear(np.asarray(x, float), y)       # the fan rides the front sweep's shear too (9 Nov 2026 merge)
        ax = np.abs(np.asarray(x, float)); y = np.asarray(y, float); yy = np.clip(y / Lp, 0, 1); xm = xmax(y)
        taper = float(P.get("pygAxisTaper", 0.85)); ayf = lambda yv: a * (1 - taper * np.clip(np.asarray(yv, float) / (Lp * ka), 0, 1)); ay = ayf(y)
        r_at = float(P.get("pygRingAt", -1.0)); dy_k = Lp * ka / (n_ + 1.5); n_post = int(round(0.4 * n_)); N = n_ + n_post
        out = np.zeros_like(ax); dists = []
        for k in range(N):
            if k < n_: yk = Lp * ka * ((k + 1) / (n_ + 1.5) if r_at < 0 else (r_at + k / (n_ + 1.5)))
            else: yk = Lax * Lp - 0.5 * float(ayf(Lax * Lp)) * (1 - (k - n_ + 0.5) / max(n_post, 1))   # from the cap, in steps
            fade = 1 - 0.6 * min(k, n_) / max(n_, 1)
            ak = float(ayf(min(yk, Lax * Lp))) * (1.0 if k < n_ else 0.9 * (1 - (k - n_ + 0.5) / max(n_post, 1))); A_ = (ak, yk)
            phi = math.radians(90.0 * ((k + 0.5) / N) ** 0.6)
            Mf = (wp * se(math.cos(phi)), Lp * se(math.sin(phi)))
            yt = min(yk + 0.5 * dy_k, 0.999 * Lp); Mt = (float(xmax(yt)), yt)        # the as-built end: pygRibFan blends the two
            M_ = ((1 - fan) * Mt[0] + fan * Mf[0], (1 - fan) * Mt[1] + fan * Mf[1])
            dx_, dy_ = M_[0] - A_[0], M_[1] - A_[1]; Ln = max(math.hypot(dx_, dy_), 1e-6)
            tpar = ((ax - A_[0]) * dx_ + (y - A_[1]) * dy_) / Ln ** 2           # along the furrow, 0 at the axis, 1 at the margin
            perp = np.abs((ax - A_[0]) * dy_ - (y - A_[1]) * dx_) / Ln           # distance from the furrow's line
            d_ = np.where(tpar >= 0, perp, np.hypot(ax - A_[0], y - A_[1]))       # before the axis: the distance to its start
            dists.append(d_)
            out -= 0.55 * F * fade * trough(d_, 1.1) * ((ax > ay + 1.0) | (y > Lax * Lp + 0.5))
        if rib > 0.01 and len(dists) >= 2:
            D = np.sort(np.stack(dists), axis=0); d1, d2 = D[0], D[1]       # 4 d1 d2 / (d1 + d2)^2: 1 midway between neighbours, 0 on a furrow
            w_ = 4 * d1 * d2 / np.maximum(d1 + d2, 1e-6) ** 2
            out += rib * (0.5 - 0.5 * np.cos(np.pi * w_)) * smoothstep(ay + 0.5, ay + 2.5, ax) * (1 - 0.6 * yy ** 2)   # a rounded crest, fading rearward
        keep = (1 - np.clip(np.where(y > 0.35 * Lp, outside(ax, y), 0.0), 0, 1))   # not on the margin spines, nor the front lip
        keep = keep * smoothstep(ovl, ovl + 2.0, y) * (1 - smoothstep(xm - 1.2, xm, ax))   # ... nor over the rolled edge
        if bwf > 0.005: keep = keep * smoothstep(bwf * wp - 0.5, bwf * wp + 1.0, np.minimum(xm - ax, (1 - yy) * Lp))   # the border stays smooth
        return out * keep
    has_fan = float(P.get("pygRibFan", 0.0)) >= 0.005 and int(P["pygRings"]) >= 1
    zfun_base = zfun
    if has_fan: zfun = lambda x, y: zfun_base(x, y) + fan_field(x, y)
    # the margin spines, for ornaments(): where each leaves the base margin and how far it runs as a bump (24 Oct 2026)
    def base_pt(phi):
        bx, by = wp * se(math.cos(phi)), Lp * se(math.sin(phi))
        x, y = (bx * neck_f(by / Lp) if nk > 0.005 else bx), by
        if fsw > 1e-9: x, y = shear(x, y)
        return float(x), float(y)
    def margin_xy(phi):
        mx, my = margin_pt(np.asarray(phi, float))
        if fsw > 1e-9: mx, my = shear(mx, my)
        return np.c_[mx, my]
    return dict(outline=outline, zfun=zfun, zfun_base=zfun_base, detail=fan_field if has_fan else None, t=t, Lp=Lp, wp=wp, a=a, margin=margin, u_a=a / max(wp, 1e-6),
                axis_len=Lax, border_frac=bwf, outline_exp=nO, spines=spines, base_pt=base_pt, margin_xy=margin_xy, tube=tube,
                front_y=fy, front_sweep=fsw)

def _base(S): return (S.get("zfun_base", S["zfun"]), S.get("detail"))   # 9 Nov 2026: the fan furrows ride on the blurred base (anatomy/head._base)


def shell(P, S, grid=GRID_TAIL):
    zb, det = _base(S)
    tail = M.heightfield_shell(S["outline"], zb, S["t"], nu=grid[0], nv=grid[1], detail=det)
    env = M.under_envelope(S["outline"], zb, nu=grid[0], nv=grid[1], detail=det)
    return tail, env

def solid(P, S, grid=GRID_TAIL):
    zb, det = _base(S)
    return M.under_envelope(S["outline"], zb, nu=grid[0], nv=grid[1], floor=0.0, detail=det)

def ornaments(P, S):
    """Terminal spine, then the posterior prong. Forks and marginal spines are the margin itself (in the plan); with
    pygSpineRise / Bend / Grow (24 Oct 2026) each of them runs on as a curved solid spine that continues the bump."""
    OR.set_from(P); out = []
    rise0, bend0, grow = float(P.get("pygSpineRise", 0.0)), float(P.get("pygSpineBend", 0.0)), float(P.get("pygSpineGrow", 0.0))
    if S.get("tube") and S.get("spines"):
        for sd in S["spines"]:
            phi_k, Lk, L_full, hh, e = sd["phi"], sd["Lk"], sd["L_full"], sd["hh"], sd["e"]
            turn = (rise0 + bend0) if sd["rise"] is None else float(sd["rise"]); roll = 0.0 if sd["roll"] is None else float(sd["roll"])
            own = sd["rise"] is not None; grow_k = 0.0 if own else grow                        # an own spine's length is its len; the preset's grows by pygSpineGrow
            x0, y0 = S["base_pt"](phi_k); r0 = math.hypot(x0, y0); dx, dy = x0 / r0, y0 / r0        # the radial direction of this spine
            # the bump's half-width across the spine at each fraction of its length, measured on the actual margin curve
            fr = np.linspace(0.0, 0.97, 24); dphi = hh * (1 - fr ** (1.0 / e))
            pa = S["margin_xy"](phi_k + dphi); pb = S["margin_xy"](phi_k - dphi); half_w = 0.5 * np.hypot(*(pa - pb).T)
            blade = Lk > 0.5                                                                          # is there a flat bump to continue, or does the spine leave the margin itself?
            s_root = 0.8 if blade else 0.0                                                            # a short overlap (the grid's staircase shows along a long one)
            rx, ry = (x0 + dx * s_root * Lk, y0 + dy * s_root * Lk) if blade else (x0 - dx * 1.0, y0 - dy * 1.0)   # no blade: root 1 mm inside the margin
            zr = float(np.asarray(S["zfun"](np.array([rx]), np.array([ry]))).ravel()[0])
            ln_ = sd.get("lean_dir", 0.0)                                                             # 26 Oct 2026: the lobe's lean toward the rear
            if ln_ > 1e-6 and abs(phi_k - math.pi / 2) > 1e-6:
                th_ = math.atan2(dy, dx) + (ln_ if phi_k < math.pi / 2 else -ln_); dx, dy = math.cos(th_), math.sin(th_)
            yaw = math.degrees(math.atan2(dx, dy))                                                   # spine_solid's +y is rear; yaw + toward +x
            def bump_section(W, ts, ra, rb, Lk=Lk, x0=x0, y0=y0, dx=dx, dy=dy, fr=fr, half_w=half_w):   # over the bump: its width and height
                if Lk <= 0.5: return ra, rb
                sdist = ((W[:, 0] - x0) * dx + (W[:, 1] - y0) * dy); frac = sdist / Lk; inside = (frac < 0.97) & (ts < 0.8)
                hw = np.interp(np.clip(frac, 0, 0.97), fr, half_w); zz = np.asarray(S["zfun"](W[:, 0], W[:, 1])).ravel()
                return np.where(inside, np.maximum(hw, 0.35), ra), np.where(inside, np.maximum(0.5 * zz + 0.075, 0.35), rb)   # crest 0.15 proud, belly on the bed (the path runs 0.075 above the bump's middle)
            L = (1 - s_root) * Lk + (L_full - Lk) + grow_k * S["Lp"] + (0.0 if blade else 1.0)       # to the bump's planned tip, the length cut from the plan, the growth
            r_leave = max(float(half_w[-1]), 0.6) if blade else max(0.45 * S["margin"] + 0.5, 0.6)   # the bump's half-width where the solid takes over; or a spine's own root
            h = curved_spine(r_leave, 0.45, L, (rx, ry, 0.5 * zr + 0.075), yaw_deg=yaw, pitch_deg=0.0, bend_deg=turn, roll_deg=roll,
                             flare=1.0, plate=max(zr + 0.15, 0.8), profile=bump_section, slender=not blade)            # tangent to the bump, curving up from it; root ellipse from the bed to 0.15 above the crest
            out.append(("pygSpineHorns", [h, M.mirror_x(h)], "union"))
    if P["termSpine"] > 0.02:
        out.append(("termSpine", curved_spine(0.5 * S["margin"], 0.6, P["termSpine"] * S["Lp"], (0, 0.85 * S["Lp"], 0.5 * S["margin"]), 0, 0.0,
                                              float(P.get("termBend", 0.0)), 0.0)))
    if int(P.get("tailProngs", 0)) > 0:
        out.append(("tailProngs", prong(int(P["tailProngs"]), P["tailProngLen"] * S["Lp"], P["tailProngSplay"],
                                        P.get("tailProngWidth", 0.45) * S["margin"] + 0.6, 0.45, (0, 0.90 * S["Lp"], 0.5 * S["margin"]),
                                        yaw_deg=0.0, pitch_deg=5.0, stem_frac=P.get("tailProngStem", 0.0), center_bias=P.get("tailProngCenter", 1.0),
                                        curl_deg=P.get("tailProngCurl", 0.0), bend_deg=float(P.get("tailProngBend", 0.0)))))
    return out

def ports(P, S):
    """One port: the front joint plane at y = 0, wide (it meets the last segment's full plate)."""
    return dict(front=Port(y=0.0, rear=False, wide=True, halfwidth=S["wp"], ring_half=S["a"], kind="tail"))

def cells(P, lap=0.06, grid=GRID_TAIL):
    """b3 (axis) / a3 (right pleural field + margin) / c3 = mirror a3, on the shared plan."""
    S = plan(P); ua = S["u_a"]
    ring_out = lambda u, v: S["outline"](u * ua, v)
    pl_out = lambda u, v: S["outline"]((ua - lap) + 0.5 * (u + 1) * (1 - (ua - lap)), v)
    zb, det = _base(S)
    axis = M.heightfield_shell(ring_out, zb, S["t"], nu=grid[0] // 3 | 1, nv=grid[1], detail=det)
    pleura = M.heightfield_shell(pl_out, zb, S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False, detail=det)
    return dict(axis=axis, pleura_right=pleura, pleura_left=M.mirror_x(pleura))
