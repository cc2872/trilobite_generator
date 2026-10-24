"""
anatomy/tail.py: the pygidium, joint-free. Same six functions as anatomy/thorax.py.
"""
import math
import numpy as np
import mesh as M
from fields import tail_halfwidth, furrow_amp
from anatomy.common import pitch, ring_top, smoothstep, plateau, trough, vault, spine_solid, curved_spine, prong, GRID_TAIL
from anatomy.port import Port

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
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P) * float(P.get("pygFurrowScale", 1.0))
    rib = float(P.get("pygRibRelief", 0.0)) * F                   # 24 Oct 2026: convex ribs between the pleural furrows
    n = int(P["pygRings"])
    def xmax(y):
        yy = np.clip(np.asarray(y, float) / Lp, 0, 1)
        return np.maximum(wp * np.clip(1 - yy ** nO, 0, 1) ** (1.0 / nO), 1.5)
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
    hphi_f = math.radians(13)
    phi_m = [math.radians(50 + 80 * (k + 0.5) / n_m) for k in range(n_m)]
    hphi_m = math.radians(80 / max(n_m, 1)) * 0.5 * 1.3
    def bump(phi, phi_k, hh, e): return np.clip(1 - np.abs(phi - phi_k) / hh, 0, 1) ** e
    def radial_extra(phi):
        ex = np.zeros_like(phi)
        if Lf > 0.5:
            for pf in (phi_f, math.pi - phi_f): ex = np.maximum(ex, Lf * bump(phi, pf, hphi_f, 1.5))
        for pk in phi_m:
            for pm in (pk, math.pi - pk): ex = np.maximum(ex, Lm * bump(phi, pm, hphi_m, 1.2))
        return ex
    def margin_pt(phi):
        x0, y0 = wp * se(np.cos(phi)), Lp * se(np.sin(phi))
        r0 = np.hypot(x0, y0); f = 1 + radial_extra(phi) / np.maximum(r0, 1e-6)
        return x0 * f, y0 * f
    PHI_MIN = math.radians(1.5)   # the margin never reaches the front line: at phi = 0 the outer grid column would lie
    def outline(u, v):            # along y = 0 and its corner triangles would be collinear (the spline fit used to hide it)
        phi = PHI_MIN + (math.pi - 2 * PHI_MIN) * 0.5 * (1 - u)
        mx, my = margin_pt(np.array([phi])); fx = u * (wp - 1.0)
        return float((1 - v) * fx + v * mx[0]), float(v * my[0])
    def outside(x, y):
        r = (np.abs(x / wp) ** nO + np.abs(y / Lp) ** nO) ** (1.0 / nO)
        Lmax = max(Lf if Lf > 0.5 else 0.0, Lm if n_m else 0.0, 1e-6)
        return np.clip((r - 1) * np.hypot(x, y) / np.maximum(r, 1e-6) / Lmax, 0, 1)
    def ov_pre(ax, y): return np.where(y > 0.35 * Lp, outside(ax, y), 0.0)   # the margin spines' region (no ribs there)
    def zfun(x, y):
        ax = np.abs(x); xm = xmax(y); yy = np.clip(y / Lp, 0, 1)
        nT = P["tailDomeExp"]
        tz = np.clip(1 - yy ** nT, 0, 1) ** (1 / nT) * 0.9 + 0.1
        z = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
        ay = a * (1 - 0.85 * np.clip(yy / ka, 0, 1))
        z += rise * plateau(x, ay) * (1 - smoothstep((Lax - 0.10) * Lp, (Lax + 0.10) * Lp, y))
        z -= F * trough(ax - (ay + 0.6), 0.9) * (y < (Lax + 0.05) * Lp)
        dy_k = Lp * ka / (n + 1.5)                                 # the furrow spacing
        for k in range(n):
            yk = Lp * ka * (k + 1) / (n + 1.5)
            fade = 1 - 0.6 * k / max(n, 1)
            z -= 0.7 * F * fade * trough(y - yk, 0.8) * plateau(x, ay + 1.5, 1.0)
            px = np.clip((ax - ay) / np.maximum(xm - ay, 1), 0, 1); ly = yk + px * 0.5 * Lp * ka / (n + 1.5)
            z -= 0.6 * F * fade * trough(y - ly, 0.9) * (ax > ay + 1.0)
            if rib > 0.01:                                         # the rib: a ridge with concave flanks and a crest, over the pleural field behind this furrow
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
    # the margin spines, for ornaments(): where each leaves the base margin and how far it runs as a bump (24 Oct 2026)
    spines = []                                                   # (phi, bump length as planned, full length asked for, hh, e)
    if Lf_full > 0.5: spines.append((phi_f, Lf, Lf_full, hphi_f, 1.5))
    for pk in phi_m: spines.append((pk, Lm, Lm_full, hphi_m, 1.2))
    def base_pt(phi): return wp * se(math.cos(phi)), Lp * se(math.sin(phi))
    def margin_xy(phi):
        mx, my = margin_pt(np.asarray(phi, float)); return np.c_[mx, my]
    return dict(outline=outline, zfun=zfun, t=t, Lp=Lp, wp=wp, a=a, margin=margin, u_a=a / max(wp, 1e-6),
                axis_len=Lax, border_frac=bwf, outline_exp=nO, spines=spines, base_pt=base_pt, margin_xy=margin_xy)

def shell(P, S, grid=GRID_TAIL):
    tail = M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=grid[0], nv=grid[1])
    env = M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1])
    return tail, env

def solid(P, S, grid=GRID_TAIL):
    return M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1], floor=0.0)

def ornaments(P, S):
    """Terminal spine, then the posterior prong. Forks and marginal spines are the margin itself (in the plan); with
    pygSpineRise / Bend / Grow (24 Oct 2026) each of them runs on as a curved solid spine that continues the bump."""
    out = []
    rise, bend, grow = float(P.get("pygSpineRise", 0.0)), float(P.get("pygSpineBend", 0.0)), float(P.get("pygSpineGrow", 0.0))
    if (abs(rise) > 1e-6 or abs(bend) > 1e-6 or grow > 0.01) and S.get("spines"):
        for phi_k, Lk, L_full, hh, e in S["spines"]:
            x0, y0 = S["base_pt"](phi_k); r0 = math.hypot(x0, y0); dx, dy = x0 / r0, y0 / r0        # the radial direction of this spine
            # the bump's half-width across the spine at each fraction of its length, measured on the actual margin curve
            fr = np.linspace(0.0, 0.97, 24); dphi = hh * (1 - fr ** (1.0 / e))
            pa = S["margin_xy"](phi_k + dphi); pb = S["margin_xy"](phi_k - dphi); half_w = 0.5 * np.hypot(*(pa - pb).T)
            blade = Lk > 0.5                                                                          # is there a flat bump to continue, or does the spine leave the margin itself?
            s_root = 0.8 if blade else 0.0                                                            # a short overlap (the grid's staircase shows along a long one)
            rx, ry = (x0 + dx * s_root * Lk, y0 + dy * s_root * Lk) if blade else (x0 - dx * 1.0, y0 - dy * 1.0)   # no blade: root 1 mm inside the margin
            zr = float(np.asarray(S["zfun"](np.array([rx]), np.array([ry]))).ravel()[0])
            yaw = math.degrees(math.atan2(dx, dy))                                                   # spine_solid's +y is rear; yaw + toward +x
            def bump_section(W, ts, ra, rb, Lk=Lk, x0=x0, y0=y0, dx=dx, dy=dy, fr=fr, half_w=half_w):   # over the bump: its width and height
                if Lk <= 0.5: return ra, rb
                sdist = ((W[:, 0] - x0) * dx + (W[:, 1] - y0) * dy); frac = sdist / Lk; inside = (frac < 0.97) & (ts < 0.8)
                hw = np.interp(np.clip(frac, 0, 0.97), fr, half_w); zz = np.asarray(S["zfun"](W[:, 0], W[:, 1])).ravel()
                return np.where(inside, np.maximum(hw, 0.35), ra), np.where(inside, np.maximum(0.5 * zz + 0.15, 0.35), rb)
            L = (1 - s_root) * Lk + (L_full - Lk) + grow * S["Lp"] + (0.0 if blade else 1.0)         # to the bump's planned tip, the length cut from the plan, the growth
            r_leave = max(float(half_w[-1]), 0.6) if blade else max(0.45 * S["margin"] + 0.5, 0.6)   # the bump's half-width where the solid takes over; or a spine's own root
            h = curved_spine(r_leave, 0.45, L, (rx, ry, 0.5 * zr), yaw_deg=yaw, pitch_deg=0.0, bend_deg=rise + bend, roll_deg=0.0,
                             flare=1.0, plate=max(zr, 0.8), profile=bump_section)                   # tangent to the bump, curving up from it
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
    axis = M.heightfield_shell(ring_out, S["zfun"], S["t"], nu=grid[0] // 3 | 1, nv=grid[1])
    pleura = M.heightfield_shell(pl_out, S["zfun"], S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False)
    return dict(axis=axis, pleura_right=pleura, pleura_left=M.mirror_x(pleura))
