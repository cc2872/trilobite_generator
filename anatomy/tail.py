"""
anatomy/tail.py: the pygidium, joint-free. Same six functions as anatomy/thorax.py.
"""
import math
import numpy as np
import mesh as M
from fields import tail_halfwidth, furrow_amp
from anatomy.common import pitch, ring_top, smoothstep, plateau, trough, vault, spine_solid, prong, GRID_TAIL
from anatomy.port import Port

def plan(P):
    """outline(u, v), zfun(x, y) and scalars for the tail. Transcribed from trilobite.build_pygidium, minus tubercles."""
    t, c, h = P["wall"] * P["tailWall"], P["clearance"], P["relief"] * P["tailRelief"]
    Lp = P["pygFrac"] * P["length"]
    ovl = P["overlap"] * pitch(P)
    wp = tail_halfwidth(P)
    a = P["axisFrac"] * wp
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    n = int(P["pygRings"])
    def xmax(y):
        yy = np.clip(np.asarray(y, float) / Lp, 0, 1)
        return np.maximum(wp * np.sqrt(np.clip(1 - yy ** 2, 0, 1)), 1.5)
    Lf = P["pygSpine"] * Lp
    n_m = int(P["pygMarginal"]); Lm = P["pygMarginalLen"] * Lp
    sp = math.radians(P["pygSplay"])
    phi_f = math.atan(wp / Lp * math.tan(math.pi / 2 - sp)) if Lf > 0.5 else None
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
        x0, y0 = wp * np.cos(phi), Lp * np.sin(phi)
        r0 = np.hypot(x0, y0); f = 1 + radial_extra(phi) / np.maximum(r0, 1e-6)
        return x0 * f, y0 * f
    PHI_MIN = math.radians(1.5)   # the margin never reaches the front line: at phi = 0 the outer grid column would lie
    def outline(u, v):            # along y = 0 and its corner triangles would be collinear (the spline fit used to hide it)
        phi = PHI_MIN + (math.pi - 2 * PHI_MIN) * 0.5 * (1 - u)
        mx, my = margin_pt(np.array([phi])); fx = u * (wp - 1.0)
        return float((1 - v) * fx + v * mx[0]), float(v * my[0])
    def outside(x, y):
        r = np.sqrt((x / wp) ** 2 + (y / Lp) ** 2)
        Lmax = max(Lf if Lf > 0.5 else 0.0, Lm if n_m else 0.0, 1e-6)
        return np.clip((r - 1) * np.hypot(x, y) / np.maximum(r, 1e-6) / Lmax, 0, 1)
    def zfun(x, y):
        ax = np.abs(x); xm = xmax(y); yy = np.clip(y / Lp, 0, 1)
        nT = P["tailDomeExp"]
        tz = np.clip(1 - yy ** nT, 0, 1) ** (1 / nT) * 0.9 + 0.1
        z = margin * 0.6 + (h - margin * 0.6) * vault(x / xm, P) * tz
        ay = a * (1 - 0.85 * yy)
        z += rise * plateau(x, ay) * (1 - smoothstep(0.75 * Lp, 0.95 * Lp, y))
        z -= F * trough(ax - (ay + 0.6), 0.9) * (y < 0.9 * Lp)
        for k in range(n):
            yk = Lp * (k + 1) / (n + 1.5)
            fade = 1 - 0.6 * k / max(n, 1)
            z -= 0.7 * F * fade * trough(y - yk, 0.8) * plateau(x, ay + 1.5, 1.0)
            px = np.clip((ax - ay) / np.maximum(xm - ay, 1), 0, 1); ly = yk + px * 0.5 * Lp / (n + 1.5)
            z -= 0.6 * F * fade * trough(y - ly, 0.9) * (ax > ay + 1.0)
        if P["borderWidth"] > 0.005:
            bw = P["borderWidth"] * wp
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
    return dict(outline=outline, zfun=zfun, t=t, Lp=Lp, wp=wp, a=a, margin=margin, u_a=a / max(wp, 1e-6))

def shell(P, S, grid=GRID_TAIL):
    tail = M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=grid[0], nv=grid[1])
    env = M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1])
    return tail, env

def solid(P, S, grid=GRID_TAIL):
    return M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1], floor=0.0)

def ornaments(P, S):
    """Terminal spine, then the posterior prong. Forks and marginal spines are the margin itself (in the plan)."""
    out = []
    if P["termSpine"] > 0.02:
        out.append(("termSpine", spine_solid(0.5 * S["margin"], 0.6, P["termSpine"] * S["Lp"], (0, 0.85 * S["Lp"], 0.5 * S["margin"]), 0)))
    if int(P.get("tailProngs", 0)) > 0:
        out.append(("tailProngs", prong(int(P["tailProngs"]), P["tailProngLen"] * S["Lp"], P["tailProngSplay"],
                                        P.get("tailProngWidth", 0.45) * S["margin"] + 0.6, 0.45, (0, 0.90 * S["Lp"], 0.5 * S["margin"]),
                                        yaw_deg=0.0, pitch_deg=5.0, stem_frac=P.get("tailProngStem", 0.0), center_bias=P.get("tailProngCenter", 1.0),
                                        curl_deg=P.get("tailProngCurl", 0.0))))
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
