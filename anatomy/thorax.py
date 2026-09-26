"""
anatomy/thorax.py: thoracic segment i, joint-free.

  plan(P, i)            outline(u, v), zfun(x, y) and the scalars (the BREP builder's, transcribed)
  shell(P, S, grid)     (shell, envelope): the plate the pin joint is cut into, doublure included
  solid(P, S, grid)     the solid wedge to the bed that a print-in-place joint is cut into
  ornaments(P, S)       [(name, solid)] grafted after the joint: the dorsal axial spine
  ports(P, S, i)        where this part meets its neighbours: front and rear Port
  cells(P, i)           the b2 / a2 / c2 cells for the site's controls

A joint gets the ports and the plate; it never reads the plan. Swap this module for another thorax as long as it
returns the same six things.
"""
import math
import numpy as np
import mesh as M
from fields import seg_halfwidth, pleural_spine_field, furrow_amp
from anatomy.common import pitch, smoothstep, plateau, trough, vault, spine_solid, GRID_SEG
from anatomy.port import Port

TIP_KEEP_MM = 3.0        # (used by the pin joint's band rule via Port.halfwidth; kept here for reference)

def pleura_lengths(P, i):
    """(inner run, outer blade, bend angle deg) of segment i's pleura in mm — the two-length profile in the sketch."""
    w = seg_halfwidth(P, i); a = P["axisFrac"] * w
    inner = P["fulcrum"] * (w - a); blade = (1 - P["fulcrum"]) * (w - a)
    bend = math.degrees(math.atan(P["pleuralSlope"] * P["relief"] / max(w - a, 1e-6)))
    return inner, blade, bend

def plan(P, i):
    """outline(u, v), zfun(x, y) and the scalars for segment i. Transcribed from trilobite.build_segment, minus tubercles."""
    t, c, h = P["wall"], P["clearance"], P["relief"]
    d = pitch(P); ovl = P["overlap"] * d; flap = max(ovl - 2.0, 1.0)
    w = seg_halfwidth(P, i)
    a = P["axisFrac"] * w
    margin = P["marginHeight"] * h
    rise = P["axisRise"] * h
    F = furrow_amp(P)
    sweep = P["tipSweep"] * d
    L0 = d + flap
    Ls = pleural_spine_field(P)[i] * w
    X_tip = w + Ls * math.cos(math.radians(P["spineSweep"]))
    S_sp = Ls * math.sin(math.radians(P["spineSweep"]))
    R_TIP = 0.7
    def q_of(x): return np.clip((np.abs(x) - a) / (w - a), 0, 1)
    def p_of(x): return np.clip((np.abs(x) - w) / max(X_tip - w, 1e-6), 0, 1) if Ls > 0.5 else np.zeros_like(np.asarray(x, float))
    def edges(x):
        q = q_of(x); p = p_of(x)
        yc = 0.5 * L0 + sweep * q ** 1.6 + S_sp * p ** 1.5
        root = smoothstep(0.0, 0.3, q)
        half = 0.5 * L0 * (1 - root) + 0.5 * min(P["bladeChord"] * d, L0) * root
        hb = half * (1 - P["tipTaper"] * q ** 2.5)
        hb = hb * (1 - p) ** 0.5 + R_TIP * p
        return yc - hb, yc + hb
    def outline(u, v):
        x = u * X_tip
        yf, yr = edges(np.array([abs(x)]))
        return x, float(yf[0] + v * (yr[0] - yf[0]))
    def zfun(x, y):
        ax = np.abs(x)
        z = margin + (h - margin) * vault(x / w, P)
        z += rise * plateau(x, a)
        z -= F * trough(ax - (a + 0.6), 0.9)
        z -= 0.7 * F * trough(y - d, 0.8) * plateau(x, a + 1.5, 1.0)
        yf0, _ = edges(x); px = q_of(x); ly = yf0 + 0.30 * d + px * 0.25 * d
        z -= 0.8 * F * trough(y - ly, 0.9) * (ax > a + 1.0)
        y0r = ovl - 1.0
        yc_r = 0.5 * (y0r + d); hr = max(0.5 * (d - y0r), 0.5)
        ring_hump = P["ringArch"] * h * np.clip(1 - ((y - yc_r) / hr) ** 2, 0, 1)
        yf_b, yr_b = edges(x); y0b = yf_b + ovl - 1.0
        yc_b = 0.5 * (y0b + yr_b); hb_b = np.maximum(0.5 * (yr_b - y0b), 0.5)
        blade_hump = P["bladeCamber"] * h * np.clip(1 - ((y - yc_b) / hb_b) ** 2, 0, 1)
        ax_w = plateau(x, a + 1.0, 1.5)
        window = smoothstep(ovl - 1.5, ovl + 0.5, y - yf_b)
        z += window * (ax_w * ring_hump + (1 - ax_w) * blade_hump)
        if Ls > 0.5:
            p = p_of(x)
            z = np.where(p > 0, np.maximum(margin * (1 - 0.55 * p), t + 0.6), z)
        yf, _ = edges(x)
        t_prev = t * (P["headWall"] if i == 0 else 1.0)
        drop = np.minimum(t_prev + c + 0.7 * F + 0.3, np.maximum(z - (t + 0.6), 0))
        z -= drop * (1 - smoothstep(ovl - 1.5, ovl, y - yf))
        return z
    return dict(outline=outline, zfun=zfun, edges=edges, t=t, w=w, a=a, d=d, ovl=ovl, margin=margin, rise=rise, h=h,
                X_tip=X_tip, u_a=a / X_tip, last=(i == int(P["segCount"]) - 1))

def _doublure(seg, S, P):
    """Vertical rim + inward lip along the blade margin, only on broad square-tipped pleurae (as in the BREP builder)."""
    yf_m, yr_m = (float(v[0]) for v in S["edges"](np.array([S["w"]])))
    rim_len = (yr_m - yf_m) - 1.0
    if rim_len > 2.0 and P["tipTaper"] < 0.3:
        t, w, margin = S["t"], S["w"], S["margin"]
        tools = []
        for s in (1, -1):
            tools.append(M.box(t, rim_len, max(margin - t, 1.0), at=(s * w, yf_m + 0.5, 0), align=("max" if s > 0 else "min", "min", "min")))
            tools.append(M.box(0.06 * P["width"], rim_len, t, at=(s * w, yf_m + 0.5, 0), align=("max" if s > 0 else "min", "min", "min")))
        seg = M.union(seg, *tools)
    return seg

def shell(P, S, grid=GRID_SEG):
    """The plate: height-field shell + doublure, and the under-envelope the joint clips its webs to."""
    seg = M.heightfield_shell(S["outline"], S["zfun"], S["t"], nu=grid[0], nv=grid[1])
    env = M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1])
    return _doublure(seg, S, P), env

def solid(P, S, grid=GRID_SEG):
    """The same plan as one solid wedge down to the bed (print-in-place joints)."""
    return M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1], floor=0.0)

def ornaments(P, S):
    """Solids unioned onto the part AFTER the joint, in this order. Plan frame."""
    out = []
    if P["axialSpine"] > 0.02:
        r = 0.45 * S["margin"]
        out.append(("axialSpine", spine_solid(0.6 * r + 0.6, 0.5, P["axialSpine"] * S["h"],
                                              (0, S["ovl"] + 0.45 * (S["d"] - S["ovl"]), S["h"] + S["rise"] - 1.0), 0, pitch_deg=60)))
    return out

def ports(P, S, i):
    """front: the joint plane at y = 0 (wide on segment 0: it meets the full-width head).
    rear: at y = pitch (wide and overspanned on the last segment: it meets the tail)."""
    return dict(front=Port(y=0.0, rear=False, wide=(i == 0), halfwidth=S["w"], ring_half=S["a"], kind="seg", shingle=S["ovl"]),
                rear=Port(y=S["d"], rear=True, wide=S["last"], halfwidth=S["w"], ring_half=S["a"], overspan=S["last"], kind="seg", shingle=S["ovl"]))

def cells(P, i, lap=0.06, grid=GRID_SEG):
    """The row as its three cells — ring (b2), right pleura (a2), left pleura (c2 = mirror a2) — each a closed shell
    on the shared plan. Roots overlap by `lap` of the plan width so a union of the three is the fused row's shell.
    This is the Flexi row's part decomposition; hooks between them are the v1.1 joint design."""
    S = plan(P, i); ua = S["u_a"]
    ring_out = lambda u, v: S["outline"](u * ua, v)
    pl_out = lambda u, v: S["outline"]((ua - lap) + 0.5 * (u + 1) * (1 - (ua - lap)), v)
    ring = M.heightfield_shell(ring_out, S["zfun"], S["t"], nu=grid[0] // 3 | 1, nv=grid[1])
    pleura = M.heightfield_shell(pl_out, S["zfun"], S["t"], nu=grid[0] // 2 | 1, nv=grid[1], symmetric=False)
    return dict(ring=ring, pleura_right=pleura, pleura_left=M.mirror_x(pleura))
