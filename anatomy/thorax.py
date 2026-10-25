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
from anatomy.common import pitch, smoothstep, plateau, trough, vault, spine_solid, curved_spine, GRID_SEG
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
                X_tip=X_tip, u_a=a / X_tip, last=(i == int(P["segCount"]) - 1), i=i, n=int(P["segCount"]),
                axial=dict(L=float(P["axialSpine"]) * h, pitch=float(P.get("axialPitch", 60.0)), bend=float(P.get("axialBend", 0.0)), grad=float(P.get("axialGrad", 0.0))),   # the row's own P reaches ornaments() this way (rows model: _P_row)
                horn=dict(L=float(P.get("pleuralHorn", 0.0)) * w, turn=float(P.get("pleuralHornPitch", 45.0)) + float(P.get("pleuralHornBend", 0.0)),
                          roll=float(P.get("pleuralHornRoll", 0.0)), yaw_off=float(P.get("_hornYaw", 0.0)), grad=float(P.get("pleuralHornGrad", 0.0))))

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
    frac = min(max(S.get("i", 0) / max(S.get("n", 1) - 1, 1), 0.0), 1.0)      # 0 on the first segment, 1 on the last
    A = S.get("axial") or dict(L=float(P["axialSpine"]) * S["h"], pitch=float(P.get("axialPitch", 60.0)), bend=float(P.get("axialBend", 0.0)), grad=float(P.get("axialGrad", 0.0)))
    if A["L"] > 0.02 * S["h"]:
        r = 0.45 * S["margin"]; L = A["L"] * (1.0 + A["grad"] * frac)
        out.append(("axialSpine", curved_spine(0.6 * r + 0.6, 0.5, L, (0, S["ovl"] + 0.45 * (S["d"] - S["ovl"]), S["h"] + S["rise"] - 1.0), 0,
                                               pitch_deg=A["pitch"], bend_deg=A["bend"], roll_deg=0.0)))   # 24 Oct 2026: a curve; defaults = the 60 deg rod
    Hn = S.get("horn") or dict(L=float(P.get("pleuralHorn", 0.0)) * S["w"], turn=float(P.get("pleuralHornPitch", 45.0)) + float(P.get("pleuralHornBend", 0.0)),
                               roll=float(P.get("pleuralHornRoll", 0.0)), yaw_off=0.0, grad=float(P.get("pleuralHornGrad", 0.0)))
    if Hn["L"] > 0.02 * S["w"] and "X_tip" in S:                           # 24 Oct 2026: a horn continuing each pleural blade (Koneprusia)
        L = Hn["L"] * (1.0 + Hn["grad"] * frac)
        # The horn leaves the plate with the section the blade has at its tip (chord across, height up) and tapers from
        # there; while its centreline is still over the plate its section IS the blade's, so nothing of the blade runs
        # on underneath. The root is 2.5 mm inboard of the tip, at the blade's mid-height, sloping as the blade does.
        # (A root at the tip was a 4.7 mm bulb on a 1.8 mm blade kinked off it at 55 deg -- the junction close-ups, 11:10.)
        xt = float(S["X_tip"]); xs = np.linspace(0.45 * S["w"], xt, 120); yf, yr = S["edges"](xs)
        chord = np.asarray(yr) - np.asarray(yf); ymid = 0.5 * (np.asarray(yf) + np.asarray(yr)); zs = np.asarray(S["zfun"](xs, ymid)).ravel()
        # The blade's MATERIAL along the mid-chord (24 Oct 2026, the spine audit): solid to the bed ahead of the rear joint
        # plane; behind it (the swept tip usually is) a plate S["t"] thick over the next part's lip. The section is that
        # material -- centre at its middle, half-height to its faces -- not bed-to-top, which filled the hollow under the
        # tip plate by 4-5 mm on every Koneprusia segment. The crest stands 0.15 proud; the belly stays on the underside.
        t_plate = float(S.get("t", 2.0)); d_ = float(S["d"])
        vs = np.linspace(0.05, 0.97, 7); YV = np.asarray(yf)[:, None] + vs[None, :] * chord[:, None]       # across the chord: the plate rises toward
        TV = np.asarray(S["zfun"](np.repeat(xs, len(vs)), YV.ravel())).reshape(len(xs), len(vs))          # the rear edge and its underside with it
        UV = np.where(YV > d_, np.maximum(TV - t_plate, 0.0), 0.0)
        U = UV.max(axis=1); T = TV.max(axis=1)
        win = max(1, int(round(2.0 / max(float(xs[1] - xs[0]), 1e-6))))    # the section at one x reaches ~2 mm inboard (the chord lies across the yaw):
        U = np.array([U[max(0, i - win):i + win + 1].max() for i in range(len(U))])   # take the plate's highest underside over that reach
        T = np.array([T[max(0, i - win):i + win + 1].max() for i in range(len(T))])
        bottom = U + 0.5; crest = T + 0.15                                   # the section: 0.5 above the material's highest underside, crest 0.15 proud
        hb = np.maximum(0.5 * (crest - bottom), 0.3); zc = 0.5 * (crest + bottom)
        x0 = max(xt - 2.5, 0.45 * S["w"]); y0 = float(np.interp(x0, xs, ymid)); z0 = float(np.interp(x0, xs, zc)); hb0 = float(np.interp(x0, xs, hb))
        r_leave = max(0.5 * float(np.interp(xt - 0.3, xs, chord)), 0.6)   # the blade's half-chord where it ends
        slope = float(np.degrees(np.arctan2(float(zc[-1]) - z0, max(xt - x0, 1e-6))))
        def blade_section(W, ts, ra, rb):                                   # over the plate: the blade's own chord and material height
            x = np.abs(W[:, 0]); inside = (x < xt) & (ts < 0.8)
            ra2 = np.where(inside, np.maximum(np.interp(x, xs, 0.5 * chord), 0.35), ra)
            rb2 = np.where(inside, np.maximum(np.interp(x, xs, hb), 0.35), rb)           # crest 0.15 proud of the blade (flush was a tangent meeting: sliver faces), belly above the underside
            return ra2, rb2
        h = curved_spine(r_leave, 0.45, L, (x0, y0, z0), yaw_deg=90.0 - float(P["spineSweep"]) + Hn["yaw_off"], pitch_deg=slope,
                         bend_deg=Hn["turn"] - slope, roll_deg=Hn["roll"],
                         flare=1.0, plate=max(2.0 * hb0, 0.8), profile=blade_section)
        out.append(("pleuralHorns", [h, M.mirror_x(h)], "union"))
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
