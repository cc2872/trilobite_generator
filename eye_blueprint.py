"""eye_blueprint.py — the eye primitive as a drawing: plan, outward (lateral) view, transverse section, and the lens
lattice in each. Drawn analytically from the parameter set, so it shows what the primitive WILL build, on the same
white-on-black sheet as blueprint.py. Usage: python eye_blueprint.py [preset] -> out10/<preset>/eye_blueprint.png"""
import sys, json, math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
INK, BG, DIM = "#ffffff", "#000000", "#9a9a9a"

def eye_spec(P):
    """Primitive parameters (fossil table, 8 Sep 2026). Missing keys fall back to the Phacops sp. scan values."""
    return dict(R=1.0, height=P.get("eyeHeight", 1.7), slope=math.radians(P.get("eyeSlope", 15)), shade=P.get("eyeShade", 0.0),
                arc=math.radians(P.get("eyeArc", 110)), elong=P.get("eyeElong", 1.3), lensD=P.get("lensD", 0.16), lensGap=P.get("lensGap", 0.3),
                lensRise=P.get("lensRise", 0.35))

def lattice(S):
    """Lens centres on the band in (theta, s): hex packing, files staggered by half a pitch. Returns (theta, s, r, z)."""
    pitch = S["lensD"] * (1 + S["lensGap"]); H = S["height"] * S["R"]
    rows = int(H / (0.87 * pitch)); out = []
    for i in range(rows):
        z = (i + 0.5) * 0.87 * pitch; r = S["R"] + (H - z) * math.tan(S["slope"])
        n = max(1, int(S["arc"] * r / pitch)); off = 0.5 * (i % 2)
        for k in range(n):
            th = -S["arc"] / 2 + (k + off + 0.5) * S["arc"] / (n + 0.5)
            if abs(th) <= S["arc"] / 2: out.append((th, z, r, z))
    return np.array(out), pitch, rows

def draw(P, name, path):
    S = eye_spec(P); R, H, E = S["R"], S["height"] * S["R"], S["elong"]
    L, pitch, rows = lattice(S); th, z, r = L[:, 0], L[:, 1], L[:, 2]
    rb = R + H * math.tan(S["slope"])                      # band radius at the base (leans outward going down)
    dl = S["lensD"] / 2
    fig = plt.figure(figsize=(16.5, 11.7), facecolor=BG)
    gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.90, bottom=0.06, wspace=0.18, hspace=0.28)
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1); bg.set_facecolor(BG); bg.set_xticks([]); bg.set_yticks([])
    for gx in np.linspace(0, 1, 41): bg.axvline(gx, color="#1c1c1c", lw=0.4)
    for gy in np.linspace(0, 1, 29): bg.axhline(gy, color="#1c1c1c", lw=0.4)
    fig.text(0.04, 0.955, f"EYE PRIMITIVE · {name.upper()} · revolved profile, scaled {E:.1f}× along the body, trimmed to {math.degrees(S['arc']):.0f}°, lens lattice on the band", color=INK, fontsize=10, family="monospace")
    fig.text(0.04, 0.935, f"R = cap radius (unit) · band {S['height']:.1f} R tall, leaning {math.degrees(S['slope']):.0f}° from vertical · shade {S['shade']:.2f} R · lens D {S['lensD']:.2f} R, gap {S['lensGap']:.1f} D → {rows} rows, {len(L)} lenses on this half-eye", color=DIM, fontsize=8, family="monospace")
    # ---------- plan (from above): cap ellipse, base ellipse, band arc, lens dots projected
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Ellipse((0, 0), 2 * R, 2 * R * E, fill=False, color=INK, lw=1.0))
    ax.add_patch(Ellipse((0, 0), 2 * rb, 2 * rb * E, fill=False, color=DIM, lw=0.6, ls=(0, (3, 2))))
    t = np.linspace(-S["arc"] / 2, S["arc"] / 2, 80)
    ax.plot(rb * np.cos(t), rb * E * np.sin(t), color=INK, lw=2.2)
    ax.scatter(r * np.cos(th), r * E * np.sin(th), s=6, color=INK, alpha=0.6, lw=0)
    ax.plot([0, 1.3 * rb * math.cos(S["arc"] / 2)], [0, 1.3 * rb * E * math.sin(S["arc"] / 2)], color=DIM, lw=0.4)
    ax.plot([0, 1.3 * rb * math.cos(-S["arc"] / 2)], [0, 1.3 * rb * E * math.sin(-S["arc"] / 2)], color=DIM, lw=0.4)
    ax.text(0, R * E + 0.12, "palpebral cap", color=DIM, fontsize=7, family="monospace", ha="center")
    ax.text(rb + 0.1, 0, "visual band\n(thick arc)", color=INK, fontsize=7, family="monospace", va="center")
    ax.text(1.32 * rb * math.cos(S["arc"] / 2), 1.32 * rb * E * math.sin(S["arc"] / 2), f"eyeArc {math.degrees(S['arc']):.0f}°", color=DIM, fontsize=7, family="monospace")
    ax.annotate("", xy=(0, 0), xytext=(0, -R * E), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7)); ax.text(0.05, -0.5 * R * E, f"{E:.1f} R", color=INK, fontsize=7, family="monospace")
    ax.annotate("", xy=(0, 0), xytext=(R, 0), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7)); ax.text(0.5 * R, 0.06, "R", color=INK, fontsize=7, family="monospace", ha="center")
    ax.set_xlim(-1.8, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_title("PLAN · from above · outward is +x, body axis is y", color=INK, fontsize=8, family="monospace", loc="left")
    # ---------- outward (lateral) view: looking at the band from +x — trapezoid drum with the hex lattice
    ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    yt = R * E * math.sin(S["arc"] / 2); yb_ = rb * E * math.sin(S["arc"] / 2)
    ax.plot([-yt, yt, yb_, -yb_, -yt], [H, H, 0, 0, H], color=INK, lw=1.0)
    if S["shade"] > 0.005:
        ax.plot([-yt - S["shade"], yt + S["shade"]], [H + 0.05, H + 0.05], color=INK, lw=2.0)
    ax.plot([-yb_ - 0.6, yb_ + 0.6], [0, 0], color=DIM, lw=0.5, ls=(0, (4, 3)))
    vis = np.cos(th) > 0.05
    for yy, zz, rr, tt in zip(r[vis] * E * np.sin(th[vis]), z[vis], r[vis], th[vis]):
        ax.add_patch(Ellipse((yy, zz), 2 * dl * max(0.25, abs(math.cos(tt))) * 1.0 * (E if False else 1), 2 * dl, fill=False, color=INK, lw=0.5, alpha=0.85))
    ax.annotate("", xy=(yb_ + 0.35, 0), xytext=(yb_ + 0.35, H), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7)); ax.text(yb_ + 0.42, H / 2, f"{S['height']:.1f} R", color=INK, fontsize=7, family="monospace", va="center")
    ax.text(0, -0.35, "cheek (librigena)", color=DIM, fontsize=7, family="monospace", ha="center")
    ax.text(0, H + 0.22, "brim / palpebral edge" if S["shade"] > 0.005 else "palpebral edge", color=DIM, fontsize=7, family="monospace", ha="center")
    ax.set_xlim(-2.4, 2.8); ax.set_ylim(-0.7, H + 0.7); ax.set_title(f"OUTWARD VIEW · from +x · {rows} rows, files staggered ½ pitch (hex packing)", color=INK, fontsize=8, family="monospace", loc="left")
    # ---------- transverse section (plane y = 0): cap, leaning band with lens caps, cheek
    ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([-1.6, -R, -R * 0.98, 0, R, R], [-0.15, -0.05, H, H + 0.02, H, H], color=INK, lw=1.0)   # inner side + cap top (schematic: cheek to cap)
    xs = np.linspace(0, 1, 40); ax.plot(R + (1 - xs) * 0 + xs * 0, H * xs, color=INK, lw=0.0)
    band_x = np.array([R, rb]); band_z = np.array([H, 0]); ax.plot(band_x, band_z, color=INK, lw=2.2)
    if S["shade"] > 0.005: ax.plot([R, R + S["shade"]], [H, H], color=INK, lw=2.0)
    ax.plot([rb, rb + 0.9], [0, -0.2], color=INK, lw=1.0); ax.plot([-1.6, rb + 1.0], [-0.3, -0.3], color=DIM, lw=0.4, ls=(0, (4, 3)))
    # lens caps along the profile: the file at theta = 0
    sel = np.abs(th) < 0.5 * pitch / R
    for zz, rr in zip(z[sel], r[sel]):
        nx, nz = math.cos(S["slope"]), math.sin(S["slope"])                       # band normal (outward, slightly down)
        ang = np.linspace(-math.pi / 2, math.pi / 2, 20)
        px = rr + S["lensRise"] * dl * np.cos(ang) * nx * 2 - dl * np.sin(ang) * nz * 0; pz = zz + dl * np.sin(ang) * 1.0
        ax.plot(rr + (S["lensRise"] * dl) * np.cos(ang) * 1.0, zz + dl * np.sin(ang), color=INK, lw=0.6)
    ax.annotate("", xy=(rb + 0.5, 0), xytext=(rb + 0.5, H), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7)); ax.text(rb + 0.56, H / 2, f"h {S['height']:.1f} R", color=INK, fontsize=7, family="monospace", va="center")
    ax.plot([R, R], [H, 0], color=DIM, lw=0.4, ls=(0, (2, 2))); ax.text(R + 0.08, 0.15, f"{math.degrees(S['slope']):.0f}°", color=DIM, fontsize=7, family="monospace")
    ax.text(-1.5, -0.45, "glabella side", color=DIM, fontsize=7, family="monospace"); ax.text(rb + 0.2, -0.45, "outward", color=DIM, fontsize=7, family="monospace")
    ax.text(0, H + 0.2, "eye axis (revolve)", color=DIM, fontsize=7, family="monospace", ha="center"); ax.plot([0, 0], [-0.3, H + 0.1], color=DIM, lw=0.4, ls=(0, (1, 3)))
    ax.set_xlim(-1.9, rb + 1.5); ax.set_ylim(-0.7, H + 0.6); ax.set_title("SECTION · transverse through the eye axis · lens caps on the band", color=INK, fontsize=8, family="monospace", loc="left")
    # ---------- the band unrolled (theta, s): the lattice as the builder sees it
    ax = fig.add_subplot(gs[1, 0:2]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    arcl = S["arc"] * rb; ax.plot([-arcl / 2, arcl / 2, arcl / 2 * R / rb, -arcl / 2 * R / rb, -arcl / 2], [0, 0, H, H, 0], color=INK, lw=0.8)
    for tt, zz, rr in zip(th, z, r): ax.add_patch(Circle((tt * rr, zz), dl, fill=False, color=INK, lw=0.5))
    ax.text(-arcl / 2, -0.25, f"θ·r along the arc → {arcl:.2f} R at the base", color=DIM, fontsize=7, family="monospace")
    ax.text(arcl / 2 + 0.1, H / 2, "s up the band", color=DIM, fontsize=7, family="monospace", va="center")
    ax.text(-arcl / 2, H + 0.15, f"pitch {pitch:.3f} R = D (1 + gap) · rows = h / (0.87 pitch) = {rows} · files per row = arc·r / pitch", color=DIM, fontsize=7, family="monospace")
    ax.set_xlim(-arcl / 2 - 0.4, arcl / 2 + 1.4); ax.set_ylim(-0.5, H + 0.4); ax.set_title("BAND UNROLLED · (θ, s) · what lattice() writes into the radius before the spline fit", color=INK, fontsize=8, family="monospace", loc="left")
    # ---------- placement on the cephalon
    ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    try:
        from fields import head_halfwidth
        wh = head_halfwidth(P); Lc = P["cephFrac"] * P["length"]; par = P["cephParallel"] * Lc; nO = P["headOutlineExp"]
        u = np.linspace(-1, 1, 200); yf = -par - (Lc - par) * np.clip(1 - np.abs(u) ** nO, 0, 1) ** (1 / nO)
        ax.plot(u * wh, yf, color=INK, lw=0.9); ax.plot([-wh, -wh, wh, wh], [-par, 0, 0, -par], color=INK, lw=0.9)
        a = P["axisFrac"] * wh; g = np.linspace(0, -0.9 * Lc, 50); gw = a * (1 + (P["glabInflate"] - 1) * np.clip(-g / Lc, 0, 1))
        ax.plot(np.r_[gw, -gw[::-1]], np.r_[g, g[::-1]], color=DIM, lw=0.6)
        from eyes import eye_geometry
        G = eye_geometry(P); eR = G["eR"]
        for sgn in (1, -1):
            ax.add_patch(Ellipse((sgn * G["xe"], G["ye"]), 2 * eR, 2 * eR * E, fill=False, color=INK, lw=1.0))
            tt = np.linspace(-S["arc"] / 2, S["arc"] / 2, 60)
            ax.plot(sgn * (G["xe"] + eR * rb * np.cos(tt)), G["ye"] + eR * rb * E * np.sin(tt), color=INK, lw=2.0)
        ax.text(G["xe"] + 1.5 * eR, G["ye"], f"lat {G['xe'] / wh:.2f} wh\npos {-G['ye'] / Lc:.2f} Lc\nR {eR:.1f} mm", color=DIM, fontsize=7, family="monospace", va="center")
        ax.set_xlim(-wh * 1.15, wh * 1.6); ax.set_ylim(-Lc * 1.1, 6)
    except Exception as ex:
        ax.text(0.1, 0.5, f"placement: {ex}", color=DIM, fontsize=7, family="monospace", transform=ax.transAxes)
    ax.set_title("PLACEMENT · on the cephalon · both eyes, front at the bottom", color=INK, fontsize=8, family="monospace", loc="left")
    fig.text(0.98, 0.018, "Claire Choi · Cornell · eye primitive, drawn from parameters (not yet built)", color=INK, fontsize=8, ha="right", family="monospace")
    fig.savefig(path, dpi=110, facecolor=BG); plt.close(fig); return path

if __name__ == "__main__":
    import schema, os
    name = sys.argv[1] if len(sys.argv) > 1 else "phacopida"
    d = json.load(open(f"presets/{name}.json")); P = schema.coerce(d["params"], base=schema.table_defaults())
    extra = dict(phacopida=dict(eyeHeight=1.7, eyeSlope=15, eyeShade=0.0, eyeArc=110, eyeElong=1.3, lensD=0.16, lensGap=0.3),
                 asaphida=dict(eyeHeight=1.0, eyeSlope=45, eyeShade=0.0, eyeArc=160, eyeElong=2.2, lensD=0.05, lensGap=0.0),
                 odontopleurida=dict(eyeHeight=2.2, eyeSlope=0, eyeShade=0.15, eyeArc=180, eyeElong=1.0, lensD=0.10, lensGap=0.3)).get(name, {})
    P = dict(P, **extra); os.makedirs(f"out10/{name}", exist_ok=True)
    print(draw(P, name, f"out10/{name}/eye_blueprint.png"))
