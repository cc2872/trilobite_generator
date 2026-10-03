"""
isopod_model/head_diagrams.py (3 Oct 2026): every head style as the site's sketch: plan and elevation, white hairlines
on black, with the handles. Drawn from shaped_head.diagram(), i.e. from the curves the solid is built from.

    python isopod_model/head_diagrams.py     -> photos/head_diagrams.png, source/head_diagrams.json (for the site)
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import schema, shaped_head as SH, head_styles as HS, presets as PR, body as BODY

LINE, DIM, INK = "#9a9a9a", "#4a4a4a", "#ffffff"

def face_for(name):
    P, face = HS.apply(PR.params("textured"), name); return BODY.face_of(P, face), P

def draw(ax, name, D, P, thorax=3):
    o = D["outline"]; ax.set_facecolor("k")
    ax.fill(np.r_[o[:, 0], -o[::-1, 0]], np.r_[o[:, 1], o[::-1, 1]], color="#0d0d0d", zorder=1)
    for s in (1, -1):
        ax.plot(s * o[:, 0], o[:, 1], color=INK if s > 0 else LINE, lw=1.1, zorder=3)
        ax.plot(s * D["glabella"][:, 0], D["glabella"][:, 1], color=LINE, lw=0.9, zorder=3)
        for k in ("eye", "brim", "boss"):
            if D[k] is not None: ax.plot(s * D[k][:, 0], D[k][:, 1], color=LINE if k != "brim" else DIM, lw=0.8, zorder=3, ls="-" if k != "brim" else (0, (3, 2)))
    w = 1.0 / D["size"]["width_vs_thorax"]; pit = 0.19                    # the thorax, to scale with this head's width
    for i in range(thorax):
        y0 = 0.04 + i * pit; wi = w * (1 - 0.05 * i); a = 0.28 * w
        ax.plot([-wi, -a, -a, a, a, wi, wi * 0.97, a, a, -a, -a, -wi * 0.97, -wi], [y0 + .02, y0, y0, y0, y0, y0 + .02, y0 + pit * .78, y0 + pit * .7, y0 + pit * .7, y0 + pit * .7, y0 + pit * .7, y0 + pit * .78, y0 + .02], color=DIM, lw=0.8, zorder=2)
    for k, (x, y) in D["handles"].items():
        ax.plot([x], [y], "o", ms=5.5, mfc="k", mec=INK, mew=1.3, zorder=5)
        ax.annotate(k.upper(), (x, y), xytext=(6, -3), textcoords="offset points", color=LINE, fontsize=5.5, zorder=5)
    # elevation, to the right, same scale, dorsal toward +x
    x0 = max(D["size"]["halfwidth"], 1.0) + 0.55; sec, pr = D["section"], D["profile"]
    ax.plot([x0, x0], [sec[0, 0], thorax * pit + 0.04], color=DIM, lw=0.8)
    ax.plot(x0 + pr[:, 1], pr[:, 0], color=DIM, lw=0.8, ls=(0, (3, 2))); ax.plot(x0 + sec[:, 1], sec[:, 0], color=INK, lw=1.1)
    zz = np.linspace(0, 1, 2 * thorax * 2 + 1) * thorax * pit + 0.04
    ax.plot(x0 + 0.30 + 0.05 * (np.arange(len(zz)) % 2), zz, color=DIM, lw=0.8)
    i = int(sec[:, 1].argmax()); ax.plot([x0 + sec[i, 1]], [sec[i, 0]], "o", ms=5.5, mfc="k", mec=INK, mew=1.3)
    ax.annotate("CROWN", (x0 + sec[i, 1], sec[i, 0]), xytext=(6, -3), textcoords="offset points", color=LINE, fontsize=5.5)
    pg = HS.STYLES[name]["p"]; ax.set_title(f"{name.upper()}" + (f" · GON P. {pg}" if pg else ""), color=INK, fontsize=8, loc="left", pad=3)
    ax.text(0, 1.0, f"W {D['size']['width_vs_thorax']:.2f} × THORAX   L {D['size']['length']:.2f} × HALF-WIDTH", transform=ax.transAxes, color=LINE, fontsize=5.5, va="top")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

if __name__ == "__main__":
    names = [n for n in HS.names() if n != "all_on"]; out = {}
    cols = 5; rows = -(-len(names) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.4 * rows), facecolor="k"); axs = axs.ravel()
    lims = []
    for a, n in zip(axs, names):
        f, P = face_for(n); D = SH.diagram(HS.outline(n), f); draw(a, n, D, P); lims.append((a.get_xlim(), a.get_ylim()))
        out[n] = dict(outline=HS.outline(n), face=f, handles={k: [round(v, 4) for v in xy] for k, xy in D["handles"].items()}, size=D["size"],
                      **{k: (None if D[k] is None else np.round(D[k], 4).tolist()) for k in ("outline_pts",) if False},
                      plan=np.round(D["outline"][::8], 4).tolist(), glabella=np.round(D["glabella"][::6], 4).tolist(),
                      eye=None if D["eye"] is None else np.round(D["eye"][::3], 4).tolist(), section=np.round(D["section"][::4], 4).tolist())
    x0 = min(l[0][0] for l in lims); x1 = max(l[0][1] for l in lims); y1 = max(l[1][0] for l in lims); y0 = min(l[1][1] for l in lims)
    for a in axs[:len(names)]: a.set_xlim(x0, x1); a.set_ylim(y1, y0)        # one scale for every panel: 1:1 across heads
    for a in axs[len(names):]: a.set_facecolor("k"); a.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.02, hspace=0.12)
    os.makedirs(os.path.join(HERE, "photos"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "photos", "head_diagrams.png"), dpi=110, facecolor="k")
    json.dump(dict(handles=SH.HANDLES, styles=out), open(os.path.join(HERE, "source", "head_diagrams.json"), "w"), indent=1)
    print("photos/head_diagrams.png, source/head_diagrams.json", len(names), "heads")
