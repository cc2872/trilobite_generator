"""Contact sheet of globe frames from out/frames.json + out/sites.json, drawn the way the page will draw them
(Fibonacci dots: land, shelf sea, high land; fossil localities alive at that frame). For checking only."""
import json, sys, base64, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "web", "static", "earth")
F = json.load(open(f"{OUT}/frames.json")); S = json.load(open(f"{OUT}/sites.json"))
N = F["meta"]["lattice"]
i = np.arange(N); LAT = np.degrees(np.arcsin(1 - (2*i + 1)/N)); LON = np.degrees((i*np.pi*(3 - 5**.5)) % (2*np.pi) - np.pi)
bits = lambda s: np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8), bitorder="little")[:N].astype(bool)


def ortho(lat, lon, lat0, lon0):
    a, b, a0, b0 = map(np.radians, (lat, lon, lat0, lon0))
    x = np.cos(a)*np.sin(b - b0); y = np.cos(a0)*np.sin(a) - np.sin(a0)*np.cos(a)*np.cos(b - b0)
    vis = np.sin(a0)*np.sin(a) + np.cos(a0)*np.cos(a)*np.cos(b - b0) > 0
    return x, y, vis


def panel(ax, t, lat0=None, lon0=None):
    fr = F["frames"][str(t)]
    land, shelf, high = bits(fr["land"]), bits(fr["shelf"]), bits(fr["high"])
    pts = [(s[10][str(t)], s[5]) for s in S["sites"] if str(t) in s[10]]
    P = np.array([p for p, _ in pts]) if pts else np.zeros((0, 2)); W = np.array([n for _, n in pts])
    if lat0 is None:                                  # face the fossils
        v = np.c_[np.cos(np.radians(P[:, 0]))*np.cos(np.radians(P[:, 1])), np.cos(np.radians(P[:, 0]))*np.sin(np.radians(P[:, 1])), np.sin(np.radians(P[:, 0]))]
        m = (v*W[:, None]).sum(0); lat0 = np.degrees(np.arcsin(m[2]/np.linalg.norm(m))); lon0 = np.degrees(np.arctan2(m[1], m[0]))
    ax.set_facecolor("black"); ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec="#666", lw=.8))
    for mask, c, s in ((shelf, "#2c4a5a", 3), (land & ~high, "#cfcfcf", 3), (high, "#ffffff", 4)):
        x, y, vis = ortho(LAT[mask], LON[mask], lat0, lon0); ax.scatter(x[vis], y[vis], s=s, c=c, lw=0)
    if len(P):
        x, y, vis = ortho(P[:, 0], P[:, 1], lat0, lon0)
        ax.scatter(x[vis], y[vis], s=np.clip(np.sqrt(W[vis])*3, 3, 40), facecolors="none", edgecolors="#ff7a1a", lw=.6)
    for name, la, lo in fr["continents"]:
        x, y, vis = ortho(np.array([la]), np.array([lo]), lat0, lon0)
        if vis[0]: ax.text(x[0], y[0], name.upper(), color="#9fd3ff", fontsize=6.5, ha="center", va="center", family="monospace")
    for la in (-60, -30, 0, 30, 60):                 # graticule
        lo = np.linspace(-180, 180, 361); x, y, vis = ortho(np.full_like(lo, la), lo, lat0, lon0)
        x[~vis] = np.nan; ax.plot(x, y, c="#333", lw=.4 if la else .8)
    ax.set_title(f"{t} Ma   {len(P)} localities", color="w", fontsize=9, family="monospace")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal"); ax.axis("off")


if __name__ == "__main__":
    ts = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [520, 500, 480, 460, 440, 420, 400, 380, 360, 330, 300, 260]
    cols = 4; rows = (len(ts) + cols - 1)//cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4.2), facecolor="black")
    for ax, t in zip(np.ravel(axs), ts): panel(ax, t)
    for ax in np.ravel(axs)[len(ts):]: ax.axis("off")
    out = sys.argv[3] if len(sys.argv) > 3 else f"{OUT}/check_frames.png"
    fig.savefig(out, dpi=130, facecolor="black", bbox_inches="tight"); print(out)
