"""
parts.py — the part library: crop anatomical patches from the atlases (heightfields in body coordinates)
and cut protrusion solids (spines) from the meshes.

Patches (all normalised: t along the part 0..1, u across -1..1, z / local relief):
  head, glabella, eye, segment (one representative thoracic segment), pygidium, brim (harpetids)
Solids: pleural spine (one segment's pair), genal spines — closed meshes with an attachment frame.
"""
import numpy as np, json

LANDMARKS = {   # s-fractions in atlas coordinates: rear of head, rear of thorax, thoracic furrows
    "olenoides": dict(s_h=0.347, s_t=0.807, furrows=[0.347,0.414,0.469,0.525,0.586,0.629,0.684,0.727,0.770,0.807], axis=0.25),
    "proetida":  dict(s_h=0.36,  s_t=0.84,  furrows=None, axis=0.22),
    "harpetid":  dict(s_h=0.35,  s_t=0.88,  furrows=None, axis=0.20),
    "phacops":   dict(s_h=0.29,  s_t=0.81,  furrows=None, axis=0.30),
}

def load_atlas(name):
    d = np.load(f"{name}_atlas.npz"); meta = json.loads(str(d["meta"]))
    return dict(s=d["s"], u=d["u"], Z=np.nan_to_num(d["Z"], nan=0.0), W=d["W"], crest=d["crest"], meta=meta)

def crop(at, s0, s1, u0=-1, u1=1, nt=48, nu=101):
    """Resample the atlas over [s0,s1]x[u0,u1] onto a (t,u) grid, z renormalised to the crop's own relief."""
    ts = np.linspace(s0, s1, nt); us = np.linspace(u0, u1, nu)
    i = np.clip(np.searchsorted(at["s"], ts) - 1, 0, len(at["s"]) - 2); j = np.clip(np.searchsorted(at["u"], us) - 1, 0, len(at["u"]) - 2)
    P = at["Z"][np.ix_(i, j)]; lo, hi = np.percentile(P, 2), np.percentile(P, 99.5)
    return dict(t=ts, u=us, z=(P - lo) / max(hi - lo, 1e-6), s_range=(s0, s1), u_range=(u0, u1), relief=float(hi - lo))

def find_eye(at, lm):
    """Eye = highest point in the cheek band of the head (u > 0 side), excluding the axis."""
    s0, s1 = 0.25 * lm["s_h"], 0.85 * lm["s_h"]; u0, u1 = lm["axis"] + 0.05, 0.85
    si = (at["s"] >= s0) & (at["s"] <= s1); ui = (at["u"] >= u0) & (at["u"] <= u1)
    sub = at["Z"][np.ix_(si, ui)]; k = np.unravel_index(np.argmax(sub), sub.shape)
    return float(at["s"][si][k[0]]), float(at["u"][ui][k[1]])

def thoracic_furrows(at, lm):
    if lm["furrows"]: return lm["furrows"]
    f = [x for x in at["meta"]["furrows_s"] if lm["s_h"] - 0.01 <= x <= lm["s_t"] + 0.01]
    return [lm["s_h"]] + [x for x in f if lm["s_h"] + 0.02 < x < lm["s_t"] - 0.02] + [lm["s_t"]]

def extract(name):
    at = load_atlas(name); lm = LANDMARKS[name]; lib = {}
    lib["head"] = crop(at, 0.0, lm["s_h"])
    lib["glabella"] = crop(at, 0.02, lm["s_h"], -lm["axis"] - 0.1, lm["axis"] + 0.1)
    es, eu = find_eye(at, lm); ds, du = 0.14 * lm["s_h"], 0.18
    lib["eye"] = crop(at, es - ds, es + ds, eu - du, eu + du); lib["eye"]["centre_su"] = (es, eu)
    fur = thoracic_furrows(at, lm); m = len(fur) // 2
    lib["segment"] = crop(at, fur[m - 1], fur[m]); lib["segment"]["furrows"] = fur
    lib["pygidium"] = crop(at, lm["s_t"], 1.0)
    if name == "harpetid": lib["brim"] = crop(at, 0.0, lm["s_h"] + 0.3, 0.55, 1.0)
    lib["meta"] = dict(at["meta"], landmarks=lm, n_segments=len(fur) - 1)
    return lib

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = ["olenoides", "proetida", "harpetid", "phacops"]; cols = ["head", "glabella", "eye", "segment", "pygidium"]
    fig, axes = plt.subplots(len(names), len(cols) + 1, figsize=(19, 12)); store = {}
    for r, n in enumerate(names):
        lib = extract(n); store[n] = lib
        for c, part in enumerate(cols):
            p = lib[part]; ax = axes[r, c]
            ax.imshow(p["z"].T, origin="lower", cmap="gray", vmin=0, vmax=1, extent=[p["s_range"][0], p["s_range"][1], p["u_range"][0], p["u_range"][1]], aspect="auto")
            ax.set_title(f"{n} · {part}" + (f" (segment {len(p['furrows'])//2} of {len(p['furrows'])-1})" if part == "segment" else ""), fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        ax = axes[r, -1]
        if "brim" in lib: ax.imshow(lib["brim"]["z"].T, origin="lower", cmap="gray", vmin=0, vmax=1, aspect="auto"); ax.set_title(f"{n} · brim", fontsize=8)
        else: ax.text(0.5, 0.5, "no brim", ha="center", va="center"); ax.set_title(f"{n} · brim", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        print(f"{n:10s} segments={lib['meta']['n_segments']:2d}  eye at (s={lib['eye']['centre_su'][0]:.2f}, u={lib['eye']['centre_su'][1]:.2f})  reliefs: head {lib['head']['relief']:.2f} seg {lib['segment']['relief']:.2f} pyg {lib['pygidium']['relief']:.2f}")
    plt.tight_layout(); plt.savefig("parts_library.png", dpi=80)
    np.savez("parts_library.npz", **{f"{n}__{k}__{kk}": vv for n, lib in store.items() for k, p in lib.items() if k != "meta" for kk, vv in p.items() if isinstance(vv, np.ndarray)})
    json.dump({n: lib["meta"] for n, lib in store.items()}, open("parts_library_meta.json", "w"), indent=1)
    print("saved parts_library.npz / parts_library_meta.json / parts_library.png")
