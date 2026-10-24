"""
scripts/trilomorph_import.py — TriloMorph cephala as saved heads for the head editor and the studio (20 Oct 2026).

TriloMorph (Serra, Balseiro, Monnet et al. 2023, Scientific Data 10:841, doi:10.1038/s41597-023-02724-9; Zenodo
doi:10.5281/zenodo.8033445; CC BY 4.0) is a database of hand-landmarked trilobite cephala in dorsal view: 16 landmarks
and four semilandmark curves per specimen (glabella LM2–LM7, facial suture LM10–LM12, anterior margin LM13–LM15,
posterior margin LM4–LM15). Their margin curves ARE a traced head in the sense of the head editor's trace tool, so this
script turns their files into saved heads: the margin as `outline.trace`, the genal spine (LM15→LM16) as the spine
keys, and the eye and glabella landmarks as the face keys the crescent head takes. Every head records its source
specimen and the licence in `doc`. Nothing of their images is used or copied.

    python scripts/trilomorph_import.py <trilomorph checkout> --genus Phacops Harpes Deiphon          # by genus
    python scripts/trilomorph_import.py <trilomorph checkout> --id AMNH_28896 1020_Liu_1977            # by specimen ID
    python scripts/trilomorph_import.py <trilomorph checkout> --all                                   # every sighted cephalon
    options: --out web/heads (where saved heads live)  --prefix tm-  --dry (print, do not write)

What is and is not read off their data (say so when you publish a head):
  * the margin: LM1 → anterior curve → LM15 → posterior curve (reversed) → LM4, in half-widths of the widest margin point
  * the genal spine: length |LM16 − LM15| / sagittal length, spread = its angle from straight back; width and taper are
    not in their protocol and keep the editor's defaults
  * eye: radius = half the LM8–LM9 length, position along the head from LM8/LM9, lateral position from LM14 (outer edge − r);
    blind files (LM8 = LM9) give eye_size 0
  * glabella: axis_frac from LM6 (occipital width), glab_inflate from LM5 (widest) over LM6; the glabella curve is not used
  * `width` (head vs thorax half-width) is unknown without a thorax and keeps the default 1.15
"""
import os, sys, re, json, glob, math, argparse, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "isopod_model"))

CURVES = ["glabella", "suture", "anterior", "posterior"]
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{0,39}\Z")


def read_txt(p):
    t = open(p, encoding="utf-8", errors="ignore").read()
    m = re.search(r"<landmarks.scaled[^>]*>(.*?)</landmarks.scaled>", t, re.S)
    if not m: return None
    lm = {}
    for line in m.group(1).strip().splitlines():
        a = line.split()
        if len(a) == 3: lm[a[0]] = (float(a[1]), float(a[2]))
    cv = {}
    mc = re.search(r"<curves.scaled[^>]*>(.*?)</curves.scaled>", t, re.S)
    if mc:
        for name in CURVES:
            mm = re.search(r"<%s [^>]*>(.*?)</%s>" % (name, name), mc.group(1), re.S)
            if mm:
                pts = [tuple(map(float, l.split())) for l in mm.group(1).strip().splitlines() if len(l.split()) == 2]
                if len(pts) >= 2: cv[name] = np.array(pts)
    return dict(lm=lm, curves=cv)


def read_tps(p):
    t = open(p, encoding="utf-8", errors="ignore").read().splitlines()
    n = int(re.match(r"LM=(\d+)", t[0]).group(1)); lm = {f"LM{i + 1}": tuple(map(float, l.split())) for i, l in enumerate(t[1:1 + n])}
    cv = {}; i = 1 + n; k = 0
    while i < len(t):
        if t[i].startswith("POINTS="):
            m = int(t[i].split("=")[1]); pts = [tuple(map(float, l.split())) for l in t[i + 1:i + 1 + m]]
            if k < len(CURVES): cv[CURVES[k]] = np.array(pts)
            k += 1; i += 1 + m
        else: i += 1
    return dict(lm=lm, curves=cv)


def resample(pts, n):
    pts = np.asarray(pts, float); d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] <= 0: return np.repeat(pts[:1], n, axis=0)
    s = np.linspace(0, d[-1], n); return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def head_from(rec, n_front=40, n_rear=30):
    """A saved-head record (outline with trace + spine keys, face keys) from one specimen's landmarks and curves."""
    L = rec["lm"]; C = rec["curves"]
    for k in ("LM1", "LM4", "LM15", "LM16"):
        if k not in L: raise ValueError(f"missing {k}")
    A = np.array(L["LM1"]); B = np.array(L["LM4"]); ax = B - A; Ls = float(np.linalg.norm(ax))
    if Ls < 1e-9: raise ValueError("LM1 = LM4")
    u = ax / Ls; nrm = np.array([-u[1], u[0]])
    def frame(P):                                                   # -> (across >= 0, along <= 0 at the front, 0 at LM4)
        P = np.atleast_2d(np.asarray(P, float)) - B
        return np.c_[np.abs(P @ nrm), P @ u]
    ant = C.get("anterior"); post = C.get("posterior")
    if ant is None or post is None: raise ValueError("no margin curves")
    # the anterior curve runs LM13 -> LM15 and the posterior LM4 -> LM15 (their template); orient each by its ends
    def oriented(cv, start, end):
        cv = np.asarray(cv, float)
        if np.linalg.norm(cv[0] - np.array(L[start])) > np.linalg.norm(cv[-1] - np.array(L[start])): cv = cv[::-1]
        return cv
    ant = oriented(ant, "LM13", "LM15"); post = oriented(post, "LM4", "LM15")
    margin = np.vstack([frame([L["LM1"]]), frame(resample(ant, n_front)), frame(resample(post[::-1], n_rear)), frame([L["LM4"]])])
    hw = float(margin[:, 0].max())
    if hw < 1e-6: raise ValueError("no width")
    tr = margin / hw; tr[0] = [0.0, tr[:, 1].min()]; tr[-1] = [0.0, 0.0]   # the front on the axis, exactly; the rear margin on the axis
    tr = [[round(float(x), 4), round(float(min(y, 0.99)), 4)] for x, y in tr]
    # the genal spine
    g15, g16 = frame([L["LM15"]])[0], frame([L["LM16"]])[0]
    sp = g16 - g15; Lg = float(np.hypot(*sp)) / Ls
    outline = dict(trace=tr)
    if Lg >= 0.05:
        outline["genal"] = round(min(Lg, 3.5), 3)
        outline["genal_spread"] = round(float(np.degrees(math.atan2(sp[0], sp[1]))), 1)    # 0 = straight back; + outward
        outline["genal_spread"] = max(0.0, min(80.0, outline["genal_spread"]))     # 24 Oct 2026: the band runs outward only
    else:
        outline["genal"] = 0.0
    # the face
    face = {}
    blind = "LM8" not in L or "LM9" not in L or np.hypot(*(np.array(L["LM8"]) - L["LM9"])) < 1e-9
    if blind: face["eye_size"] = 0.0
    else:
        e8, e9 = frame([L["LM8"]])[0], frame([L["LM9"]])[0]; eR = 0.5 * float(np.hypot(*(e8 - e9))) / hw
        yc = 0.5 * (e8[1] + e9[1]); face["eye_size"] = round(max(0.02, min(0.5, eR)), 3)
        face["eye_pos"] = round(max(0.1, min(0.95, -yc / Ls)), 3)
        if "LM14" in L:
            x14 = frame([L["LM14"]])[0][0] / hw; face["eye_lat"] = round(max(0.05, min(0.95, x14 - eR)), 3)
    if "LM6" in L and "LM5" in L:
        w6 = frame([L["LM6"]])[0][0] / hw; w5 = frame([L["LM5"]])[0][0] / hw
        face["axis_frac"] = round(max(0.15, min(0.45, w6)), 3)
        face["glab_inflate"] = round(max(0.5, min(2.0, w5 / max(w6, 1e-6))), 3)
    return dict(outline=outline, face=face, sagittal_length=float(Ls), halfwidth=float(hw), blind=bool(blind))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--genus", nargs="*", default=[]); ap.add_argument("--id", nargs="*", default=[])
    ap.add_argument("--all", action="store_true"); ap.add_argument("--out", default=os.path.join(ROOT, "web", "heads")); ap.add_argument("--prefix", default="tm-"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    import yaml, shaped_head as SH
    docs = yaml.safe_load(open(os.path.join(a.root, "trilomorph.yaml"), encoding="utf-8")); meta = {d["ID"]: d for d in docs}
    by_genus = {}
    for d in docs: by_genus.setdefault(str(d["taxonomy"].get("genus")).lower(), []).append(d["ID"])
    ids = list(a.id)
    for g in a.genus: ids += by_genus.get(g.lower(), []) or [f"?{g}"]
    files = {os.path.basename(p).rsplit("_C.", 1)[0]: p for p in glob.glob(os.path.join(a.root, "Cephala", "landmarks", "*"))}
    if a.all: ids = sorted(files)
    done = 0; os.makedirs(a.out, exist_ok=True); used = set()
    for i in ids:
        if i.startswith("?"): print(f"{i[1:]}: no such genus in trilomorph.yaml"); continue
        p = files.get(i)
        if not p: print(f"{i}: no cephalon landmark file"); continue
        rec = read_txt(p) if p.endswith(".txt") else read_tps(p)
        if not rec: print(f"{i}: unreadable"); continue
        try: h = head_from(rec); SH.outline_of(h["outline"])
        except (ValueError, KeyError) as ex: print(f"{i}: skipped ({ex})"); continue
        m = meta.get(i, {}); genus = str(m.get("taxonomy", {}).get("genus") or i)
        name = re.sub(r"[^A-Za-z0-9_\-]", "-", a.prefix + genus)[:40]
        if not NAME_RE.match(name): name = "tm-" + re.sub(r"[^A-Za-z0-9]", "", genus)[:36] or "tm-head"
        base, k = name, 2
        while name in used: name, k = f"{base[:37]}-{k}", k + 1                            # a second specimen of the genus
        used.add(name)
        doc = f"TriloMorph {i} ({genus}; {m.get('stratigraphy', {}).get('min_age', '?')}): margin, genal spine, eye and glabella from Serra et al. 2023, CC BY 4.0"
        out = dict(name=name, style="plain", outline=h["outline"], face=h["face"], doc=doc[:200], saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   source=dict(database="TriloMorph", id=i, genus=genus, ref_pic=m.get("ref.pic"), citation="Serra et al. 2023 Sci Data 10:841, doi:10.1038/s41597-023-02724-9; Zenodo doi:10.5281/zenodo.8033445; CC BY 4.0"))
        print(f"{i:28s} -> {name:28s} genal {h['outline']['genal']:.2f}  {'blind' if h['blind'] else 'eye r ' + str(h['face'].get('eye_size'))}  trace {len(h['outline']['trace'])} pts")
        if not a.dry:
            tmp = os.path.join(a.out, name + ".json.part"); json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(a.out, name + ".json")); done += 1
    print(f"{done} heads written to {a.out}" if not a.dry else "dry run")


if __name__ == "__main__":
    main()
