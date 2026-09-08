"""Re-save every preset with all schema keys, a schema stamp, table-default fallback (no inherited crescent),
and the Gon-ten order naming. 8 Sep 2026. Old files archived in presets_v1_archive/."""
import json, os, glob, schema
def load(name):
    if name in schema.PRESETS: return dict(schema.PRESETS[name])
    d = json.load(open(f"presets_v1_archive/{name}.json")); return d.get("params", d)
def pitch(p): return p["length"] * (1 - p["cephFrac"] - p["pygFrac"]) / p["segCount"]
def fit_length(p, minpitch=8.05):
    """Scale policy: length is the smallest value ≥ the stated one that gives pitch ≥ 8 (print rule)."""
    need = minpitch * p["segCount"] / (1 - p["cephFrac"] - p["pygFrac"])
    return max(p["length"], round(need))
T = load("textured")
ORDERS = {
 # order            base            overrides                                                              blurb (Gon page: fill in)
 "agnostida":      ("agnostid",     {},                                                                      "Agnostida — Gon p. __. Two segments, head and tail alike, blind."),
 "redlichiida":    ("olenellid",    {},                                                                      "Redlichiida (olenellid) — Gon p. __. Long genal horns, 14 segments, macropleural spine, tiny tail."),
 "corynexochida":  ("reference",    dict(effacement=0.5, furrowDepth=0.9, glabInflate=1.7, glabRise=0.2, segCount=8, cephFrac=0.30, pygFrac=0.30, pygWidth=0.95, eyeSize=0.12, genalSpine=0.25), "Corynexochida — Gon p. __. Forward-expanding glabella, effaced, large tail, 8 segments."),
 "lichida":        ("textured",     dict(segCount=10, cephFrac=0.30, pygFrac=0.26, pygWidth=1.0, pygMarginal=6, pygMarginalLen=0.6, tubercles=0.7, tubercleSize=1.8, genalSpine=0.45, glabInflate=1.6, spineBase=0.15, spineGrad=0.15, tipTaper=0.75), "Lichida — Gon p. __. Tuberculate, wide spinose tail, inflated glabella, 10 segments."),
 "odontopleurida": ("spiny",        dict(spineBase=0.8, spineGrad=0.3, pygMarginal=6, pygMarginalLen=0.9, occipitalSpine=0.5, axialSpine=0.4, genalSpine=0.8, tubercles=0.5, segCount=9, cephFrac=0.32, pygFrac=0.16), "Odontopleurida — Gon p. __. Spines everywhere: pleural, marginal, occipital, axial; 9 segments."),
 "phacopida":      ("phacopid",     {},                                                                      "Phacopida (Phacops, fitted 6 Sep) — Gon p. __. Schizochroal eyes, 11 segments, no genal spines."),
 "proetida":       ("textured",     dict(segCount=9, cephFrac=0.30, pygFrac=0.24, pygWidth=0.95, genalSpine=0.2, genalWidthMM=3, eyeSize=0.16, eyePos=0.55, glabInflate=1.15, borderWidth=0.12, pygRings=6, pygSpine=0.0, spineBase=0.0), "Proetida — Gon p. __. Wide bordered head, short genal spines, 9 segments, well-furrowed tail."),
 "asaphida":       ("asaphid",      {},                                                                      "Asaphida — Gon p. __. Effaced, broad, 8 segments, isopygous."),
 "harpetida":      ("harpetid",     {},                                                                      "Harpetida (fitted 6 Sep) — Gon p. __. Horseshoe brim, prolongations to the tail, narrow thorax. Brim pits not modelled."),
 "ptychopariida":  ("textured",     dict(segCount=12, cephFrac=0.28, pygFrac=0.14, genalSpine=0.3, eyeSize=0.12, glabInflate=1.05, widthThoraxRear=0.6), "Ptychopariida — Gon p. __. Generalist: 12 segments, small tail, backward-tapering glabella."),
}
os.makedirs("presets", exist_ok=True)
for f in glob.glob("presets/*.json"): os.remove(f)
rows = []
for name, (base, ov, blurb) in ORDERS.items():
    p = dict(schema.table_defaults()); p.update(load(base)); p.update(ov)
    p = schema.coerce(p, base=schema.table_defaults())
    L0 = p["length"]; p["length"] = fit_length(p)
    json.dump(dict(schema=schema.SCHEMA_VERSION, order=name, base=base, blurb=blurb, length_policy=f"pitch>=8: {L0}->{p['length']} mm",
                   expect={}, params=p), open(f"presets/{name}.json", "w"), indent=1, sort_keys=True)
    rows.append((name, base, L0, p["length"], p["segCount"], round(pitch(p), 2), len(p)))
for r in rows: print("%-15s base=%-10s L %5.0f -> %5.0f  n=%2d pitch %5.2f  keys %d" % r)
