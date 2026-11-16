"""Credits for the Earth layout's data, from the same downloads build_globe.py reads.

Writes
  web/static/earth/pbdb_references.txt   every publication behind the PBDB records used (PBDB's own formatting)
  web/static/earth/data_credits.txt      plain-text credits: sources, PBDB contributors, every GBIF dataset citation
  web/static/earth/credits_section.html  the "Data" section for the Sources panel of web/tree.html (pasted in by hand)
"""
import os, sys, html, re
import json, gzip
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbif_exclude import NOT_FOSSIL

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.environ.get("EARTH_RAW", os.path.join(ROOT, "earth_raw"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "web", "static", "earth")
FULL = os.path.join(ROOT, "assets", "earth", "fossils")
ACCESSED = os.environ.get("EARTH_ACCESSED", "15 November 2026 (UTC)")
e = html.escape


def pbdb():
    refs = pd.concat([pd.read_csv(f"{RAW}/pbdb_trilobita_refs.csv"), pd.read_csv(f"{RAW}/pbdb_agnostida_refs.csv")])
    refs = refs.drop_duplicates("reference_no").sort_values(["author1last", "pubyr"])
    ent = pd.concat([pd.read_csv(f"{RAW}/pbdb_{t}_ent.csv", low_memory=False) for t in ("Trilobita", "Agnostida")])
    auth = ent.authorizer.value_counts(); entr = ent.enterer.value_counts()
    return refs, auth, entr


def gbif():
    src = os.path.join(RAW, "gbif_datasets.csv")       # written by gbif_pull.py; falls back to the shipped copy
    g = pd.read_csv(src if os.path.exists(src) else os.path.join(FULL, "gbif_datasets.csv"))
    g = g[~g.datasetKey.isin(NOT_FOSSIL)]
    used = {}                                           # records per dataset actually on the globe
    for s in json.load(gzip.open(os.path.join(FULL, "sites_full.json.gz"), "rt")):
        if s["src"] == "GBIF": used[s["datasetKey"]] = used.get(s["datasetKey"], 0) + s["n"]
    g["n"] = g.datasetKey.map(used).fillna(0).astype(int)
    g = g[g.n > 0]
    g[["datasetKey", "n", "license", "citation"]].to_csv(os.path.join(FULL, "gbif_datasets.csv"), index=False)
    def clean(c):
        c = str(c) if isinstance(c, str) else ""
        c = re.sub(r"\s+https://doi\.org/\S+$", "", c.strip()) if c.count("https://doi.org/") > 1 else c.strip()
        return c
    g["citation"] = g.citation.map(clean)
    lic = lambda u: ("CC BY-NC 4.0" if "by-nc" in u else "CC BY 4.0" if "/by/" in u or "CC by/" in u else "CC0 1.0" if "zero" in u or "CC0" in u else u)
    g["lic"] = g.license.astype(str).map(lic)
    return g.sort_values("n", ascending=False)


def main():
    refs, auth, entr = pbdb(); g = gbif()
    with open(f"{OUT}/pbdb_references.txt", "w", encoding="utf-8") as f:
        f.write(f"Publications behind the Paleobiology Database records of Trilobita and Agnostida used on the Earth layout "
                f"({len(refs)} references; downloaded {ACCESSED}). Formatting is PBDB's.\n\n")
        for x in refs.formatted: f.write(str(x).strip() + "\n")

    nc = g[g.lic.str.contains("NC")]
    lines = [
        "DATA CREDITS - Earth layout (fossil localities and ancient geography)", "",
        f"Fossil occurrences: Paleobiology Database (paleobiodb.org), released CC0 (credited here as scholarly practice). Data were downloaded from the Paleobiology "
        f"Database on {ACCESSED}, using the taxon names Trilobita and Agnostida and the parameters show=full,paleoloc,class,"
        f"strat,env,ref,coords; {len(refs)} source publications (pbdb_references.txt). Peters, S. E. & McClennen, M. 2016. "
        f"The Paleobiology Database application programming interface. Paleobiology 42: 1-7.",
        "PBDB authorizers: " + ", ".join(auth.index), "PBDB data enterers: " + ", ".join(entr.index), "",
        f"Museum specimens: GBIF.org, {int(g.n.sum())} georeferenced fossil specimen records from {len(g)} datasets, "
        f"accessed {ACCESSED}. Each dataset's own citation and licence:",
        *[f"  [{r.lic}] ({int(r.n)} records) {r.citation}" for r in g.itertuples()], "",
        "Ancient coastlines and continental margins: Kocsis, A. T. & Scotese, C. R. 2021. Mapping paleocoastlines and "
        "continental flooding during the Phanerozoic. Earth-Science Reviews 213: 103463. Data: PaleoMAP PaleoCoastlines "
        "v7.3, Zenodo doi:10.5281/zenodo.7994000, CC BY 4.0.",
        "Elevation (high land): Scotese, C. R. & Wright, N. 2018. PALEOMAP Paleodigital Elevation Models (PaleoDEMS) for "
        "the Phanerozoic. PALEOMAP Project. doi:10.5281/zenodo.5460860, CC BY 4.0.",
        "Plate model: Scotese, C. R. PALEOMAP Global Plate Model, rotation file v19o_r1c (distributed with PaleoCoastlines "
        "v7.3); Scotese, C. R. 2016. PALEOMAP PaleoAtlas for GPlates and the PaleoData Plotter Program.",
        "Reconstruction software: pyGPlates (GPlates, EarthByte Group, University of Sydney).",
    ]
    with open(f"{OUT}/data_credits.txt", "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")

    a = lambda href, text: f'<a href="{e(href)}" target="_blank" rel="noopener">{e(text)}</a>'
    rows = "\n".join(f'      <li>{e(r.citation)} <span class="dim">{int(r.n):,} records &middot; {e(r.lic)}</span></li>' for r in g.itertuples())
    sec = f'''    <h2 id="src-data">Data<span class="cap">Where the Earth layout's fossils and ancient geography come from. Downloaded {e(ACCESSED)}.</span></h2>
    <p class="ref">The Paleobiology Database. Occurrences of Trilobita and Agnostida, {a("https://paleobiodb.org", "paleobiodb.org")}, released CC0. Data were downloaded from the Paleobiology Database on {e(ACCESSED)}, using the taxon names Trilobita and Agnostida. Peters, S. E. &amp; McClennen, M. 2016. The Paleobiology Database application programming interface. Paleobiology 42: 1-7. {a("https://doi.org/10.1017/pab.2015.39", "doi:10.1017/pab.2015.39")}.
      <span class="in"><b>On the Earth layout:</b> {int(len(refs)):,} source publications ({a("/static/earth/pbdb_references.txt", "the full list")}), compiled by the PBDB's contributors: authorizers {e(", ".join(auth.index[:12]))} and {len(auth) - 12} others; data entry by {e(", ".join(entr.index[:12]))} and {len(entr) - 12} others ({a("/static/earth/data_credits.txt", "everyone")}).</span></p>
    <p class="ref">GBIF.org. Georeferenced fossil specimen records of Trilobita and Agnostida from {len(g)} museum datasets, accessed {e(ACCESSED)}.
      <span class="in"><b>On the Earth layout:</b> {int(g.n.sum()):,} specimens. {len(nc)} of the datasets are licensed CC BY-NC 4.0 (non-commercial), among them the Museum of Comparative Zoology, Harvard University.</span></p>
    <details class="ref"><summary>The {len(g)} datasets, with their citations</summary>
    <ul class="datasets">
{rows}
    </ul></details>
    <p class="ref">Kocsis, Á. T. &amp; Scotese, C. R. 2021. Mapping paleocoastlines and continental flooding during the Phanerozoic. Earth-Science Reviews 213: 103463. {a("https://doi.org/10.1016/j.earscirev.2020.103463", "doi:10.1016/j.earscirev.2020.103463")}. Data: PaleoMAP PaleoCoastlines v7.3, {a("https://doi.org/10.5281/zenodo.7994000", "doi:10.5281/zenodo.7994000")}, CC BY 4.0.
      <span class="in"><b>On the Earth layout:</b> the land and shelf seas of every frame, 540 to 250 Ma.</span></p>
    <p class="ref">Scotese, C. R. &amp; Wright, N. 2018. PALEOMAP Paleodigital Elevation Models (PaleoDEMS) for the Phanerozoic. PALEOMAP Project. {a("https://doi.org/10.5281/zenodo.5460860", "doi:10.5281/zenodo.5460860")}, CC BY 4.0.
      <span class="in"><b>On the Earth layout:</b> the high land.</span></p>
    <p class="ref">Scotese, C. R. PALEOMAP Global Plate Model (rotation file v19o_r1c, distributed with PaleoCoastlines v7.3); Scotese, C. R. 2016. PALEOMAP PaleoAtlas for GPlates and the PaleoData Plotter Program. Reconstructed with pyGPlates (GPlates, EarthByte Group).
      <span class="in"><b>On the Earth layout:</b> where each fossil site was at each frame, and the continent labels. Early Palaeozoic positions differ between plate models by tens of degrees; these are PALEOMAP's.</span></p>
'''
    with open(f"{OUT}/credits_section.html", "w", encoding="utf-8") as f: f.write(sec)
    print("wrote", [p for p in os.listdir(OUT) if p.endswith((".txt", ".html"))], len(refs), "refs,", len(g), "datasets")


if __name__ == "__main__":
    main()
