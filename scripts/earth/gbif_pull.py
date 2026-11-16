"""Pull every georeferenced trilobite / agnostid fossil specimen record from GBIF, dataset by dataset
(the search API caps paging at 100k per query; every dataset here is under that).
Skips the GBIF copy of PBDB (pulled directly) and datasets that match the name but are not fossils."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, sys, time, concurrent.futures as cf, urllib.request, urllib.parse, csv, os

TAXA = [9273948, 9538015]                       # Trilobita, Agnostida (GBIF backbone)
from gbif_exclude import NOT_FOSSIL
SKIP = {"bb5b30b4-827e-4d5e-a86a-825d65cb6583"} | NOT_FOSSIL  # PBDB (pulled directly) + non-fossil name matches
KEEP_BASIS = {"FOSSIL_SPECIMEN", "PRESERVED_SPECIMEN", "MATERIAL_CITATION"}   # not MATERIAL_SAMPLE (DNA)
API = "https://api.gbif.org/v1/occurrence/search"
FIELDS = ["key", "datasetKey", "datasetName", "institutionCode", "collectionCode", "catalogNumber", "basisOfRecord",
          "scientificName", "acceptedScientificName", "taxonRank", "kingdom", "phylum", "class", "order", "family", "genus",
          "species", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters", "country", "countryCode",
          "stateProvince", "locality", "verbatimLocality", "earliestEonOrLowestEonothem", "earliestEraOrLowestErathem",
          "earliestPeriodOrLowestSystem", "earliestEpochOrLowestSeries", "earliestAgeOrLowestStage",
          "latestPeriodOrHighestSystem", "latestEpochOrHighestSeries", "latestAgeOrHighestStage", "group", "formation",
          "member", "bed", "lithostratigraphicTerms", "typeStatus", "recordedBy", "eventDate", "year", "references",
          "license", "rightsHolder", "issues"]


def get(params, tries=6):
    url = API + "?" + urllib.parse.urlencode(params, doseq=True)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def count(ds, lo, hi):
    return get({"taxonKey": TAXA, "datasetKey": ds, "hasCoordinate": "true", "limit": 0,
                "decimalLatitude": f"{lo:.6f},{hi:.6f}"})["count"]


def bands(ds, lo=-90.0, hi=90.0, n=None):
    """Latitude bands holding at most 3000 records each, so no request pages deep (GBIF stalls past ~10k)."""
    n = count(ds, lo, hi) if n is None else n
    if n == 0: return []
    if n <= 3000 or hi - lo < 1e-4: return [(lo, hi, n)]
    mid = (lo + hi) / 2
    # ranges are inclusive at both ends in GBIF; nudge the upper half to avoid double counting the midline
    return bands(ds, lo, mid) + bands(ds, mid + 1e-6, hi)


def page(ds, lo, hi, off):
    d = get({"taxonKey": TAXA, "datasetKey": ds, "hasCoordinate": "true", "limit": 300, "offset": off,
             "decimalLatitude": f"{lo:.6f},{hi:.6f}"})
    return d["results"]


def main(out):
    facets = get({"taxonKey": TAXA, "hasCoordinate": "true", "limit": 0, "facet": "datasetKey", "facetLimit": 200})
    sets = [(c["name"], c["count"]) for c in facets["facets"][0]["counts"] if c["name"] not in SKIP]
    jobs = []
    for ds, n in sets:
        for lo, hi, m in bands(ds):
            jobs += [(ds, lo, hi, off) for off in range(0, m, 300)]
    print(f"{len(sets)} datasets, {sum(n for _, n in sets)} records, {len(jobs)} pages", flush=True)
    rows, done, seen = [], 0, set()
    with cf.ThreadPoolExecutor(6) as ex:
        for res in ex.map(lambda j: page(*j), jobs):
            for r in res:
                if r["key"] in seen: continue
                seen.add(r["key"])
                if r.get("basisOfRecord") not in KEEP_BASIS:
                    continue
                row = {k: r.get(k) for k in FIELDS}
                row["issues"] = ";".join(r.get("issues") or [])
                rows.append(row)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} pages, {len(rows)} kept", flush=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows))
    n = {}
    for r in rows: n[r["datasetKey"]] = n.get(r["datasetKey"], 0) + 1
    with open(os.path.join(os.path.dirname(out), "gbif_datasets.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["datasetKey", "n", "license", "citation"])
        for k, c in sorted(n.items(), key=lambda x: -x[1]):
            d = json.load(urllib.request.urlopen(f"https://api.gbif.org/v1/dataset/{k}", timeout=60))
            w.writerow([k, c, d.get("license", ""), (d.get("citation") or {}).get("text", "")])


if __name__ == "__main__":
    main(sys.argv[1])
