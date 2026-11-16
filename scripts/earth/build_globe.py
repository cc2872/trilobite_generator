"""Build the Earth layout's data: ancient geography per 5 Myr frame and every georeferenced trilobite locality,
reconstructed to where it was when the animals lived.

Sources (all fetched by the scripts in this folder, see SOURCES.md):
  PBDB occurrences of Trilobita + Agnostida (paleobiodb.org, CC BY 4.0)
  GBIF fossil specimen records of Trilobita + Agnostida (museum collections; per-dataset licences)
  PALEOMAP PaleoCoastlines v7 (Kocsis & Scotese 2021; CC BY 4.0): coastlines (CS) and continental margins (CM)
  PALEOMAP PaleoDEMs (Scotese & Wright 2018): elevation, used only for "high" land
  PALEOMAP plate model v19o_r1c (rotation + partitioning polygons), the version the coastlines follow
Outputs (into OUT):
  frames.json    per frame t: land / shelf / high bit masks on the page's Fibonacci lattice + continent labels
  sites.json     compact per-site arrays for the globe (positions per frame, lineage, age, counts)
  sites_full.json.gz / sites_full.csv.gz (into assets/earth/fossils)  every site with its full detail
Run: sh scripts/earth/fetch.sh && python scripts/earth/build_globe.py   (about a minute after the downloads)
"""
import os, sys, re, json, gzip, base64, math, glob, collections, hashlib
import numpy as np, pandas as pd, geopandas as gpd, shapely, pygplates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbif_exclude import NOT_FOSSIL

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.environ.get("EARTH_RAW", os.path.join(ROOT, "earth_raw"))          # fetch.sh puts the downloads here (git-ignored)
COAST, DEMDIR = f"{RAW}/coast", f"{RAW}/paleomap/dem/PaleoDEMS_long_lat_elev_csv_v2.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "web", "static", "earth")     # what the page loads
FULL = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "assets", "earth", "fossils")  # full detail
os.makedirs(OUT, exist_ok=True); os.makedirs(FULL, exist_ok=True)
ROT = f"{COAST}/v19o/m06c9h_3id_forPgeog_19o_r1c.rot"
POLY = f"{COAST}/v19o/PALEOMAP_PlatePolygons__forPgeog_v19o.gpml"
N_LAT = 14000                                        # the page's lattice (assets/earth/make_mask.py)
FRAMES = list(range(540, 249, -5))                   # 540, 535, ... 250 Ma
T_LO, T_HI = 251.902, 541.0                          # trilobites: Fortunian to end-Permian
MAX_SPAN = 50                                        # Myr; a locality dated more coarsely than about one period is not placed

# ---------------------------------------------------------------------------------------------- the tree's lineages
TREE = ["redlichiida", "olenellina", "redlichiina", "agnostida", "eodiscina", "agnostina", "corynexochida",
        "corynexochina", "leiostegiina", "illaenina", "lichida", "odontopleurida", "ptychopariida", "ptychopariina",
        "olenina", "asaphida", "harpetida", "proetida", "phacopida", "calymenina", "cheirurina", "phacopina",
        "phacopoidea", "dalmanitoidea", "acastoidea"]
TREE_NAME = {n.capitalize(): n for n in TREE}
# PBDB orders that Gon's scheme files elsewhere: Olenida = Gon's Olenina (Ptychopariida); Trinucleida sits in Asaphida
# (Trinucleioidea); Harpida = Harpetida.
ORDER_FALLBACK = {"Olenida": "olenina", "Trinucleida": "asaphida", "Harpida": "harpetida", "Harpetida": "harpetida"}
# PBDB hangs these families straight off the order, without the suborders Gon uses; place them in Gon's suborders.
# Illaenina = the illaenids and scutelluids (Styginidae); Ptychopariida minus Olenina = Ptychopariina in Gon's tree.
FAMILY_TO_NODE = {"Illaenidae": "illaenina", "Styginidae": "illaenina", "Scutelluidae": "illaenina",
                  "Panderiidae": "illaenina", "Ptychopariida": "ptychopariina"}


def lineage_mapper():
    tx = pd.concat([pd.read_csv(f"{RAW}/pbdb_trilobita_taxa.csv"), pd.read_csv(f"{RAW}/pbdb_agnostida_taxa.csv")])
    tx = tx.drop_duplicates("taxon_no")
    parent = dict(zip(tx.taxon_no, tx.parent_no)); name = dict(zip(tx.taxon_no, tx.taxon_name))
    acc = dict(zip(tx.taxon_no, tx.accepted_no))
    by_name = {}
    for no, nm, a in zip(tx.taxon_no, tx.taxon_name, tx.accepted_no):
        by_name.setdefault(nm, a)
    cache = {}

    def node_of_no(no):
        if no in cache: return cache[no]
        seen, cur, hit = 0, acc.get(no, no), None
        while cur in name and seen < 40:
            n = name[cur]
            if n in FAMILY_TO_NODE: hit = FAMILY_TO_NODE[n]; break
            if n in TREE_NAME: hit = TREE_NAME[n]; break
            if n in ORDER_FALLBACK: hit = ORDER_FALLBACK[n]; break
            cur = parent.get(cur); seen += 1
        cache[no] = hit
        return hit

    def node_of(no=None, names=()):
        if no is not None and not pd.isna(no):
            h = node_of_no(int(no))
            if h: return h
        for nm in names:                              # genus, family, order text as a fallback (GBIF)
            if isinstance(nm, str) and nm:
                if nm in FAMILY_TO_NODE: return FAMILY_TO_NODE[nm]
                if nm in TREE_NAME: return TREE_NAME[nm]
                if nm in ORDER_FALLBACK: return ORDER_FALLBACK[nm]
                a = by_name.get(nm)
                if a is not None and not pd.isna(a):
                    h = node_of_no(int(a))
                    if h: return h
        return None
    return node_of


# ---------------------------------------------------------------------------------------------- ages for named intervals
def interval_ages():
    iv = pd.read_csv(f"{RAW}/pbdb_intervals.csv")
    d = {}
    for n, b, t in zip(iv.interval_name, iv.b_age, iv.t_age):
        d.setdefault(str(n).strip().lower(), (float(b), float(t)))
    # common GBIF spellings
    for k, v in list(d.items()):
        for pre in ("early ", "middle ", "late ", "lower ", "upper "):
            pass
    syn = {"lower cambrian": "early cambrian", "upper cambrian": "late cambrian", "lower ordovician": "early ordovician",
           "upper ordovician": "late ordovician", "lower silurian": "early silurian", "upper silurian": "late silurian",
           "lower devonian": "early devonian", "upper devonian": "late devonian", "lower carboniferous": "mississippian",
           "upper carboniferous": "pennsylvanian", "lower permian": "cisuralian", "upper permian": "lopingian",
           "middle permian": "guadalupian", "early permian": "cisuralian", "late permian": "lopingian",
           "middle cambrian": "miaolingian", "cambrian series 2": "series 2", "furongian": "furongian"}
    for a, b in syn.items():
        if b in d and a not in d: d[a] = d[b]
    return d


def text_age(row, IV):
    """GBIF chronostratigraphy text -> (max_ma, min_ma, precision label)."""
    lo_keys = ["earliestAgeOrLowestStage", "earliestEpochOrLowestSeries", "earliestPeriodOrLowestSystem"]
    hi_keys = ["latestAgeOrHighestStage", "latestEpochOrHighestSeries", "latestPeriodOrHighestSystem"]

    def look(keys):
        for k in keys:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                v = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", v.strip())          # "UpperOrdovician" -> "Upper Ordovician"
                s = re.sub(r"\s+", " ", v.lower().replace("stage", "").replace("series", "").strip())
                s = s.replace("lower/early", "early").replace("upper/late", "late")
                if s in IV: return IV[s], k
                s2 = re.sub(r"^(lower|upper)\b", lambda m: "early" if m.group(1) == "lower" else "late", s)
                if s2 in IV: return IV[s2], k
        return None, None
    a, ka = look(lo_keys); b, kb = look(hi_keys)
    if a is None and b is None: return None
    a = a or b; b = b or a
    return max(a[0], b[0]), min(a[1], b[1]), (ka or kb).replace("earliest", "").replace("OrLowest", "/")


# ---------------------------------------------------------------------------------------------- localities
def load_pbdb(node_of):
    cols = None
    frames = []
    for f in ("pbdb_trilobita_occs.csv", "pbdb_agnostida_occs.csv"):
        d = pd.read_csv(f"{RAW}/{f}", low_memory=False)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["lng", "lat", "max_ma", "min_ma"])
    d["node"] = [node_of(a, (g, fa, o)) for a, g, fa, o in zip(d.accepted_no, d.genus, d.family, d.order)]
    sites = []
    for cno, g in d.groupby("collection_no", sort=False):
        r = g.iloc[0]
        place = ", ".join(str(x) for x in (r.get("county"), r.get("state"), r.get("cc")) if isinstance(x, str) and x and x != "__")
        sites.append(dict(
            id=f"pbdb:{int(cno)}", src="PBDB", name=str(r.collection_name), lat=float(r.lat), lon=float(r.lng),
            max_ma=float(g.max_ma.max()), min_ma=float(g.min_ma.min()),
            interval=" - ".join(x for x in dict.fromkeys([r.early_interval, r.late_interval]) if isinstance(x, str)),
            place=place, cc=r.get("cc") if isinstance(r.get("cc"), str) else "",
            formation=r.get("formation") if isinstance(r.get("formation"), str) else "",
            group=r.get("geological_group") if isinstance(r.get("geological_group"), str) else "",
            member=r.get("member") if isinstance(r.get("member"), str) else "",
            environment=r.get("environment") if isinstance(r.get("environment"), str) else "",
            lithology=r.get("lithology1") if isinstance(r.get("lithology1"), str) else "",
            lagerstatte=r.get("lagerstatten") if isinstance(r.get("lagerstatten"), str) else "",
            preservation=r.get("preservation_quality") if isinstance(r.get("preservation_quality"), str) else "",
            museum=r.get("museum") if isinstance(r.get("museum"), str) else "",
            precision=r.get("latlng_precision") if isinstance(r.get("latlng_precision"), str) else "",
            n=int(len(g)),
            nodes=dict(collections.Counter(x for x in g.node if isinstance(x, str))),
            taxa=sorted(set(str(x) for x in g.accepted_name.dropna()))[:200],
            genera=sorted(set(str(x) for x in g.genus.dropna())),
            refs=sorted(set(str(x) for x in g.primary_reference.dropna()))[:20] if "primary_reference" in g else [],
            pbdb_paleo=[float(r.paleolat), float(r.paleolng)] if not pd.isna(r.paleolat) else None,
        ))
    return sites, d


def fm_key(name, cc=""):
    if not isinstance(name, str) or not name.strip(): return None
    n = re.sub(r"\b(formation|fm\.?|group|gp\.?|member|mbr\.?|shale|limestone|ls\.?)\b", "", name.lower())
    n = re.sub(r"[^a-z0-9]+", " ", n).strip()
    return n or None


def formation_ages(pbocc):
    """Formation -> (max_ma, min_ma, n collections) from PBDB's dated collections (median of their bounds)."""
    c = pbocc.drop_duplicates("collection_no")
    c = c.assign(k=[fm_key(f) for f in c.formation]).dropna(subset=["k"])
    out = {}
    for k, g in c.groupby("k"):
        out[k] = (float(g.max_ma.median()), float(g.min_ma.median()), int(len(g)))
    return out


def genus_ranges():
    tx = pd.concat([pd.read_csv(f"{RAW}/pbdb_trilobita_taxa.csv"), pd.read_csv(f"{RAW}/pbdb_agnostida_taxa.csv")])
    tx = tx[tx.accepted_rank == "genus"].dropna(subset=["firstapp_max_ma", "lastapp_min_ma"])
    return {n: (float(a), float(b)) for n, a, b in zip(tx.taxon_name, tx.firstapp_max_ma, tx.lastapp_min_ma)}


def load_gbif(node_of, IV, FM, GR):
    p = f"{RAW}/gbif_trilobita_fossils.csv"
    if not os.path.exists(p): return [], 0, 0
    d = pd.read_csv(p, low_memory=False)
    d = d.dropna(subset=["decimalLatitude", "decimalLongitude"])
    d = d[~d.datasetKey.isin(NOT_FOSSIL) & (d.basisOfRecord != "MATERIAL_SAMPLE")]
    d = d[~d.issues.fillna("").str.contains("ZERO_COORDINATE|COORDINATE_INVALID|COORDINATE_OUT_OF_RANGE")]
    ages = []
    for r in d.to_dict("records"):
        a = text_age(r, IV)
        f = FM.get(fm_key(r.get("formation")) or "")
        if f and (a is None or (f[0] <= a[0] + 1 and f[1] >= a[1] - 1 and (f[0] - f[1]) < (a[0] - a[1]))):
            a = (f[0], f[1], f"formation age from {f[2]} PBDB collections")     # finer, and inside the record's own range
        ages.append(a)
    # last resort for records with neither chronostratigraphy nor a known formation: the PBDB range of the record's own
    # genus (flagged; coarse). Not pooled by place: museum "localities" can be a whole county holding many horizons.
    for i, gname in enumerate(d.genus):
        if ages[i] is None and isinstance(gname, str) and gname in GR:
            hi, lo = GR[gname]; ages[i] = (hi, lo, f"inferred from the PBDB range of {gname}")
    d["max_ma"] = [a[0] if a else np.nan for a in ages]; d["min_ma"] = [a[1] if a else np.nan for a in ages]
    d["age_from"] = [a[2] if a else "" for a in ages]
    n_all = len(d)
    d = d[(d.min_ma.isna()) | ((d.min_ma < T_HI) & (d.max_ma > T_LO))]
    d["node"] = [node_of(None, (g, f, o)) for g, f, o in zip(d.genus, d.family, d["order"])]
    S = lambda x: x.astype(object).where(x.notna(), "").astype(str)
    key = (S(d.datasetKey) + "|" + S(d.decimalLatitude.round(3)) + "|" + S(d.decimalLongitude.round(3))
           + "|" + S(d.formation) + "|" + S(d.max_ma.round(1)) + "|" + S(d.min_ma.round(1)))
    sites = []
    for k, g in d.groupby(key, sort=False):
        r = g.iloc[0]
        place = ", ".join(str(x) for x in (r.get("stateProvince"), r.get("countryCode")) if isinstance(x, str) and x)
        loc = r.get("locality") if isinstance(r.get("locality"), str) else (r.get("verbatimLocality") if isinstance(r.get("verbatimLocality"), str) else "")
        sites.append(dict(
            id="gbif:" + hashlib.md5(k.encode()).hexdigest()[:12], src="GBIF", datasetKey=str(r.datasetKey),
            dataset=r.datasetName if isinstance(r.datasetName, str) else "",
            institution=r.institutionCode if isinstance(r.institutionCode, str) else "", name=(loc or str(r.formation or "") or place)[:160],
            lat=float(r.decimalLatitude), lon=float(r.decimalLongitude),
            max_ma=None if pd.isna(r.max_ma) else float(r.max_ma), min_ma=None if pd.isna(r.min_ma) else float(r.min_ma),
            interval=" - ".join(x for x in dict.fromkeys([r.get("earliestAgeOrLowestStage") or r.get("earliestEpochOrLowestSeries") or r.get("earliestPeriodOrLowestSystem"),
                                                          r.get("latestAgeOrHighestStage") or r.get("latestEpochOrHighestSeries") or r.get("latestPeriodOrHighestSystem")]) if isinstance(x, str)),
            place=place, cc=r.get("countryCode") if isinstance(r.get("countryCode"), str) else "",
            formation=r.get("formation") if isinstance(r.get("formation"), str) else "",
            group=r.get("group") if isinstance(r.get("group"), str) else "",
            member=r.get("member") if isinstance(r.get("member"), str) else "",
            uncertainty_m=None if pd.isna(r.coordinateUncertaintyInMeters) else float(r.coordinateUncertaintyInMeters),
            n=int(len(g)), nodes=dict(collections.Counter(x for x in g.node if isinstance(x, str))),
            taxa=sorted(set(str(x) for x in g.acceptedScientificName.dropna()))[:200],
            genera=sorted(set(str(x) for x in g.genus.dropna())),
            types=sorted(set(f"{t} {c}" for t, c in zip(g.typeStatus, g.catalogNumber) if isinstance(t, str) and t))[:50],
            catalog=sorted(set(f"{a} {b}" for a, b in zip(g.institutionCode, g.catalogNumber) if isinstance(b, str)))[:30],
            gbif_keys=[int(x) for x in g.key[:30]], license=str(r.license or ""), age_from=str(r.age_from or ""),
        ))
    return sites, n_all, len(d)


# ---------------------------------------------------------------------------------------------- reconstruction
R = pygplates.RotationModel(ROT)
PP = pygplates.FeatureCollection(POLY)


def plate_ids(lat, lon):
    feats = []
    for la, lo in zip(lat, lon):
        f = pygplates.Feature(); f.set_geometry(pygplates.PointOnSphere(float(la), float(lo))); feats.append(f)
    part = pygplates.partition_into_plates(PP, R, feats, properties_to_copy=[pygplates.PartitionProperty.reconstruction_plate_id])
    return np.array([f.get_reconstruction_plate_id() for f in part])


def _unit(lat, lon):
    a, b = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)]


def rotate(lat, lon, pids, t):
    """Finite rotation per plate, applied to all its points at once (Rodrigues' formula)."""
    out = np.full((len(lat), 2), np.nan)
    v = _unit(np.asarray(lat, float), np.asarray(lon, float))
    for pid in np.unique(pids):
        idx = np.where(pids == pid)[0]
        rot = R.get_rotation(float(t), int(pid), anchor_plate_id=0)
        if rot.represents_identity_rotation():
            w = v[idx]
        else:
            plat, plon, ang = rot.get_lat_lon_euler_pole_and_angle_degrees()
            k = _unit(np.array([plat]), np.array([plon]))[0]; th = np.radians(ang)
            x = v[idx]
            w = x*np.cos(th) + np.cross(k, x)*np.sin(th) + np.outer(x @ k, k)*(1 - np.cos(th))
        out[idx, 0] = np.degrees(np.arcsin(np.clip(w[:, 2], -1, 1)))
        out[idx, 1] = np.degrees(np.arctan2(w[:, 1], w[:, 0]))
    return out


# ---------------------------------------------------------------------------------------------- geography frames
def lattice(n):
    i = np.arange(n); lat = np.degrees(np.arcsin(1 - (2*i + 1)/n)); lon = np.degrees((i*np.pi*(3 - 5**.5)) % (2*np.pi) - np.pi)
    return lat, lon


def pack(a): return base64.b64encode(np.packbits(a.astype(np.uint8), bitorder="little").tobytes()).decode()


def nearest(t, avail): return min(avail, key=lambda a: (abs(a - t), a))


def frames_geo():
    cs_av = sorted(int(os.path.basename(p).split("Ma")[0]) for p in glob.glob(f"{COAST}/paleocoastlines_v7_shapefiles/CS/*.shp"))
    dem_av = {}
    for p in glob.glob(f"{DEMDIR}/*.csv"):
        m = re.search(r"_(\d+)Ma\.csv$", p); dem_av[int(m.group(1))] = p
    la, lo = lattice(N_LAT); pts = shapely.points(lo, la)
    out = {}
    for t in FRAMES:
        tc = nearest(t, cs_av)
        cs = gpd.read_file(f"{COAST}/paleocoastlines_v7_shapefiles/CS/{tc}Ma_CS_v7.shp").make_valid().union_all()
        cm = gpd.read_file(f"{COAST}/paleocoastlines_v7_shapefiles/CM/{tc}Ma_CM_v7.shp").make_valid().union_all()
        land = shapely.contains(cs, pts) | shapely.touches(cs, pts)
        shelf = (shapely.contains(cm, pts) | shapely.touches(cm, pts)) & ~land
        td = nearest(t, sorted(dem_av))
        dem = pd.read_csv(dem_av[td], comment=None, header=0, names=["lon", "lat", "elev"], skiprows=1)
        grid = np.full((181, 361), np.nan)
        grid[(dem.lat.values + 90).astype(int), (dem.lon.values + 180).astype(int)] = dem.elev.values
        e = grid[np.clip(np.round(la + 90).astype(int), 0, 180), np.clip(np.round(lo + 180).astype(int), 0, 360)]
        high = land & (e > 1000)
        out[t] = dict(coast_slice=tc, dem_slice=td, land=pack(land), shelf=pack(shelf), high=pack(high),
                      n_land=int(land.sum()), n_shelf=int(shelf.sum()))
        print(f"  frame {t} Ma <- coast {tc} Ma, dem {td} Ma: land {land.mean():.1%} shelf {shelf.mean():.1%}", flush=True)
    return out


# Palaeocontinent labels: a present-day anchor point well inside each block, moved with the same plate model.
# Names and spans follow the standard early-Palaeozoic usage (e.g. Torsvik & Cocks 2017); anchors are chosen here.
CONTINENTS = [
    ("Laurentia", 58, -97, 541, 420), ("Laurussia", 58, -97, 420, 300), ("Baltica", 59, 38, 541, 420),
    ("Avalonia", 52.5, -1.5, 500, 420), ("Siberia", 64, 105, 541, 250), ("Gondwana", 5, 20, 541, 300),
    ("Pangaea", 5, 20, 300, 250), ("North China", 37, 113, 541, 250), ("South China", 27, 108, 541, 250),
    ("Kazakhstania", 47, 72, 541, 300), ("Tarim", 40, 83, 541, 290),
]


def continent_labels():
    lat = np.array([c[1] for c in CONTINENTS]); lon = np.array([c[2] for c in CONTINENTS])
    pids = plate_ids(lat, lon)
    lab = {}
    for t in FRAMES:
        xy = rotate(lat, lon, pids, t)
        lab[t] = [[c[0], round(float(p[0]), 1), round(float(p[1]), 1)] for c, p in zip(CONTINENTS, xy) if c[4] <= t <= c[3]]
    return lab, {c[0]: int(p) for c, p in zip(CONTINENTS, pids)}


# ---------------------------------------------------------------------------------------------- main
def main():
    print("lineage map ..."); node_of = lineage_mapper(); IV = interval_ages()
    print("PBDB ..."); pb, pbocc = load_pbdb(node_of)
    print(f"  {len(pb)} PBDB localities from {len(pbocc)} occurrences")
    print("GBIF ..."); gb, gb_all, gb_kept = load_gbif(node_of, IV, formation_ages(pbocc), genus_ranges())
    print(f"  {len(gb)} GBIF localities from {gb_kept} specimen records ({gb_all} georeferenced before the age filter)")
    sites = pb + gb
    lat = np.array([s["lat"] for s in sites]); lon = np.array([s["lon"] for s in sites])
    print("partitioning into plates ..."); pids = plate_ids(lat, lon)
    for s, p in zip(sites, pids): s["plate"] = int(p)
    # positions: every frame inside the site's age range (+-2.5 Myr), or its nearest frame when the range is narrower
    want = collections.defaultdict(list)
    for i, s in enumerate(sites):
        if s["max_ma"] is None: continue
        if s["max_ma"] - s["min_ma"] > MAX_SPAN:              # dated no closer than "Cambrian to Devonian": kept, not placed
            s["not_placed"] = f"age range {s['max_ma'] - s['min_ma']:.0f} Myr is wider than {MAX_SPAN} Myr"; continue
        fr = [t for t in FRAMES if s["min_ma"] - 2.5 <= t <= s["max_ma"] + 2.5]
        if not fr: fr = [nearest((s["max_ma"] + s["min_ma"]) / 2, FRAMES)]
        s["frames"] = fr
        for t in fr: want[t].append(i)
    print("reconstructing ...")
    for t in FRAMES:
        idx = np.array(want[t], dtype=int)
        if not len(idx): continue
        xy = rotate(lat[idx], lon[idx], pids[idx], t)
        for i, p in zip(idx, xy): sites[i].setdefault("paleo", {})[t] = [round(float(p[0]), 2), round(float(p[1]), 2)]
    print("geography frames ..."); geo = frames_geo()
    labels, label_pids = continent_labels()
    for t in FRAMES: geo[t]["continents"] = labels[t]

    # ---- write
    meta = dict(model="PALEOMAP v19o_r1c (Scotese), as used by PaleoCoastlines v7 (Kocsis & Scotese 2021)",
                lattice=N_LAT, frames=FRAMES, built=pd.Timestamp.now().isoformat(timespec="seconds"),
                continents_plate=label_pids)
    with open(f"{OUT}/frames.json", "w") as f: json.dump(dict(meta=meta, frames={str(t): geo[t] for t in FRAMES}), f, separators=(",", ":"))
    NODE_IX = {n: i for i, n in enumerate(TREE)}
    compact = []
    for s in sites:
        top = max(s["nodes"], key=s["nodes"].get) if s["nodes"] else None
        compact.append([s["id"], round(s["lat"], 3), round(s["lon"], 3),
                        None if s["max_ma"] is None else round(s["max_ma"], 1), None if s["min_ma"] is None else round(s["min_ma"], 1),
                        s["n"], [NODE_IX[k] for k in s["nodes"]], NODE_IX.get(top, -1),
                        1 if s.get("lagerstatte") else 0, 0 if s["src"] == "PBDB" else 1,
                        {str(t): v for t, v in (s.get("paleo") or {}).items()}])
    with open(f"{OUT}/sites.json", "w") as f:
        json.dump(dict(meta=dict(fields=["id", "lat", "lon", "max_ma", "min_ma", "n", "nodes", "top", "lagerstatte", "gbif", "paleo"],
                                 nodes=TREE, count=len(compact)), sites=compact), f, separators=(",", ":"))
    with gzip.open(f"{FULL}/sites_full.json.gz", "wt", encoding="utf-8") as f:
        json.dump(sites, f, separators=(",", ":"), default=str)
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in s.items()} for s in sites]).to_csv(f"{FULL}/sites_full.csv.gz", index=False, compression="gzip")
    print("done:", {p: os.path.getsize(f"{OUT}/{p}") for p in ("frames.json", "sites.json")}, "+ full detail in", FULL)


if __name__ == "__main__":
    main()
