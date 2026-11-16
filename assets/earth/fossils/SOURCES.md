# Earth layout data: ancient geography and every georeferenced trilobite locality

Built 14 Nov 2026 by `scripts/earth/build_globe.py` from the downloads made by `scripts/earth/fetch.sh`
(PBDB and GBIF fetched on 14 Nov 2026 US Eastern, 15 Nov UTC). Rebuild: `sh scripts/earth/fetch.sh && python scripts/earth/build_globe.py`.

## What is in it

| file | what | size (gzip) |
|---|---|---|
| `web/static/earth/frames.json` | 59 frames, 540 to 250 Ma every 5 Myr: land, shelf sea and high land as bit masks on the page's 14,000-point Fibonacci lattice (the same lattice as `assets/earth/make_mask.py`), plus palaeocontinent labels | 82 KB |
| `web/static/earth/sites.json` | every locality, compact: id, present lat/lon, age range, record count, tree lineages, Lagerstätte flag, source, and its reconstructed position at every frame inside its age range | 0.8 MB |
| `assets/earth/fossils/sites_full.json.gz` (+ `.csv.gz`) | the same localities with everything known: name, place, formation/group/member, interval, environment, lithology, Lagerstätte, museum, taxa, genera, type specimens, catalogue numbers, references, licence, how the age was obtained | 3.3 MB |
| `assets/earth/fossils/gbif_datasets.csv` | the 54 GBIF datasets used, with record counts, licence and the dataset's own citation (DOI) | |
| `assets/earth/fossils/check_frames.png` | a contact sheet of 12 frames drawn from the two files above, for checking | |

## Numbers

- **PBDB**: 63,471 occurrences (56,897 Trilobita + 6,574 Agnostida; PBDB files Agnostida outside Trilobita) in 18,079 dated
  collections (Trilobita alone: 9,459 accepted names, 2,990 genera, 2,784 references, 81 countries).
- **GBIF**: 150,892 georeferenced fossil specimen records from 54 museum datasets, grouped into 12,693 localities. Excluded:
  the GBIF copy of PBDB, and 16 datasets that match trilobite names but hold living organisms (eDNA samples, herbaria,
  seaweed and insect collections; listed in `scripts/earth/gbif_exclude.py`). Largest: Swedish Museum of Natural History
  29,902; Harvard MCZ 23,526; Kansas 22,555; Oslo 20,419; Yale Peabody 18,339. 6,305 specimens carry a type status.
- **Together**: 30,772 localities, 214,363 records. Placed on the ancient globe: 27,071 localities (181,088 records). Kept but
  not placed: 2,902 dated more coarsely than 50 Myr (24,039 records) and 799 undated (9,236 records).
- **Lineages**: 93% of records map onto a node of the tree page (the rest are identified only as "Trilobita").

## How each piece was made

**Geography.** Coastlines (land) and continental margins (shelf, the −1400 m isobath) are the PaleoMAP PaleoCoastlines v7.3
(Kocsis & Scotese 2021), drawn every 5 Myr; where a 5 Myr slice is missing the nearest slice is used (`coast_slice` in each
frame). "High" land is land above 1000 m in the PALEOMAP PaleoDEMs (Scotese & Wright 2018). Each lattice point is sampled
inside or outside those polygons; nothing is drawn by hand.

**Plate model.** All positions use PALEOMAP v19o_r1c, the rotation file shipped with the coastlines. Tested against v3: with
v19o 95% of PBDB trilobite collections fall on the reconstructed continental margins (97% at 510 Ma), with v3 91% (60% at
510 Ma) (`scripts/earth/model_match.py`). Our positions agree with PBDB's own Scotese-model palaeocoordinates to a median of
0.6° latitude.

**Ages.** PBDB collections keep their own max/min ages. GBIF records: the chronostratigraphy fields on the specimen
(stage, then series, then system; 72,899 records); else, or when finer and inside the record's own range, the median age of
the PBDB collections from the same formation (36,363); else the PBDB range of the record's own genus
(32,394, coarse, flagged `inferred from the PBDB range of <genus>`); 9,236 remain undated and are kept but not placed. Every GBIF
locality records which of these it used (`age_from`).

**Positions in time.** A locality is reconstructed at every frame inside its age range (±2.5 Myr), so a site dated
"Darriwilian" appears in two frames and one dated only "Ordovician" in about nine. A locality whose age range is wider
than 50 Myr (about one period, e.g. "Cambrian to Devonian") is kept in the data but not placed (`not_placed`).

**Lineages.** Each record's PBDB parent chain is climbed to the nearest node of the tree page. PBDB's classification differs
from Gon's in four places, fixed in `build_globe.py`: Olenida → Olenina; Trinucleida → Asaphida; Illaenidae, Styginidae,
Scutelluidae, Panderiidae → Illaenina (PBDB hangs them straight off Corynexochida); other Ptychopariida → Ptychopariina.

**Continent labels.** One present-day anchor point inside each block, moved with the same model, named for the span the
block is usually given (Laurentia to 420 Ma then Laurussia; Gondwana to 300 Ma then Pangaea; Baltica, Avalonia, Siberia,
North and South China, Kazakhstania, Tarim). The anchors and spans are chosen in `build_globe.py`, not taken from a source.

## Caveats to state wherever this is shown

- **Early Palaeozoic reconstructions are model-dependent.** Against PBDB's GPlates-model palaeocoordinates, ours differ by a
  median 14° in latitude, and by ~50° great-circle in the Cambrian. Palaeolatitude is constrained by palaeomagnetism;
  palaeolongitude largely is not. The page should name the model.
- **Sampling is uneven**: US, Canada, China, UK, Czechia, Russia and Sweden dominate; the Southern Hemisphere and Africa are
  thin. Absence of dots is absence of collecting as much as absence of trilobites.
- Some museum localities are glacial erratics (e.g. the Hulterstad erratics on Öland): their present position is not where
  the rock formed.
- GBIF records are not deduplicated against PBDB: a well-studied site can appear from both.
- Museum labels can use older series names. The Trenton (Walcott-Rust Quarry, Harvard MCZ) is labelled "Middle Ordovician"
  in the old North American sense and so dated 471-458 Ma here, where the current timescale puts it in the Katian (~450 Ma).

## Licences and citation

The credits shown on the site are generated by `scripts/earth/credits.py`: `web/static/earth/data_credits.txt` (everyone,
including every PBDB contributor and every GBIF dataset citation), `pbdb_references.txt` (the 2,839 source publications)
and the Sources panel's Data section.

- PBDB data: released CC0 (PBDB's download header, checked 15 Nov 2026 UTC); credited anyway. Cite: Paleobiology Database (paleobiodb.org), downloaded 15 Nov 2026 (UTC); Peters, S. E. & McClennen, M.
  2016. The Paleobiology Database application programming interface. *Paleobiology* 42: 1–7.
- GBIF: per dataset (`gbif_datasets.csv`): 20 CC BY 4.0, 16 CC0, **18 CC BY-NC 4.0 (49,407 records), including Harvard MCZ,
  TalTech, Cincinnati Museum Center and IANIGLA**. Non-commercial terms cover anything sold (prints, an exhibition shop).
  For a publication GBIF asks for a derived-dataset DOI listing the datasets used (gbif.org/derived-dataset).
- Kocsis, Á. T. & Scotese, C. R. 2021. Mapping paleocoastlines and continental flooding during the Phanerozoic. *Earth-Science
  Reviews* 213: 103463, doi:10.1016/j.earscirev.2020.103463. Data: PaleoMAP PaleoCoastlines v7.3, Zenodo 10.5281/zenodo.7994000 (record 4017820), CC BY 4.0.
- Scotese, C. R. & Wright, N. 2018. PALEOMAP Paleodigital Elevation Models (PaleoDEMS) for the Phanerozoic. PALEOMAP Project,
  earthbyte.org; Zenodo 10.5281/zenodo.5460860, CC BY 4.0.
- Scotese, C. R. 2016. PALEOMAP PaleoAtlas for GPlates and the PaleoData Plotter Program (plate model).
- pygplates (GPlates, EarthByte group) for the plate partitioning and rotations.
