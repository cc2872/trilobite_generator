#!/bin/sh
# Downloads everything build_globe.py reads into earth_raw/ (git-ignored, ~300 MB). Re-run to refresh PBDB/GBIF.
set -e
cd "$(dirname "$0")/../.."
R=earth_raw; mkdir -p $R/coast $R/paleomap
P="https://paleobiodb.org/data1.2"
SHOW="full,paleoloc,class,strat,stratext,env,ref,coords,loc,geo,lith,abund,ident,acconly,crmod&pgm=gplates,scotese,seton"
for T in Trilobita Agnostida; do
  t=$(echo $T | tr A-Z a-z)
  curl -fsS -o $R/pbdb_${t}_occs.csv "$P/occs/list.csv?base_name=$T&vocab=pbdb&show=$SHOW&limit=all"
  curl -fsS -o $R/pbdb_${t}_taxa.csv "$P/taxa/list.csv?base_name=$T&vocab=pbdb&show=class,app,size,attr&limit=all"
done
curl -fsS -o $R/pbdb_trilobita_refs.csv "$P/occs/refs.csv?base_name=Trilobita&vocab=pbdb&show=both&limit=all"
curl -fsS -o $R/pbdb_intervals.csv "$P/intervals/list.csv?all_records&vocab=pbdb"
python3 scripts/earth/gbif_pull.py $R/gbif_trilobita_fossils.csv
Z="https://zenodo.org/api/records/7994000/files"            # PaleoCoastlines v7.3 (Kocsis & Scotese), CC BY 4.0
for f in Readme.md paleocoastlines_v7_shapefiles.zip paleomap_global_plate_model_v19o_r1c.zip paleomap_political_boundaries_v3.zip; do
  curl -fsSL -o $R/coast/$f "$Z/$f/content"; done
( cd $R/coast && unzip -qo paleocoastlines_v7_shapefiles.zip -d paleocoastlines_v7_shapefiles && unzip -qo paleomap_global_plate_model_v19o_r1c.zip -d v19o \
  && unzip -qo paleomap_political_boundaries_v3.zip -d paleomap_political_boundaries_v3 )
E="https://www.earthbyte.org/webdav/ftp/Data_Collections/Scotese_Wright_2018_PaleoDEM"
curl -fsSL -o $R/paleomap/PaleoDEMS_long_lat_elev_csv_v2.zip "$E/PaleoDEMS_long_lat_elev_csv_v2.zip"
( cd $R/paleomap && unzip -qo PaleoDEMS_long_lat_elev_csv_v2.zip -d dem )
echo "fetched into $R"
