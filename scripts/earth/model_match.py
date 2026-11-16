# Which PALEOMAP rotation version do the Kocsis & Scotese palaeocoastlines follow? Reconstruct PBDB collections to a slice
# with each version and count how many land inside that slice's continental-margin polygons (marine fossils should).
import pandas as pd, numpy as np, pygplates, geopandas as gpd, shapely
from shapely import points
import os
RAW=os.environ.get('EARTH_RAW', os.path.join(os.path.dirname(__file__),'..','..','earth_raw'))
C=RAW+'/coast/'
occ=pd.concat([pd.read_csv(RAW+'/pbdb_trilobita_occs.csv',usecols=['collection_no','lng','lat','max_ma','min_ma'],low_memory=False),
               pd.read_csv(RAW+'/pbdb_agnostida_occs.csv',usecols=['collection_no','lng','lat','max_ma','min_ma'],low_memory=False)])
col=occ.drop_duplicates('collection_no').dropna(subset=['lng','lat'])
col['mid']=(col.max_ma+col.min_ma)/2
models={'v3':(C+'paleomap_global_plate_model_v3/PALEOMAP_PlateModel.rot',C+'paleomap_global_plate_model_v3/PALEOMAP_PlatePolygons.gpml'),
        'v19o':(C+'v19o/m06c9h_3id_forPgeog_19o_r1c.rot',C+'v19o/PALEOMAP_PlatePolygons__forPgeog_v19o.gpml')}
for name,(rot,pp) in models.items():
    R=pygplates.RotationModel(rot); P=pygplates.FeatureCollection(pp)
    res=[]
    for t in (510,485,450,425,400,375,340,300,270):
        sub=col[(col.mid-t).abs()<=2.5]
        if len(sub)<30: continue
        feats=[pygplates.Feature() for _ in range(len(sub))]
        for f,(lo,la) in zip(feats,zip(sub.lng,sub.lat)): f.set_geometry(pygplates.PointOnSphere(la,lo))
        pf=pygplates.partition_into_plates(P,R,feats,properties_to_copy=[pygplates.PartitionProperty.reconstruction_plate_id])
        rec=[]; pygplates.reconstruct(pf,R,rec,float(t))
        xy=np.array([r.get_reconstructed_geometry().to_lat_lon()[::-1] for r in rec])
        cm=gpd.read_file(f'{C}paleocoastlines_v7_shapefiles/CM/{t}Ma_CM_v7.shp').union_all()
        cs=gpd.read_file(f'{C}paleocoastlines_v7_shapefiles/CS/{t}Ma_CS_v7.shp').union_all()
        pts=points(xy); inm=shapely.contains(cm,pts).mean() if len(xy) else np.nan
        res.append((t,len(xy),round(inm,3)))
    print(name,res, 'mean in-margin', round(np.mean([r[2] for r in res]),3))
