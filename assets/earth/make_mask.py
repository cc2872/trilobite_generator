"""Land mask for the tree page's Earth layout, from assets/earth/Earth.stl (uploaded 3 Oct 2026).

The model is a globe of radius ~39.3 (sea) with the land raised (>= ~40.2). It is z-up and its
longitude zero is 90 degrees west of Greenwich, so true longitude = model longitude + 90.
Samples N Fibonacci points; a point is land if the model's surface there is above 39.8, and
high land above 40.75. Prints the two bit strings web/tree.html carries (EARTH = {...}).
"""
import os, json, base64, numpy as np, trimesh
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
N, SEA, HIGH = 14000, 39.8, 40.75
def mask():
    m = trimesh.load(os.path.join(HERE, "Earth.stl")); v = m.vertices; r = np.linalg.norm(v, axis=1)
    i = np.arange(N); lat = np.arcsin(1 - (2*i + 1)/N); lon = (i*np.pi*(3 - 5**.5)) % (2*np.pi) - np.pi
    ls = lon - np.pi/2
    p = np.c_[np.cos(lat)*np.cos(ls), np.cos(lat)*np.sin(ls), np.sin(lat)]
    rr = r[cKDTree(v/r[:, None]).query(p, k=3)[1]].mean(1)
    b = lambda a: base64.b64encode(np.packbits(a.astype(np.uint8), bitorder="little").tobytes()).decode()
    return {"N": N, "land": b(rr > SEA), "high": b(rr > HIGH)}
if __name__ == "__main__":
    print(json.dumps(mask()))
