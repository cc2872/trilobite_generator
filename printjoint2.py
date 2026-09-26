"""
printjoint2.py: compatibility names for the flexi print joint. The joint itself is joints/flexi.py; building and the
overhang pass are assemble.py. New code calls assemble.animal(P, "flexi") / assemble.part(P, name, "flexi").
"""
import assemble
from joints import flexi as _FX
from joints.flexi import DEFAULTS, MIN_PITCH, clean, geometry, transforms_deg

def print_head(P, J=None):    return assemble.part(P, "head", _FX, opts=J)
def print_tail(P, J=None):    return assemble.part(P, "tail", _FX, opts=J)
def print_segment(P, i, J=None, pocket_on_first=False): return assemble.part(P, f"seg{i}", _FX, opts=J)
def print_animal(P, J=None, report=None): return assemble.animal(P, _FX, opts=J, report=report)
def restore(P, base, J=None, report=None): return assemble.restore(P, base, _FX, J, report)
def chain(P, J=None, deg=0.0, n=None):
    """All thoracic print segments posed at `deg` per joint about the print pivot (no overhang pass)."""
    import numpy as np, math, trimesh
    Jg, d, zj, y_piv = geometry(P, J); n = n or int(P["segCount"])
    segs = [assemble.part(P, f"seg{i}", _FX, opts=J) for i in range(n)]
    out, T = [], np.eye(4)
    for s in segs:
        out.append(s.copy().apply_transform(T))
        T = T @ trimesh.transformations.translation_matrix((0, d, 0)) @ trimesh.transformations.rotation_matrix(math.radians(-deg), (1, 0, 0), (0, y_piv - d, zj))
    return out
