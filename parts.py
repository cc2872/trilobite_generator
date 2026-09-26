"""
parts.py: compatibility names for the flat builder (11 to 26 Sep 2026). The code now lives in anatomy/ (the parts,
joint-free), joints/pin.py (the measured joint) and assemble.py (where they meet). New code imports those; this
file keeps the old names working: parts.cephalon / segment / pygidium build the pin-jointed parts exactly as before.
"""
import numpy as np
import assemble
from joints import pin as _PIN
from anatomy.common import pitch, ring_top, smoothstep, plateau, trough, vault, spine_solid, prong, safe_expr, GRID_SEG, GRID_TAIL
from anatomy.thorax import plan as segment_plan, cells as segment_cells, pleura_lengths
from anatomy.tail import plan as pygidium_plan, cells as pygidium_cells
from anatomy.head import plan as cephalon_plan, cells as cephalon_cells, eye_geometry, fov, eye_solid, lens_centres, eye_params, FOV_VERSION
from joints.pin import hinge_z, hinge_width, joint_offsets, WEDGE_REACH, WEDGE_REACH_WIDE, TIP_KEEP_MM, WIDE_OVERSPAN_MM

def cephalon(P, bevel_deg=None, grid=(121, 61), notes=None): return assemble.part(P, "head", _PIN, bevel_deg=bevel_deg, grid=grid, notes=notes)
def segment(P, i, bevel_deg=None, grid=GRID_SEG):            return assemble.part(P, f"seg{i}", _PIN, bevel_deg=bevel_deg, grid=grid)
def pygidium(P, bevel_deg=None, grid=GRID_TAIL):             return assemble.part(P, "tail", _PIN, bevel_deg=bevel_deg, grid=grid)

def add_hinge(part, env, P, y_axis, rear, wide=False, bevel_deg=None, halfwidth=None, overspan=False):
    """The pin joint on one edge (old signature)."""
    from anatomy.port import Port
    out, _ = _PIN.cut(part, env, P, Port(y=y_axis, rear=rear, wide=wide, halfwidth=halfwidth, ring_half=0.0, overspan=overspan), bevel_deg=bevel_deg)
    return out
