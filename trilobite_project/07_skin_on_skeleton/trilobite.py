"""Shim: what instrument.py needs from trilobite.py, sourced from fields.py (no CAD kernel)."""
from fields import pitch, ring_top, hinge_z, hinge_width
def joint_offsets(P): return [0] + [pitch(P)] * P["segCount"]
