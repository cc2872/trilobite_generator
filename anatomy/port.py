"""
anatomy/port.py: the one thing a part tells a joint about itself.

A Port is a joint plane on a part: where it is, which way the part lies, and how wide the plate is there. A joint
module reads a Port and the parameter table; it never reads a part's plan. If a new head or thorax describes its
ports the same way, every joint fits it.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Port:
    y: float               # the joint plane, part frame (mm)
    rear: bool             # True: this is the part's rear edge (the next part follows at +y)
    wide: bool             # full-width plate on both sides (head-seg0, last seg-tail): the pin's wide band rule
    halfwidth: float       # the plate's local half-width at the port (to the pleural tip)
    ring_half: float       # half-width of the axial ring at the port
    overspan: bool = False # last segment's rear only: the pin band overspans the plate so no tip is stranded
    kind: str = "seg"      # "head", "seg" or "tail": a print joint trims a segment's run but cuts a head/tail at the plane
    shingle: float = 0.0   # length of the plan's front band that shingles under the previous part (a trimming joint starts its run there)
