"""printfill.py — thicken the tracked (pin-jointed) parts for printing without touching the measured geometry.

The tracked builder makes a 2 mm shell; printed, it is flimsy. `thicken(part, plan, depth)` adds the layer between the
dorsal surface and the same surface shifted down by `depth`, i.e. a thicker shell that follows the animal exactly and
leaves the hinge barrels, bores and knuckles untouched (they sit under the ring at ring_top - 3.4 mm and below, so
depths <= 3 mm never reach them; deeper fills are clipped 0.5 mm clear of the hinge band). depth = 0 is identity.
Print only: /api/build applies it when the request carries "fill": <mm>; the cache key gets a -fill<mm> suffix.
"""
import numpy as np, mesh as M, assemble
from anatomy import head as HEAD, thorax as THORAX, tail as TAIL
from anatomy.common import pitch, GRID_SEG, GRID_TAIL
from anatomy.port import Port
from joints import pin as PIN


def thicken(part, S, depth, P=None, joints_y=(), no_fill_y=None, grid=GRID_SEG):
    """joints_y: y of each hinge line on this part; a box around each (hinge width x wedge reach, below the barrel
    tops) is kept clear so the fill never touches barrels, bores or knuckles. Everywhere else the fill goes down
    `depth` under the surface — pleurae included, which is where the printed shell is thinnest."""
    if depth <= 0.05: return part
    if P is not None: depth = min(float(depth), P["marginHeight"] * P["relief"] - 1.5)   # never fill the pleurae down to the bed
    if depth <= 0.05: return part
    env = M.to_manifold(M.under_envelope(S["outline"], S["zfun"], nu=grid[0], nv=grid[1], floor=-60.0))
    layer = env - env.translate((0, 0, -float(depth)))            # surface .. surface-depth
    layer = layer ^ M.to_manifold(M.box(1000, 1000, 1000, at=(0, 0, 0.6), align=("c", "c", "min")))   # and never below z = 0.6
    if P is not None:
        for y in joints_y:                                             # the pin joint says what to leave clear
            for box in PIN.print_keepout(P, Port(y=y, rear=True, wide=False, halfwidth=0.0, ring_half=0.0)):
                layer = layer - M.to_manifold(box)
    if no_fill_y is not None:                                      # the flap that rides OVER the next part stays a shell:
        y0, y1 = no_fill_y                                         # thickening it downward would land on that part
        layer = layer - M.to_manifold(M.box(1000, y1 - y0, 1000, at=(0, y0, 0), align=("c", "min", "c")))
    return M.from_manifold(M.to_manifold(part) + layer)


def cephalon(P, depth):
    ovl = P["overlap"] * pitch(P)
    return thicken(assemble.part(P, "head", PIN), HEAD.plan(P), depth, P, joints_y=(0.0,), no_fill_y=(-ovl - 0.5, 5.0))
def segment(P, i, depth):
    d = pitch(P)
    return thicken(assemble.part(P, f"seg{i}", PIN), THORAX.plan(P, i), depth, P, joints_y=(0.0, d), no_fill_y=(d - 0.5, d + 20.0))
def pygidium(P, depth): return thicken(assemble.part(P, "tail", PIN), TAIL.plan(P), depth, P, joints_y=(0.0,), grid=GRID_TAIL)
