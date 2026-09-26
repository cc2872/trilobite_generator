"""
joints/pin.py: the measured joint. Interleaving-knuckle pin hinge with a ventral bevel stop, exactly the geometry
the frozen readings (tests/references) were made with. MEASURED: changing anything here is an instrument change.

The hinge line, knuckle span, band and reach rules are trilobite.add_hinge's (8 to 11 Sep 2026), moved here from
parts.py unchanged. The band follows the port's local half-width (11 Sep: harpetida seg7) and the last segment's
rear band overspans the plate (12 Sep: asaphida seg10). The geometry itself is cut by mesh.hinge().
"""
import math
import numpy as np
import mesh as M
from fields import seg_halfwidth, furrow_amp
from anatomy.common import pitch, ring_top

NAME = "pin"; MEASURED = True; BASE = "shell"
WEDGE_REACH = 0.15        # seg-seg joints: bevel reach past the hinge line, fraction of the pitch (trilobite.py, 8 Sep)
WEDGE_REACH_WIDE = 0.5    # head-seg0 and last-seg-tail joints (full-width plates, no tips to sever)
TIP_KEEP_MM = 3.0         # the bevel band stops this far short of a segment's pleural tip (11 Sep: harpetida seg7)
WIDE_OVERSPAN_MM = 4.0    # last-seg rear band overspans the local plate by this much so no tip is stranded outboard

# ---------------------------------------------------------------- where the hinge is (trilobite.py, unchanged)
def hinge_z(P): return ring_top(P) - P["barrelR"] - P["wall"] - P["clearance"] - 0.4 - 0.7 * furrow_amp(P)
def hinge_width(P): return 2 * P["axisFrac"] * seg_halfwidth(P, P["segCount"] - 1) - 2
def joint_offsets(P): return [0] + [pitch(P)] * int(P["segCount"])

def fits(P):
    """The pin always builds; the instrument's own gates (rest overlap, unsane parts) judge it."""
    return True
def pivot(P): return dict(z=round(hinge_z(P), 2), y_beyond_plane=0.0)
def stop_deg(P): return float(P["maxAngle"])

# ---------------------------------------------------------------- the bevel band (parts._hinge_geometry, unchanged)
def _band(P, port):
    """Bevel band width and reach. 11 Sep 2026: the band follows the LOCAL half-width of the plate (the BREP builder
    sized it from the animal's maximum width, so on a narrow rear segment the 'axial band + 0.12 W' covered nearly the
    whole pleura and the wedge severed its tips; OCC's grid-jitter retries had been hiding it).
    Fix: seg-seg bands are capped 3 mm short of the local pleural tip, so the tips stay for the instrument to find.
    Wide joints (head-seg0, last seg-tail) keep the global-width rule the readings were validated on."""
    W = P["width"]; wide = port.wide; halfwidth = port.halfwidth
    if P["bladeChord"] < 1.0 and not wide: band = 2 * (P["axisFrac"] * (W / 2)) + 0.12 * W
    else:                                    band = 2 * (P["axisFrac"] * (W / 2)) + 0.45 * W
    if not wide and halfwidth is not None:                      # seg-seg joints: never within 3 mm of the local tip
        band = min(band, 2 * (halfwidth - TIP_KEEP_MM))         # (wide joints keep the validated global-width rule)
    if port.overspan and halfwidth is not None:                 # last-segment rear joint ONLY: the wide wedge must
        band = max(band, 2 * halfwidth + WIDE_OVERSPAN_MM)      # span the whole tapered plate, or the outboard
        #  pleural tip is stranded as an orphan body (never cut, just disconnected: 68 mm3 pair, 12 Sep asaphida
        #  seg10). Head and pygidium are NOT overspanned: their bands were validated on the frozen readings.
    reach = (WEDGE_REACH_WIDE if wide else WEDGE_REACH) * pitch(P)
    return band, reach

# ---------------------------------------------------------------- the template
def cut(body, env, P, port, bevel_deg=None):
    """parts.add_hinge, unchanged: knuckles, barrel, bore, stop block and the bevel wedge on one edge.
    bevel_deg = None uses the printed stop P['maxAngle']; the instrument passes its own fixed bevel."""
    band, reach = _band(P, port)
    out = M.hinge(body, env, y_axis=port.y, rear=port.rear, zh=hinge_z(P), Wh=hinge_width(P), barrel_r=P["barrelR"],
                  clearance=P["clearance"], n_knuckles=int(P["nKnuckles"]), ring_top=ring_top(P), wall=P["wall"],
                  bore_d=P["boreDia"], band=band, reach=reach,
                  bevel_deg=(P["maxAngle"] if bevel_deg is None else float(bevel_deg)), wide=port.wide)
    return out, np.eye(4)

def overhang(P, part, S, port, opts=None):
    """The pin never trims a plate: nothing to put back."""
    return None

def _rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])
def _trans(x, y, z):
    m = np.eye(4); m[:3, 3] = (x, y, z); return m

def pose(P, angles_deg):
    """4x4 per part, joint k bent by angles_deg[k] about the hinge line (head = identity). instrument.transforms_deg."""
    zh = hinge_z(P); offs = joint_offsets(P)
    mats = [np.eye(4)]; Mx = np.eye(4)
    for i in range(len(offs)):
        Mx = Mx @ _trans(0, offs[i], 0) @ _trans(0, 0, zh) @ _rot_x(-float(angles_deg[i])) @ _trans(0, 0, -zh)
        mats.append(Mx)
    return mats

def transforms_deg(P, theta_deg): return pose(P, [theta_deg] * len(joint_offsets(P)))

def print_keepout(P, port):
    """The box a shell thickener must leave clear around this hinge (printfill): hinge width x wedge reach, below
    the barrel tops."""
    z_keep = hinge_z(P) + P["barrelR"] + P["clearance"] + 0.5
    reach = max(WEDGE_REACH_WIDE, WEDGE_REACH) * pitch(P) + P["barrelR"] + 1.0
    return [M.box(hinge_width(P) + 2.0, 2 * reach, 200.0, at=(0, port.y, z_keep), align=("c", "c", "max"))]
