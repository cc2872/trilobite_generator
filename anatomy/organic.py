"""
anatomy/organic.py (1 Nov 2026): the ORGANIC protocol -- one set of rules every builder applies to the thin parts of an
animal, so nothing comes out as a needle, a wire, a sawn-off blade or a hard corner. One strength, k = P["organic"]
(0..1; 1 by default; 0 = the geometry exactly as built before, which the instrument and the frozen references use).

  1. SLENDERNESS   a spine's root radius is at least its length / SLENDER: a long spine is a horn, not a wire.
  2. TAPER         the radius falls along a full, convex profile (thick for longer, then easing off), not a straight cone,
                   and ends in a round tip no thinner than TIP_FRAC of the root -- never a needle. A lobe drawn on a
                   margin (the tail's) ends round too: its apex is eased over LOBE_ROUND of its half-width.
  3. ROOT          every spine grows out of its plate through a flared collar at least FLARE x its root radius.
  4. SECTION       a spine drawn in a plate's plan (the thorax's pleural spines, the genal piece) keeps rules 1 and 2 in its
                   plan (SLENDER_PLAN, a round cap), and is a rounded ridge: its crest stands SECTION x its half-width
                   above its edges.
  5. BENDS         a kink is a curve: the genal piece bends through a fillet of at least BEND_W x its width.
  6. EDGES         a plate's free edge is rolled, not cut square: the top falls to the edge through a quarter round of up
                   to EDGE_R (a solid part's) or, on a plate lifted off the bed, a bullnose its own thickness round.

Where it applies: anatomy/common.curved_spine and spine_solid (every tube: occipital, axial, pleural horns, genal horns,
prongs and their tines, tail spines, the studio's free spines), anatomy/thorax.plan (pleural spines in the plan),
anatomy/tail.plan (the margin's lobes), anatomy/genal.Piece (tip, kink), anatomy/head.plan (the genal piece's section), anatomy/head.sphere_eye (the stalk) and
isopod_model/body.build_part (rule 6, the studio's parts). The builders read STRENGTH, which the
entry points set from P (assemble.part, isopod_model/body.build); instrument.py builds with organic 0.
"""
import numpy as np

SLENDER = 40.0      # length / root radius, at most (a 160 mm spine is at least 4 mm in radius at its root)
TIP_FRAC = 0.4      # tip radius / root radius, at least
TAPER_EXP = 0.6     # radius = tip + (root - tip) (1 - t)^TAPER_EXP at full strength (1 = the straight cone)
FLARE = 1.5         # root collar width / root radius, at least
SLENDER_PLAN = 24.0 # the same for a spine drawn in a plan (length / root HALF-width): flatter than a tube, so a little wider
SECTION = 0.6       # a plan spine's crest stands SECTION x its half-width above its edges (a rounded ridge, not a flat strip)
BEND_W = 1.5        # a kink's fillet radius / the piece's width, at least
LOBE_ROUND = 0.3    # a lobe drawn on a margin (the tail's): its apex eased over this fraction of its half-width
EDGE_R = 1.6        # the roll on a solid part's free edge, model mm (at most; never more than a third of the plate's width there)

STRENGTH = 1.0      # set per build by set_from(P)


def k_of(P=None):
    """The strength for a parameter table (its own 'organic' when it has one), else the module's."""
    if P is not None and "organic" in P:
        try: return float(np.clip(float(P["organic"]), 0.0, 1.0))
        except (TypeError, ValueError): return STRENGTH
    return STRENGTH


def set_from(P):
    """Set the module's strength from P when P says (entry points call this before building); returns it."""
    global STRENGTH
    if P is not None and "organic" in P: STRENGTH = k_of(P)
    return STRENGTH


def strength(k=None):
    """k when given (clipped to 0..1), else the build's."""
    return STRENGTH if k is None else float(np.clip(float(k), 0.0, 1.0))


def spine_radii(base_r, tip_r, length, k=None):
    """Rules 1 and 2 on a tube's radii: (root, tip)."""
    k = STRENGTH if k is None else k
    if k <= 0: return float(base_r), float(tip_r)
    r0 = max(float(base_r), k * float(length) / SLENDER)
    return r0, max(float(tip_r), k * TIP_FRAC * r0)


def taper(ts, base_r, tip_r, k=None):
    """Rule 2: the radius along t in [0, 1] -- the straight cone at k = 0, the convex profile at k = 1."""
    k = STRENGTH if k is None else k
    ts = np.asarray(ts, float)
    e = 1.0 - k * (1.0 - TAPER_EXP)
    return tip_r + (base_r - tip_r) * (1.0 - ts) ** e


def flare(f, k=None):
    """Rule 3: the collar."""
    k = STRENGTH if k is None else k
    return max(float(f), 1.0 + k * (FLARE - 1.0))


def plan_spine(x, x0, x1, r_root, L, k=None, cap=True):
    """Rules 1, 2 and 4 for a spine drawn in a plan from x0 (where it leaves the plate; r_root its half-width there) to
    x1 (its tip), along x: its half-width at x (perpendicular to it), and its root half-width after rule 1."""
    k = STRENGTH if k is None else k
    x = np.asarray(x, float)
    r0 = max(float(r_root), k * float(L) / SLENDER_PLAN); rt = k * TIP_FRAC * r0
    c = min(rt, 0.5 * max(x1 - x0, 1e-9)) if cap else 0.0              # the round cap's length along x
    xc = x1 - c
    p = np.clip((x - x0) / max(xc - x0, 1e-9), 0, 1)
    hw = taper(p, r0, rt, k)
    if c > 1e-9:
        qc = np.clip((x - xc) / c, 0, 0.97); hw = np.where(x > xc, rt * np.sqrt(np.maximum(1 - qc * qc, 0.0)), hw)   # (never quite a point:
                                                                                                                          # a grid's end column stays a column)
    return hw, r0


def dome(s_across, hw, k=None):
    """Rule 4's section: how far above its edges a plan spine stands at s_across (-1 .. 1 across it), hw its half-width."""
    k = STRENGTH if k is None else k
    s_across = np.clip(np.asarray(s_across, float), -1, 1)
    return k * SECTION * np.asarray(hw, float) * np.sqrt(np.maximum(1 - s_across ** 2, 0.0))


def edge_drop(d, R, k=None):
    """Rule 6: how far a top surface falls at distance d from a free edge -- a quarter round of radius R (R at the edge,
    0 from R in). R may be an array."""
    k = STRENGTH if k is None else k
    R = k * np.asarray(R, float)
    q = np.clip(1 - np.asarray(d, float) / np.maximum(R, 1e-9), 0, 1)
    return np.where(R > 1e-6, R * (1 - np.sqrt(np.maximum(1 - q * q, 0.0))), 0.0)
