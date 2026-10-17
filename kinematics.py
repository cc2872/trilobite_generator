"""
kinematics.py: the enrolment bend profile for the PRINT model and the site (17 Oct 2026).

Instrument 2.1 bends every joint by the same angle and is NOT changed by this file (instrument.py, joints/pin.py,
tests/references and the pre-registration stay as they are). The print model (isopod_model), the sheet and the
site's curl follow the expression Esteve, Rubio, Zamora & Rahman 2017 (Palaeontology 60: 423-432) fitted to
CT-scanned enrolled conocoryphids, restated in Esteve & Hughes 2023 (Proc. R. Soc. B 290: 20230871, Methods):

    theta(i) = Ln[a * i] * b          i = 1, 2, ... counts the thoracic joints from the one behind the first segment

"where a and b are two variables and depend on the particular enrolment style ... i is a counter for successive
segments". Equivalently theta(i) = c + b ln i with c = b ln a: a straight line in ln i, c the first thoracic
joint's angle, b the change per e-fold of joint number (negative: the angles fall towards the tail). The
cephalon / first-segment joint is not on the curve; they set it from the enrolled fossils (about 44.6 and 46.2 deg).

The constants here are the fit to their Fig. 5A (the spiral model, Bailiaspis? glabrata, 15 segments): a = 0.00194,
b = -3.686 deg, so c = 23.0 deg. Against the fourteen angles read off that figure the fit's largest residual is
0.05 deg, under the reading error; the total over 14 thoracic joints is 14 c + b ln(14!) = 229 deg, plus 45 at the
head = 274 deg, their total. Their sphaeroidal curve (Fig. 5B, 21.8 falling to 3.3 deg over 13 thoracic joints)
does NOT follow this form as read: its drop accelerates towards the tail. Their a, b per style are in the 2023
paper's table S1, not seen; until then the sphaeroidal case is not represented here.

Because b is a rate per e-fold, the last joint's angle and the total depend on the joint count: a 15-joint chain
ends at 13.3 deg and sums to 274; a 10-joint chain ends at 14.9 deg and sums to 205. That is the mechanism
behind their result that 15 segments cannot close sphaeroidally and 14 can.

    profile(n)            degrees per joint for n joints, head joint first: [HEAD_DEG, theta(1), ..., theta(n-1)]
    angles(n, head_deg)   the profile scaled so the head joint is at head_deg (the one scalar sweeps and sliders use)
    total(n, head_deg)    its sum
    c_first()             the first thoracic joint's angle at full profile, b ln a

The physical print joint has its own limit (the isopod kit: about 36 deg per joint before neighbours touch,
segment/dome.py), so on the print model the head joint may stop before 45: the reading says so, the shape holds.
"""
import math

HEAD_DEG = 45.0        # cephalon / first segment, set from the fossils, not from the expression
A = 0.00194            # Esteve et al. 2017 spiral model (fit to Fig. 5A)
B = -3.686             # deg per e-fold of thoracic joint number
SOURCE = "Esteve et al. 2017, theta(i) = Ln[a i] b, a = 0.00194, b = -3.686 (Fig. 5A fit)"


def theta(i, a=A, b=B):
    """The expression itself: thoracic joint i (1 = behind the first segment), degrees, never below zero."""
    return max(b * math.log(a * i), 0.0)


def c_first(a=A, b=B): return b * math.log(a)


def profile(n, head_deg=HEAD_DEG, a=A, b=B):
    """Degrees per joint for a chain with n joints, head joint first."""
    n = int(n)
    if n <= 0: return []
    return [float(head_deg)] + [theta(i, a, b) for i in range(1, n)]


def angles(n, head_deg, **kw):
    """The profile scaled so the head joint turns head_deg (the one scalar the sweeps, sliders and checks use)."""
    s = float(head_deg) / HEAD_DEG
    return [s * x for x in profile(n, **kw)]


def total(n, head_deg, **kw): return float(sum(angles(n, head_deg, **kw)))


def label(n, head_deg):
    """One line for sheets and readouts."""
    x = angles(n, head_deg)
    if len(x) <= 1: return f"{head_deg:g} deg"
    return f"head {x[0]:.0f} deg, then {x[1]:.0f} to {x[-1]:.0f} deg, {sum(x):.0f} deg over {n} joints"


# ---------------------------------------------------------------- where the axis is: Esteve's dorsal pivot vs the kit's ball
def dorsal_pivots(pieces, half_width=2.0, edge_band=1.0):
    """Esteve et al. 2017 (after Bruton & Haas 1997) rotate each segment about the anterior edge of its pleura, which fits
    below the posterior edge of the segment ahead: a DORSAL axis at the plate overlap. For a chain of pieces in one frame
    (head first, y forward, z up, x the midline), joint k's dorsal pivot is the rear edge of piece k's plate on the
    midline: y = the rearmost point of piece k within |x| < half_width, z = the underside of the plate at that edge (the
    lowest point within edge_band of it). Returns [(y, z)] per joint, in the pieces' units.

    The print model's kit pivots at the ball instead (a ventral axis, under the plates). Posing the printed pieces about
    these dorsal pivots shows what the fossils do with our dorsal geometry; it is not a pose the printed object can take."""
    import numpy as np
    out = []
    for k in range(len(pieces) - 1):
        V = np.asarray(pieces[k].vertices); mid = V[np.abs(V[:, 0]) < half_width]
        if len(mid) == 0: mid = V
        y_rear = float(mid[:, 1].max()); edge = mid[mid[:, 1] > y_rear - edge_band]
        out.append((y_rear, float(edge[:, 2].min())))
    return out
