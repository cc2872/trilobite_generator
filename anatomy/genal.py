"""
anatomy/genal.py (1 Nov 2026): the genal piece -- a spine, a prolongation or a lobe at the genal angle -- as a shape
assembled from three parts, shared by the generator's head (anatomy/head.py) and the studio's (isopod_model/shaped_head.py).

  1. A capsule from the root on the margin, aimed anywhere: angle 0 = straight back, +90 = straight out, 180 = forward,
     -90 = in over the body (the full circle).
  2. An optional kink: from the kink (a fraction of the length) a second capsule runs on, aimed anywhere on its own.
     The two capsules meet in a round joint.
  3. The tip, from pointed (tip 0: the width runs down to a point) to round (tip 1: a parallel band with a round end).
  4. A web: an edge from an anchor on the rear margin to the tip, filling the corner between the margin and the spine.
     Its roundness: 0 = a straight edge, +1 = a curve that hugs the corner (a fillet), -1 = a curve that bulges out.

Frame: x across (outward +), y along (rear +), the RIGHT side; the caller mirrors. Units are the caller's.

    Piece(root, L, W, angle, kink=0, kink_angle=None, tip=0, web=None, web_round=0, hw_min=0.3)
      .bands       [(name, outline(u, v), s_of(u))] -- each a plan band for mesh.heightfield_shell(symmetric=False):
                   u in [-1, 1] runs along it, v in [0, 1] across; s_of(u) is the position along the whole piece (0 root,
                   1 tip) for the caller's height profile. The first band starts a little inside the head (overlap).
      .tip, .kink, .root, .anchor   points;  .skeleton  (n, 2) the centreline root -> tip
      .polygon()   the plan outline (a shapely Polygon) of everything above, for drawings
      .notes       what had to be refused
"""
import math
import numpy as np

def dirv(deg):
    """Unit vector for an angle in this frame: 0 = +y (back), +90 = +x (out)."""
    a = math.radians(float(deg)); return np.array([math.sin(a), math.cos(a)])

def _wrap(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


class Piece:
    def __init__(self, root, L, W, angle, kink=0.0, kink_angle=None, tip=0.0, web=None, web_round=0.0, hw_min=0.3,
                 inward=(-0.6, -0.8), stub=None):
        self.notes = []
        R = np.asarray(root, float); L = float(max(L, 0.0)); W = float(max(W, 2 * hw_min)); tip = float(np.clip(tip, 0.0, 1.0))
        kink = float(np.clip(kink, 0.0, 0.9)); a1 = _wrap(angle); a2 = _wrap(a1 if kink_angle is None else kink_angle)
        self.root, self.L, self.W, self.tip_round = R, L, W, tip
        self.hw_min = hw_min
        rc = 0.5 * W * tip                                         # the round end's radius (0 for a point)
        Ls = max(L - rc, 0.25 * L)                                 # the centreline up to where the round end starts
        rc = min(rc, L - Ls) if L > 0 else 0.0
        self.Ls, self.rc = Ls, rc
        def hw(s):                                                 # half-width along the centreline, s = 0 .. Ls
            q = np.clip(np.asarray(s, float) / max(Ls, 1e-9), 0, 1)
            return np.maximum(0.5 * W * ((1 - tip) * (1 - q) + tip), hw_min)
        self.hw = hw
        d1 = dirv(a1); d2 = dirv(a2)
        if kink > 0.01 and Ls > 0:
            L1 = kink * Ls; L2 = Ls - L1; K = R + L1 * d1; E = K + L2 * d2
            segs = [(R, K, 0.0, L1, d1), (K, E, L1, Ls, d2)]
        else:
            K = R + 0.5 * Ls * d1; E = R + Ls * d1; segs = [(R, E, 0.0, Ls, d1)]; a2 = a1
        self.angles = (a1, a2); self.kinked = len(segs) == 2
        self.kink = K; self.end = E
        dl = segs[-1][4]; self.tip = E + rc * dl                   # the furthest point: the round end's apex (or the point)
        self.skeleton = np.vstack([R] + [s[1] for s in segs] + ([self.tip] if rc > 1e-9 else []))
        self.bands = []
        # the root stub: a short round-ended band from inside the head out to the root, so the piece is joined to the head
        # whichever way it points (a spine aimed forward or out leaves the margin at an angle; the stub fills the corner)
        iv = np.asarray(inward, float); iv = iv / max(np.hypot(*iv), 1e-9); Ls_stub = (0.8 * W if stub is None else stub)
        if Ls_stub > 1e-6:
            Rin = R + Ls_stub * iv; dstub = -iv; h0 = float(hw(0.0))
            self.bands.append(("root", self._capsule(Rin, dstub, 0.0, Ls_stub, 0.0, h0, hfun=lambda s_: h0), lambda u: 0.0))
        for i, (p0, p1, s0, s1, d) in enumerate(segs):
            last = i == len(segs) - 1
            back = 0.3 * float(hw(s0)) if i == 0 else 0.6 * float(hw(s0))   # each band starts a little inside the one before
            cap = rc if last else float(hw(s1))                    # every segment ends round: the joint, or the tip
            if last and rc <= 1e-9: cap = 0.0                      # ... except a pointed tip
            self.bands.append((f"segment{i + 1}", self._capsule(p0, d, s0, s1, back, cap), self._s_of(s0, s1, back, cap)))
        self.anchor = None
        if web is not None:
            A = np.asarray(web, float)
            if np.hypot(*(A - R)) > 0.25 * W:
                b = self._web(A, float(np.clip(web_round, -1, 1)))
                if b is not None: self.anchor = A; self.bands.append(("web", b, lambda u: 0.5 * (u + 1)))

    # a straight band from p0 along d, s0 .. s1 on the centreline, starting `back` behind p0, ending in a round cap
    def _capsule(self, p0, d, s0, s1, back, cap, hfun=None):
        n = np.array([d[1], -d[0]])                                # the band's left (outer for a spine running back)
        Lseg = s1 - s0; T0, T1 = -back, Lseg + cap
        hw = self.hw if hfun is None else (lambda s_: hfun(s_ - s0)); hm = self.hw_min
        def outline(u, v):
            t = T0 + (T1 - T0) * 0.5 * (u + 1)
            if t <= 0: h = float(hw(s0))
            elif t <= Lseg: h = float(hw(s0 + t))
            else:                                                  # the round end (an exact half-disc of radius cap)
                q = min((t - Lseg) / max(cap, 1e-9), 1.0); h = max(float(hw(s1)) * math.sqrt(max(1 - q * q, 0.0)), hm if q < 1 else 0.0)
                if cap <= 1e-9: h = float(hw(s1))
            c = p0 + t * d
            return float(c[0] + n[0] * (v - 0.5) * 2 * h), float(c[1] + n[1] * (v - 0.5) * 2 * h)
        return outline

    def _s_of(self, s0, s1, back, cap):
        Lseg = s1 - s0; T0, T1 = -back, Lseg + cap; Ltot = max(self.Ls + self.rc, 1e-9)
        return lambda u: float(np.clip((s0 + T0 + (T1 - T0) * 0.5 * (u + 1)) / Ltot, -0.5, 1.0))

    # the web: inner edge the anchor -> tip curve, outer edge the centreline root -> tip; both run to the tip
    def _web(self, A, rho):
        R, T = self.root, self.tip
        M = 0.5 * (A + T); C = M + rho * (R - M)                   # quadratic Bezier A -> C -> T: rho 1 hugs the corner (C = R)
        tt = np.linspace(0, 1, 121)[:, None]
        inner = (1 - tt) ** 2 * A + 2 * (1 - tt) * tt * C + tt ** 2 * T
        sk = self.skeleton; seg = np.r_[0, np.cumsum(np.hypot(*np.diff(sk, axis=0).T))]
        tq = np.linspace(0, seg[-1], 121); outer = np.c_[np.interp(tq, seg, sk[:, 0]), np.interp(tq, seg, sk[:, 1])]
        def arclen_resample(P_, n=121):
            s = np.r_[0, np.cumsum(np.hypot(*np.diff(P_, axis=0).T))]; q = np.linspace(0, s[-1], n)
            return np.c_[np.interp(q, s, P_[:, 0]), np.interp(q, s, P_[:, 1])]
        inner = arclen_resample(inner); outer = arclen_resample(outer)
        try:
            from shapely.geometry import Polygon
            poly = Polygon(np.vstack([inner, outer[::-1]]))
            if not poly.is_valid or poly.area < 1e-6:
                self.notes.append("web left off: its edge crosses the spine (aim the spine back, or move the anchor)"); return None
        except ImportError:
            pass
        lead = 0.5 * self.W                                        # start a little inside the head (toward the front: -y)
        fwd = np.array([0.0, -lead])
        def outline(u, v):
            q = 0.5 * (u + 1)
            if q < 0.08:                                           # the first stretch reaches forward into the head
                f = (0.08 - q) / 0.08; a_, b_ = inner[0] + f * fwd, outer[0] + f * fwd
            else:
                k = (q - 0.08) / 0.92 * (len(inner) - 1); i = int(min(k, len(inner) - 2)); w = k - i
                a_ = inner[i] * (1 - w) + inner[i + 1] * w; b_ = outer[i] * (1 - w) + outer[i + 1] * w
            p = a_ + v * (b_ - a_); return float(p[0]), float(p[1])
        self.web_inner = inner
        return outline

    def polygon(self, n_u=61, n_v=7):
        """The plan outline of every band, unioned (shapely)."""
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        polys = []
        for _, outline, _ in self.bands:
            us = np.linspace(-1, 1, n_u)
            left = [outline(u, 0.0) for u in us]; right = [outline(u, 1.0) for u in us[::-1]]
            p = Polygon(left + right)
            if not p.is_valid: p = p.buffer(0)
            polys.append(p)
        return unary_union(polys)

    def handles(self):
        """Where the studio draws its grips: root, kink (the middle when there is none), tip, web anchor."""
        return dict(root=self.root, kink=self.kink, tip=self.tip, anchor=self.anchor)


# ---------------------------------------------------------------- the old keys (schema <= 6.5, studio outline <= 31 Oct)
def from_old_path(xs, ys):
    """An old centreline (root first) as (angle, kink, kink_angle): two chords, root -> middle and middle -> tip
    (one straight piece when they differ by under 3 degrees). The old spines were smooth curves; this keeps both ends
    and the middle where they were."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    if s[-1] < 1e-9: return 0.0, 0.0, 0.0
    m = np.interp(0.5 * s[-1], s, xs), np.interp(0.5 * s[-1], s, ys)
    a1 = math.degrees(math.atan2(m[0] - xs[0], m[1] - ys[0])); a2 = math.degrees(math.atan2(xs[-1] - m[0], ys[-1] - m[1]))
    if abs(_wrap(a2 - a1)) < 3.0:
        a = math.degrees(math.atan2(xs[-1] - xs[0], ys[-1] - ys[0])); return round(a, 1), 0.0, round(a, 1)
    return round(a1, 1), 0.5, round(a2, 1)

def tip_from_taper(kT):
    """schema <= 6.5 genalTaper (0.5 blunt .. 3 drawn to a point) -> genalTip (1 round .. 0 pointed)."""
    return float(np.clip((2.0 - float(kT)) / 1.5, 0.0, 1.0))
