
import numpy as np
 
class Curve:
    """Monotone-safe piecewise-cubic (Catmull-Rom with clamped ends) through (s, v) control points."""
    def __init__(self, pts):
        pts = sorted(pts)
        self.s = np.array([p[0] for p in pts], float)
        self.v = np.array([p[1] for p in pts], float)
 
    def __call__(self, s):
        s = np.clip(np.asarray(s, float), self.s[0], self.s[-1])
        # tangents: finite differences, clamped at the ends
        n = len(self.s)
        m = np.zeros(n)
        for i in range(n):
            if i == 0: m[i] = (self.v[1] - self.v[0]) / (self.s[1] - self.s[0])
            elif i == n - 1: m[i] = (self.v[-1] - self.v[-2]) / (self.s[-1] - self.s[-2])
            else: m[i] = 0.5 * ((self.v[i+1] - self.v[i]) / (self.s[i+1] - self.s[i]) + (self.v[i] - self.v[i-1]) / (self.s[i] - self.s[i-1]))
        idx = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, n - 2)
        s0, s1 = self.s[idx], self.s[idx + 1]; h = s1 - s0; t = (s - s0) / h
        h00 = 2*t**3 - 3*t**2 + 1; h10 = t**3 - 2*t**2 + t; h01 = -2*t**3 + 3*t**2; h11 = t**3 - t**2
        return h00*self.v[idx] + h10*h*m[idx] + h01*self.v[idx+1] + h11*h*m[idx+1]
 
# ---------------- landmarks
def landmarks(P):
    s_h = P["cephFrac"]
    s_t = 1 - P["pygFrac"]
    return dict(s_h=s_h, s_t=s_t, pitch_s=(s_t - s_h) / P["segCount"])
 
def seg_s(P, i):
    """Body coordinate of the middle of thoracic segment i."""
    L = landmarks(P)
    return L["s_h"] + (i + 0.5) * L["pitch_s"]
 
# ---------------- width field
def width_curve(P):
    """Half-width of the body as a function of s (mm). Widest at widthMaxPos."""
    L = landmarks(P); W = P["width"] / 2
    smax = P["widthMaxPos"]
    s0 = L["s_h"] + 0.5 * L["pitch_s"]                      # centre of segment 0
    pts = [(0.0, 0.08 * W),
           (min(0.10, smax - 0.02), P["widthHeadFront"] * W),
           (smax, W)]
    if s0 > smax + 0.02: pts.append((s0, P["widthThoraxFront"] * W))
    pts += [(L["s_t"] if L["s_t"] > s0 + 0.02 else s0 + 0.02, P["widthThoraxRear"] * W),
            (1.0, P["widthTail"] * W)]
    return Curve(pts)
 
def seg_halfwidth(P, i):
    return float(width_curve(P)(seg_s(P, i)))
 
def head_halfwidth(P):
    """Half-width at the rear of the head (where it meets segment 0)."""
    return float(width_curve(P)(landmarks(P)["s_h"]))
 
def tail_halfwidth(P):
    return float(width_curve(P)(landmarks(P)["s_t"])) * P["pygWidth"]
 
# ---------------- pleural spine field
def pleural_spine_field(P):
    """Spine length factor (× local half-width) per segment: base + gradient + one macropleural bump."""
    n = P["segCount"]
    out = []
    for i in range(n):
        f = P["spineBase"] + P["spineGrad"] * (i / max(n - 1, 1))
        if int(P["macroIndex"]) == i: f += P["macroAmp"]
        out.append(max(0.0, f))
    return out
 
def furrow_amp(P):
    """Furrow depth after effacement (mm)."""
    return P["furrowDepth"] * (1.0 - P["effacement"])
 
if __name__ == "__main__":
    import schema
    P = schema.defaults()
    wc = width_curve(P)
    print("half-width along s:", [round(float(wc(s)), 1) for s in np.linspace(0, 1, 11)])
    print("segment half-widths:", [round(seg_halfwidth(P, i), 1) for i in range(P["segCount"])])
    P2 = dict(P, macroIndex=2, spineGrad=-0.2); print("spine field:", [round(v, 2) for v in pleural_spine_field(P2)])
