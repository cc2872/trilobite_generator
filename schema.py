"""
schema.py — the parameter table as data.

Every knob of the trilobite generator is declared once here. Everything else — defaults,
validation, UI sliders, dataset columns, JSON round-trip, the parameter hash on every
artifact — derives from this table. Add a parameter here and it exists everywhere.
"""
from dataclasses import dataclass, asdict
import json, hashlib

SCHEMA_VERSION = "4.0"

@dataclass(frozen=True)
class Param:
    key: str
    label: str
    default: float
    lo: float
    hi: float
    step: float
    group: str
    doc: str = ""
    kind: str = "float"          # "float" | "int" | "odd_int"
    unit: str = ""

PARAMS = [
    # ---- body proportions
    Param("length", "Body length", 130, 60, 250, 1, "Body", "Whole animal, head tip to tail shield tip (spines extra)", unit="mm"),
    Param("width", "Max width", 65, 30, 140, 1, "Body", "Widest point of the body", unit="mm"),
    Param("relief", "Relief", 16, 6, 40, 0.5, "Body", "Dorsal height of the vault at the axis", unit="mm"),
    Param("segCount", "Thoracic segments", 6, 2, 16, 1, "Body", "Number of articulating thoracic segments", kind="int"),
    Param("cephFrac", "Head length fraction", 0.33, 0.18, 0.45, 0.005, "Body", "Cephalon length / body length"),
    Param("pygFrac", "Tail shield fraction", 0.14, 0.05, 0.40, 0.005, "Body", "Pygidium length / body length (spines extra)"),
    Param("marginHeight", "Margin height", 0.34, 0.05, 0.60, 0.01, "Body", "Pleural margin height / relief"),
    Param("fulcrum", "Fulcrum", 0.55, 0.3, 0.9, 0.01, "Body", "Where the pleura bends down, as a fraction of half-width"),
    Param("overlap", "Shingle overlap", 0.5, 0.1, 0.8, 0.01, "Body", "Rear flap length as a fraction of segment pitch"),
    Param("wall", "Shell wall", 2.0, 1.2, 4.0, 0.1, "Body", "Exoskeleton thickness (keep ≥ 1.5 for FDM)", unit="mm"),
    # ---- outline (width curve)
    Param("widthMaxPos", "Widest point", 0.30, 0.10, 0.80, 0.01, "Outline", "Position of max width along the body (0 head, 1 tail)"),
    Param("widthHeadFront", "Head front width", 0.85, 0.3, 1.0, 0.01, "Outline", "Width at the head's front third / max width"),
    Param("widthThoraxFront", "Shoulder width", 0.92, 0.6, 1.0, 0.01, "Outline", "Width of segment 0 / max width (the head–thorax shoulder)"),
    Param("widthThoraxRear", "Thorax rear width", 0.62, 0.3, 1.0, 0.01, "Outline", "Width at the last segment / max width"),
    Param("widthTail", "Tail tip width", 0.25, 0.0, 0.8, 0.01, "Outline", "Width near the tail tip / max width"),
    # ---- axis and furrows
    Param("axisFrac", "Axial ring width", 0.33, 0.20, 0.50, 0.01, "Axis & furrows", "Axial lobe width / local body width"),
    Param("axisRise", "Axial ring rise", 0.15, 0.0, 0.40, 0.01, "Axis & furrows", "Ring stands this fraction of relief above the vault"),
    Param("furrowDepth", "Furrow depth", 1.2, 0.0, 3.0, 0.1, "Axis & furrows", "Depth of the furrows at full expression", unit="mm"),
    Param("effacement", "Effacement", 0.0, 0.0, 1.0, 0.01, "Axis & furrows", "0 = fully sculpted, 1 = smooth (Nileus)"),
    # ---- head
    Param("cephParallel", "Head parallel rear", 0.45, 0.0, 0.8, 0.01, "Head", "Fraction of head length that is parallel-sided"),
    Param("glabInflate", "Glabella expansion", 1.25, 0.6, 2.0, 0.01, "Head", "Glabella front width / rear width (club shape)"),
    Param("glabRise", "Glabella rise", 0.18, 0.0, 0.6, 0.01, "Head", "Glabella height above the cheeks / relief"),
    Param("glabLobes", "Glabellar furrow pairs", 3, 0, 4, 1, "Head", "Pairs of lateral glabellar furrows", kind="int"),
    Param("eyeSize", "Eye size", 0.14, 0.0, 0.45, 0.005, "Head", "Eye radius / head half-width (0 = blind)"),
    Param("eyePos", "Eye position", 0.45, 0.10, 0.90, 0.01, "Head", "Along the head, 0 rear → 1 front"),
    Param("eyeArc", "Eye arc", 150, 60, 300, 5, "Head", "Angular extent of the visual surface", unit="deg"),
    Param("eyeHeight", "Eye height", 0.8, 0.2, 1.6, 0.05, "Head", "Dome height of the eye / eye radius"),
    Param("genalSweep", "Cheek sweep", 0.8, 0.0, 2.5, 0.05, "Head", "How far the cheeks sweep back along the shoulder / segment pitch"),
    Param("borderWidth", "Border width", 0.10, 0.0, 0.30, 0.01, "Head", "Raised border / head half-width (0 = none)"),
    Param("genalSpine", "Genal spine length", 0.35, 0.0, 1.5, 0.01, "Head", "Genal spine length / head length (0 = none)"),
    Param("genalCurve", "Genal spine curve", 20, 0, 60, 1, "Head", "Inward curl of the genal spines", unit="deg"),
    # ---- head skin (blend of registered real-specimen skins from skins.py, laid over the parametric head)
    Param("headSkin", "Head skin blend", 1.0, 0.0, 1.0, 0.01, "Skin", "0 = pure parametric head, 1 = fully the blended real skin (outline stays yours)"),
    Param("skinOlenoides", "Skin: Olenoides", 1.0, 0.0, 1.0, 0.01, "Skin", "Blend weight for the registered Olenoides skin"),
    Param("skinGltf", "Skin: purchased sculpt", 0.0, 0.0, 1.0, 0.01, "Skin", "Blend weight for the registered purchased-sculpt skin"),
    Param("skinHarpetida", "Skin: Harpetida", 0.0, 0.0, 1.0, 0.01, "Skin", "Blend weight for the registered Harpetida skin"),
    Param("skinProetida", "Skin: Proetida", 0.0, 0.0, 1.0, 0.01, "Skin", "Blend weight for the registered Proetida skin"),
    # ---- thorax pleurae and spines (fields along the thorax)
    Param("tipSweep", "Pleural tip sweep", 0.5, 0.0, 2.0, 0.01, "Thorax", "How far the pleural blades sweep back / pitch"),
    Param("tipTaper", "Pleural tip taper", 0.55, 0.0, 0.95, 0.01, "Thorax", "How much the blade narrows toward its tip (0 = square, 0.95 = needle)"),
    Param("spineBase", "Pleural spine base", 0.0, 0.0, 1.2, 0.01, "Thorax", "Extra needle spine beyond the blade tip, segment 0 / half-width"),
    Param("spineGrad", "Pleural spine gradient", 0.0, -1.0, 1.0, 0.01, "Thorax", "Change in spine length from first to last segment"),
    Param("macroIndex", "Macropleural segment", -1, -1, 15, 1, "Thorax", "Index of a segment with extra-long spines (-1 = none)", kind="int"),
    Param("macroAmp", "Macropleural extra", 0.8, 0.0, 2.0, 0.05, "Thorax", "Extra spine length on that segment / half-width"),
    Param("spineSweep", "Spine sweep", 45, 0, 80, 1, "Thorax", "Pleural spines sweep back by this angle", unit="deg"),
    # ---- tail
    Param("pygWidth", "Tail width", 0.90, 0.5, 1.1, 0.01, "Tail", "Tail shield width / last segment width"),
    Param("pygRings", "Tail axial rings", 4, 0, 12, 1, "Tail", "Axial rings on the pygidium", kind="int"),
    Param("pygSpine", "Tail spine length", 0.9, 0.0, 2.0, 0.05, "Tail", "Paired tail spines / shield length (0 = none)"),
    Param("pygSplay", "Tail spine splay", 20, 0, 45, 1, "Tail", "Outward angle of the tail spines", unit="deg"),
    Param("termSpine", "Terminal spine", 0.0, 0.0, 2.5, 0.05, "Tail", "Single median terminal spine / shield length (0 = none)"),
    # ---- extra spine families and ornament
    Param("axialSpine", "Axial spines", 0.0, 0.0, 2.5, 0.05, "Spines", "Dorsal spine on every thoracic ring / relief (0 = none)"),
    Param("occipitalSpine", "Occipital spine", 0.0, 0.0, 1.5, 0.05, "Spines", "Spine on the occipital ring / head length (0 = none)"),
    Param("pygMarginal", "Tail marginal spines", 0, 0, 10, 1, "Spines", "Spines around the tail margin (0 = none)", kind="int"),
    Param("pygMarginalLen", "Tail marginal length", 0.5, 0.1, 1.5, 0.05, "Spines", "Marginal spine length / shield length"),
    Param("tubercles", "Tubercles", 0.0, 0.0, 1.0, 0.01, "Ornament", "Density of granules on the shell (0 = smooth)"),
    Param("tubercleSize", "Tubercle size", 1.6, 0.8, 3.0, 0.1, "Ornament", "Granule radius", unit="mm"),
    Param("seed", "Ornament seed", 1, 0, 999, 1, "Ornament", "Random seed for ornament placement", kind="int"),
    # ---- articulation (held constant across a sweep: the fixed ruler)
    Param("maxAngle", "Stop angle", 18, 4, 40, 0.5, "Hinge", "Ventral flexion per joint before the stop engages", unit="deg"),
    Param("clearance", "Clearance", 0.3, 0.15, 0.6, 0.01, "Hinge", "Gap between moving parts", unit="mm"),
    Param("boreDia", "Pin bore", 1.95, 1.6, 3.0, 0.05, "Hinge", "For 1.75 mm filament pins", unit="mm"),
    Param("barrelR", "Knuckle radius", 2.6, 1.8, 4.0, 0.1, "Hinge", "", unit="mm"),
    Param("nKnuckles", "Knuckles", 3, 3, 7, 2, "Hinge", "Odd number across the hinge", kind="odd_int"),
]
# ---- macro knobs: one slider drives several parameters linearly between (value at 0, value at 1)
MACROS = [
    ("spikiness", "Spikiness", 0.25, [("spineBase", 0.0, 1.0), ("genalSpine", 0.0, 1.4), ("pygSpine", 0.0, 1.8), ("axialSpine", 0.0, 1.5),
                                       ("pygMarginal", 0, 8), ("tipTaper", 0.35, 0.9), ("tipSweep", 0.3, 1.4), ("occipitalSpine", 0.0, 0.8)]),
    ("headSize", "Head size", 0.5, [("cephFrac", 0.20, 0.45), ("widthMaxPos", 0.20, 0.40), ("cephParallel", 0.2, 0.6)]),
    ("tailSize", "Tail size", 0.3, [("pygFrac", 0.05, 0.40), ("pygWidth", 0.7, 1.05), ("pygRings", 1, 10)]),
    ("elongation", "Elongation", 0.4, [("width", 100, 38), ("segCount", 4, 14), ("taper", 0.90, 0.97)]),
    ("sculpture", "Sculpture", 0.6, [("effacement", 1.0, 0.0), ("furrowDepth", 0.3, 2.0), ("tubercles", 0.0, 0.8), ("glabRise", 0.05, 0.35)]),
    ("eyes", "Eyes", 0.4, [("eyeSize", 0.0, 0.40), ("eyeHeight", 0.3, 1.5)]),
]

def apply_macro(P, key, value):
    """Set every parameter a macro drives; value in [0, 1]."""
    for k, label, default, maps in MACROS:
        if k != key: continue
        for pk, lo, hi in maps:
            P[pk] = lo + (hi - lo) * float(value)
    return coerce(P)

BY_KEY = {p.key: p for p in PARAMS}
GROUPS = []
for _p in PARAMS:
    if _p.group not in GROUPS: GROUPS.append(_p.group)

def defaults():
    return {p.key: p.default for p in PARAMS}

def coerce(P):
    """Fill missing keys with defaults, clamp to range, enforce int/odd-int kinds. Unknown keys are dropped."""
    Q = defaults()
    for k, v in (P or {}).items():
        if k in BY_KEY:
            p = BY_KEY[k]
            v = float(v)
            v = min(max(v, p.lo), p.hi)
            if p.kind in ("int", "odd_int"):
                v = int(round(v))
                if p.kind == "odd_int" and v % 2 == 0: v = min(v + 1, int(p.hi))
            Q[k] = v
    return Q

def to_json(P):
    return json.dumps({"schema": SCHEMA_VERSION, "params": coerce(P)}, sort_keys=True)

def from_json(s):
    return coerce(json.loads(s)["params"])

def param_hash(P):
    return hashlib.md5(json.dumps(coerce(P), sort_keys=True).encode()).hexdigest()[:10]

def table():
    return [asdict(p) for p in PARAMS]

if __name__ == "__main__":
    for g in GROUPS:
        print(f"[{g}]")
        for p in PARAMS:
            if p.group == g: print(f"  {p.key:16s} {p.default!s:>6} [{p.lo}..{p.hi}] {p.unit:3s} {p.doc}")
    print("hash of defaults:", param_hash(defaults()))
