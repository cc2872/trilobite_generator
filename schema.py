"""
schema.py — the parameter table as data.

Every knob of the trilobite generator is declared once here. Everything else — defaults,
validation, UI sliders, dataset columns, JSON round-trip, the parameter hash on every
artifact — derives from this table. Add a parameter here and it exists everywhere.
"""
from dataclasses import dataclass, asdict
import json, hashlib

SCHEMA_VERSION = "5.0"

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
    kind: str = "float"          # "float" | "int" | "odd_int" | "expr" (a text formula in s ∈ [0,1])
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
    # ---- surface form (fitted from the reference sculpt, 2026-09-05: seg2 crest rms 0.12 mm, head dome rms 0.33 mm)
    Param("tent", "Tent vault", 1.0, 0.0, 1.0, 0.01, "Form", "0 = fulcrum vault (v4), 1 = axial dome + straight pleural slope (reference sculpt)"),
    Param("axisSigma", "Axial dome width", 0.55, 0.25, 1.0, 0.01, "Form", "Gaussian half-width of the axial dome / half-width (tent vault)"),
    Param("pleuralSlope", "Pleural slope", 0.74, 0.2, 1.2, 0.01, "Form", "Linear drop of the pleural slope over the half-width / relief (tent vault)"),
    Param("ringArch", "Ring arch", 0.22, 0.0, 0.5, 0.01, "Form", "Convexity of each axial ring between its two hinges / relief"),
    Param("bladeCamber", "Blade camber", 0.14, 0.0, 0.4, 0.01, "Form", "Convexity of each pleural blade along its chord / relief"),
    Param("headOutlineExp", "Head outline exponent", 2.15, 1.2, 5.0, 0.05, "Form", "Superellipse exponent of the head front (2 = ellipse, higher = squarer)"),
    Param("headDomeExp", "Head dome exponent", 1.5, 1.2, 4.0, 0.05, "Form", "Superellipsoid exponent of the head vault (1.5 = tent-like, 2 = ellipsoid, higher = flatter top)"),
    Param("headDomeFill", "Head dome fill", 0.82, 0.5, 1.0, 0.01, "Form", "Dome semi-axes / head half-width and length (the rest is flat border)"),
    # ---- head / tail shell: thickness and curve, independent of the thorax
    Param("headWall", "Head thickness", 1.0, 0.6, 2.5, 0.05, "Form", "Head shell thickness / wall (segment 0's stepped front follows it)"),
    Param("tailWall", "Tail thickness", 1.0, 0.6, 2.5, 0.05, "Form", "Tail shell thickness / wall"),
    Param("headRelief", "Head height", 1.0, 0.6, 1.6, 0.02, "Form", "Head apex height / relief (the occipital ring always holds the hinge height)"),
    Param("tailRelief", "Tail height", 1.0, 0.5, 1.4, 0.02, "Form", "Tail apex height / relief (the axis always holds the hinge height over the front joint)"),
    Param("tailDomeExp", "Tail curve", 2.0, 1.2, 5.0, 0.05, "Form", "Fall-off of the tail along the body: 2 = elliptical, higher = flat top with a steep rear"),
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
    Param("headRearArc", "Head rear arc", 0.0, 0.0, 0.7, 0.01, "Head", "Crescent: the rear edge bows back from the axis to the genal angles by this fraction of head length (0 = straight rear, v4)"),
    Param("headRearExp", "Head rear arc shape", 2.2, 1.0, 5.0, 0.05, "Head", "1 = the rear edge leaves the axis at once (thin band); higher = it stays straight and bows late (thick band, short horns)"),
    Param("genalWidth", "Genal arm thickness", 0.12, 0.03, 0.5, 0.01, "Head", "Thickness of the crescent's arm across, / head half-width (harpetid 0.3)"),
    Param("genalTaper", "Genal arm end", 2.0, 0.5, 3.0, 0.05, "Head", "End of the arm: 0.5 = blunt/rounded, 3 = drawn to a point"),
    Param("genalPath", "Genal arm path", "s**2", 0, 0, 0, "Head",
          "Sideways offset of the arm's centreline as a formula in s (0 = root, 1 = tip), scaled by genalCurve. "
          "e.g. 's**2' flares late, 's' straight, 'sin(pi*s/2)' flares early, '-0.5*s+s**2' hugs the thorax then flares", kind="expr"),
    Param("genalWidthMM", "Genal arm thickness (mm)", 0.0, 0.0, 30.0, 0.5, "Head", "Absolute arm thickness in mm; 0 = use genalWidth (fraction of half-width)"),
    Param("genalCurve", "Genal spine curve", 20, -60, 60, 1, "Head", "Inward curl of the genal spines", unit="deg"),
    # ---- head skin (blend of registered real-specimen skins from skins.py, laid over the parametric head)
    Param("headSkin", "Head skin blend", 1.0, 0.0, 1.0, 0.01, "Skin", "0 = pure parametric head, 1 = fully the blended real skin (outline stays yours)"),
    Param("skinOlenoides", "Skin: Olenoides", 1.0, 0.0, 1.0, 0.01, "Skin", "Blend weight for the registered Olenoides skin"),
    Param("skinGltf", "Skin: purchased sculpt", 0.0, 0.0, 1.0, 0.01, "Skin", "Blend weight for the registered purchased-sculpt skin"),
    Param("skinHarpetida", "Skin: Harpetida", 0.0, 0.0, 1.0, 0.01, "Skin", "Blend weight for the registered Harpetida skin"),
    Param("skinProetida", "Skin: Proetida", 0.0, 0.0, 1.0, 0.01, "Skin", "Blend weight for the registered Proetida skin"),
    # ---- thorax pleurae and spines (fields along the thorax)
    Param("tipSweep", "Pleural tip sweep", 0.5, 0.0, 2.0, 0.01, "Thorax", "How far the pleural blades sweep back / pitch"),
    Param("bladeChord", "Blade chord", 1.3, 0.5, 1.3, 0.01, "Thorax", "Fore-aft width of the pleural blade beyond its root / segment pitch (1.3 = the v4 full plate, 0.9 = separate ribs)"),
    Param("tipTaper", "Pleural tip taper", 0.55, 0.0, 0.95, 0.01, "Thorax", "How much the blade narrows toward its tip (0 = square, 0.95 = needle)"),
    Param("spineBase", "Pleural spine base", 0.0, 0.0, 1.2, 0.01, "Thorax", "Extra needle spine beyond the blade tip, segment 0 / half-width"),
    Param("spineGrad", "Pleural spine gradient", 0.0, -1.0, 1.0, 0.01, "Thorax", "Change in spine length from first to last segment"),
    Param("macroIndex", "Macropleural segment", -1, -1, 15, 1, "Thorax", "Index of a segment with extra-long spines (-1 = none)", kind="int"),
    Param("macroAmp", "Macropleural extra", 0.8, 0.0, 2.0, 0.05, "Thorax", "Extra spine length on that segment / half-width"),
    Param("spineSweep", "Spine sweep", 50, 0, 80, 1, "Thorax", "Pleural spines sweep back by this angle", unit="deg"),
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
    # the six morphospace dials (Raup-style: few semantic axes). Each maps 0..1 linearly onto its parameters.
    ("sculpt", "Sculpt", 1.0, [("furrowDepth", 0.3, 1.2), ("effacement", 0.85, 0.0), ("glabRise", 0.05, 0.18), ("glabLobes", 0, 3),
                               ("eyeSize", 0.0, 0.16), ("borderWidth", 0.0, 0.10), ("pygRings", 0, 4)]),
    ("headSize", "Head", 0.45, [("cephFrac", 0.22, 0.42), ("widthMaxPos", 0.12, 0.35), ("headRearArc", 0.0, 0.25)]),
    ("tailSize", "Tail", 0.5, [("pygFrac", 0.05, 0.38), ("pygWidth", 0.7, 1.05), ("widthTail", 0.30, 0.80)]),
    ("elongation", "Elongation", 0.5, [("segCount", 4, 14), ("width", 95, 48), ("length", 120, 220), ("widthThoraxRear", 0.75, 0.45)]),
    ("spikiness", "Spines", 0.15, [("spineBase", 0.0, 0.9), ("spineGrad", 0.0, 0.5), ("genalSpine", 0.0, 0.9), ("pygSpine", 0.0, 1.6),
                                   ("axialSpine", 0.0, 0.9), ("pygMarginal", 0, 8), ("tipTaper", 0.5, 0.9)]),
    ("eyes", "Eyes", 0.4, [("eyeSize", 0.04, 0.30), ("eyeHeight", 0.6, 1.2), ("glabInflate", 1.1, 1.7)]),
]
# ---- the website's three layers (everything else stays in the schema as a fixed or macro-driven value)
UI = {
    "dials": ["sculpt", "headSize", "tailSize", "elongation", "spikiness", "eyes"],
    "parts": {
        "Body":   ["length", "width", "relief", "wall"],
        "Head":   ["headOutlineExp", "headDomeExp", "headDomeFill", "headRearArc", "genalSpine", "genalWidthMM", "eyeSize", "eyePos"],
        "Thorax": ["segCount", "bladeChord", "tipSweep", "tipTaper", "spineBase", "spineGrad", "spineSweep"],
        "Tail":   ["pygFrac", "pygWidth", "pygSpine", "pygSplay", "pygRings", "pygMarginal"],
    },
    "ruler": ["maxAngle", "clearance", "boreDia", "barrelR", "nKnuckles"],
    "arm": {"path": "genalPath", "curve": "genalCurve", "taper": "genalTaper"},
}

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


# ---- named presets. "reference" reproduces the 2026-09-05 reference sculpt (smooth, effaced) on our pin-hinge
# mechanism; "textured" is our model: the same body form with the sculpt (furrows, glabella, eyes, border) switched on.
# Every form parameter (tent vault, ring arch, blade camber, head dome/outline, blade chord) is shared.
_FORM = dict(length=130, width=69.5, relief=17.1, segCount=6, cephFrac=0.335, pygFrac=0.20,
             marginHeight=0.14, overlap=0.45,
             widthMaxPos=0.20, widthHeadFront=0.75, widthThoraxFront=0.93, widthThoraxRear=0.62, widthTail=0.50,
             axisFrac=0.28, axisRise=0.0, cephParallel=0.05, genalSweep=0.3, genalSpine=0.12, genalCurve=10,
             headSkin=0.0, headOutlineExp=2.15, headDomeExp=1.5, headDomeFill=0.82,
             tent=1.0, axisSigma=0.55, pleuralSlope=0.74, ringArch=0.22, bladeCamber=0.14,
             tipSweep=0.75, tipTaper=0.85, bladeChord=0.9, spineBase=0.0, spineSweep=45,
             pygWidth=0.82, pygSpine=1.0, pygSplay=14, termSpine=0.0)
def _sculpt(level):
    """Parameter values the 'sculpt' macro sets at `level` in [0, 1]."""
    out = {}
    for k, label, default, maps in MACROS:
        if k == "sculpt":
            for pk, lo, hi in maps: out[pk] = lo + (hi - lo) * level
    return out
PRESETS = {
    "reference": dict(_FORM, tubercles=0.0, **_sculpt(0.0)),
    # the default animal (6 Sep 2026 evening): the reference form with our sculpt, re-proportioned to read as a classic
    # trilobite — 9 segments at 150 mm (pitch 8.5 ≥ 8), short forks, small formula genal spines, slight crescent rear.
    "textured":  dict(dict(_FORM, **_sculpt(1.0)), tubercles=0.0, eyeArc=150,
                      length=150, width=72, segCount=9, cephFrac=0.30, pygFrac=0.22, pygSpine=0.35, pygSplay=16, pygRings=4,
                      genalSpine=0.32, genalCurve=12, genalPath="s**2", genalWidthMM=5, genalTaper=2.2, headRearArc=0.12, headRearExp=2.0,
                      eyeSize=0.16, eyeHeight=0.85, eyePos=0.62, glabInflate=1.35, widthThoraxRear=0.62, widthTail=0.55, tipTaper=0.8),
}
# ---- species presets fitted from scans (6 Sep 2026). Each is the sculpted default plus measured form; nothing here
#      changes _FORM or "textured".
PRESETS["harpetid"] = dict(PRESETS["textured"],
    # Harpetid STL (98 × 144 × 23 mm): dome m 1.42, semi-axes 0.61 × 0.62 of the head, rim 0.2 relief (0.98 mm rms);
    # front outline n 1.89; W/L 1.61; prolongations 1.2 Lc past the occipital ring, 0.3 wh wide, rounded.
    cephFrac=0.42, pygFrac=0.10, segCount=9, width=72, headDomeExp=1.42, headDomeFill=0.61, headOutlineExp=1.89,
    marginHeight=0.33, genalSpine=1.2, genalCurve=2, genalWidth=0.30, genalTaper=0.6, genalSweep=0.05, cephParallel=0.02,
    widthThoraxFront=0.58,          # harpid thorax is narrow inside the horseshoe: 55 / 96 mm on the scan
    eyeSize=0.05, eyeHeight=1.0, eyePos=0.45, glabRise=0.12, borderWidth=0.0, pygSpine=0.0, pygRings=2,
    spineBase=0.0, widthMaxPos=0.12, widthThoraxRear=0.42, widthTail=0.30, tipTaper=0.7)
PRESETS["phacopid"] = dict(PRESETS["textured"],
    # Phacops sp. scan (25 × 15 mm, matrix masked): eyes at 0.55 ± 0.1 of half-width, 0.75 Lc from the front,
    # 0.84–0.96 of glabella height; glabella ~0.55 of head width in plan, only 8 % above the rings in z; 11 rings.
    length=205, width=95, segCount=11, cephFrac=0.34, pygFrac=0.22, eyeSize=0.22, eyeHeight=0.9, eyePos=0.75, glabInflate=1.7,   # 205 mm: 11 segments need pitch ≥ 8 mm for this hinge
    glabRise=0.12, genalSpine=0.0, pygSpine=0.0, pygRings=7, spineBase=0.0, headDomeExp=1.9, widthTail=0.6,
    tipTaper=0.6, bladeChord=1.0)
DEFAULT_PRESET = "textured"

def table_defaults():
    """The raw per-parameter defaults from the table (no preset applied)."""
    return {p.key: p.default for p in PARAMS}

def preset(name):
    """Full parameter dict for a named preset (missing keys filled from the table, then coerced)."""
    P = table_defaults(); P.update(PRESETS[name]); return coerce(P, base=P)

def defaults():
    return preset(DEFAULT_PRESET)

def coerce(P, base=None):
    """Fill missing keys with defaults, clamp to range, enforce int/odd-int kinds. Unknown keys are dropped."""
    Q = dict(base) if base is not None else defaults()
    for k, v in (P or {}).items():
        if k in BY_KEY:
            p = BY_KEY[k]
            if p.kind == "expr":
                Q[k] = str(v)[:120]; continue
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
