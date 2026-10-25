"""
scripts/build_ornaments.py: the printable (isopod-model) animal with the curved-spine ornaments (23 Oct 2026).

    python scripts/build_ornaments.py OUT                      # the default animal + five ornamented ones (~15 min)
    python scripts/build_ornaments.py OUT dicranurus ceratarges

Writes OUT/<name>/<name>.stl (print mm, every piece), the pieces, checks.json, and OUT/summary.json. The default
animal built here must match the default built on main: same piece count, extents and head volume (the new keys are
all 0 on it). Each ornamented animal must build with every piece one watertight body; the checks say whether the
pieces touch at rest and whether the curl clears. If a case fails, the traceback is in summary.json.
"""
import os, sys, json, time, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "isopod_model"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, trimesh, schema, mesh as M
import body as BODY, check as CHECK, head_styles as HS

CASES = {
  "default":        ("shaped", "diagram", {}),   # 25 Oct 2026: the generator default = the diagram animal
  "crescent":       ("isopod", None, {}),         # the old default: the crescent on the isopod head piece
  "diagram":        ("shaped", "diagram", "PRESET"),   # 25 Oct 2026: the textbook trilobite -- the studio's starting animal (schema.PRESETS['diagram'] + head style 'diagram')
  "walliserops":    ("shaped", "walliserops", dict(headProngs=3, headProngLen=0.9, headProngStem=0.5, headProngSplay=40, headProngBend=55, occipitalSpine=0.5)),
  "dicranurus":     ("shaped", "odontopleura", dict(occipitalSpine=1.3, occipitalTines=2, occipitalSplay=60, occipitalBend=160, spineBase=0.6, spineSweep=55)),
  "ceratarges":     ("shaped", "lichas", dict(genalHorn=1.5, genalHornPitch=50, genalHornBend=130, genalHornRoll=-25, occipitalSpine=0.8, occipitalBend=120)),
  "ceratarges_tail": ("shaped", "lichas", dict(genalHorn=1.5, genalHornPitch=50, genalHornBend=130, genalHornRoll=-25, occipitalSpine=0.8, occipitalBend=120,
                                               pygFrac=0.18, pygRings=2, pygRibRelief=1.2, pygFurrowScale=1.8, pygSpine=2.0, pygSplay=35, pygMarginal=6, pygMarginalLen=0.25,
                                               pygSpineRise=35, pygSpineBend=80, pygSpineGrow=0.6, pygSpineBlade=0.0)),   # 24 Oct 2026: a tail of its own
  "acanthopyge":    ("shaped", "lichas", dict(pygFrac=0.30, pygWidth=1.0, pygRings=3, pygRibRelief=1.6, pygFurrowScale=2.2, pygSpine=1.0, pygSplay=30, pygMarginal=2, pygMarginalLen=0.7,
                                               pygSpineRise=10, pygSpineBend=15, pygSpineGrow=0.3, pygSpineBlade=0.7, tailOutlineExp=2.4, spineBase=0.4, tipTaper=0.5)),   # lichid: a big ribbed tail, broad lobate spines
  "odontocephalus": ("shaped", "dalmanites", dict(frontSpine=0.7, frontSpineBend=35)),
  "boedaspis":      ("shaped", "odontopleura", dict(genalHorn=1.0, genalHornPitch=35, genalHornBend=90, genalHornRoll=60, occipitalSpine=1.0, occipitalBend=100, spineBase=0.8)),
  # 24 Oct 2026: the spiny Devonian forms whose thoracic spines leave the plane (Koneprusia, Comura, Ceratonurus)
  "koneprusia":     ("shaped", "odontopleura", dict(genalHorn=1.2, genalHornPitch=40, genalHornBend=110, genalHornRoll=-35, genalHornYaw=25,
                                                   occipitalSpine=1.1, occipitalBend=130,
                                                   axialSpine=2.5, axialPitch=70, axialBend=45, axialGrad=0.8,
                                                   pleuralHorn=1.0, pleuralHornPitch=55, pleuralHornBend=75, pleuralHornRoll=-20, pleuralHornGrad=0.4,
                                                   spineBase=0.0, tipTaper=0.65, spineSweep=55, segCount=9,
                                                   pygFrac=0.12, pygRings=3, pygRibRelief=0.8, pygFurrowScale=1.6, pygSpine=1.7, pygSplay=28, pygMarginal=4, pygMarginalLen=0.3,
                                                   pygSpineRise=30, pygSpineBend=45, pygSpineGrow=0.4, pygSpineBlade=0.0)),   # the horn IS the pleural spine: no flat blade spine under it
}

def main():
    out = os.path.abspath(sys.argv[1]); only = sys.argv[2:] or list(CASES); os.makedirs(out, exist_ok=True)
    D = schema.defaults(); summary = {}
    for name in only:
        kind, style, extra = CASES[name]; t0 = time.time(); rec = dict(kind=kind, style=style, extra=extra)
        try:
            if extra == "PRESET": extra = dict(schema.PRESETS[name])
            P = schema.coerce(dict(D, **extra))
            if kind == "isopod": parts, plans, PV, K = BODY.build(P, "isopod")
            else:
                P, face = HS.apply(P, style); P = schema.coerce(dict(P, **extra))           # the style's params, then ours
                parts, plans, PV, K = BODY.build(P, "shaped", face=face, outline=HS.outline(style))
            s = K["s"]; pieces = [p.copy().apply_scale(1.0 / s) for p in parts]
            names = ["head"] + [f"thorax{i + 1}" for i in range(len(pieces) - 2)] + ["tail"]
            d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
            whole = trimesh.util.concatenate(pieces); whole.export(os.path.join(d, f"{name}.stl"))
            for k, (n, p) in enumerate(zip(names, pieces)): p.export(os.path.join(d, f"{k:02d}_{n}.stl"))
            pv = [tuple(CHECK.fit_ball(pieces[k + 1], (0, y / s, z / s), 0.0)[0][1:]) for k, (y, z) in enumerate(PV)]
            chk = CHECK.run(pieces, pv, names)
            rec.update(ok=True, pieces=len(pieces), extents_mm=np.round(whole.extents, 1).tolist(), faces=int(len(whole.faces)),
                       bodies_per_piece=[len(M.bodies(p)) for p in pieces], watertight=[bool(p.is_watertight) for p in pieces],
                       head_volume=round(float(pieces[0].volume), 1), checks=CHECK.summary(chk), seconds=round(time.time() - t0))
            json.dump(chk, open(os.path.join(d, "checks.json"), "w"), indent=1, default=str)
        except Exception:
            rec.update(ok=False, error=traceback.format_exc()[-1500:], seconds=round(time.time() - t0))
        summary[name] = rec
        print(f"{name:15s} {'OK' if rec['ok'] else 'FAILED'}  {rec['seconds']} s" + (f"  pieces {rec['pieces']} bodies {rec['bodies_per_piece']}" if rec["ok"] else ""), flush=True)
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    print(f"\nwrote {out}/summary.json")

if __name__ == "__main__":
    main()
