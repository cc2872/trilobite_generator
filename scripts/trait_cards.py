"""Render the tree page's character cards (14 Nov 2026) from the generator itself, replacing the guide's figures:
one plain animal, then the same animal with only that character switched on (schema.CHARACTERS, the bundles the
cards' ?chars= links send), so each card shows exactly what the character changes. Eye characters are drawn as the
head alone, with the lens packing, so the eye fills the card. The cephalic brim is a head style, not a parameter.
Characters the generator does not model (vincular furrow, natant hypostome, tubercles) get no image; the page shows
them as dotted, "not modelled" cards.
    python scripts/trait_cards.py [key ...]            -> web/static/traits/<key>.png"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "isopod_model")): sys.path.insert(0, p)
os.chdir(ROOT)
import trimesh, schema, blueprint, assemble, joints

OUT = os.path.join(ROOT, "web", "static", "traits"); os.makedirs(OUT, exist_ok=True)
# The plain animal: the generator's defaults with every spine off, so a spine card adds a spine rather than lengthening one.
BASE = schema.coerce(dict(schema.coerce({}), genalSpine=0.0, spineBase=0.0, spineGrad=0.0, pygSpine=0.0, pygMarginal=0,
                          termSpine=0.0, tailProngs=0))
# tree page card key -> (schema.CHARACTERS key or None, head style or None, draw the head alone)
CARDS = {
    "plain": (None, None, False),
    "eye":   ("holochroalEyes", None, True),   "schiz": ("schizochroalEyes", None, True),
    "blind": ("eyesLost", None, True),         "big":   ("pelagicEyes", None, True),
    "genal": ("genalSpines", None, False),     "pleur": ("pleuralSpines", None, False),
    "pyg":   ("pygidialSpines", None, False),  "iso":   ("isopygous", None, False),
    "eff":   ("effaced", None, False),         "olen":  ("olenimorph", None, False),
    "prop":  ("proparian", None, True),        "brim":  (None, "harpes", False),
}


def pin_parts(P):
    parts = assemble.animal(P, joint="pin")
    mats = joints.get("pin").transforms_deg(P, 0.0)                     # parts are built in their own frames: place at rest
    return [m.copy().apply_transform(T) for m, T in zip(parts, mats)]


def card(key):
    ch, head, head_only = CARDS[key]
    P = schema.express(BASE, ch) if ch else BASE
    if head:                                                            # a head style: the printable animal's shaped head
        import body as IBODY
        parts, plans, PV, K = IBODY.build(P, head); s = K["s"]
        parts = [p.copy().apply_scale(1.0 / s) for p in parts]
    else:
        parts = pin_parts(P)
    blueprint.EYE_LENSES = head_only                                    # the lens packing, on the eye cards only
    blueprint.plan_card(parts[:1] if head_only else parts, P, os.path.join(OUT, f"{key}.png"), px=480)


if __name__ == "__main__":
    for k in sys.argv[1:] or CARDS:
        t = time.time()
        try: card(k); print(f"{k:6s} {time.time() - t:5.1f} s", flush=True)
        except Exception as ex: print(f"{k:6s} FAILED {type(ex).__name__}: {str(ex)[:160]}", flush=True)
