"""Render the tree page's blueprint menu cards (9 Nov 2026): one dorsal plan per order preset, white on black, into
web/static/blueprints/<preset>.png, drawn AS the finished animal: the preset's drawing block (its saved head + rows,
as the studio opens it). A preset without a drawing block is not finished and gets no card -- the menu shows it as a
placeholder until it is drawn.
    python scripts/blueprint_cards.py [preset ...]        (no names: every preset with a drawing block)"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "web"), os.path.join(ROOT, "isopod_model")): sys.path.insert(0, p)
os.chdir(ROOT)
import app as A, blueprint, body as IBODY

OUT = os.path.join(ROOT, "web", "static", "blueprints"); os.makedirs(OUT, exist_ok=True)
names = sys.argv[1:] or sorted(A._presets())
for name in names:
    t = time.time()
    try:
        rec = A._preset_record(name)[1]
        if not rec.get("drawing"):
            print(f"{name:16s} skipped: no drawing block yet", flush=True); continue
        R = A._drawing_resolve(name, rec["drawing"])
        parts, plans, PV, K = IBODY.build(R["P"], "shaped", face=R["head"]["face"], outline=R["head"]["outline"], rows=(R["body"] or None))
        s = K["s"]; pieces = [p.copy().apply_scale(1.0 / s) for p in parts]; P = R["P"]; how = "drawing"
        blueprint.plan_card(pieces, P, os.path.join(OUT, f"{name}.png"))
        print(f"{name:16s} {how:8s} {time.time() - t:5.0f} s", flush=True)
    except Exception as ex:
        print(f"{name:16s} FAILED {type(ex).__name__}: {str(ex)[:160]}", flush=True)
