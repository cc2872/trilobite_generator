"""spine_battery.py: build every animal the generator ships and audit every spine on it (body.audit_ornaments):
root buried in its part, nothing in the hollow under its own plate, no contact with another part at rest.
Writes a summary (markdown) and one photo sheet per animal with failures.  python spine_battery.py OUTDIR [names...]"""
import os, sys, json, glob, time, traceback, warnings, shutil
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); os.chdir(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/isopod_model"); sys.path.insert(0, ROOT + "/scripts")
warnings.filterwarnings("ignore")
import numpy as np, trimesh
import schema, mesh as M, body as BODY, head_styles as HS, render as RD
from PIL import Image, ImageDraw
from build_ornaments import CASES

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); only = [a for a in sys.argv[2:] if not a.startswith("--")]
MODE = next((a for a in sys.argv[2:] if a.startswith("--")), "")                   # --list: print the animals; --summary: collect; --one <label>: build one (fresh process each: memory)
if MODE != "--summary": shutil.rmtree(os.path.join(ROOT, "web", "heads_cache"), ignore_errors=True)   # fresh heads: the audit records a head's spines only when it is built
STYLE_OF = dict(agnostida="agnostus", asaphida="isotelus", corynexochida="olenoides", harpetida="harpes", lichida="lichas", odontopleurida="odontopleura",
                phacopida="phacops", proetida="modocia", ptychopariida="modocia", redlichiida="paradoxides")
animals = []
for name, (kind, style, extra) in CASES.items(): animals.append((f"case:{name}", kind, style, dict(extra)))
for name in schema.PRESETS: animals.append((f"preset:{name}", "shaped", "plain", dict(schema.preset(name))))
for f in sorted(glob.glob("presets/*.json")):
    n = os.path.basename(f)[:-5]
    if n.startswith("_"): continue
    st = STYLE_OF.get(n); st = st if (st in HS.STYLES) else None
    animals.append((f"order:{n}", "shaped" if st else "isopod", st, dict(json.load(open(f))["params"])))
if only: animals = [a for a in animals if any(o == a[0] or o in a[0] for o in only)]
if MODE == "--list":
    for a in animals: print(a[0])
    sys.exit(0)
if MODE == "--summary":
    rows = [tuple(json.load(open(os.path.join(OUT, f)))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("| animal | spines | audit findings | mesh | s |\n|---|---|---|---|---|\n")
        for label, n, audit, mesh_, t in rows: f.write(f"| {label} | {n} | {'<br>'.join(audit) if audit else 'all connected and clear'} | {mesh_} | {t} |\n")
    print(f"{len(rows)} animals, {sum(1 for r in rows if not r[2])} clean"); sys.exit(0)
rows = []
for label, kind, style, extra in animals:
    t0 = time.time(); D = schema.defaults()
    try:
        P = schema.coerce(dict(D, **extra), quiet=True)
        if kind == "isopod": parts, plans, PV, K = BODY.build(P, "isopod")
        else:
            P, face = HS.apply(P, style); P = schema.coerce(dict(P, **extra), quiet=True)
            parts, plans, PV, K = BODY.build(P, "shaped", face=face, outline=HS.outline(style))
        n_sp = len(BODY.LAST_ORNAMENTS); audit = list(BODY.LAST_AUDIT); wt = sum(p.is_watertight for p in parts); nb = sum(len(M.bodies(p)) == 1 for p in parts)
        rows.append((label, n_sp, audit, f"{wt}/{len(parts)} watertight, {nb}/{len(parts)} one body", round(time.time() - t0)))
        print(f"{label}: {n_sp} spines, {len(audit)} audit findings, {wt}/{len(parts)} watertight, {time.time()-t0:.0f}s", flush=True)
        for a in audit: print("   -", a, flush=True)
        s = K["s"]; whole = trimesh.util.concatenate([p.copy().apply_scale(1.0 / s) for p in parts])
        ims = [RD.render(whole, 90, -90, 620, 620), RD.render(whole, 35, -55, 620, 620)]
        im = Image.new("RGB", (1248, 660)); d = ImageDraw.Draw(im)
        for i, a in enumerate(ims): im.paste(Image.fromarray(a), (2 + 624 * i, 36))
        d.text((6, 6), f"{label}: {n_sp} spines; audit: {len(audit)} finding(s)", fill=(246, 112, 79))
        for i, a in enumerate(audit[:3]): d.text((6, 18 + 0 * i), "", fill=(0, 0, 0))
        d.text((6, 20), " | ".join(audit[:3])[:230], fill=(200, 200, 200))
        im.save(os.path.join(OUT, label.replace(":", "_") + ".png"))
    except Exception:
        rows.append((label, 0, ["BUILD FAILED: " + traceback.format_exc()[-300:]], "", round(time.time() - t0))); print(f"{label}: BUILD FAILED", flush=True)
    json.dump(rows[-1], open(os.path.join(OUT, label.replace(":", "_") + ".json"), "w"))
print("done", flush=True)
