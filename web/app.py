"""
web/app.py — the trilobite site on the mesh builder (prompt 7, 11 Sep 2026). Flask, no OpenCascade.

    python web/app.py            # http://localhost:8765 (PORT env overrides)

Controls are the 3x3 CELLS contract from schema.py (three primary parameters per cell, more one tap deeper), plus the
prong pickers. The viewer loads one GLB per part; hovering a cell highlights its tagma row. "Curl it" runs the
instrument (instrument.read: build at the fixed measurement bevel, probe, measure — ~30 s) and returns the reading and
the hinge kinematics so the browser animates the animal to its stop with the limiting pair flashing.
"""
import os, sys, json, glob, time, threading, hashlib, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from flask import Flask, request, jsonify, send_from_directory, send_file
import schema, parts, instrument as I

app = Flask(__name__, static_folder=None)
CACHE = os.path.join(ROOT, "web", "cache"); os.makedirs(CACHE, exist_ok=True)
LOCK = threading.Lock()          # one build or measurement at a time (the lab workstation has one job's worth of RAM to spare)
MEASURED = {}                    # param hash -> last instrument result (so the sheet can carry the reading and the enrolled pose)
PORT = int(os.environ.get("PORT", 8765))   # the lab tunnel (trilomorph.org) points at 8765, the legacy port

# ---- maintenance mode: flip to False (or delete this block) to bring the generator back. While True, every
# route - the page, every /api/*, every /files/* - returns this instead of running any real code.
MAINTENANCE = False
MAINTENANCE_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Trilobite Morphospace</title>
<style>html,body{height:100%;margin:0;background:#000;color:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}
body{display:flex;align-items:center;justify-content:center}
h1{font-size:1.6rem;letter-spacing:.18em;text-transform:uppercase;font-weight:normal}</style></head>
<body><h1>Under maintenance</h1></body></html>"""

@app.before_request
def _maintenance_gate():
    if MAINTENANCE:
        return MAINTENANCE_HTML, 503

def _params_meta():
    return [dict(key=p.key, label=p.label, default=p.default, lo=p.lo, hi=p.hi, step=p.step, kind=p.kind, doc=p.doc, unit=p.unit, group=p.group)
            for p in schema.PARAMS]

def _presets():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "presets", "*.json"))):
        n = os.path.basename(f)[:-5]
        if not n.startswith("_"): out[n] = json.load(open(f))["params"]
    for n in schema.PRESETS: out.setdefault(n, schema.preset(n))
    return out

@app.get("/")
def index(): return send_file(os.path.join(ROOT, "web", "index.html"))

@app.get("/api/schema")
def api_schema():
    return jsonify(dict(version=schema.SCHEMA_VERSION, instrument=I.INSTRUMENT_VERSION, params=_params_meta(), cells=schema.CELLS,
                        vertical_spines=schema.VERTICAL_SPINES, presets=sorted(_presets()), default=schema.DEFAULT_PRESET))

@app.get("/api/preset/<name>")
def api_preset(name):
    P = _presets().get(name)
    if P is None: return jsonify(error="no such preset"), 404
    return jsonify(schema.coerce(P, base=schema.table_defaults()))

def _kinematics(P):
    return dict(hinge_z=parts.hinge_z(P), offsets=parts.joint_offsets(P), names=I.part_names(P))

@app.post("/api/build")
def api_build():
    """Print geometry: every part at the printed stop bevel (P['maxAngle']), one GLB each, cached by parameter hash."""
    P, notes = schema.coerce_report(request.get_json(force=True).get("P", {}), base=schema.table_defaults())
    key = schema.param_hash(P); folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
    if os.path.exists(manifest): return send_file(manifest)
    with LOCK:
        if os.path.exists(manifest): return send_file(manifest)
        t0 = time.time(); os.makedirs(folder, exist_ok=True); bnotes = []; out = []
        builders = [("head", lambda: parts.cephalon(P, notes=bnotes))] + [(f"seg{i}", (lambda i=i: parts.segment(P, i))) for i in range(int(P["segCount"]))] + [("tail", lambda: parts.pygidium(P))]
        for name, fn in builders:
            try:
                m = fn(); m.export(os.path.join(folder, f"{name}.glb")); m.export(os.path.join(folder, f"{name}.stl"))
                out.append(dict(name=name, url=f"/files/{key}/{name}.glb", stl=f"/files/{key}/{name}.stl", bodies=len(__import__("mesh").bodies(m)), volume=round(m.volume, 1)))
            except Exception as ex:
                out.append(dict(name=name, error=str(ex)[:120]))
        man = dict(key=key, parts=out, kinematics=_kinematics(P), print=I.print_validity(P), schema_notes=notes, build_notes=bnotes,
                   build_seconds=round(time.time() - t0, 1), schema=schema.SCHEMA_VERSION)
        json.dump(man, open(manifest, "w")); _evict()
        return jsonify(man)

@app.post("/api/measure")
def api_measure():
    """The instrument: fixed measurement bevel, probe, sweep. Synchronous (~30 s); the page shows the wait."""
    P, notes = schema.coerce_report(request.get_json(force=True).get("P", {}), base=schema.table_defaults())
    with LOCK:
        r, B = I.read(P)
    r = {k: v for k, v in r.items() if k not in ("scan_trace",)}
    r.update(kinematics=_kinematics(P), schema_notes=notes)
    MEASURED[schema.param_hash(P)] = json.loads(json.dumps(r, default=str))
    return jsonify(MEASURED[schema.param_hash(P)])

@app.post("/api/sheet")
def api_sheet():
    """The blueprint sheet (A3, blueprint.sheet) for the current parameters: the flat animal from the cached build, the
    last instrument reading for these parameters if there is one (with the animal enrolled to its stop superimposed)."""
    import trimesh, blueprint
    P, notes = schema.coerce_report(request.get_json(force=True).get("P", {}), base=schema.table_defaults())
    key = schema.param_hash(P); folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
    if not os.path.exists(manifest):
        with app.test_request_context(json={"P": P}): api_build()
    man = json.load(open(manifest)); meas = MEASURED.get(key)
    tag = "measured" if meas else "flat"; png = os.path.join(folder, f"sheet_{tag}.png")
    if os.path.exists(png): return jsonify(url=f"/files/{key}/sheet_{tag}.png", measured=bool(meas))
    with LOCK:
        names = [p["name"] for p in man["parts"] if "error" not in p]
        meshes = [trimesh.load(os.path.join(folder, f"{n}.stl")) for n in names]
        flat = trimesh.util.concatenate(meshes)
        m = dict(meas or {}); m.setdefault("limited_by", "not measured"); m.setdefault("enroll_class", "—")
        m.update(hinge_z=round(parts.hinge_z(P), 2), pitch=round(parts.pitch(P), 2), knuckle=round(parts.hinge_width(P) / int(P["nKnuckles"]) - P["clearance"], 2),
                 e_max=m.get("v1_equivalent_e_max", "—"), params=key, print_valid=man["print"]["print_valid"])
        enrolled = None
        if meas and meas.get("theta_joint_deg") is not None:
            mats = I.transforms_deg(P, float(meas["theta_joint_deg"]))
            enrolled = trimesh.util.concatenate([mm.copy().apply_transform(T) for mm, T in zip(meshes, mats)])
        blueprint.sheet(flat, P, m, png, enrolled=enrolled)
    return jsonify(url=f"/files/{key}/sheet_{tag}.png", measured=bool(meas))

@app.get("/api/stl/<key>")
def api_stl(key):
    """One combined flat STL of the cached build — the download."""
    import trimesh
    folder = os.path.join(CACHE, key); man = json.load(open(os.path.join(folder, "manifest.json")))
    out = os.path.join(folder, "flat.stl")
    if not os.path.exists(out):
        trimesh.util.concatenate([trimesh.load(os.path.join(folder, f"{p['name']}.stl")) for p in man["parts"] if "error" not in p]).export(out)
    return send_file(out, as_attachment=True, download_name=f"trilobite_{key}.stl")

@app.get("/files/<key>/<path:name>")
def files(key, name): return send_from_directory(os.path.join(CACHE, key), name)

def _evict(max_dirs=60):
    dirs = sorted(glob.glob(os.path.join(CACHE, "*")), key=os.path.getmtime)
    for d in dirs[:-max_dirs]: shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
