"""
web/app.py — the trilobite site on the mesh builder (prompt 7, 11 Sep 2026). Flask, no OpenCascade.

    python web/app.py            # http://localhost:8765 (PORT env overrides)

Controls are the 3x3 CELLS contract from schema.py (three primary parameters per cell, more one tap deeper), plus the
prong pickers. The viewer loads one GLB per part; hovering a cell highlights its tagma row. "Curl it" runs the
instrument (instrument.read: build at the fixed measurement bevel, probe, measure — ~30 s) and returns the reading and
the hinge kinematics so the browser animates the animal to its stop with the limiting pair flashing.
"""
import os, sys, json, glob, time, threading, hashlib, shutil
import trimesh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from flask import Flask, request, jsonify, send_from_directory, send_file
import schema, instrument as I, assemble, joints
from joints import pin as PIN

# manifold3d is pinned to 3.0.1 (PREREG §9): other versions' booleans shed sliver flakes, pinch thin envelopes to
# non-manifold edges, and spike cylinder seams. A wrong-version image is what shipped the 21 Sep head pinch. Warn
# loudly at startup so a mis-built deploy is visible in the logs instead of silently exporting crumbs.
import importlib.metadata as _md
try: _MANIFOLD_V = _md.version("manifold3d")
except Exception: _MANIFOLD_V = "unknown"
if _MANIFOLD_V != "3.0.1":
    sys.stderr.write(f"[trilobite] WARNING: manifold3d {_MANIFOLD_V} != pinned 3.0.1 — print/measure geometry may carry "
                     f"boolean artifacts (non-manifold pinches, sliver flakes, seam spikes). `pip install -r requirements.txt`.\n")
    sys.stderr.flush()

app = Flask(__name__, static_folder=None)
CACHE = os.path.join(ROOT, "web", "cache"); os.makedirs(CACHE, exist_ok=True)

def _build_sig(*names):
    """A short hash of the geometry-builder source. Folded into every print cache key so any change to how parts
    are built (printjoint2 gaps/chamfer/clean(), printfill, the parts/mesh/fields stack) invalidates the cache on
    its own — no manual web/cache/ clear after a deploy. Params live in param_hash(P); this is the code half."""
    h = hashlib.sha1()
    for n in names:
        try:
            with open(os.path.join(ROOT, n), "rb") as f: h.update(f.read().replace(b"\r\n", b"\n"))   # LF-normalize: same key on Windows and in the Docker (LF) build
        except OSError: pass
    return h.hexdigest()[:8]
BUILD_SIG = _build_sig("assemble.py", "joints/pin.py", "joints/flexi.py", "joints/ball.py", "printfill.py", "anatomy/head.py", "anatomy/thorax.py", "anatomy/tail.py", "anatomy/common.py", "mesh.py", "fields.py")
# the isopod model is a separate builder (isopod_model/); its own source hash keys its cache half
ISOPOD_SIG = _build_sig("isopod_model/body.py", "isopod_model/crescent_head.py", "anatomy/head_crescent.py", "joints/ball.py", "mesh.py", "fields.py")
def _isopod_body():
    """Import isopod_model/body.py (its intra-package `import crescent_head` needs isopod_model on sys.path)."""
    d = os.path.join(ROOT, "isopod_model")
    if d not in sys.path: sys.path.insert(0, d)
    import body as IBODY
    return IBODY

def _build_isopod(P, folder):
    """Build the isopod model (isopod body + ball joints + crescent-on-isopod head) at print mm. Returns (parts, man-fields)."""
    IBODY = _isopod_body()
    parts, plans, PV, K = IBODY.build(P, "isopod")
    s = K["s"]                                                  # model units -> the isopod's own print mm
    pieces = [p.copy().apply_scale(1.0 / s) for p in parts]
    names = ["head"] + [f"seg{i}" for i in range(len(pieces) - 2)] + ["tail"]
    pivots = [[float(y / s), float(z / s)] for (y, z) in PV]    # per-joint pivot (y, z) in print mm, for the viewer's pose
    out = []
    for name, m in zip(names, pieces):
        m.export(os.path.join(folder, f"{name}.glb")); m.export(os.path.join(folder, f"{name}.stl"))
        out.append(dict(name=name, url=f"/files/{os.path.basename(folder)}/{name}.glb", stl=f"/files/{os.path.basename(folder)}/{name}.stl",
                        bodies=len(__import__("mesh").bodies(m)), watertight=bool(m.is_watertight), volume=round(float(m.volume), 1)))
    kin = dict(names=names, pivots=pivots)                      # pivots present => the viewer poses about per-joint pivots (ball joints)
    return out, kin
LOCK = threading.Lock()          # one build or measurement at a time (the lab workstation has one job's worth of RAM to spare)
MEASURED = {}                    # param hash -> last instrument result (so the sheet can carry the reading and the enrolled pose)
PROGRESS = {"done": 0, "total": 0}   # parts finished in the build now running; the page polls it for its loading count
                                     # (one build at a time, held by LOCK, so a single global is enough)
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

@app.get("/tree")
def tree(): return send_file(os.path.join(ROOT, "web", "tree.html"))                  # the ring/spiral family tree

@app.get("/family-tree")
def family_tree(): return send_file(os.path.join(ROOT, "web", "tree.html"))            # alias -> the same page

@app.get("/loading.gif")
def loading_gif(): return send_file(os.path.join(ROOT, "web", "loading.gif"))

@app.get("/api/progress")
def api_progress(): return jsonify(PROGRESS)

# eye configurations offered as a single dropdown in the head cell, built from the schema's own CHARACTERS bundles
EYE_TYPES = ["holochroalEyes", "schizochroalEyes", "pelagicEyes", "stalkedEyes", "eyesLost"]

@app.get("/api/schema")
def api_schema():
    eye_types = [dict(key=k, name=schema.CHARACTERS[k]["name"], set=schema.CHARACTERS[k]["set"]) for k in EYE_TYPES]
    return jsonify(dict(version=schema.SCHEMA_VERSION, instrument=I.INSTRUMENT_VERSION, params=_params_meta(), cells=schema.CELLS,
                        vertical_spines=schema.VERTICAL_SPINES, presets=sorted(_presets()), default=schema.DEFAULT_PRESET,
                        eye_types=eye_types))

@app.get("/api/preset/<name>")
def api_preset(name):
    P = _presets().get(name)
    if P is None: return jsonify(error="no such preset"), 404
    return jsonify(schema.coerce(P, base=schema.table_defaults()))

def _kinematics(P):
    return dict(hinge_z=PIN.hinge_z(P), offsets=PIN.joint_offsets(P), names=I.part_names(P))

@app.post("/api/build")
def api_build():
    """Print geometry: every part at the printed stop bevel (P['maxAngle']), one GLB each, cached by parameter hash."""
    body = request.get_json(force=True)
    P, notes = schema.coerce_report(body.get("P", {}), base=schema.table_defaults())
    joint = body.get("joint", "isopod")                    # "isopod" = the default (isopod model); "pin" = measured; "flexi"/"ball" = print joints
    fill = float(body.get("fill", 0.0) or 0.0)              # pin builds only: thicken the shell for printing (mm), print-only
    if joint == "isopod":
        key = schema.param_hash(P) + "-i" + ISOPOD_SIG      # the isopod model has its own builder-source hash
        folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
        PROGRESS.update(done=0, total=1)
        if os.path.exists(manifest):
            PROGRESS.update(done=1); return send_file(manifest)
        with LOCK:
            if os.path.exists(manifest):
                PROGRESS.update(done=1); return send_file(manifest)
            t0 = time.time(); os.makedirs(folder, exist_ok=True); bnotes = []
            try:
                out, kin = _build_isopod(P, folder)
            except Exception as ex:
                shutil.rmtree(folder, ignore_errors=True)               # never cache a failed build — a retry must rebuild, not serve the error
                PROGRESS.update(done=1)
                return jsonify(key=key, parts=[], kinematics=_kinematics(P), joint="isopod",
                               build_notes=[("isopod", "build failed", str(ex)[:160])], build_seconds=round(time.time() - t0, 1)), 500
            man = dict(key=key, parts=out, kinematics=kin, print=dict(print_valid=True, notes=[]), schema_notes=notes,
                       build_notes=bnotes, build_seconds=round(time.time() - t0, 1), schema=schema.SCHEMA_VERSION, joint="isopod", fill_mm=0)
            json.dump(P, open(os.path.join(folder, "params.json"), "w")); json.dump(man, open(manifest, "w")); _evict()
            PROGRESS.update(done=1); return jsonify(man)
    jtag = ("" if joint == "pin" else f"-{joint}") + (f"-fill{fill:g}" if fill > 0.05 and joint == "pin" else "")  # pin stays untagged (matches the sheet key)
    key = schema.param_hash(P) + "-b" + BUILD_SIG + jtag   # params + builder-source hash: a code change never serves stale parts
    folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
    n_parts = int(P["segCount"]) + 2
    PROGRESS.update(done=0, total=n_parts)
    if os.path.exists(manifest):
        PROGRESS.update(done=n_parts); return send_file(manifest)
    with LOCK:
        if os.path.exists(manifest):
            PROGRESS.update(done=n_parts); return send_file(manifest)
        t0 = time.time(); os.makedirs(folder, exist_ok=True); bnotes = []; out = []
        JM = joints.get(joint)                                          # the joint module: pin (measured) or a print joint
        if fill > 0.05 and JM.MEASURED:
            import printfill as F
            builders = [("head", lambda: F.cephalon(P, fill))] + [(f"seg{i}", (lambda i=i: F.segment(P, i, fill))) for i in range(int(P["segCount"]))] + [("tail", lambda: F.pygidium(P, fill))]
        else:
            builders = assemble.builders(P, JM, notes=bnotes)
        clean = getattr(JM, "clean", lambda m: m)
        if not JM.MEASURED: PROGRESS.update(total=n_parts + 1)          # + the overhang pass (restore), which needs every part
        built = []
        for name, fn in builders:
            try:
                built.append((name, clean(fn()), None))                 # drop ghost shells + thin border flakes (< 5 mm3 or < 0.5 mm)
            except Exception as ex:
                built.append((name, None, str(ex)[:120]))
            PROGRESS.update(done=len(built))
        overhangs = None
        if not JM.MEASURED and all(m is not None for _, m, _ in built):
            try:                                                        # put back the spines/arms the joint trim removed,
                overhangs = []                                          # cleared against every later part through the curl
                fixed = assemble.restore(P, [m for _, m, _ in built], JM, report=overhangs)
                built = [(nm, f, None) for (nm, _, _), f in zip(built, fixed)]
            except Exception as ex:
                bnotes.append(("print", "overhangs not restored", str(ex)[:120])); overhangs = None
            PROGRESS.update(done=n_parts + 1)
        for name, m, err in built:
            if err is not None:
                out.append(dict(name=name, error=err)); continue
            m.export(os.path.join(folder, f"{name}.glb")); m.export(os.path.join(folder, f"{name}.stl"))
            out.append(dict(name=name, url=f"/files/{key}/{name}.glb", stl=f"/files/{key}/{name}.stl", bodies=len(__import__("mesh").bodies(m)), watertight=bool(m.is_watertight), volume=round(m.volume, 1)))
        man = dict(key=key, parts=out, kinematics=_kinematics(P), print=I.print_validity(P), schema_notes=notes, build_notes=bnotes,
                   build_seconds=round(time.time() - t0, 1), schema=schema.SCHEMA_VERSION, joint=joint, fill_mm=fill)
        if not JM.MEASURED:
            man["overhangs"] = overhangs; pv = JM.pivot(P)
            man["print_joint"] = dict(pivot_z=pv["z"], pivot_y_beyond_joint=pv["y_beyond_plane"], scale=pv.get("scale", 1.0), gaps=pv.get("gaps"))
        json.dump(P, open(os.path.join(folder, "params.json"), "w")); json.dump(man, open(manifest, "w")); _evict()
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
    body = request.get_json(force=True)
    if body.get("joint") == "isopod":                              # the isopod model has its own sheet (ball joints, crescent head, print-in-place)
        P, notes = schema.coerce_report(body.get("P", {}), base=schema.table_defaults())
        key = schema.param_hash(P) + "-i" + ISOPOD_SIG
        folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
        if not os.path.exists(manifest):
            with app.test_request_context(json={"P": P, "joint": "isopod"}): api_build()
        man = json.load(open(manifest)); png = os.path.join(folder, "sheet_isopod.png")
        if os.path.exists(png): return jsonify(url=f"/files/{key}/sheet_isopod.png", measured=False)
        with LOCK:
            parts = [p for p in man.get("parts", []) if "error" not in p]
            if not parts: return jsonify(error="isopod build failed — no parts to draw"), 500
            meshes = [trimesh.load(os.path.join(folder, f"{p['name']}.stl")) for p in parts]
            IB = _isopod_body(); kin = man.get("kinematics", {})
            info = dict(names=[p["name"] for p in parts], pivots=kin.get("pivots", []), curl_deg=getattr(IB, "CURL", 30.0),
                        params=key, schema=man.get("schema", ""), build_seconds=man.get("build_seconds"),
                        volumes=[p.get("volume") for p in parts])
            blueprint.isopod_sheet(meshes, P, info, png)
        return jsonify(url=f"/files/{key}/sheet_isopod.png", measured=False)
    P, notes = schema.coerce_report(body.get("P", {}), base=schema.table_defaults())
    phash = schema.param_hash(P); key = phash + "-b" + BUILD_SIG   # the sheet draws the default (pin) build; key must match api_build's
    folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
    if not os.path.exists(manifest):
        with app.test_request_context(json={"P": P, "joint": "pin"}): api_build()   # the sheet draws the pin build; be explicit now the default is isopod
    man = json.load(open(manifest)); meas = MEASURED.get(phash)   # MEASURED is keyed by the plain param hash (set in /api/measure)
    tag = "measured" if meas else "flat"; png = os.path.join(folder, f"sheet_{tag}.png")
    if os.path.exists(png): return jsonify(url=f"/files/{key}/sheet_{tag}.png", measured=bool(meas))
    with LOCK:
        names = [p["name"] for p in man["parts"] if "error" not in p]
        meshes = [trimesh.load(os.path.join(folder, f"{n}.stl")) for n in names]
        # parts are built in their own frames (front hinge at y = 0): place them at rest before drawing
        mats0 = joints.get(man.get("joint", "pin")).transforms_deg(P, 0.0)
        flat = trimesh.util.concatenate([mm.copy().apply_transform(T) for mm, T in zip(meshes, mats0)])
        m = dict(meas or {}); m.setdefault("limited_by", "not measured"); m.setdefault("enroll_class", "—")
        m.update(hinge_z=round(PIN.hinge_z(P), 2), pitch=round(PIN.joint_offsets(P)[1], 2), knuckle=round(PIN.hinge_width(P) / int(P["nKnuckles"]) - P["clearance"], 2),
                 e_max=m.get("v1_equivalent_e_max", "—"), params=phash, print_valid=man["print"]["print_valid"])
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
        P = schema.coerce(json.load(open(os.path.join(folder, "params.json")))) if os.path.exists(os.path.join(folder, "params.json")) else None
        idx = [k for k, p in enumerate(man["parts"]) if "error" not in p]
        meshes = [trimesh.load(os.path.join(folder, f"{man['parts'][k]['name']}.stl")) for k in idx]
        if man.get("joint") == "isopod":
            whole = trimesh.util.concatenate(meshes)                       # isopod parts are already in one frame at print mm
        else:
            JM = joints.get(man.get("joint", "pin"))
            mats0 = JM.transforms_deg(P, 0.0) if P is not None else [__import__("numpy").eye(4)] * len(man["parts"])
            whole = trimesh.util.concatenate([mm.copy().apply_transform(mats0[k]) for mm, k in zip(meshes, idx)])
            if P is not None and hasattr(JM, "model_scale"):
                whole.apply_scale(JM.model_scale(P))                       # scale the print to the joint's native (isopod) size
        V = whole.vertices                                                 # refuse a file a slicer would read as enormous
        if not __import__("numpy").isfinite(V).all() or (V.max(0) - V.min(0)).max() > 800:   # 800: the ball joint scales to isopod size
            return jsonify(error="export failed its size check (non-finite or > 800 mm); rebuild this animal"), 500
        tmp = out + ".part"; whole.export(tmp, file_type="stl"); os.replace(tmp, out)   # never serve a half-written file
    return send_file(out, as_attachment=True, download_name=f"trilobite_{key}.stl")

@app.get("/files/<key>/<path:name>")
def files(key, name): return send_from_directory(os.path.join(CACHE, key), name)

def _evict(max_dirs=60):
    dirs = sorted(glob.glob(os.path.join(CACHE, "*")), key=os.path.getmtime)
    for d in dirs[:-max_dirs]: shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
