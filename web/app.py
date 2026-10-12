"""
web/app.py — the trilobite site on the mesh builder (prompt 7, 11 Sep 2026). Flask, no OpenCascade.

    python web/app.py            # http://localhost:8765 (PORT env overrides)

Controls are the 3x3 CELLS contract from schema.py (three primary parameters per cell, more one tap deeper), plus the
prong pickers. The viewer loads one GLB per part; hovering a cell highlights its tagma row. "Curl it" runs the
instrument (instrument.read: build at the fixed measurement bevel, probe, measure — ~30 s) and returns the reading and
the hinge kinematics so the browser animates the animal to its stop with the limiting pair flashing.
"""
import os, sys, re, json, glob, time, threading, hashlib, shutil
import trimesh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from flask import Flask, request, jsonify, send_from_directory, send_file
import schema, instrument as I, assemble, joints
from joints import pin as PIN
sys.path.insert(0, os.path.join(ROOT, 'web')); import recorder as RECORDER

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
threading.Thread(target=RECORDER.flush, daemon=True).start()   # send on any measurements stored while the receiver was unreachable
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
ISOPOD_SIG = _build_sig("isopod_model/body.py", "isopod_model/rows.py", "isopod_model/crescent_head.py", "anatomy/head_crescent.py", "joints/ball.py", "mesh.py", "fields.py")
SHAPED_SIG = _build_sig("isopod_model/body.py", "isopod_model/rows.py", "isopod_model/shaped_head.py", "isopod_model/head_styles.py", "anatomy/head_crescent.py", "joints/ball.py", "mesh.py", "fields.py")
def _isopod_body():
    """Import isopod_model/body.py (its intra-package `import crescent_head` needs isopod_model on sys.path)."""
    d = os.path.join(ROOT, "isopod_model")
    if d not in sys.path: sys.path.insert(0, d)
    import body as IBODY
    return IBODY

def _head_styles():
    """isopod_model/head_styles.py (the named heads: a silhouette + a sculpted face each)."""
    _isopod_body(); import head_styles as HS
    return HS

def _head_style(name):
    """A valid head-style name, or None for the approved crescent on the isopod head piece. A SAVED head's name
    (web/heads/<name>.json, from the head editor) or a dict ({"style", "outline", "face"}, the editor's live head) is a
    custom shaped head and is returned resolved (see _head_custom)."""
    if isinstance(name, dict): return _head_custom(name)
    if name and name in _saved_heads(): return _head_custom({"style": name})
    return name if name and name != "crescent" and name in _head_styles().STYLES else None

# ---- saved heads: the head editor's "save". One JSON per head in web/heads/ (git-ignored, like records/): the guide
# style it started from and the outline / face edits on top. A saved head is addressed by its name wherever a style
# name goes (the head picker, ?head=, /api/build, /api/sheet, the editor's ?style=), and is re-read at every use, so
# re-saving a name re-keys every build made with it (the cache key hashes the resolved numbers).
HEADS = os.path.join(ROOT, "web", "heads")
_HEAD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{0,39}$")

def _saved_heads():
    out = {}
    for f in sorted(glob.glob(os.path.join(HEADS, "*.json"))):
        try: d = json.load(open(f)); out[d["name"]] = d
        except Exception: pass
    return out

def _head_custom(spec):
    """A custom shaped head: a guide style's outline and face with edits on top. `spec["style"]` is a guide style, the
    crescent, or a SAVED head (whose own edits then come first, `spec`'s on top). Returns dict(style, base, outline,
    face, id, saved): `base` the guide style the geometry rests on, the outline fully resolved, `face` as edits on the
    crescent's own face, both validated (unknown keys raise KeyError), and `id`, a short hash of the resolved numbers
    for the cache key; `saved` the saved head's record when `style` named one."""
    HS = _head_styles(); import shaped_head as SH; from anatomy import head_crescent as HC
    style = spec.get("style") or "crescent"; saved = None
    o_edits = {k: float(v) for k, v in (spec.get("outline") or {}).items()}
    f_edits = {k: float(v) for k, v in (spec.get("face") or {}).items()}
    if style in _saved_heads():
        saved = _saved_heads()[style]; style = saved.get("style") or "crescent"
        o_edits = dict(saved.get("outline") or {}, **o_edits); f_edits = dict(saved.get("face") or {}, **f_edits)
    # the base face is what the BUILD would use: the default preset with the style's schema values (eyeSize, glabInflate ...)
    # mapped through body.face_of, then the style's own face keys on top -- so the drawing matches the solid
    IBODY = _isopod_body(); P0 = schema.coerce(dict(schema.preset(schema.DEFAULT_PRESET), **(HS.style(style)[1] if style in HS.STYLES else {})), base=schema.table_defaults())
    base_face = IBODY.face_of(P0, HS.style(style)[0] if style in HS.STYLES else None)
    base_out = HS.outline(style) if style in HS.STYLES else {}
    outline = SH.outline_of(dict(base_out, **o_edits))
    face = HC.face_of(dict(base_face, **f_edits))
    face_edits = {k: v for k, v in face.items() if v != HC.FACE.get(k)}        # only what differs from the crescent's own face
    blob = json.dumps(dict(o=outline, f=face), sort_keys=True).encode()
    return dict(style=spec.get("style") or "crescent", base=style, outline=outline, face=face_edits, id=hashlib.sha1(blob).hexdigest()[:10],
                saved=(dict(name=saved["name"], style=saved.get("style") or "crescent", outline=saved.get("outline") or {}, face=saved.get("face") or {}) if saved else None))

def _iso_key(P, head=None, body=None):
    """Cache key of an isopod-model build. A head style has its own key and its own builder-source hash, so the
    crescent's cached builds are untouched by anything in the shaped-head code. A custom head (dict) keys on the hash
    of its resolved outline + face."""
    h = _head_style(head); spec = _body_spec(body)
    b = "" if spec is None else f"-b-{_body_id(P, spec)}"          # a rows-model body: its own id in the key
    if isinstance(h, dict): return schema.param_hash(P) + f"-h-custom-{h['id']}{b}-" + SHAPED_SIG
    return schema.param_hash(P) + ("-i" + ISOPOD_SIG if h is None else f"-h-{h}-" + SHAPED_SIG) + b

def _build_isopod(P, folder, head=None, body_spec=None):
    """Build the isopod model at print mm: isopod body + ball joints + either the crescent on the isopod head piece
    (head None) or a head style drawn on its own silhouette (3 Oct 2026). The style's outline and sculpt keys are
    build options; P is used as given (the page applies a style's parameter values to P when it is picked, so the
    sliders show them and stay editable). Returns (parts, man-fields)."""
    IBODY = _isopod_body(); h = _head_style(head)
    rows = _body_spec(body_spec)
    if h is None: parts, plans, PV, K = IBODY.build(P, "isopod", rows=rows)
    elif isinstance(h, dict):                                   # the head editor's head: its own outline + face
        parts, plans, PV, K = IBODY.build(P, "shaped", face=h["face"], outline=h["outline"], rows=rows)
    else:
        HS = _head_styles(); face, _ = HS.style(h)
        parts, plans, PV, K = IBODY.build(P, "shaped", face=face, outline=HS.outline(h), rows=rows)
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
    # the checks (11 Oct 2026): the build has already refused invalid parts (body.MeshError); here what a slicer and the
    # chain would still notice -- parts touching at rest, and the debris the clean-up removed
    import check as CHECK
    touching = [dict(a=names[k], b=names[j], mm3=round(v / s ** 3, 3)) for k, j, v in CHECK.contacts(parts, [(y, z) for y, z in PV], 0.0)]
    dropped = {names[k]: round(v / s ** 3, 3) for k, v in getattr(IBODY, "LAST_DROPPED", {}).items() if v > 0}
    cut = {k: v for k, v in dropped.items() if v > 20.0}                  # more than debris: the clearances severed anatomy (parts overlapped)
    checks = dict(print_valid=all(p["watertight"] and p["bodies"] == 1 for p in out) and not touching and not cut,
                  notes=[f"{t['a']} and {t['b']} touch at rest: {t['mm3']} mm3" for t in touching]
                        + [f"{k}: {v:.0f} mm3 cut away by the clearances (it overlapped the part ahead; a forward sweep or a long overlap?)" for k, v in cut.items()]
                        + ([f"debris dropped (mm3): {dropped}"] if dropped and not cut else []),
                  touching=touching, debris_mm3=dropped)
    return out, kin, checks
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

@app.get("/head")
def head_page(): return send_file(os.path.join(ROOT, "web", "head.html"))              # the head editor (p5): plan + elevation with handles

@app.get("/body")
def body_page(): return send_file(os.path.join(ROOT, "web", "body.html"))              # the body editor (p5): the rows, head to tail tip

@app.get("/studio")
def studio_page(): return send_file(os.path.join(ROOT, "web", "studio.html"))          # the combined editor (p5): head + body on one whole-animal drawing

# ---- saved bodies (the body editor's "save"): web/bodies/<name>.json = {name, preset, spec, doc}. A saved body's
# name goes wherever a body spec goes (?body=name, /api/build "body").
BODIES = os.path.join(ROOT, "web", "bodies")

def _saved_bodies():
    out = {}
    for f in sorted(glob.glob(os.path.join(BODIES, "*.json"))):
        try: d = json.load(open(f)); out[d["name"]] = d
        except Exception: pass
    return out

def _body_spec(body):
    """A body spec (dict), or a saved body's name -> its spec; None/''/'preset' = the preset's rows untouched (None)."""
    if body is None or body == "" or body == "preset": return None
    if isinstance(body, str):
        d = _saved_bodies().get(body)
        if d is None: raise KeyError(f"no saved body '{body}'")
        return d.get("spec") or {}
    return dict(body)

def _iso_problems(P, head, spec):
    """Why this animal cannot be built: the rows' problems (the kit does not fit, the print too big) and the head's
    (its spines would make the print longer than the bed). Checked before a build is started (11 Oct 2026)."""
    _isopod_body(); import rows as RW
    from anatomy.common import pitch
    from fields import seg_halfwidth
    out = []
    if spec is not None:
        R = RW.plans(P, spec, strict=False); out += R["problems"]; scale = R["scale"]; L_body = R["length"]
    else: scale = BALL_SCALE(P); L_body = P["length"] * (1 - P["cephFrac"])
    h = _head_style(head)
    if isinstance(h, dict):
        o = h["outline"]; L_head = o["length"] * o["width"] * seg_halfwidth(P, 0) * (1.0 + o["genal"])   # axial length + the genal spines
        if (L_head + L_body) * scale > RW.PRINT_MAX_MM: out.append(f"with this head the print would be {(L_head + L_body) * scale:.0f} mm long; at most {RW.PRINT_MAX_MM:g}")
    return out

def BALL_SCALE(P):
    from joints import ball as BALL
    return BALL.model_scale(P)

def _iso_n_parts(P, spec):
    """How many pieces the isopod build makes: head + tail + the thorax (the rows model's pieces after fusion)."""
    if spec is None: return int(P["segCount"]) + 2
    _isopod_body(); import rows as RW
    return len(RW.plans(P, spec, strict=False)["parts"]) + 1

def _body_id(P, spec):
    """Short hash of the RESOLVED rows (every row's numbers, the groups, the scale), for the cache key."""
    _isopod_body(); import rows as RW
    R = RW.plans(P, spec, strict=False)
    blob = json.dumps(dict(rows=R["rows"], groups=R["groups"], scale=R["scale"], ts=R["term_spine"], model=R["model"]), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]

@app.route("/api/body_diagram", methods=["GET", "POST"])
def api_body_diagram():
    """The body editor's drawing. GET ?preset=<order>[&body=<saved>]; POST {"preset", "spec": {...}, "head": <style|saved>}.
    Answers the rows diagram (isopod_model/rows.diagram: outline per row, pieces after fusion, joints, handles,
    the axis profile, the print size) and the head's diagram scaled to the first row, for the page to draw."""
    _isopod_body(); import rows as RW; import shaped_head as SH
    q = request.get_json(force=True, silent=True) or {} if request.method == "POST" else dict(request.args)
    preset = q.get("preset") or schema.DEFAULT_PRESET
    Pp = _presets().get(preset)
    if Pp is None: return jsonify(error="no such preset"), 404
    P = schema.coerce(Pp, base=schema.table_defaults())
    try:
        spec = _body_spec(q.get("spec") if request.method == "POST" else q.get("body")) or {}
        if request.method == "GET" and q.get("start") == "boring": spec = {"list": RW.boring(P)}    # the editor's plain starting animal
        D = RW.diagram(P, spec)
        h = _head_custom({"style": q.get("head") or "crescent"}); hd = _diagram_json(SH.diagram(h["outline"], h["face"]))
    except (KeyError, ValueError) as ex:
        return jsonify(error=str(ex)[:200]), 400
    return jsonify(preset=preset, spec=spec, diagram=D, head=dict(style=h["style"], diagram=hd), id=_body_id(P, spec),
                   presets=sorted(_presets()), saved_bodies=[dict(name=n, preset=d.get("preset"), doc=d.get("doc", "")) for n, d in _saved_bodies().items()])

@app.get("/api/bodies")
def api_bodies():
    return jsonify([dict(name=n, preset=d.get("preset"), head=d.get("head") or "crescent", spec=d.get("spec") or {}, doc=d.get("doc", ""), saved_at=d.get("saved_at")) for n, d in _saved_bodies().items()])

@app.post("/api/macro")
def api_macro():
    """The opening page's macro model made to ROUGHLY mirror the editors (11 Oct 2026). {"P": the page's parameters,
    "preset"?, "head": a style or saved head, "body": a saved body's name or a spec} -> {"P": {schema key: value}}:
    the head's projection (its guide style's values, face keys, the outline's genal spine / front / dome / length)
    laid over P, then the body's (row counts and lengths, the width field, medians of the row numbers, the spine
    field). Nothing is built; the page writes the values into its sliders. Only a SAVED or custom head projects its
    outline: a guide style picked from the list keeps the page's old rule (its parameter values alone)."""
    _isopod_body(); import rows as RW
    q = request.get_json(force=True, silent=True) or {}
    base = _presets().get(q.get("preset")) if q.get("preset") else None
    P = schema.coerce(q.get("P") or (base or {}), base=schema.table_defaults())
    out = {}; info = {}
    try:
        head = q.get("head")
        if head and head != "crescent":
            h = _head_style(head); HS = _head_styles()
            if h is None: return jsonify(error=f"no such head '{head}'"), 404
            if isinstance(h, dict):
                hm = HS.macro(P, h["outline"], h["face"], HS.STYLES.get(h["base"], {}).get("params", {}))
                out.update(hm); P = schema.coerce(dict(P, **hm), base=P); info["head"] = dict(style=h["style"], base=h["base"], keys=sorted(hm))
            else:
                hm = dict(HS.style(h)[1]); out.update(hm); P = schema.coerce(dict(P, **hm), base=P); info["head"] = dict(style=h, keys=sorted(hm))
        spec = _body_spec(q.get("body"))
        if spec is not None:
            bm = RW.macro(P, spec); out.update(bm); info["body"] = dict(keys=sorted(bm), rows=len(RW.plans(P, spec)["rows"]))
    except (KeyError, ValueError) as ex:
        return jsonify(error=str(ex)[:200]), 400
    return jsonify(P=out, info=info)

@app.post("/api/bodies")
def api_bodies_save():
    """Save a body: {"name", "preset", "spec", "doc"?}. Name rules as for heads. Re-saving replaces."""
    _isopod_body(); import rows as RW
    q = request.get_json(force=True, silent=True) or {}
    name = str(q.get("name", "")).strip()
    if not _HEAD_NAME.match(name): return jsonify(error="name: letters, digits, _ and -, 1 to 40 characters, starting with a letter"), 400
    if name in _presets() or name == "preset": return jsonify(error=f"'{name}' is a preset's name"), 400
    preset = q.get("preset") or schema.DEFAULT_PRESET
    if preset not in _presets(): return jsonify(error="no such preset"), 404
    try: spec = dict(q.get("spec") or {}); RW.plans(schema.coerce(_presets()[preset], base=schema.table_defaults()), spec, strict=False)   # work in progress may be saved; the build refuses it
    except (KeyError, ValueError) as ex: return jsonify(error=str(ex)[:200]), 400
    head = str(q.get("head") or "crescent")                                                # the head it was drawn with (a style or a saved head)
    if head != "crescent" and _head_style(head) is None: return jsonify(error=f"no such head '{head}'"), 404
    rec = dict(name=name, preset=preset, head=head, spec=spec, doc=str(q.get("doc", ""))[:200], saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(BODIES, exist_ok=True); tmp = os.path.join(BODIES, name + ".json.part")
    json.dump(rec, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(BODIES, name + ".json"))
    return jsonify(saved=rec)

@app.delete("/api/bodies/<name>")
def api_bodies_delete(name):
    if not _HEAD_NAME.match(name) or name not in _saved_bodies(): return jsonify(error="no such saved body"), 404
    os.remove(os.path.join(BODIES, name + ".json")); return jsonify(deleted=name)

STATIC = os.path.join(ROOT, "web", "static")                                            # self-hosted libraries (p5.min.js)
@app.get("/static/<path:name>")
def static_file(name): return send_from_directory(STATIC, name)

def _diagram_json(d):
    """shaped_head.diagram's arrays as rounded lists (None stays None)."""
    import numpy as np
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray): out[k] = np.round(v, 4).tolist()
        elif isinstance(v, dict): out[k] = {kk: ([round(float(x), 4) for x in vv] if isinstance(vv, (tuple, list)) else vv) for kk, vv in v.items()}
        else: out[k] = v
    return out

@app.route("/api/head_diagram", methods=["GET", "POST"])
def api_head_diagram():
    """The head editor's drawing. GET ?style=<name> gives a style's resolved outline + face and its diagram;
    POST {"style", "outline": {edits}, "face": {edits}} redraws with the edits on top (about 0.2 s: the same Shape
    and head_crescent functions the solid is built from, so what is drawn is what builds). The answer carries the
    resolved outline and face, the diagram polylines (plan right half, glabella, eye, brim, boss, section, profile,
    all in half-width units), the handle positions, which keys each handle drags, and the defaults."""
    HS = _head_styles(); import shaped_head as SH; from anatomy import head_crescent as HC
    spec = request.get_json(force=True, silent=True) or {} if request.method == "POST" else {"style": request.args.get("style", "crescent")}
    try:
        h = _head_custom(spec)
        d = SH.diagram(h["outline"], h["face"])
    except (KeyError, ValueError) as ex:
        return jsonify(error=str(ex)[:200]), 400
    face_full = HC.face_of(h["face"])
    return jsonify(style=h["style"], base=h["base"], saved=h["saved"], outline=h["outline"], face=face_full, id=h["id"], diagram=_diagram_json(d),
                   handles=SH.HANDLES, outline_defaults=SH.OUTLINE, face_defaults=HC.FACE,
                   styles=[dict(name=n, doc=s.get("doc", ""), p=s.get("p", 0), params=s.get("params", {})) for n, s in HS.STYLES.items()],
                   saved_heads=[dict(name=n, style=r.get("style") or "crescent", doc=r.get("doc", "")) for n, r in _saved_heads().items()])

@app.get("/loading.gif")
def loading_gif(): return send_file(os.path.join(ROOT, "web", "loading.gif"))

@app.get("/api/progress")
def api_progress(): return jsonify(PROGRESS)

# eye configurations offered as a single dropdown in the head cell, built from the schema's own CHARACTERS bundles
EYE_TYPES = ["holochroalEyes", "schizochroalEyes", "pelagicEyes", "stalkedEyes", "eyesLost"]

# the morphological characters (schema.CHARACTERS) as selectable "features", each tagged with the 3x3 cell its
# parameters live in (so the picker can sit in the right cell). The cell is the most common cell among the set's params.
_PARAM_CELL = {}
for _cid, _c in schema.CELLS.items():
    for _k in list(_c.get("primary", [])) + list(_c.get("more", [])): _PARAM_CELL.setdefault(_k, _cid)
def _char_cell(ch):
    from collections import Counter
    cells = [_PARAM_CELL[k] for k in ch["set"] if k in _PARAM_CELL]
    return Counter(cells).most_common(1)[0][0] if cells else "a1"
def _characters_list():
    return [dict(key=k, name=c["name"], set=c["set"], cell=_char_cell(c)) for k, c in schema.CHARACTERS.items()]

@app.get("/api/head_styles")
def api_head_styles():
    """The head styles for the page's picker: name, what it is, the guide page, and the parameter values it sets."""
    HS = _head_styles()
    saved = [dict(name=n, doc=f"saved head, from {d.get('style') or 'crescent'}" + (f": {d['doc']}" if d.get("doc") else ""), p=0,
                  params=HS.STYLES.get(d.get("style") or "", {}).get("params", {}), saved=True, style=d.get("style") or "crescent") for n, d in _saved_heads().items()]
    return jsonify([dict(name="crescent", doc="the approved crescent, on the isopod head piece", p=0, params={})] +
                   [dict(name=n, doc=s.get("doc", ""), p=s.get("p", 0), params=s.get("params", {})) for n, s in HS.STYLES.items()] + saved)

@app.get("/api/heads")
def api_heads():
    """The saved heads (the head editor's "save"): name, the guide style each rests on, its edits, when it was saved."""
    return jsonify([dict(name=n, style=d.get("style") or "crescent", outline=d.get("outline") or {}, face=d.get("face") or {},
                         doc=d.get("doc", ""), saved_at=d.get("saved_at")) for n, d in _saved_heads().items()])

@app.post("/api/heads")
def api_heads_save():
    """Save a head: {"name", "style", "outline": {edits}, "face": {edits}, "doc"?}. `style` may be a guide style, the
    crescent, or another saved head (then it is flattened: the stored record rests on the guide style). The name is
    [A-Za-z][A-Za-z0-9_-]{0,39} and must not be a guide style's. Re-saving a name replaces it."""
    spec = request.get_json(force=True, silent=True) or {}
    name = str(spec.get("name", "")).strip()
    if not _HEAD_NAME.match(name): return jsonify(error="name: letters, digits, _ and -, 1 to 40 characters, starting with a letter"), 400
    if name == "crescent" or name in _head_styles().STYLES: return jsonify(error=f"'{name}' is a guide style's name"), 400
    try: h = _head_custom(spec)
    except (KeyError, ValueError) as ex: return jsonify(error=str(ex)[:200]), 400
    # store the edits against the guide style, flattened (a head saved from a saved head rests on the same guide style)
    HS = _head_styles(); from anatomy import head_crescent as HC
    base_out = HS.outline(h["base"]) if h["base"] in HS.STYLES else {}
    import shaped_head as SH
    o_edits = {k: v for k, v in h["outline"].items() if v != dict(SH.OUTLINE, **base_out).get(k)}
    base_face = _head_custom({"style": h["base"]})["face"]
    f_edits = {k: v for k, v in h["face"].items() if v != base_face.get(k)}
    f_edits.update({k: HC.FACE[k] for k in base_face if k not in h["face"]})   # an edit back to the crescent's own value is an edit too
    rec = dict(name=name, style=h["base"], outline=o_edits, face=f_edits, doc=str(spec.get("doc", ""))[:200], saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(HEADS, exist_ok=True); tmp = os.path.join(HEADS, name + ".json.part")
    json.dump(rec, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(HEADS, name + ".json"))
    return jsonify(saved=rec, id=h["id"])

@app.delete("/api/heads/<name>")
def api_heads_delete(name):
    if not _HEAD_NAME.match(name) or name not in _saved_heads(): return jsonify(error="no such saved head"), 404
    os.remove(os.path.join(HEADS, name + ".json")); return jsonify(deleted=name)

@app.get("/api/schema")
def api_schema():
    eye_types = [dict(key=k, name=schema.CHARACTERS[k]["name"], set=schema.CHARACTERS[k]["set"]) for k in EYE_TYPES]
    return jsonify(dict(version=schema.SCHEMA_VERSION, instrument=I.INSTRUMENT_VERSION, params=_params_meta(), cells=schema.CELLS,
                        vertical_spines=schema.VERTICAL_SPINES, presets=sorted(_presets()), default=schema.DEFAULT_PRESET,
                        eye_types=eye_types, characters=_characters_list(),
                        colormaps=["magma", "viridis", "twilight", "inferno", "cividis", "gray"]))

@app.post("/api/tps")
def api_tps():
    """Dorsal homologous landmarks for the current specimen as a TPS file (geomorph / tpsDig readable). No build needed."""
    import landmarks as LM
    P, _ = schema.coerce_report(request.get_json(force=True).get("P", {}), base=schema.table_defaults())
    h = schema.param_hash(P)
    return app.response_class(LM.tps_string([(h, P)]), mimetype="text/plain",
                              headers={"Content-Disposition": f'attachment; filename="trilobite_{h}.tps"'})

_PRESET_CHARS = None
@app.post("/api/classify")
def api_classify():
    """The morphological characters the current animal expresses (schema.characters), and the order whose character
    set it most resembles (max Jaccard overlap). Cheap — no geometry — so the page can call it live as you edit."""
    global _PRESET_CHARS
    P = request.get_json(force=True).get("P", {})
    ch = schema.characters(P); cur = set(ch)
    if _PRESET_CHARS is None:
        _PRESET_CHARS = {n: set(schema.characters(pp)) for n, pp in _presets().items()}
    best, best_score = None, 0.0
    for n, pc in _PRESET_CHARS.items():
        union = len(cur | pc) or 1; score = len(cur & pc) / union
        if score > best_score: best, best_score = n, score
    return jsonify(characters=[dict(key=k, name=schema.CHARACTERS[k]["name"]) for k in ch],
                   nearest=best if best_score > 0 else None, nearest_score=round(best_score, 2))

@app.get("/api/preset_chars")
def api_preset_chars():
    """For the tree page: the characters each preset's own parameters express (schema.characters), so an option
    button can tell a lineage whose model already shows the feature from one that only inherits it on the tree."""
    return jsonify({n: schema.characters(pp) for n, pp in _presets().items()})

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
        head = _head_style(body.get("head"))                # a head style (isopod_model/head_styles.py), or None = the crescent
        try: bspec = _body_spec(body.get("body"))           # a rows-model body (the body editor): a spec or a saved name
        except KeyError as ex: return jsonify(error=str(ex)), 404
        try: key = _iso_key(P, head, bspec); problems = _iso_problems(P, head, bspec)   # the isopod model has its own builder-source hash;
        except (KeyError, ValueError) as ex: return jsonify(error=f"body or head spec: {str(ex)[:200]}"), 400   # resolving validates (bounds, keys)
        if problems: return jsonify(error="cannot build: " + "; ".join(problems)[:400]), 400
        folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
        n_parts = _iso_n_parts(P, bspec)                    # head + the body's printed pieces + tail (fused rows count once)
        PROGRESS.update(done=0, total=n_parts)
        if os.path.exists(manifest):
            PROGRESS.update(done=n_parts); return send_file(manifest)
        with LOCK:
            if os.path.exists(manifest):
                PROGRESS.update(done=n_parts); return send_file(manifest)
            t0 = time.time(); os.makedirs(folder, exist_ok=True); bnotes = []
            _isopod_body().PROGRESS_HOOK = lambda done, total: PROGRESS.update(done=done, total=total)   # the build reports parts as they finish
            try:
                out, kin, checks = _build_isopod(P, folder, head, bspec)
            except Exception as ex:
                shutil.rmtree(folder, ignore_errors=True)               # never cache a failed build — a retry must rebuild, not serve the error
                PROGRESS.update(done=n_parts)
                return jsonify(key=key, parts=[], kinematics=_kinematics(P), joint="isopod",
                               build_notes=[("isopod", "build failed", str(ex)[:160])], build_seconds=round(time.time() - t0, 1)), 500
            man = dict(key=key, parts=out, kinematics=kin, print=checks, schema_notes=notes,
                       build_notes=bnotes, build_seconds=round(time.time() - t0, 1), schema=schema.SCHEMA_VERSION, joint="isopod", fill_mm=0,
                       head=head or "crescent")
            json.dump(P, open(os.path.join(folder, "params.json"), "w")); json.dump(man, open(manifest, "w")); _evict()
            PROGRESS.update(done=n_parts); return jsonify(man)
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
    return jsonify(_measure(P, notes, "site"))

def _measure(P, notes, source):
    """Run the instrument on P (one at a time), remember the reading for the sheet, and record the animal. The page
    and the agent interface (web/agent.py) both measure through here."""
    with LOCK:
        r, B = I.read(P)
    r = {k: v for k, v in r.items() if k not in ("scan_trace",)}
    r.update(kinematics=_kinematics(P), schema_notes=notes)
    MEASURED[schema.param_hash(P)] = json.loads(json.dumps(r, default=str))
    RECORDER.record(schema.coerce(P), MEASURED[schema.param_hash(P)], schema.SCHEMA_VERSION, I.INSTRUMENT_VERSION, schema.param_hash(P), source=source)   # one line per distinct animal, kept and sent on (web/recorder.py)
    return MEASURED[schema.param_hash(P)]

@app.post("/api/sheet")
def api_sheet():
    """The blueprint sheet (A3, blueprint.sheet) for the current parameters: the flat animal from the cached build, the
    last instrument reading for these parameters if there is one (with the animal enrolled to its stop superimposed)."""
    import trimesh, blueprint
    body = request.get_json(force=True)
    cmap = body.get("cmap", "none")
    if cmap not in blueprint.CMAPS and cmap != "none": cmap = "none"   # 'none' = black & white blueprint (default)
    if body.get("joint") == "isopod":                              # the isopod model has its own sheet (ball joints, crescent head, print-in-place)
        P, notes = schema.coerce_report(body.get("P", {}), base=schema.table_defaults())
        try: head = _head_style(body.get("head")); bspec = _body_spec(body.get("body")); key = _iso_key(P, head, bspec)
        except (KeyError, ValueError) as ex: return jsonify(error=f"body or head spec: {str(ex)[:200]}"), 400
        folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
        if not os.path.exists(manifest):
            with app.test_request_context(json={"P": P, "joint": "isopod", "head": head, "body": bspec}): api_build()
        if not os.path.exists(manifest): return jsonify(error="the build failed; nothing to draw"), 500   # a refused build leaves no manifest
        man = json.load(open(manifest)); png = os.path.join(folder, f"sheet_isopod_{cmap}.png")   # cache per colormap
        if os.path.exists(png): return jsonify(url=f"/files/{key}/sheet_isopod_{cmap}.png", measured=False)
        with LOCK:
            parts = [p for p in man.get("parts", []) if "error" not in p]
            if not parts: return jsonify(error="isopod build failed — no parts to draw"), 500
            meshes = [trimesh.load(os.path.join(folder, f"{p['name']}.stl")) for p in parts]
            IB = _isopod_body(); kin = man.get("kinematics", {})
            info = dict(names=[p["name"] for p in parts], pivots=kin.get("pivots", []), curl_deg=getattr(IB, "CURL", 30.0),
                        params=key, schema=man.get("schema", ""), build_seconds=man.get("build_seconds"),
                        volumes=[p.get("volume") for p in parts])
            blueprint.isopod_sheet(meshes, P, info, png, cmap=cmap)
        return jsonify(url=f"/files/{key}/sheet_isopod_{cmap}.png", measured=False)
    P, notes = schema.coerce_report(body.get("P", {}), base=schema.table_defaults())
    phash = schema.param_hash(P); key = phash + "-b" + BUILD_SIG   # the sheet draws the default (pin) build; key must match api_build's
    folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
    if not os.path.exists(manifest):
        with app.test_request_context(json={"P": P, "joint": "pin"}): api_build()   # the sheet draws the pin build; be explicit now the default is isopod
    man = json.load(open(manifest)); meas = MEASURED.get(phash)   # MEASURED is keyed by the plain param hash (set in /api/measure)
    tag = "measured" if meas else "flat"; png = os.path.join(folder, f"sheet_{tag}_{cmap}.png")   # cache per colormap
    if os.path.exists(png): return jsonify(url=f"/files/{key}/sheet_{tag}_{cmap}.png", measured=bool(meas))
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
        blueprint.sheet(flat, P, m, png, enrolled=enrolled, cmap=cmap)
    return jsonify(url=f"/files/{key}/sheet_{tag}_{cmap}.png", measured=bool(meas))

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

import agent as AGENT
app.register_blueprint(AGENT.make_blueprint(app, schema, I, _presets, _measure, LOCK))   # /llms.txt and /api/agent/*: the way in for programs

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
