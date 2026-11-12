"""
web/app.py — the trilobite site on the mesh builder (prompt 7, 11 Sep 2026). Flask, no OpenCascade.

    python web/app.py            # http://localhost:8765 (PORT env overrides)

Controls are the 3x3 CELLS contract from schema.py (three primary parameters per cell, more one tap deeper), plus the
prong pickers. The viewer loads one GLB per part; hovering a cell highlights its tagma row. "Curl it" runs the
instrument (instrument.read: build at the fixed measurement bevel, probe, measure — ~30 s) and returns the reading and
the hinge kinematics so the browser animates the animal to its stop with the limiting pair flashing.
"""
import os, sys, re, json, glob, time, threading, hashlib, shutil
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
    if isinstance(name, str) and name.endswith("-drawing"): _drawing_info(name[:-len("-drawing")])   # 10 Nov 2026: write a missing drawing mirror from its preset's block
    if name and name in _saved_heads(): return _head_custom({"style": name})
    if name == "crescent": return None                                    # the crescent on the isopod head piece, by name
    if not name: name = _isopod_body().DEFAULT_HEAD                          # 25 Oct 2026: no head given = the default head style ('diagram')
    return name if name in _head_styles().STYLES else None

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
    o_edits = {k: (v if k in ("trace", "dots") else float(v)) for k, v in (spec.get("outline") or {}).items()}   # 20 Oct 2026: "trace" is a point list (7 Nov: "dots" too)
    if spec.get("trace") is not None: o_edits["trace"] = spec.get("trace")                               # (also accepted at the top level)
    f_edits = {k: float(v) for k, v in (spec.get("face") or {}).items()}
    if style in _saved_heads():
        saved = _saved_heads()[style]; style = saved.get("style") or "crescent"
        o_edits = dict(saved.get("outline") or {}, **o_edits); f_edits = dict(saved.get("face") or {}, **f_edits)
    if o_edits.get("trace") in ([], None) and "trace" in o_edits: o_edits.pop("trace")                  # [] = "drop the trace"
    if o_edits.get("dots") in ([], None) and "dots" in o_edits: o_edits.pop("dots")                     # [] = "no dots"
    # the base face is what the BUILD would use: the default preset with the style's schema values (eyeSize, glabInflate ...)
    # mapped through body.face_of, then the style's own face keys on top -- so the drawing matches the solid
    IBODY = _isopod_body(); P0 = schema.coerce(dict(schema.preset(schema.DEFAULT_PRESET), **(HS.style(style)[1] if style in HS.STYLES else {})), base=schema.table_defaults())
    base_face = IBODY.face_of(P0, HS.style(style)[0] if style in HS.STYLES else None)
    base_out = HS.outline(style)                                  # "crescent" = the kit's silhouette (head_styles.OUTLINES); unknown = {}
    outline = SH.outline_of(dict(base_out, **o_edits))
    face = HC.face_of(dict(base_face, **f_edits))
    face_edits = {k: v for k, v in face.items() if v != HC.FACE.get(k) or v != base_face.get(k)}   # what differs from the crescent's own face -- and (9 Nov 2026) what
    # the page set back to the crescent's value over a style that differs (Terataspis' tubercles 0 on the lichas style came back as 0.45)
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

def _build_isopod(P, folder, head=None, body_spec=None, preview=False):
    """Build the isopod model at print mm: isopod body + ball joints + either the crescent on the isopod head piece
    (head None) or a head style drawn on its own silhouette (3 Oct 2026). The style's outline and sculpt keys are
    build options; P is used as given (the page applies a style's parameter values to P when it is picked, so the
    sliders show them and stay editable). Returns (parts, man-fields)."""
    IBODY = _isopod_body(); h = _head_style(head)
    rows = _body_spec(body_spec)
    if h is None: parts, plans, PV, K = IBODY.build(P, "isopod", rows=rows, preview=preview)
    elif isinstance(h, dict):                                   # the head editor's head: its own outline + face
        parts, plans, PV, K = IBODY.build(P, "shaped", face=h["face"], outline=h["outline"], rows=rows, preview=preview)
    else:
        HS = _head_styles(); face, _ = HS.style(h)
        parts, plans, PV, K = IBODY.build(P, "shaped", face=face, outline=HS.outline(h), rows=rows, preview=preview)
    s = K["s"]                                                  # model units -> print mm
    pieces = [p.copy().apply_scale(1.0 / s) for p in parts]
    ri = getattr(IBODY, "ROWS_INFO", {}) or {}
    names = ["head"] + [f"seg{i}" for i in range(len(pieces) - 2)] + ["tail"]
    pivots = [[float(y / s), float(z / s)] for (y, z) in PV]    # per-joint pivot (y, z) in print mm, for the viewer's pose
    out = []
    for name, m in zip(names, pieces):
        if not preview: m.export(os.path.join(folder, f"{name}.glb"))           # the viewer reads the STL; the GLB is for downloads
        m.export(os.path.join(folder, f"{name}.stl"))
        out.append(dict(name=name, url=f"/files/{os.path.basename(folder)}/{name}.glb", stl=f"/files/{os.path.basename(folder)}/{name}.stl",
                        bodies=len(__import__("mesh").bodies(m)), watertight=bool(m.is_watertight), volume=round(float(m.volume), 1)))
    import kinematics as KIN
    kit = float(K.get("kit", 1.0))                              # 10 Nov 2026: the joint kit as printed, x the isopod's (the sheet's ball and clearance)
    kin = dict(names=names, pivots=pivots, kit=round(kit, 4),   # pivots present => the viewer poses about per-joint pivots (ball joints)
               profile=[round(a / KIN.HEAD_DEG, 4) for a in KIN.profile(len(pivots))],   # each joint's share of the head angle (Esteve profile)
               profile_source=KIN.SOURCE)
    try:                                                        # 26 Oct 2026: where the builder put the eye (print mm), for the sheet's lens drawing
        Sh = plans[0][2]; Eh = Sh.get("eye") if isinstance(Sh, dict) else None
        if Eh and not Eh.get("blind") and float(P.get("eyeSolid", 0)) > 0.5 and float(P.get("eyeSize", 0)) > 0.01:
            kin["eye"] = [round(float(Eh["xe"]) / s, 3), round((float(Eh["ye"]) + float(plans[0][3])) / s, 3), round(float(Eh["eR"]) / s, 3)]
    except Exception: pass
    # the checks (11 Oct 2026): the build has already refused invalid parts (body.MeshError); here what a slicer and the
    # chain would still notice -- parts touching at rest, and the debris the clean-up removed
    if preview: return out, kin, dict(print_valid=False, preview=True, notes=["preview: the pieces as drawn, before the joint clearances -- the printable pieces are being finished"], touching=[], debris_mm3={})
    import check as CHECK
    touching = [dict(a=names[k], b=names[j], mm3=round(v / s ** 3, 3)) for k, j, v in CHECK.contacts(parts, [(y, z) for y, z in PV], 0.0)]
    dropped = {names[k]: round(v / s ** 3, 3) for k, v in getattr(IBODY, "LAST_DROPPED", {}).items() if v > 0}
    cut = {k: v for k, v in dropped.items() if v > 20.0}                  # more than debris: the clearances severed anatomy (parts overlapped)
    checks = dict(print_valid=all(p["watertight"] and p["bodies"] == 1 for p in out) and not touching and not cut,
                  notes=_size_notes(pieces, kit, ri) + [f"{t['a']} and {t['b']} touch at rest: {t['mm3']} mm3" for t in touching]
                        + [f"free {n}" for n in (getattr(IBODY, "ROWS_INFO", {}) or {}).get("free_notes", [])]
                        + [f"spine audit: {n}" for n in getattr(IBODY, "LAST_AUDIT", [])]
                        + [f"{names[k]}: {v / s ** 3:.1f} mm3 below the bed cut off" for k, v in getattr(IBODY, "LAST_BED_CUT", {}).items() if v / s ** 3 > 0.05]
                        + (lambda r: [] if not r[1] else [f"enrolment: the pieces collide before the tail tucks under the head -- {r[1][0][0]} hits {r[1][0][1]} ({r[1][0][2]} mm3); it enrols to {r[0]:.0f} of 180 deg"])(IBODY.enrol_check(parts, [(nm,) for nm in names], PV, s))
                        + [f"{k}: {v:.0f} mm3 cut away by the clearances (it overlapped the part ahead; a forward sweep or a long overlap?)" for k, v in cut.items()]
                        + ([f"debris dropped (mm3): {dropped}"] if dropped and not cut else []),
                  touching=touching, debris_mm3=dropped)
    return out, kin, checks
LOCK = threading.Lock()          # one build or measurement at a time (the lab workstation has one job's worth of RAM to spare)
MEASURED = {}                    # param hash -> last instrument result (so the sheet can carry the reading and the enrolled pose)
PROGRESS = {"done": 0, "total": 0}   # parts finished in the build now running; the page polls it for its loading count
                                     # (one build at a time, held by LOCK, so a single global is enough)
PORT = int(os.environ.get("PORT", 8765))   # the lab tunnel (trilomorph.org) points at 8765, the legacy port

# ---- prism points (31 Oct 2026): every distinct animal built is logged as its hash + the four Raup-prism coordinates
# (c=cephFrac, p=pygFrac, n=segCount, r=widthThoraxRear) so /prism accumulates a live cloud over every build anyone does.
# Just the five numbers per animal (~40 bytes), never the mesh -- the STLs would be gigabytes. Deduped by hash; the flat
# file survives restarts. Point it at a persistent volume with TRILO_PRISM_POINTS on the deploy box.
PRISM_POINTS = os.environ.get("TRILO_PRISM_POINTS") or os.path.join(ROOT, "data", "prism_points.csv")
_PRISM_SEEN = set(); _PRISM_LOCK = threading.Lock()
def _prism_init():
    try:
        with open(PRISM_POINTS, encoding="utf-8") as f:
            for line in f:
                h = line.split(",", 1)[0].strip()
                if h: _PRISM_SEEN.add(h)
    except OSError: pass
_prism_init()
def _record_point(P):
    """Log this animal's hash + Raup-prism coordinates once. Cheap, deduped, best-effort: a build is never failed
    because the point could not be written."""
    try:
        h = schema.param_hash(P)
        if h in _PRISM_SEEN: return
        with _PRISM_LOCK:
            if h in _PRISM_SEEN: return
            _PRISM_SEEN.add(h)
            os.makedirs(os.path.dirname(PRISM_POINTS) or ".", exist_ok=True)
            with open(PRISM_POINTS, "a", encoding="utf-8") as f:
                f.write(f"{h},{float(P['cephFrac']):.4f},{float(P['pygFrac']):.4f},{int(P['segCount'])},{float(P['widthThoraxRear']):.4f}\n")
    except Exception: pass

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

# ---------------------------------------------------------------- the site password (17 Oct 2026)
# TRILO_SITE_PASSWORD set => every request needs the signed cookie /login issues (or, on /api/agent/* and /llms.txt, the
# agent token). Unset => open, with a warning in the log. The cookie is signed with TRILO_SECRET_KEY (set it in .env so
# logins survive a restart; otherwise a key is drawn per process and everyone logs in again after a deploy).
import hmac, secrets
SITE_PASSWORD = os.environ.get("TRILO_SITE_PASSWORD", "").strip()
app.secret_key = os.environ.get("TRILO_SECRET_KEY") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30)
if not SITE_PASSWORD:
    sys.stderr.write("[trilobite] WARNING: TRILO_SITE_PASSWORD is not set: the site is open to anyone.\n"); sys.stderr.flush()
LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>trilobite morphospace</title>
<style>:root{--bg:#000;--ink:#fff;--dim:#9a9a9a;--line:#333}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Helvetica,Arial,sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center}
form{width:min(360px,calc(100vw - 32px));display:grid;gap:18px}
h1{font-size:.8rem;letter-spacing:.22em;text-transform:uppercase;font-weight:normal;margin:0;color:var(--dim)}
input{background:transparent;border:0;border-bottom:1px solid var(--ink);color:var(--ink);font:inherit;font-size:1rem;padding:8px 0;outline:none;width:100%}
button{background:var(--ink);color:var(--bg);border:0;font:inherit;font-size:.8rem;letter-spacing:.18em;text-transform:uppercase;padding:12px;cursor:pointer}
p{margin:0;font-size:.75rem;color:var(--dim);letter-spacing:.04em}</style></head>
<body><form method="post" action="/login"><h1>Trilobite morphospace</h1>
<input type="password" name="password" placeholder="password" autofocus autocomplete="current-password">
<input type="hidden" name="next" value="%NEXT%"><button type="submit">Enter</button>%MSG%</form></body></html>"""

def _logged_in():
    from flask import session
    return bool(SITE_PASSWORD) and session.get("site") == hashlib.sha256(SITE_PASSWORD.encode()).hexdigest()[:16]

@app.route("/login", methods=["GET", "POST"])
def login():
    from flask import session, redirect
    nxt = request.values.get("next") or "/"
    if not re.fullmatch(r"/[A-Za-z0-9_\-/.?=&%]*", nxt): nxt = "/"                      # same-site paths only
    if not SITE_PASSWORD: return redirect(nxt)
    if request.method == "POST":
        given = (request.form.get("password") or "").strip()
        if hmac.compare_digest(given.encode(), SITE_PASSWORD.encode()):
            session.permanent = True; session["site"] = hashlib.sha256(SITE_PASSWORD.encode()).hexdigest()[:16]
            return redirect(nxt)
        time.sleep(0.5)                                                                   # slow a guesser down
        return LOGIN_HTML.replace("%NEXT%", nxt).replace("%MSG%", "<p>wrong password</p>"), 401
    return LOGIN_HTML.replace("%NEXT%", nxt).replace("%MSG%", "")

@app.get("/logout")
def logout():
    from flask import session, redirect
    session.pop("site", None); return redirect("/login")

@app.before_request
def _password_gate():
    if not SITE_PASSWORD or _logged_in(): return None
    p = request.path
    if p in ("/login", "/logout", "/loading.gif") or p.startswith("/static/"): return None
    if p.startswith("/api/agent/") or p == "/llms.txt":                                   # programs: the agent token stands in for the cookie
        tok = os.environ.get("TRILO_AGENT_TOKEN", "").strip()
        if not tok:
            try: tok = open(os.environ.get("TRILO_AGENT_TOKEN_FILE") or os.path.join(ROOT, "web", "agent_token.txt"), encoding="utf-8").read().strip()
            except OSError: tok = ""
        if tok and (request.args.get("token") == tok or request.headers.get("Authorization") == "Bearer " + tok): return None
    if p.startswith("/api/") or p.startswith("/files/"):
        return jsonify(error="this site needs a password: open / in a browser and log in"), 401
    from flask import redirect
    return redirect("/login?next=" + __import__("urllib.parse").parse.quote(request.full_path.rstrip("?"), safe="/?=&"))

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
def index(): return send_file(os.path.join(ROOT, "web", "tree.html"))                   # front page = the ten order presets (the tree page's blueprint menu); the editor is at /studio

@app.get("/generator")
def generator(): return send_file(os.path.join(ROOT, "web", "index.html"))             # the parametric generator (blueprint / 3D / measure / params), now the build backend

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

@app.get("/prism")
def prism_page(): return send_file(os.path.join(ROOT, "web", "prism.html"))            # the Raup prism (30 Oct 2026): click a point, build it, see where it sits

@app.get("/api/prism/points")
def api_prism_points():
    """The live cloud (31 Oct 2026): every distinct animal built here, as [hash, c, p, n, r]. Read fresh from the file
    so a just-built animal shows up without a restart. Numbers only -- the meshes are never kept."""
    out = []
    try:
        with open(PRISM_POINTS, encoding="utf-8") as f:
            for line in f:
                q = line.rstrip("\n").split(",")
                if len(q) >= 5:
                    try: out.append([q[0], float(q[1]), float(q[2]), int(float(q[3])), float(q[4])])
                    except ValueError: pass
    except OSError: pass
    return jsonify(points=out)

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
        if body.endswith("-drawing"): _drawing_info(body[:-len("-drawing")])                 # 10 Nov 2026: as in _head_style
        d = _saved_bodies().get(body)
        if d is None: raise KeyError(f"no saved body '{body}'")
        return d.get("spec") or {}
    return dict(body)

def _iso_problems(P, head, spec):
    """Why this animal cannot be built: the rows' problems (the kit does not fit, the print too big) and the head's
    (its spines would make the print longer than the bed). Checked before a build is started (11 Oct 2026)."""
    _isopod_body(); import rows as RW
    from fields import seg_halfwidth
    out = []
    IB = _isopod_body()
    auto = float(P.get("printLength", 0.0) or 0.0) <= 0 and IB.HAND_SIZE and not (spec or {}).get("print_scale")
    if spec is not None or auto:
        if auto:                                                # 10 Nov 2026: the hand-sized print scales the joint kit with the animal, so
            R = RW.plans(P, spec or {}, strict=False)           # the kit fits wherever it fits at the isopod's pitch (kit 1); the size check
            s18 = RW.default_scale(R["rows"]); scale = s18 * IB.hand_ratio(1e9, IB.kit_fit18(R["rows"]))   # is at the smallest the build may go (the joint floor)
            out += RW.problems(R["rows"], scale, R["term_spine"], kit=IB.KIT_MIN); L_body = R["length"]
        else:
            R = RW.plans(P, spec, strict=False); out += R["problems"]; scale = R["scale"]; L_body = R["length"]
    else: scale = BALL_SCALE(P); L_body = P["length"] * (1 - P["cephFrac"])
    h = _head_style(head)
    if isinstance(h, dict):
        o = h["outline"]; L_ceph = o["length"] * o["width"] * seg_halfwidth(P, 0)      # the head's axial length
        L_tot = L_ceph + max(L_body, L_ceph * o["genal"])       # 9 Nov 2026: the genal spines run BESIDE the body, so the print is the head
        if L_tot * scale > RW.PRINT_MAX_MM: out.append(f"with this head the print would be {L_tot * scale:.0f} mm long; at most {RW.PRINT_MAX_MM:g}")   # plus the longer of body and spines, not all three (a harpetid's prolongations were counted on top of the thorax they lie along)
    return out

def _size_notes(pieces, kit, ri):
    """10 Nov 2026: the print's size and its joints, as notes on the manifest: the whole animal (printed in place, one
    piece on the bed), the kit's ball and clearance at this size, and whether the joint floor kept it above hand size."""
    import trimesh
    IB = _isopod_body(); W, L, H = trimesh.util.concatenate(pieces).extents
    out = [f"print {L:.0f} x {W:.0f} x {H:.0f} mm, one print-in-place piece; joints {kit:.2f} x the isopod's (ball {IB.ISO_BALL_D * kit:.1f} mm, clearance {IB.ISO_CLEAR * kit:.2f} mm)"]
    if ri.get("kit_floor") and max(W, L) > IB.HAND_MM * (1 + IB.SIZE_TOL): out.append(f"held at the smallest joints ({IB.KIT_MIN:.2f} x the isopod's): at {IB.HAND_MM:.0f} mm they would be smaller")
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
    key = dict(rows=R["rows"], groups=R["groups"], scale=R["scale"], ts=R["term_spine"], model=R["model"])
    for k in ("free_spines", "margin_spines", "domes", "textures", "grooves"):                  # 9 Nov 2026: the features drawn on the body are part of the animal too;
        if R.get(k): key[k] = R[k]                                      # added only when present, so every existing key (and its cached build) stays
    if (spec or {}).get("tail_spines") is not None: key["tail_spines"] = spec["tail_spines"]
    blob = json.dumps(key, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]

@app.route("/api/body_diagram", methods=["GET", "POST"])
def api_body_diagram():
    """The body editor's drawing. GET ?preset=<order>[&body=<saved>]; POST {"preset", "spec": {...}, "head": <style|saved>}.
    Answers the rows diagram (isopod_model/rows.diagram: outline per row, pieces after fusion, joints, handles,
    the axis profile, the print size) and the head's diagram scaled to the first row, for the page to draw."""
    _isopod_body(); import rows as RW; import shaped_head as SH
    q = request.get_json(force=True, silent=True) or {} if request.method == "POST" else dict(request.args)
    preset = q.get("preset") or schema.DEFAULT_PRESET
    P = _studio_params(preset)                                                   # 8 Nov 2026: with the drawing block's params, when the preset has one
    if P is None: return jsonify(error="no such preset"), 404
    if request.method == "POST" and isinstance(q.get("P"), dict):             # 24 Oct 2026: the studio's edited P (spines, tail), so the drawing shows what Build will build
        try: P = schema.coerce(q["P"], base=P)
        except (KeyError, ValueError, TypeError) as ex: return jsonify(error=f"P: {str(ex)[:160]}"), 400
    try:
        spec = _body_spec(q.get("spec") if request.method == "POST" else q.get("body")) or {}
        if request.method == "GET" and q.get("start") == "boring": spec = {"list": RW.boring(P)}    # the editor's plain starting animal
        D = RW.diagram(P, spec)
        h = _head_custom({"style": q.get("head") or "crescent"}); hd = _diagram_json(SH.diagram(h["outline"], h["face"]))
    except (KeyError, ValueError) as ex:
        return jsonify(error=str(ex)[:200]), 400
    return jsonify(preset=preset, spec=spec, diagram=D, head=dict(style=h["style"], diagram=hd), id=_body_id(P, spec), drawing=_drawing_info(preset),
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
    base = _studio_params(q.get("preset")) if q.get("preset") else None
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

def _save_body(q):
    """The body editor's save, shared with the drawing block (8 Nov 2026): {"name", "preset", "spec", "head"?, "doc"?} ->
    the stored record. Raises ValueError (a bad name or spec: 400) or KeyError (no such preset or head: 404)."""
    _isopod_body(); import rows as RW
    name = str(q.get("name", "")).strip()
    if not _HEAD_NAME.match(name): raise ValueError("name: letters, digits, _ and -, 1 to 40 characters, starting with a letter")
    if name in _presets() or name == "preset": raise ValueError(f"'{name}' is a preset's name")
    preset = q.get("preset") or schema.DEFAULT_PRESET
    if preset not in _presets(): raise KeyError("no such preset")
    try: spec = dict(q.get("spec") or {}); RW.plans(_studio_params(preset), spec, strict=False)   # work in progress may be saved; the build refuses it. 9 Nov 2026: validated against the preset AS THE STUDIO HAS IT (its drawing block's segCount etc.), as the body is built
    except (KeyError, ValueError) as ex: raise ValueError(str(ex)[:200])
    head = str(q.get("head") or "crescent")                                                # the head it was drawn with (a style or a saved head)
    if head != "crescent" and _head_style(head) is None: raise KeyError(f"no such head '{head}'")
    rec = dict(name=name, preset=preset, head=head, spec=spec, doc=str(q.get("doc", ""))[:200], saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(BODIES, exist_ok=True); tmp = os.path.join(BODIES, name + ".json.part")
    json.dump(rec, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(BODIES, name + ".json"))
    return rec

@app.post("/api/bodies")
def api_bodies_save():
    """Save a body: {"name", "preset", "spec", "doc"?}. Name rules as for heads. Re-saving replaces."""
    q = request.get_json(force=True, silent=True) or {}
    try: rec = _save_body(q)
    except ValueError as ex: return jsonify(error=str(ex)[:200]), 400
    except KeyError as ex: return jsonify(error=_msg(ex)), 404
    return jsonify(saved=rec)

def _msg(ex):
    """An exception's message (a KeyError's str() wraps it in quotes)."""
    return str(ex.args[0] if ex.args else ex)[:200]

@app.delete("/api/bodies/<name>")
def api_bodies_delete(name):
    if not _HEAD_NAME.match(name) or name not in _saved_bodies(): return jsonify(error="no such saved body"), 404
    os.remove(os.path.join(BODIES, name + ".json")); return jsonify(deleted=name)

# ---- the drawing block (8 Nov 2026): a preset may carry the animal matched to a reference drawing on the isopod model,
# presets/<order>.json["drawing"] = {"doc", "schema", "model": "isopod", "style": a head style, "params": {schema key:
# value} EDITS over the preset's params, "outline": edits over the style's silhouette, "face": edits over the style's
# face keys, "rows": the body spec's rows, "head"/"body": the saved head and body the block is mirrored to, "page": the
# page that edits it, "saved_at"}. The preset's own params are never touched: they are the instrument's frozen
# reference. The block is what the order's page (/agnostid for agnostida) loads, builds and saves; the mirrored saved
# head/body let the studio open the same head and rows (its parameters stay the preset's: the drawing's live here).
PRESETS_DIR = os.path.join(ROOT, "presets")
DRAWING_PAGES = {"agnostida": "agnostid.html"}                                            # order -> web/<page>

def _preset_record(name):
    f = os.path.join(PRESETS_DIR, f"{name}.json")
    if not _HEAD_NAME.match(name or "") or not os.path.exists(f): raise KeyError("no such preset")
    return f, json.load(open(f))

def _studio_params(name):
    """The preset as the STUDIO and the generator page see it (8 Nov 2026): its params with the drawing block's
    parameter edits laid over them when it has one -- so picking 'agnostida' gives the drawn agnostid. The
    instrument (sweep.py, the library, isopod_model/presets.py) keeps reading presets/*.json["params"] untouched."""
    Pp = _presets().get(name)
    if Pp is None: return None
    try: D = _preset_record(name)[1].get("drawing") or {}
    except KeyError: D = {}
    return schema.coerce(dict(Pp, **(D.get("params") or {})), base=schema.table_defaults())

_MIRRORING = set()

def _drawing_mirror(name, D):
    """Write the drawing block's saved head and body <name>-drawing (web/heads, web/bodies) from the block itself.
    10 Nov 2026: those folders are git-ignored, docker-ignored and bind-mounted on the live box, so a pushed preset
    arrives WITHOUT its mirror and the studio fell back to the plain generator animal. The block is the source of truth:
    rebuild the mirror whenever it is missing or older than the block. Same code path as the drawing save."""
    R = _drawing_resolve(name, D)
    doc = f"the {name} preset's drawing block ({str(D.get('saved_at', ''))[:10]}); its parameters live on {D.get('page') or '/' + name}"
    _save_head(dict(name=D["head"], style=R["style"], outline={k: v for k, v in R["outline"].items() if k != "trace" or v}, face=R["head"]["face"], doc=doc))
    _save_body(dict(name=D["body"], preset=name, head=D["head"], spec=R["body"] or {}, doc=doc))

def _drawing_info(name):
    """{"head", "body"}: the saved head and body a preset's drawing block is mirrored to (written from the block if missing)."""
    try: D = _preset_record(name)[1].get("drawing") or {}
    except KeyError: return None
    if not (D.get("head") and D.get("body")): return None
    h, b = _saved_heads().get(D["head"]), _saved_bodies().get(D["body"])
    stamp = str(D.get("saved_at") or "")
    if name not in _MIRRORING and (h is None or b is None or str(h.get("saved_at") or "") < stamp or str(b.get("saved_at") or "") < stamp):
        _MIRRORING.add(name)                                  # _save_body looks the head up again: no re-entry
        try: _drawing_mirror(name, D)
        except (KeyError, ValueError): return None
        finally: _MIRRORING.discard(name)
    if D["head"] in _saved_heads() and D["body"] in _saved_bodies(): return dict(head=D["head"], body=D["body"], saved_at=D.get("saved_at"))
    return None

def _drawing_resolve(name, D):
    """The block's edits resolved: the full P (preset + params), the full outline (the style's + edits), the page's face
    (the style's face keys + edits), the rows, and the head spec the build takes -- its face completed from P through
    body.face_of, so the drawing's glabInflate / eyeSize win over the style's own values (which _head_custom would
    otherwise lay over the page's parameters)."""
    HS = _head_styles(); import shaped_head as SH; IBODY = _isopod_body()
    style = D.get("style") or IBODY.DEFAULT_HEAD
    if style not in HS.STYLES: raise KeyError(f"no such head style '{style}'")
    P = schema.coerce(dict(_presets()[name], **(D.get("params") or {})), base=schema.table_defaults())
    outline = SH.outline_of(dict(dict(SH.OUTLINE, **HS.outline(style)), **{k: (v if k in ("trace", "dots") else float(v)) for k, v in (D.get("outline") or {}).items()}))
    face = dict(HS.style(style)[0], **{k: float(v) for k, v in (D.get("face") or {}).items()})
    from anatomy import head_crescent as HC
    head = dict(style=style, outline=outline, face=HC.face_of(IBODY.face_of(P, face)))   # face_of validates the keys
    rows = dict(D.get("rows") or {})
    # 9 Nov 2026: the block's rows may also carry the body's own features (domes, spike textures, tail lobes, free spines --
    # any rows.BODY_KEYS key); the per-row edits are the digit keys (or a nested "rows")
    per = {k: v for k, v in rows.items() if str(k).isdigit()}; extra = {k: v for k, v in rows.items() if not str(k).isdigit()}
    if isinstance(extra.get("rows"), dict): per.update(extra.pop("rows"))
    body = (dict(extra, **({"rows": per} if per else {}))) if rows else None
    return dict(P=P, style=style, outline=outline, face=face, rows=per, head=head, body=body)

def _drawing_block(name, q, old):
    """The page's state {"P", "outline", "face", "rows", "style"?, "doc"?} -> the stored block (edits only)."""
    HS = _head_styles(); import shaped_head as SH
    style = q.get("style") or old.get("style") or _isopod_body().DEFAULT_HEAD
    if style not in HS.STYLES: raise KeyError(f"no such head style '{style}'")
    base_P = schema.coerce(_presets()[name], base=schema.table_defaults())
    P = schema.coerce(dict(base_P, **(q.get("P") or {})), base=schema.table_defaults())
    base_out = dict(SH.OUTLINE, **HS.outline(style)); base_face = HS.style(style)[0]
    D = dict(old)
    D.update(doc=str(q.get("doc", old.get("doc", "")))[:300], schema=schema.SCHEMA_VERSION, model="isopod", style=style,
             params={k: v for k, v in P.items() if v != base_P.get(k)},
             outline={k: v for k, v in (q.get("outline") or old.get("outline") or {}).items() if v != base_out.get(k)},
             face={k: v for k, v in (q.get("face") or old.get("face") or {}).items() if v != base_face.get(k)},
             rows=(q.get("rows") if q.get("rows") is not None else old.get("rows") or {}),
             head=f"{name}-drawing", body=f"{name}-drawing", page=old.get("page") or f"/{name}", saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _drawing_resolve(name, D)                                                               # validates every key and value
    return D

@app.get("/api/drawing/<name>")
def api_drawing(name):
    """The preset's drawing block, resolved for the page: {"preset", "drawing": the block, "full": {"P", "outline",
    "face", "rows"}, "head": the build's head spec, "body": the build's body spec}."""
    try: f, rec = _preset_record(name)
    except KeyError: return jsonify(error="no such preset"), 404
    D = rec.get("drawing")
    if not D: return jsonify(error=f"the {name} preset has no drawing block"), 404
    try: R = _drawing_resolve(name, D)
    except (KeyError, ValueError) as ex: return jsonify(error=_msg(ex)), 400
    return jsonify(preset=name, drawing=D, full=dict(P=R["P"], outline=R["outline"], face=R["face"], rows=dict(rows=R["rows"])), head=R["head"], body=R["body"])

@app.post("/api/drawing/<name>")
def api_drawing_save(name):
    """Save the page's state as the preset's drawing block: {"P": the parameters (full or partial), "outline", "face",
    "rows", "style"?, "doc"?}. Stored as edits (see above); the preset's params are left as they are. Then mirrored to
    the saved head and body <preset>-drawing (web/heads, web/bodies) so the studio can open them (?body=<name>-drawing)."""
    q = request.get_json(force=True, silent=True) or {}
    try: f, rec = _preset_record(name)
    except KeyError: return jsonify(error="no such preset"), 404
    try:
        D = _drawing_block(name, q, rec.get("drawing") or {}); R = _drawing_resolve(name, D)
        _iso_key(R["P"], R["head"], R["body"])                                              # the build's own validation (rows, head)
    except (KeyError, ValueError) as ex: return jsonify(error=_msg(ex)), 400
    rec["drawing"] = D
    with LOCK:
        tmp = f + ".part"; json.dump(rec, open(tmp, "w"), indent=1, sort_keys=True); os.replace(tmp, f)
    mirrored = {}
    try:
        doc = f"the {name} preset's drawing block ({D['saved_at'][:10]}); its parameters live on {D['page']}"
        hrec, _ = _save_head(dict(name=D["head"], style=R["style"], outline={k: v for k, v in R["outline"].items() if k != "trace" or v}, face=R["head"]["face"], doc=doc))
        brec = _save_body(dict(name=D["body"], preset=name, head=D["head"], spec=R["body"] or {}, doc=doc))
        mirrored = dict(head=hrec["name"], body=brec["name"])
    except (KeyError, ValueError) as ex:
        mirrored = dict(error=_msg(ex))                                                     # the block is saved either way
    return jsonify(saved=D, full=dict(P=R["P"], outline=R["outline"], face=R["face"], rows=dict(rows=R["rows"])), mirrored=mirrored)

def _drawing_request(name, q):
    """The page's state (or the saved block where the page sent nothing) as /api/build's json."""
    f, rec = _preset_record(name); D = dict(rec.get("drawing") or {})
    if q.get("P") is not None: D["params"] = q["P"]
    for k in ("outline", "face", "rows", "style"):
        if q.get(k) is not None: D[k] = q[k]
    R = _drawing_resolve(name, D)
    return dict(P=R["P"], joint="isopod", head=R["head"], body=R["body"])

@app.post("/api/drawing/<name>/build")
def api_drawing_build(name):
    """Build the page's state through /api/build (the head spec completed here): {"P", "outline", "face", "rows",
    "preview"?, "peek"?} -> the build manifest (parts with .stl links, print checks, key for /api/stl/<key>)."""
    q = request.get_json(force=True, silent=True) or {}
    try: req = _drawing_request(name, q)
    except KeyError as ex: return jsonify(error=_msg(ex)), 404
    except ValueError as ex: return jsonify(error=_msg(ex)), 400
    req.update(preview=bool(q.get("preview")), peek=bool(q.get("peek")))
    with app.test_request_context(json=req): return api_build()

@app.post("/api/drawing/<name>/sheet")
def api_drawing_sheet(name):
    """The blueprint sheet of the page's state (built first if need be): {"P", "outline", "face", "rows", "cmap"?} -> {"url"}."""
    q = request.get_json(force=True, silent=True) or {}
    try: req = _drawing_request(name, q)
    except KeyError as ex: return jsonify(error=_msg(ex)), 404
    except ValueError as ex: return jsonify(error=_msg(ex)), 400
    req.update(cmap=q.get("cmap", "none"))
    with app.test_request_context(json=req): return api_sheet()

@app.get("/agnostid")
def agnostid_page(): return send_file(os.path.join(ROOT, "web", DRAWING_PAGES["agnostida"]))   # the agnostid's own plan editor (8 Nov 2026)

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
    return jsonify([dict(name=n, doc=s.get("doc", ""), p=s.get("p", 0), params=s.get("params", {})) for n, s in HS.STYLES.items() if n == "plain"] +
                   [dict(name="crescent", doc="the print kit's crescent, on the isopod head piece (the default until 20 Oct 2026)", p=0, params={})] +
                   [dict(name=n, doc=s.get("doc", ""), p=s.get("p", 0), params=s.get("params", {})) for n, s in HS.STYLES.items() if n != "plain"] + saved)

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
    try: rec, hid = _save_head(spec)
    except (KeyError, ValueError) as ex: return jsonify(error=_msg(ex)), 400
    return jsonify(saved=rec, id=hid)

def _save_head(spec):
    """The head editor's save, shared with the drawing block (8 Nov 2026): the record stored and the resolved head's id.
    Raises ValueError (name) or KeyError / ValueError (the spec, from _head_custom)."""
    name = str(spec.get("name", "")).strip()
    if not _HEAD_NAME.match(name): raise ValueError("name: letters, digits, _ and -, 1 to 40 characters, starting with a letter")
    if name == "crescent" or name in _head_styles().STYLES: raise ValueError(f"'{name}' is a guide style's name")
    h = _head_custom(spec)
    # store the edits against the guide style, flattened (a head saved from a saved head rests on the same guide style)
    HS = _head_styles(); from anatomy import head_crescent as HC
    base_out = HS.outline(h["base"])
    import shaped_head as SH
    o_edits = {k: v for k, v in h["outline"].items() if v != dict(SH.OUTLINE, **base_out).get(k)}   # a trace survives: no base has one
    base_face = _head_custom({"style": h["base"]})["face"]
    f_edits = {k: v for k, v in h["face"].items() if v != base_face.get(k)}
    f_edits.update({k: HC.FACE[k] for k in base_face if k not in h["face"]})   # an edit back to the crescent's own value is an edit too
    rec = dict(name=name, style=h["base"], outline=o_edits, face=f_edits, doc=str(spec.get("doc", ""))[:200], saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(HEADS, exist_ok=True); tmp = os.path.join(HEADS, name + ".json.part")
    json.dump(rec, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(HEADS, name + ".json"))
    return rec, h["id"]

@app.post("/api/heads/import")
def api_heads_import():
    """20 Oct 2026: a TriloMorph-protocol landmark file (tpsDig .tps or StereoMorph .txt, Serra et al. 2023: 16 cephalic landmarks
    and the anterior / posterior margin curves) -> a head spec for the editor {style, outline (with trace + spine), face, doc}.
    The file comes as the request body (text) or as the form field "file"; ?name= gives the head a name without saving it.
    Nothing is stored: the page loads the spec and the user saves it like any other head."""
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("trilomorph_import", os.path.join(ROOT, "scripts", "trilomorph_import.py"))
    TI = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(TI)
    f = request.files.get("file"); raw = f.read() if f else request.get_data()
    if not raw or len(raw) > 4_000_000: return jsonify(error="a .tps or StereoMorph .txt file, under 4 MB"), 400
    import tempfile
    text = raw.decode("utf-8", errors="ignore"); kind = "tps" if text.lstrip().upper().startswith("LM=") else "txt"
    with tempfile.NamedTemporaryFile("w", suffix="." + kind, delete=False, encoding="utf-8") as t: t.write(text); path = t.name
    try:
        rec = TI.read_tps(path) if kind == "tps" else TI.read_txt(path)
        if not rec or not rec.get("lm"): return jsonify(error="no landmarks found: the file must follow the TriloMorph protocol (LM1-LM16 and the margin curves)"), 400
        h = TI.head_from(rec)
        import shaped_head as SH; SH.outline_of(h["outline"])
    except (ValueError, KeyError) as ex:
        return jsonify(error=f"could not read a head from this file: {str(ex)[:160]}"), 400
    finally:
        try: os.remove(path)
        except OSError: pass
    src = (f.filename if f else request.args.get("name") or "landmarks")
    return jsonify(style="plain", outline=h["outline"], face=h["face"], blind=h["blind"], landmarks=len(rec["lm"]), curves=sorted(rec["curves"]),
                   doc=f"landmarked in tpsDig / StereoMorph with the TriloMorph protocol (Serra et al. 2023); file {os.path.basename(src)[:60]}")

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
    """The TriloMorph landmark protocol (Serra et al. 2023) for the current specimen as a TPS file, read off the head and
    tail plans (landmarks.py). ?part=C (cephalon, default) or P (pygidium); their shapRead() wants one part per file."""
    import landmarks as LM
    body = request.get_json(force=True); part = "P" if str(body.get("part", request.args.get("part", "C"))).upper().startswith("P") else "C"
    P, _ = schema.coerce_report(body.get("P", {}), base=schema.table_defaults())
    h = schema.param_hash(P)
    return app.response_class(LM.tps_string([(h, P)], part), mimetype="text/plain",
                              headers={"Content-Disposition": f'attachment; filename="trilobite_{h}_{part}.tps"'})

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
    P = _presets().get(name) if request.args.get("raw") else _studio_params(name)   # ?raw=1: the file's params alone (the instrument's)
    if P is None: return jsonify(error="no such preset"), 404
    return jsonify(schema.coerce(P, base=schema.table_defaults()))

def _kinematics(P):
    return dict(hinge_z=PIN.hinge_z(P), offsets=PIN.joint_offsets(P), names=I.part_names(P))

import contextvars
_SYNC_BUILD = contextvars.ContextVar("trilo_sync_build", default=False)   # set while the server builds for itself (the sheet)
KEEPALIVE_S = 5.0

def _keepalive(work):
    """10 Nov 2026: a long build answered without a timeout. The site's tunnel drops any request whose answer has not
    STARTED within ~100 s (it then sends the browser its own HTML error page, which the studio could not read as JSON),
    and a hand-sized build of a spiny animal can take longer. The work runs in a thread; the answer starts at once and
    sends a space every KEEPALIVE_S seconds until the build is done, then the manifest. Leading spaces are valid JSON,
    so the pages read it exactly as before. A build that fails says so in the JSON ("error", build_notes; no parts) --
    the status line is already sent by then. If the browser goes away the build still finishes and is cached."""
    import threading
    res = {}
    def run():
        try: res["r"] = work()
        except Exception as ex: res["r"] = (500, json.dumps(dict(parts=[], error=f"build failed: {str(ex)[:200]}", build_notes=[("isopod", "build failed", str(ex)[:200])])))
    th = threading.Thread(target=run, daemon=True); th.start()
    def gen():
        yield " "
        while True:
            th.join(KEEPALIVE_S)
            if not th.is_alive(): break
            yield " "
        yield res["r"][1]
    r = app.response_class(gen(), mimetype="application/json")
    r.headers["X-Accel-Buffering"] = "no"; r.headers["Cache-Control"] = "no-store"
    return r

@app.post("/api/build")
def api_build():
    """Print geometry: every part at the printed stop bevel (P['maxAngle']), one GLB each, cached by parameter hash."""
    body = request.get_json(force=True)
    P, notes = schema.coerce_report(body.get("P", {}), base=schema.table_defaults())
    _record_point(P)                                       # log this animal's hash + prism coords (31 Oct 2026); deduped, no mesh
    joint = body.get("joint", "isopod")                    # "isopod" = the default (isopod model); "pin" = measured; "flexi"/"ball" = print joints
    fill = float(body.get("fill", 0.0) or 0.0)              # pin builds only: thicken the shell for printing (mm), print-only
    if joint == "isopod":
        head = _head_style(body.get("head"))                # a head style (isopod_model/head_styles.py), or None = the crescent
        try: bspec = _body_spec(body.get("body"))           # a rows-model body (the body editor): a spec or a saved name
        except KeyError as ex: return jsonify(error=str(ex)), 404
        preview = bool(body.get("preview"))                 # 25 Oct 2026: the pieces as drawn, no joint clearances -- the viewer's first look, in seconds
        try: key = _iso_key(P, head, bspec) + ("-preview" if preview else ""); problems = _iso_problems(P, head, bspec)   # the isopod model has its own builder-source hash;
        except (KeyError, ValueError) as ex: return jsonify(error=f"body or head spec: {str(ex)[:200]}"), 400   # resolving validates (bounds, keys)
        if problems: return jsonify(error="cannot build: " + "; ".join(problems)[:400]), 400
        folder = os.path.join(CACHE, key); manifest = os.path.join(folder, "manifest.json")
        n_parts = _iso_n_parts(P, bspec)                    # head + the body's printed pieces + tail (fused rows count once)
        PROGRESS.update(done=0, total=n_parts)
        if os.path.exists(manifest):
            PROGRESS.update(done=n_parts); return send_file(manifest)
        if body.get("peek"): return jsonify(pending=True)          # 25 Oct 2026: the viewer asks first; a cached build is shown at once, no preview
        def work():
            """The build itself -> (HTTP status, the manifest's JSON text)."""
            with LOCK:
                if os.path.exists(manifest):
                    PROGRESS.update(done=n_parts); return 200, open(manifest, encoding="utf-8").read()
                t0 = time.time(); os.makedirs(folder, exist_ok=True); bnotes = []
                _isopod_body().PROGRESS_HOOK = lambda done, total: PROGRESS.update(done=done, total=total)   # the build reports parts as they finish
                try:
                    out, kin, checks = _build_isopod(P, folder, head, bspec, preview=preview)
                except Exception as ex:
                    shutil.rmtree(folder, ignore_errors=True)               # never cache a failed build — a retry must rebuild, not serve the error
                    PROGRESS.update(done=n_parts)
                    return 500, json.dumps(dict(key=key, parts=[], kinematics=_kinematics(P), joint="isopod", error=f"build failed: {str(ex)[:160]}",
                                                build_notes=[("isopod", "build failed", str(ex)[:160])], build_seconds=round(time.time() - t0, 1)), default=str)
                man = dict(key=key, parts=out, kinematics=kin, print=checks, schema_notes=notes,
                           build_notes=bnotes, build_seconds=round(time.time() - t0, 1), schema=schema.SCHEMA_VERSION, joint="isopod", fill_mm=0,
                           head=head or "crescent", body=bspec)
                json.dump(P, open(os.path.join(folder, "params.json"), "w")); json.dump(man, open(manifest, "w")); _evict()
                PROGRESS.update(done=n_parts); return 200, json.dumps(man, default=str)
        if _SYNC_BUILD.get():                                     # called from inside the server (the sheet): wait for it here
            st, txt = work(); return app.response_class(txt, status=st, mimetype="application/json")
        return _keepalive(work)                                    # 10 Nov 2026: the answer starts at once and the build finishes behind it
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
    _record_point(P)                                       # the measured animal is a prism point too (31 Oct 2026)
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
            tok = _SYNC_BUILD.set(True)
            try:
                with app.test_request_context(json={"P": P, "joint": "isopod", "head": head, "body": bspec}): api_build()
            finally: _SYNC_BUILD.reset(tok)
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
                        volumes=[p.get("volume") for p in parts], meta=[dict(bodies=p.get("bodies"), watertight=p.get("watertight")) for p in parts],
                        checks=man.get("print", {}), head=man.get("head"), body=man.get("body"),   # the editors' specs, as the manifest records them
                        eye=kin.get("eye"),                                                     # the eye as built (print mm), for the lens drawing
                        kit=kin.get("kit"),                                                     # 10 Nov 2026: the joint kit as printed (x the isopod's)
                        date=time.strftime("%Y-%m-%d"))
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
    if not _CACHE_KEY.fullmatch(key) or not os.path.exists(os.path.join(CACHE, key, "manifest.json")): return jsonify(error="no such build"), 404
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

_CACHE_KEY = re.compile(r"[0-9a-f]{10}(-[A-Za-z0-9.]+)*")
@app.get("/files/<key>/<path:name>")
def files(key, name):
    if not _CACHE_KEY.fullmatch(key) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name): return jsonify(error="no such file"), 404
    return send_from_directory(os.path.join(CACHE, key), name)

def _evict(max_dirs=60):
    dirs = sorted(glob.glob(os.path.join(CACHE, "*")), key=os.path.getmtime)
    for d in dirs[:-max_dirs]: shutil.rmtree(d, ignore_errors=True)

import agent as AGENT
app.register_blueprint(AGENT.make_blueprint(app, schema, I, _presets, _measure, LOCK))   # /llms.txt and /api/agent/*: the way in for programs

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
