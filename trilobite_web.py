"""
trilobite_web.py — the generator as a website (v5: sheet-first, poll-based build pipeline).

    python trilobite_web.py        →  http://localhost:8765     (Render/Railway set PORT)

Design: a build runs in a background thread (some parameter combinations take minutes) and moves
through building -> measuring -> done/error. The client polls a single /api/status/<key> endpoint;
there is no per-part streaming into a live 3D scene. Once done, the server has already rendered the
blueprint technical sheet (the default result view) and a combined flat STL (the download) as the
last two steps of the pipeline, so both are instantly available the moment status flips to done. The
interactive 3D viewer is an opt-in, client-lazy-loaded secondary view that loads every part once, all
at once, only after the build is already finished.
"""
import json, os, time, threading, multiprocessing, io, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import schema, fields, instrument
import trilobite as T

PORT = int(os.environ.get("PORT", 8765))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web_out"); os.makedirs(OUT, exist_ok=True)

PRESETS = dict(schema.PRESETS)                                    # the four fitted presets first…
_pdir = os.path.join(HERE, "presets")
if os.path.isdir(_pdir):                                          # …then the measured species in presets/*.json
    for f in sorted(os.listdir(_pdir)):
        if f.endswith(".json"):
            try:
                d = json.load(open(os.path.join(_pdir, f)))
                PRESETS.setdefault(f[:-5], d.get("params", d))
            except Exception: pass
KNOBS = [(p.key, p.label, p.lo, p.hi, p.step, p.group, p.kind, p.doc) for p in schema.PARAMS]
INT_KEYS = [p.key for p in schema.PARAMS if p.kind in ("int", "odd_int")]

def derived(P):
    v = instrument.print_validity(P)
    joints = P["segCount"] + 1; total = joints * P["maxAngle"]
    ratio = P["pygFrac"] / P["cephFrac"]
    v.update(joints=joints, total_curl=round(total), pyg_class="micropygous" if ratio < 0.6 else ("isopygous" if ratio < 1.15 else "macropygous"),
             head_mm=round(P["cephFrac"] * P["length"]), thorax_mm=round(v["pitch"] * P["segCount"]), tail_mm=round(P["pygFrac"] * P["length"]),
             last_width=round(2 * fields.seg_halfwidth(P, P["segCount"] - 1)), warnings=v["violations"])
    return v

# RLock (not Lock) as a defensive backstop: nothing below is written to call a LOCK-acquiring function
# while already holding LOCK (that was a real self-deadlock bug in the previous design — a thread
# re-acquiring a lock it already holds blocks forever), but an RLock makes that class of mistake
# harmless even if a future edit reintroduces it by accident.
CACHE, LOCK = {}, threading.RLock()
# trilobite.py's retry-on-garbage machinery (_build_checked, GRID_JITTER, BUILD_NOTES) is module-global,
# not thread-safe: two builds running their CAD/CPU work at once would race on that state. Since a single
# OpenCascade-heavy build already saturates one core, there is nothing to gain from true concurrency here —
# serialize the CPU-heavy construction portion of every build behind one lock so only one build ever runs
# it at a time. Measuring (below) is deliberately kept outside this lock so a second build's construction
# can start as soon as this one's parts are done, instead of waiting out a multi-minute measuring pass.
BUILD_CPU_LOCK = threading.Lock()

# instrument.measure()'s own time budget (instrument.MEASURE_BUDGET_S) only checks the clock between
# pairwise checks, so it cannot interrupt a single already-in-flight OpenCascade/trimesh call. Running it
# in a subprocess lets the parent hard-kill it — regardless of what C call it's stuck inside — if it's
# still running after MEASURE_HARD_TIMEOUT_S.
MEASURE_HARD_TIMEOUT_S = instrument.MEASURE_BUDGET_S + 90

def _measure_hard_bounded(P, folder, names_all, parts):
    """Run instrument.measure() in a subprocess and hard-kill it if it's still running after
    MEASURE_HARD_TIMEOUT_S. Reloads the STL files _run_build() already exported to `folder` rather than
    re-tessellating; the subprocess has no build123d `parts` to run instrument.py's volume-sanity checks
    against (those need the live CAD objects), so print_validity() is re-run here in the parent — cheap,
    and unaffected by mesh quality — to fill that back in regardless of whether the subprocess finished
    or was killed."""
    mesh_paths = [os.path.join(folder, f"{n}.stl") for n in names_all]
    result_path = os.path.join(folder, "_measure_result.json")
    if os.path.exists(result_path):
        try: os.remove(result_path)
        except Exception: pass
    joints = P["segCount"] + 1
    proc = multiprocessing.Process(target=instrument.measure_worker,
                                    args=(P, mesh_paths, instrument.MEASURE_BUDGET_S, result_path))
    proc.start()
    proc.join(MEASURE_HARD_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate(); proc.join(5)
        if proc.is_alive(): proc.kill(); proc.join(5)
        meas = dict(instrument=instrument.INSTRUMENT_VERSION, params=schema.param_hash(P), measure_timed_out=True,
                    e_max=None, free_curl_deg=None, total_curl_deg=round(joints * P["maxAngle"], 1),
                    closure_gap_mm=None, enroll_class="unknown", stopped_by=[], touching_at_zero=None,
                    seconds=MEASURE_HARD_TIMEOUT_S)
    else:
        try:
            with open(result_path) as f: meas = json.load(f)
        except Exception as ex:
            meas = dict(instrument=instrument.INSTRUMENT_VERSION, params=schema.param_hash(P), measure_timed_out=True,
                        e_max=None, free_curl_deg=None, total_curl_deg=round(joints * P["maxAngle"], 1),
                        closure_gap_mm=None, enroll_class="unknown", stopped_by=[], touching_at_zero=None,
                        seconds=MEASURE_HARD_TIMEOUT_S, error=f"measure result unreadable: {ex}")
    meas.update(instrument.print_validity(P, parts))
    return meas

def _generate_outputs(P, names_all, blobs, meas, folder):
    """Render the technical sheet and a combined flat STL — the two things the client shows/downloads by
    default. This is fast (posing + concatenating already-tessellated meshes, plus a matplotlib figure —
    no CAD or collision work), so it runs directly in the build thread rather than a subprocess."""
    import trimesh, blueprint
    raw = [trimesh.load(io.BytesIO(blobs[n]), file_type="stl") for n in names_all]
    mats0 = instrument.transforms(P, 0.0)
    flat = trimesh.util.concatenate([r.copy().apply_transform(M) for r, M in zip(raw, mats0)])
    flat.export(os.path.join(folder, "combined.stl"))
    e_val = meas.get("e_max") or 0.0
    mats_e = instrument.transforms(P, float(e_val))
    enrolled = trimesh.util.concatenate([r.copy().apply_transform(M) for r, M in zip(raw, mats_e)])
    blueprint.sheet(flat, P, meas, os.path.join(folder, "sheet.png"), enrolled=enrolled)

def _run_build(k, P, folder, t0):
    """Background thread: build every part (CAD + tessellate, serialized under BUILD_CPU_LOCK so
    trilobite.py's retry-on-garbage globals stay safe), measure enrollment in a hard-bounded subprocess,
    then render the sheet and combined STL. status moves building -> measuring -> done (or error)."""
    try:
        with BUILD_CPU_LOCK:
            names_all = T.PART_NAMES(P)
            fns = ([lambda: T.build_cephalon(P)] + [(lambda i=i: T.build_segment(P, i)) for i in range(int(P["segCount"]))]
                   + [lambda: T.build_pygidium(P)])
            offs = T.joint_offsets(P)
            with LOCK:
                e = CACHE[k]; e["offsets"] = offs; e["hinge_z"] = T.hinge_z(P)
            parts = []
            for n, fn in zip(names_all, fns):
                p = T._build_checked(fn, n); parts.append(p)
                # _build_checked() already tessellated p at this exact tolerance to validate it (see
                # trilobite._sane()) and cached the result; reusing it avoids tessellating twice and
                # guarantees this is the same watertight-checked mesh, not a fresh, possibly different one.
                m = getattr(p, "_checked_mesh", None) or T.to_trimesh(p, *T.SANE_MESH_TOL)
                blob = m.export(file_type="stl")
                try: m.export(os.path.join(folder, f"{n}.stl"))         # measuring subprocess reloads from here
                except Exception: pass
                with LOCK:
                    e = CACHE[k]; e["names"].append(n); e["blobs"][n] = blob
        with LOCK:
            e = CACHE[k]; e["mesh_seconds"] = round(time.time() - t0, 1); e["status"] = "measuring"

        meas = _measure_hard_bounded(P, folder, names_all, parts)
        with LOCK:
            e = CACHE[k]; e["measure"] = meas; blobs_snapshot = dict(e["blobs"])

        try:
            _generate_outputs(P, names_all, blobs_snapshot, meas, folder)
            with LOCK:
                e = CACHE[k]
                e["sheet_ready"] = os.path.exists(os.path.join(folder, "sheet.png"))
                e["stl_ready"] = os.path.exists(os.path.join(folder, "combined.stl"))
        except Exception as ex:
            with LOCK:
                CACHE[k]["error"] = f"model measured OK, but sheet/STL generation failed: {ex}"

        with LOCK:
            e = CACHE[k]; e["seconds"] = round(time.time() - t0, 1); e["status"] = "done"
    except Exception as ex:
        with LOCK:
            CACHE[k]["status"] = "error"; CACHE[k]["error"] = str(ex)

def _status_snapshot(e):
    """Build the client-facing status dict from an already-fetched CACHE entry. Never touches LOCK —
    callers must already hold it. This is the piece that avoids reintroducing the self-deadlock: build()
    calls this directly on its already-cached-and-done fast path instead of calling build_status()
    (which acquires LOCK itself) from inside its own `with LOCK:` block."""
    out = {kk: vv for kk, vv in e.items() if kk not in ("blobs", "P", "names")}
    out["done_parts"] = len(e["names"]); out["maxAngle"] = e["P"]["maxAngle"]
    if e["status"] in ("measuring", "done"):
        out["derived"] = derived(e["P"])
    if e["status"] == "done":
        out["sheet_url"] = f"/api/sheet/{e['key']}.png" if e.get("sheet_ready") else None
        out["stl_url"] = f"/api/download/{e['key']}.stl" if e.get("stl_ready") else None
        out["part_names"] = e["names"]
    return out

def build_status(k):
    """Non-blocking snapshot of a build for the client. Acquires LOCK itself — never call this from
    inside a block that already holds LOCK; use _status_snapshot(e) there instead."""
    with LOCK:
        e = CACHE.get(k)
        if e is None: return dict(status="unknown", key=k)
        return _status_snapshot(e)

def build(P):
    """Start a build if one for this exact P isn't already running or cached; always returns immediately."""
    k = schema.param_hash(P)
    with LOCK:
        e = CACHE.get(k)
        if e is not None and e["status"] != "error":
            return _status_snapshot(e)
        CACHE[k] = dict(key=k, P=P, status="building", names=[], blobs={}, offsets=None, hinge_z=None,
                        total_parts=len(T.PART_NAMES(P)), measure=None, seconds=None, mesh_seconds=None,
                        error=None, sheet_ready=False, stl_ready=False)
        snap = _status_snapshot(CACHE[k])
    folder = os.path.join(OUT, k); os.makedirs(folder, exist_ok=True)
    threading.Thread(target=_run_build, args=(k, P, folder, time.time()), daemon=True).start()
    return snap

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw): super().__init__(*a, directory=HERE, **kw)
    def log_message(self, *a): pass
    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/": self.path = "/index.html"
        if path.startswith("/api/sheet/"):                                   # already-generated by _run_build
            key = path.split("/")[-1].replace(".png", "")
            png = os.path.join(OUT, key, "sheet.png")
            if not os.path.exists(png): return self.send_json(dict(error="sheet not ready"), 404)
            data = open(png, "rb").read()
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(data))); self.end_headers()
            return self.wfile.write(data)
        if path.startswith("/api/download/"):                                # already-generated by _run_build
            key = path.split("/")[-1].replace(".stl", "")
            stl = os.path.join(OUT, key, "combined.stl")
            if not os.path.exists(stl): return self.send_json(dict(error="download not ready"), 404)
            data = open(stl, "rb").read()
            self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="trilobite_{key}.stl"')
            self.send_header("Content-Length", str(len(data))); self.end_headers()
            return self.wfile.write(data)
        if path.startswith("/api/mesh/"):                              # /api/mesh/<key>/<name>.stl — lazy 3D view only
            _, _, _, key, fname = path.split("/", 4)
            entry = CACHE.get(key); name = fname[:-4] if fname.endswith(".stl") else fname
            if not entry or name not in entry["blobs"]:
                return self.send_json(dict(error="mesh not in cache (server restarted?) — press Build again"), 404)
            data = entry["blobs"][name]
            self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(data); return
        if path.startswith("/api/status/"):                                  # poll
            key = path.split("/")[-1]
            return self.send_json(build_status(key))
        if path == "/api/health":
            with LOCK:
                building = [k for k, v in CACHE.items() if v["status"] in ("building", "measuring")]
                cached = list(CACHE)
            return self.send_json(dict(ok=True, cached=cached, building=building, web_out=os.path.isdir(OUT), files=sum(len(f) for _, _, f in os.walk(OUT))))
        if path == "/api/config":
            return self.send_json(dict(knobs=KNOBS, presets=PRESETS, defaults=schema.defaults(), int_keys=INT_KEYS, ui=schema.UI,
                                       macros=schema.MACROS, schema=schema.SCHEMA_VERSION))
        return super().do_GET()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        P = schema.coerce(body.get("P", {}))
        try:
            if self.path == "/api/derived": return self.send_json(derived(P))
            if self.path == "/api/build": return self.send_json(build(P))
        except Exception as ex:
            return self.send_json(dict(error=str(ex)), 500)
        self.send_json(dict(error="unknown endpoint"), 404)

if __name__ == "__main__":
    print(f"Trilobite generator at http://localhost:{PORT}  (Ctrl+C to stop)")
    if not os.environ.get("PORT"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
