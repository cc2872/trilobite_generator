"""
trilobite_web.py — the generator as a website (v4: schema-driven, instrument-backed).

    python trilobite_web.py        →  http://localhost:8765     (Render/Railway set PORT)
"""
import json, os, time, threading, multiprocessing, io, contextlib, webbrowser
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

CACHE, LOCK = {}, threading.Lock()
# trilobite.py's retry-on-garbage machinery (_build_checked, GRID_JITTER, BUILD_NOTES) is module-global,
# not thread-safe: two builds running their CAD/CPU work at once would race on that state (and could hand
# a jittered grid to the wrong build mid-part, or corrupt the other's garbage-detection). Since a single
# OpenCascade-heavy build already saturates one core, there is nothing to gain from true concurrency here —
# serialize the CPU-heavy portion of every build behind one lock so only one build ever runs it at a time.
BUILD_CPU_LOCK = threading.Lock()

# instrument.measure()'s own time budget (instrument.MEASURE_BUDGET_S) only checks the clock between
# pairwise checks, so it cannot interrupt a single already-in-flight OpenCascade/trimesh call - and in
# production that alone has hung the whole single-process server for over an hour on a pathological
# mesh. MEASURE_HARD_TIMEOUT_S bounds that: it's the wall-clock ceiling _measure_hard_bounded() will
# wait for the measuring subprocess before forcibly killing it. Set comfortably above
# instrument.MEASURE_BUDGET_S to give a normal timeout its own soft budget a chance to return cleanly.
MEASURE_HARD_TIMEOUT_S = instrument.MEASURE_BUDGET_S + 90

def _measure_hard_bounded(P, folder, names_all, parts):
    """Run instrument.measure() in a subprocess and hard-kill it if it's still running after
    MEASURE_HARD_TIMEOUT_S, regardless of what C call it's stuck inside. Reloads the STL files
    _run_build() already exported to `folder` rather than re-tessellating; the subprocess has no
    build123d `parts` to run instrument.py's volume-sanity checks against (those need the live CAD
    objects), so print_validity() is re-run here in the parent — cheap, and unaffected by mesh quality
    — to fill that back in regardless of whether the subprocess finished or was killed."""
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

def _run_build(k, P, mode, folder, t0):
    """Runs in a background thread so no HTTP request ever blocks on it (some parameter combinations
    take minutes). Parts are appended to CACHE[k] one at a time as they finish, so the client can
    stream them into the viewer instead of waiting on a black screen; status moves
    building -> measuring -> done, and the enrollment instrument only runs once every part exists.
    Every part goes through trilobite._build_checked(), the same sanity-check-and-retry wrapper the
    offline/console build already uses: OpenCascade occasionally hands back garbage (a failed fuse, or
    a stray inside-out sliver alongside the real part) for no parameter-related reason, and retrying with
    a slightly different surface grid reliably fixes it. Streaming the raw, unchecked part (as this used
    to do) could hand the client — and the instrument's collision measurement — a badly broken mesh, which
    is slow or pathological enough to make the whole single-process server look hung.

    BUILD_CPU_LOCK covers only the CAD-construction loop above, not the measuring step below: it exists
    to serialize access to trilobite.py's GRID_JITTER/BUILD_NOTES globals (mutated by _build_checked and
    friends), and instrument.measure() never touches those. Keeping measuring outside the lock lets a
    second build's part construction start as soon as this one's parts are done, instead of waiting out
    a potentially multi-minute measuring pass it has nothing to do with."""
    try:
        with BUILD_CPU_LOCK:
            if mode == "segment":
                names_all, fns, offs = ["segment"], [lambda: T.build_segment(P, 2)], []
            else:
                names_all = T.PART_NAMES(P)
                fns = ([lambda: T.build_cephalon(P)] + [(lambda i=i: T.build_segment(P, i)) for i in range(int(P["segCount"]))]
                       + [lambda: T.build_pygidium(P)])
                offs = T.joint_offsets(P)
            with LOCK:
                CACHE[k]["offsets"] = offs; CACHE[k]["hinge_z"] = T.hinge_z(P)
            parts = []
            for n, fn in zip(names_all, fns):
                p = T._build_checked(fn, n); parts.append(p)
                # _build_checked() already tessellated p at this exact tolerance to validate it (see
                # trilobite._sane()) and cached the result; reusing it avoids tessellating twice and
                # guarantees this is the same watertight-checked mesh, not a fresh, possibly different one.
                m = getattr(p, "_checked_mesh", None) or T.to_trimesh(p, *T.SANE_MESH_TOL)
                blob = m.export(file_type="stl")
                try: m.export(os.path.join(folder, f"{n}.stl"))         # also on disk, for downloads while it lasts
                except Exception: pass
                with LOCK:
                    e = CACHE[k]; e["names"].append(n); e["urls"].append(f"/api/mesh/{k}/{n}.stl"); e["blobs"][n] = blob
        with LOCK:
            e = CACHE[k]; e["mesh_seconds"] = round(time.time() - t0, 1)
            e["status"] = "done" if mode == "segment" else "measuring"
            if mode == "segment": e["seconds"] = e["mesh_seconds"]
        if mode != "segment":
            meas = _measure_hard_bounded(P, folder, names_all, parts)
            with LOCK:
                e = CACHE[k]; e["measure"] = meas; e["seconds"] = round(time.time() - t0, 1); e["status"] = "done"
    except Exception as ex:
        with LOCK:
            CACHE[k]["status"] = "error"; CACHE[k]["error"] = str(ex)

def build_status(k):
    """Non-blocking snapshot of a build: building (mesh streaming in) / measuring (mesh done,
    instrument running) / done / error. Never includes the raw STL bytes."""
    with LOCK:
        e = CACHE.get(k)
        if e is None: return dict(status="unknown", key=k)
        return {kk: vv for kk, vv in e.items() if kk != "blobs"}

def build(P, mode):
    """Start a build if one for this exact P isn't already running or cached; always returns immediately."""
    k = schema.param_hash(P) + ("s" if mode == "segment" else "a")
    with LOCK:
        e = CACHE.get(k)
        if e is not None and e["status"] != "error": return build_status(k)
        CACHE[k] = dict(key=k, P=P, status="building", names=[], urls=[], offsets=[], hinge_z=None,
                        maxAngle=P["maxAngle"], blobs={}, measure=None, seconds=None, mesh_seconds=None, error=None)
    folder = os.path.join(OUT, k); os.makedirs(folder, exist_ok=True)
    threading.Thread(target=_run_build, args=(k, P, mode, folder, time.time()), daemon=True).start()
    return build_status(k)

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
        if path.startswith("/api/sheet/"):                                  # the blueprint sheet of a built animal
            key = path.split("/")[-1].replace(".png", "")
            b = CACHE.get(key)
            if b is None: return self.send_json(dict(error="not built"), 404)
            png = os.path.join(OUT, key, "sheet.png")
            if not os.path.exists(png):
                import trimesh, blueprint
                P = b["P"]; mats = instrument.transforms(P, 0.0)
                raw = [trimesh.load(io.BytesIO(b["blobs"][n]), file_type="stl") for n in b["names"]]
                flat = trimesh.util.concatenate([r.copy().apply_transform(M) for r, M in zip(raw, mats)])
                e = (b["measure"] or {}).get("e_max", 1.0) or 0.0
                en = trimesh.util.concatenate([r.copy().apply_transform(M) for r, M in zip(raw, instrument.transforms(P, float(e)))])
                blueprint.sheet(flat, P, dict(b["measure"] or {}, **instrument.print_validity(P)), png, enrolled=en)
            data = open(png, "rb").read()
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(data))); self.end_headers()
            return self.wfile.write(data)
        if path.startswith("/api/mesh/"):                              # /api/mesh/<key>/<name>.stl from memory
            _, _, _, key, fname = path.split("/", 4)
            entry = CACHE.get(key); name = fname[:-4] if fname.endswith(".stl") else fname
            if not entry or name not in entry["blobs"]:
                return self.send_json(dict(error="mesh not in cache (server restarted?) — press Build again"), 404)
            data = entry["blobs"][name]
            self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(data); return
        if path.startswith("/api/build/"):                                   # poll: parts stream in as they finish
            key = path.split("/")[-1]
            b = build_status(key)
            if b["status"] in ("measuring", "done"): b = dict(b, derived=derived(b["P"]))
            return self.send_json(b)
        if path == "/api/health":
            building = [k for k, v in CACHE.items() if v["status"] in ("building", "measuring")]
            return self.send_json(dict(ok=True, cached=list(CACHE), building=building, web_out=os.path.isdir(OUT), files=sum(len(f) for _, _, f in os.walk(OUT))))
        if path == "/api/config":
            return self.send_json(dict(knobs=KNOBS, presets=PRESETS, defaults=schema.defaults(), int_keys=INT_KEYS, ui=schema.UI,
                                       macros=schema.MACROS, schema=schema.SCHEMA_VERSION))
        return super().do_GET()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        P = schema.coerce(body.get("P", {}))
        try:
            if self.path == "/api/derived": return self.send_json(derived(P))
            if self.path == "/api/build":
                b = build(P, body.get("mode", "animal"))
                if b["status"] in ("measuring", "done"): b = dict(b, derived=derived(P))
                return self.send_json(b)
        except Exception as ex:
            return self.send_json(dict(error=str(ex)), 500)
        self.send_json(dict(error="unknown endpoint"), 404)

if __name__ == "__main__":
    print(f"Trilobite generator at http://localhost:{PORT}  (Ctrl+C to stop)")
    if not os.environ.get("PORT"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
