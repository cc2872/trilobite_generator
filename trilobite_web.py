"""
trilobite_web.py — the generator as a website (v4: schema-driven, instrument-backed).

    python trilobite_web.py        →  http://localhost:8765     (Render/Railway set PORT)
"""
import json, os, time, threading, io, contextlib, webbrowser
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

def build(P, mode):
    k = schema.param_hash(P) + ("s" if mode == "segment" else "a")
    with LOCK:
        if k in CACHE: return CACHE[k]
        t0 = time.time(); folder = os.path.join(OUT, k); os.makedirs(folder, exist_ok=True)
        if mode == "segment":
            parts = [T.build_segment(P, 2)]; names = ["segment"]; offs = []; meas = None
        else:
            parts = T.parts_list(P); names = T.PART_NAMES(P); offs = T.joint_offsets(P)
            meas = instrument.measure(P, instrument.part_meshes(P, parts))
        urls, blobs = [], {}
        for n, p in zip(names, parts):
            m = T.to_trimesh(p, 0.12, 0.15)
            blobs[n] = m.export(file_type="stl")                      # bytes, kept in memory
            urls.append(f"/api/mesh/{k}/{n}.stl")
            try: m.export(os.path.join(folder, f"{n}.stl"))         # also on disk, for downloads while it lasts
            except Exception: pass
        CACHE[k] = dict(key=k, P=P, names=names, urls=urls, offsets=offs, hinge_z=T.hinge_z(P), maxAngle=P["maxAngle"],
                        seconds=round(time.time() - t0, 1), measure=meas, blobs=blobs)
        return CACHE[k]

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
        if path == "/api/health":
            return self.send_json(dict(ok=True, cached=list(CACHE), web_out=os.path.isdir(OUT), files=sum(len(f) for _, _, f in os.walk(OUT))))
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
                return self.send_json(dict({k: v for k, v in b.items() if k != "blobs"}, derived=derived(P)))
        except Exception as ex:
            return self.send_json(dict(error=str(ex)), 500)
        self.send_json(dict(error="unknown endpoint"), 404)

if __name__ == "__main__":
    print(f"Trilobite generator at http://localhost:{PORT}  (Ctrl+C to stop)")
    if not os.environ.get("PORT"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
