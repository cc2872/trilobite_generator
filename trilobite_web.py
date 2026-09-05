"""
trilobite_web.py — the generator as a website (v4: schema-driven, instrument-backed).

    python trilobite_web.py        →  http://localhost:8765     (Render/Railway set PORT)
"""
import json, os, time, threading, io, contextlib, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import schema, fields, instrument

class _LazyT:
    """Import the CAD kernel only when the parametric family is actually built (saves ~460 MB otherwise)."""
    def __getattr__(self, name):
        import trilobite; return getattr(trilobite, name)
T = _LazyT()

PORT = int(os.environ.get("PORT", 8765))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web_out"); os.makedirs(OUT, exist_ok=True)

PRESETS = {
    "Reference": {},
    "Sculpted paradoxidid": dict(family=1),
    "Sculpted, with legs": dict(family=1, legs=1),
    "Smooth (effaced)": dict(effacement=0.85, spineBase=0.0, genalSpine=0.0, pygSpine=0.0, tubercles=0.0),
    "Spiny": dict(spineBase=0.9, spineSweep=25, axialSpine=0.8, pygMarginal=6, genalSpine=0.8, tubercles=0.5),
    "Phacopid": dict(eyeSize=0.2, eyePos=0.55, glabInflate=1.6, glabRise=0.3, pleuralSpine=0.0, spineBase=0.0,
                     pygSpine=0.0, segCount=11, maxAngle=30, taper=0.96, tubercles=0.7),
    "Olenellid": dict(segCount=14, taper=0.95, genalSpine=1.2, macroIndex=2, macroAmp=1.2, termSpine=1.8, pygFrac=0.06, maxAngle=8),
    "Many segments, tight roll": dict(segCount=12, maxAngle=12, taper=0.97, overlap=0.4),
}
KNOBS = [(p.key, p.label, p.lo, p.hi, p.step, p.group) for p in schema.PARAMS]
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
        if int(P.get("family", 0)) == 1 and mode != "segment":
            import skin
            pieces, axes, SP = skin.build_skinned(dict(maxAngle=P["maxAngle"], clearance=P["clearance"], wall=P["wall"],
                                                       barrelR=P["barrelR"], nKnuckles=P["nKnuckles"]), legs=bool(int(P.get("legs", 0))), verbose=False)
            names = ["head"] + [f"seg{i}" for i in range(len(pieces) - 2)] + ["tail"]
            meas = skin.measure(pieces, axes, SP["maxAngle"]); meas.update(instrument="skin-1.0", print_valid=True, violations=[], params=schema.param_hash(P))
            blobs = {n: p.export(file_type="stl") for n, p in zip(names, pieces)}
            urls = [f"/api/mesh/{k}/{n}.stl" for n in names]
            CACHE[k] = dict(key=k, names=names, urls=urls, offsets=[], axes=[dict(y=a["y"], zh=a["zh"]) for a in axes],
                            hinge_z=axes[0]["zh"], maxAngle=SP["maxAngle"], seconds=round(time.time() - t0, 1), measure=meas, blobs=blobs)
            return CACHE[k]
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
        CACHE[k] = dict(key=k, names=names, urls=urls, offsets=offs, hinge_z=T.hinge_z(P), maxAngle=P["maxAngle"],
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
            return self.send_json(dict(knobs=KNOBS, presets=PRESETS, defaults=schema.defaults(), int_keys=INT_KEYS,
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
            if self.path == "/api/check":
                b = build(P, "animal"); m = b["measure"]
                txt = (f"instrument {m['instrument']}  params {m['params']}\\n"
                       f"e_max = {m['e_max']}   free curl = {m['free_curl_deg']} deg of {m['total_curl_deg']}\\n"
                       f"closure gap = {m['closure_gap_mm']} mm   class = {m['enroll_class']}\\n"
                       f"stopped by = {m['stopped_by']}   touching at zero = {m['touching_at_zero']}\\n"
                       f"print valid = {m['print_valid']}  {m['violations']}")
                return self.send_json(dict(text=txt, measure=m))
        except Exception as ex:
            return self.send_json(dict(error=str(ex)), 500)
        self.send_json(dict(error="unknown endpoint"), 404)

if __name__ == "__main__":
    print(f"Trilobite generator at http://localhost:{PORT}  (Ctrl+C to stop)")
    if not os.environ.get("PORT"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
