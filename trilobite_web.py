
import json, os, time, threading, io, contextlib, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import schema, fields, instrument
import trilobite as T
 
PORT = int(os.environ.get("PORT", 8765))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web_out"); os.makedirs(OUT, exist_ok=True)
 
PRESETS = {
    "Reference": {},
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
        if mode == "segment":
            parts = [T.build_segment(P, 2)]; names = ["segment"]; offs = []; meas = None
        else:
            parts = T.parts_list(P); names = T.PART_NAMES(P); offs = T.joint_offsets(P)
            meas = instrument.measure(P, instrument.part_meshes(P, parts))
        urls = []
        for n, p in zip(names, parts):
            T.save_mesh(p, os.path.join(folder, f"{n}.stl"), 0.12, 0.15); urls.append(f"/web_out/{k}/{n}.stl")
        CACHE[k] = dict(key=k, names=names, urls=urls, offsets=offs, hinge_z=T.hinge_z(P), maxAngle=P["maxAngle"],
                        seconds=round(time.time() - t0, 1), measure=meas)
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
        if path == "/api/config":
            return self.send_json(dict(knobs=KNOBS, presets=PRESETS, defaults=schema.defaults(), int_keys=INT_KEYS, schema=schema.SCHEMA_VERSION))
        return super().do_GET()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        P = schema.coerce(body.get("P", {}))
        try:
            if self.path == "/api/derived": return self.send_json(derived(P))
            if self.path == "/api/build":
                b = build(P, body.get("mode", "animal"))
                return self.send_json(dict(b, derived=derived(P)))
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
