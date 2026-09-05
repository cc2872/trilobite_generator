"""
trilobite_web.py — the generator as a local website.

    python trilobite_web.py          then open  http://localhost:8765

Sliders + number boxes for every knob (driven by schema.py — the single source of truth for
every parameter), a family/preset dropdown, an instant readout with printability warnings,
"Build" (runs build123d, 30-60 s), a full-screen 3D view with an enrollment scrub slider and
play button (posing happens in the browser — no rebuild needed to roll), the collision check,
and STL downloads. Needs trilobite.py, schema.py, fields.py and instrument.py next to it.
"""
import json, os, time, threading, io, contextlib, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import schema, instrument
import trilobite as T

PORT = int(os.environ.get("PORT", 8765))          # hosts like Render/Railway set PORT
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web_out")
os.makedirs(OUT, exist_ok=True)

# (key, label, min, max, step, group) — derived straight from schema.py, the single source of truth
KNOBS = [(p.key, f"{p.label} ({p.unit})" if p.unit else p.label, p.lo, p.hi, p.step, p.group) for p in schema.PARAMS]
INT_KEYS = sorted(p.key for p in schema.PARAMS if p.kind in ("int", "odd_int"))

# Families: named regions of parameter space, from the plain reference body to named real groups.
FAMILY = {
    "Reference (defaults)": {},
    "Slender & smooth": dict(width=56, segCount=8, effacement=0.7, spineBase=0.0, pygSpine=0.3, marginHeight=0.20),
    "Broad, macropygous": dict(width=80, cephFrac=0.26, pygFrac=0.30, pygWidth=1.0, pygSpine=0.0, segCount=5),
    "Spiny": dict(spineBase=0.8, spineSweep=55, pygSpine=1.4, pygSplay=30, eyeSize=0.12, genalSpine=0.6, termSpine=0.6),
    "Many segments, tight roll": dict(segCount=13, length=160, maxAngle=12, overlap=0.4),
    "Phacopid (big eyes, blunt)": dict(eyeSize=0.18, eyeArc=200, eyePos=0.55, glabInflate=1.15,
                                        spineBase=0.10, pygSpine=0.0, cephParallel=0.30, borderWidth=0.05),
    "Sculpted paradoxidid": dict(segCount=9, cephFrac=0.20, pygFrac=0.08, genalSpine=1.1, genalCurve=8,
                                  glabInflate=1.35, glabLobes=3, glabRise=0.22, effacement=0.0, furrowDepth=1.6,
                                  tubercles=0.7, tubercleSize=1.1, eyeSize=0.07, eyeArc=110, spineBase=0.12,
                                  spineGrad=0.1, pygSpine=0.0, pygSplay=15, width=72, length=180, wall=2.2),
}

def derived(P):
    """Readout shown under the sliders: geometry summary + printability/enrollment warnings."""
    d = T.pitch(P); joints = P["segCount"] + 1; total = joints * P["maxAngle"]
    ratio = P["pygFrac"] / P["cephFrac"]
    pyg = "micropygous" if ratio < 0.6 else ("isopygous" if ratio < 1.15 else "macropygous")
    v = instrument.print_validity(P)
    warns = list(v["violations"])
    if total > 360: warns.append(f"total curl {total:.0f}° > 360° — tail would pass the head; lower the stop angle")
    return dict(
        pitch=round(d, 1), joints=joints, total_curl=round(total), pyg_class=pyg,
        hinge_z=v["hinge_z"], hinge_width=v["hinge_width"], knuckle=v["knuckle"], pin_wall=v["pin_wall"],
        head_mm=round(P["cephFrac"] * P["length"]), thorax_mm=round(d * P["segCount"]), tail_mm=round(P["pygFrac"] * P["length"]),
        last_width=round(2 * T.seg_halfwidth(P, P["segCount"] - 1)), warnings=warns)

# ---- build cache: parts keyed by the parameter hash
CACHE = {}
LOCK = threading.Lock()

def build(P, mode):
    k = schema.param_hash(P) + ("s" if mode == "segment" else "a")
    with LOCK:
        if k in CACHE: return CACHE[k]
        t0 = time.time()
        folder = os.path.join(OUT, k); os.makedirs(folder, exist_ok=True)
        if mode == "segment":
            parts = [T.build_segment(P, min(2, P["segCount"] - 1))]; names = ["segment"]; offs = []
        else:
            parts = T.parts_list(P)
            names = T.PART_NAMES(P)
            offs = T.joint_offsets(P)
        files = []
        for n, p in zip(names, parts):
            f = os.path.join(folder, f"{n}.stl"); T.save_mesh(p, f); files.append(f"/web_out/{k}/{n}.stl")
        CACHE[k] = dict(key=k, parts=parts, names=names, urls=files, offsets=offs, hinge_z=T.hinge_z(P),
                        maxAngle=P["maxAngle"], seconds=round(time.time() - t0, 1))
        return CACHE[k]

def check(P):
    b = build(P, "animal")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.check_enrollment(P, parts=b["parts"])
    return buf.getvalue()

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
            return self.send_json(dict(knobs=KNOBS, family=FAMILY, defaults=schema.defaults(), int_keys=INT_KEYS))
        return super().do_GET()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        P = schema.coerce(body.get("P", {}))
        try:
            if self.path == "/api/derived": return self.send_json(derived(P))
            if self.path == "/api/build":
                b = build(P, body.get("mode", "animal"))
                return self.send_json(dict(key=b["key"], names=b["names"], urls=b["urls"], offsets=b["offsets"],
                                           hinge_z=b["hinge_z"], maxAngle=b["maxAngle"], seconds=b["seconds"], derived=derived(P)))
            if self.path == "/api/check": return self.send_json(dict(text=check(P)))
        except Exception as ex:
            return self.send_json(dict(error=str(ex)), 500)
        self.send_json(dict(error="unknown endpoint"), 404)

if __name__ == "__main__":
    print(f"Trilobite generator at http://localhost:{PORT}  (Ctrl+C to stop)")
    if not os.environ.get("PORT"):                                   # local run: open the browser
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
