"""
trilobite_web.py — the generator as a local website.

    python trilobite_web.py          then open  http://localhost:8765

Sliders + number boxes for every knob, presets, an instant readout with printability warnings,
"Build" (runs build123d, 30-60 s), a full-screen 3D view with an enrollment scrub slider and
play button (posing happens in the browser — no rebuild needed to roll), the collision check,
and STL downloads. Needs only trilobite.py next to it and the Python standard library.
"""
import json, os, time, threading, hashlib, io, contextlib, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import trilobite as T

PORT = int(os.environ.get("PORT", 8765))          # hosts like Render/Railway set PORT
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web_out")
os.makedirs(OUT, exist_ok=True)

# (key, label, min, max, step, group)
KNOBS = [
    ("length","Body length (mm)",60,250,1,"Body"), ("width","Body width (mm)",30,140,1,"Body"),
    ("relief","Relief / dorsal height (mm)",6,40,0.5,"Body"), ("segCount","Thoracic segments",2,16,1,"Body"),
    ("taper","Taper per segment",0.80,1.00,0.005,"Body"), ("axisFrac","Axial ring width fraction",0.20,0.50,0.01,"Body"),
    ("axisRise","Axial ring rise",0.00,0.40,0.01,"Body"), ("marginHeight","Pleural margin height",0.00,0.60,0.01,"Body"),
    ("overlap","Shingle overlap (× pitch)",0.10,0.80,0.01,"Body"),
    ("cephFrac","Head length fraction",0.18,0.45,0.005,"Head"), ("cephParallel","Head: parallel rear fraction",0.00,0.80,0.01,"Head"),
    ("glabInflate","Glabella inflation",0.70,1.60,0.01,"Head"), ("eyeSize","Eye size",0.00,0.20,0.005,"Head"),
    ("eyePos","Eye position (rear→front)",0.10,0.90,0.01,"Head"),
    ("pygFrac","Tail shield length fraction",0.05,0.40,0.005,"Tail"), ("pygWidth","Tail width (× last segment)",0.50,1.10,0.01,"Tail"),
    ("pygSpine","Tail spine length (× shield)",0.00,2.00,0.05,"Tail"), ("pygSplay","Tail spine splay (°)",0,45,1,"Tail"),
    ("pleuralSpine","Pleural spine length",0.00,1.00,0.01,"Spines"), ("spineSweep","Pleural spine sweep (°)",0,80,1,"Spines"),
    ("maxAngle","Stop angle per joint (°)",4,40,0.5,"Hinge"), ("clearance","Joint clearance (mm)",0.15,0.60,0.01,"Hinge"),
    ("wall","Shell wall (mm)",1.2,4.0,0.1,"Hinge"), ("barrelR","Knuckle radius (mm)",1.8,4.0,0.1,"Hinge"),
    ("nKnuckles","Knuckles (odd)",3,7,2,"Hinge"),
]
INT_KEYS = {"segCount", "nKnuckles"}
PRESETS = {
    "Reference (Weaver STL)": {},
    "Slender & smooth": dict(width=48, taper=0.95, pleuralSpine=0.0, pygSpine=0.3, marginHeight=0.15, segCount=8),
    "Broad, macropygous": dict(width=80, cephFrac=0.26, pygFrac=0.30, pygWidth=1.0, pygSpine=0.0, segCount=5),
    "Spiny": dict(pleuralSpine=0.8, spineSweep=55, pygSpine=1.4, pygSplay=30, eyeSize=0.12),
    "Many segments, tight roll": dict(segCount=12, maxAngle=12, taper=0.97, overlap=0.4),
    "Phacopid (big eyes, blunt)": dict(eyeSize=0.16, eyePos=0.55, glabInflate=1.3, pleuralSpine=0.15, pygSpine=0.0, cephParallel=0.3),
}

def clean(P):
    """Coerce a client dict into a valid parameter set (ints, odd knuckles, defaults for anything missing)."""
    Q = dict(T.P); Q.update({k: float(v) for k, v in P.items() if k in T.P})
    for k in INT_KEYS: Q[k] = int(round(Q[k]))
    if Q["nKnuckles"] % 2 == 0: Q["nKnuckles"] += 1
    return Q

def derived(P):
    d = T.pitch(P); joints = P["segCount"] + 1; total = joints * P["maxAngle"]
    zh, Wh = T.hinge_z(P), T.hinge_width(P)
    knuckle = Wh / P["nKnuckles"] - P["clearance"]; pin_wall = P["barrelR"] - P["boreDia"] / 2
    ratio = P["pygFrac"] / P["cephFrac"]
    pyg = "micropygous" if ratio < 0.6 else ("isopygous" if ratio < 1.15 else "macropygous")
    warns = []
    if P["wall"] < 1.5: warns.append("wall < 1.5 mm — fragile on FDM")
    if pin_wall < 1.2: warns.append(f"only {pin_wall:.1f} mm around the pin bore — raise knuckle radius")
    if knuckle < 3.0: warns.append(f"knuckles {knuckle:.1f} mm long — too small: fewer segments, wider axis, or fewer knuckles")
    if zh < P["barrelR"] + 3: warns.append("hinge axis too low — raise relief or lower knuckle radius")
    if d < 8: warns.append(f"segment pitch {d:.1f} mm — very short: fewer segments or longer body")
    if total > 360: warns.append(f"total curl {total:.0f}° > 360° — tail would pass the head; lower the stop angle")
    if P["pleuralSpine"] > 0 and P["marginHeight"] * P["relief"] < P["wall"] + 0.5: warns.append("margin rim thinner than the spine base — spines will be weak")
    return dict(
        pitch=round(d, 1), joints=joints, total_curl=round(total), pyg_class=pyg,
        hinge_z=round(zh, 1), hinge_width=round(Wh, 1), knuckle=round(knuckle, 1), pin_wall=round(pin_wall, 1),
        head_mm=round(P["cephFrac"] * P["length"]), thorax_mm=round(d * P["segCount"]), tail_mm=round(P["pygFrac"] * P["length"]),
        last_width=round(2 * T.seg_halfwidth(P, P["segCount"] - 1)), warnings=warns)

# ---- build cache: parts keyed by the parameter set
CACHE = {}
LOCK = threading.Lock()

def key_of(P): return hashlib.md5(json.dumps(P, sort_keys=True).encode()).hexdigest()[:10]

def build(P, mode):
    k = key_of(P) + ("s" if mode == "segment" else "a")
    with LOCK:
        if k in CACHE: return CACHE[k]
        t0 = time.time()
        folder = os.path.join(OUT, k); os.makedirs(folder, exist_ok=True)
        if mode == "segment":
            parts = [T.build_segment(P, 2)]; names = ["segment"]; offs = []
        else:
            parts = T.parts_list(P)
            names = ["head"] + [f"seg{i}" for i in range(P["segCount"])] + ["tail"]
            offs = T.joint_offsets(P)
        files = []
        for n, p in zip(names, parts):
            f = os.path.join(folder, f"{n}.stl"); T.export_stl(p, f); files.append(f"/web_out/{k}/{n}.stl")
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
            return self.send_json(dict(knobs=KNOBS, presets=PRESETS, defaults=T.P, int_keys=sorted(INT_KEYS)))
        return super().do_GET()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        P = clean(body.get("P", {}))
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
