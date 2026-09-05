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
    "Olenoides (base sculpt)": dict(length=130, width=98, relief=42.6, cephFrac=0.348, pygFrac=0.191, segCount=9, widthMaxPos=0.465, widthHeadFront=0.607, widthThoraxFront=0.971, widthThoraxRear=0.651, widthTail=0.276, spineBase=0.3, genalSpine=0.35),
    "Phacopid-like": dict(length=130, width=98, relief=42.6, cephFrac=0.32, pygFrac=0.20, segCount=11, widthMaxPos=0.465, widthHeadFront=0.607, widthThoraxFront=0.971, widthThoraxRear=0.651, widthTail=0.276,
                          glabBlend=1.0, eyeType=2, eyeSize=0.2, segBlend=1.0, pygBlend=1.0, spineBase=0.05, genalSpine=0.1),
    "Harpetid-like": dict(length=130, width=98, relief=42.6, cephFrac=0.38, pygFrac=0.12, segCount=7, widthMaxPos=0.465, widthHeadFront=0.607, widthThoraxFront=0.971, widthThoraxRear=0.651, widthTail=0.276,
                          brim=1, brimWidth=0.35, effacement=0.6, eyeType=0, spineBase=0.05, genalSpine=0.1),
    "Slender, 13 segments": dict(length=160, width=70, relief=30, cephFrac=0.28, pygFrac=0.12, segCount=13, spineBase=0.6, widthMaxPos=0.465, widthHeadFront=0.607, widthThoraxFront=0.971, widthThoraxRear=0.651, widthTail=0.276, genalSpine=0.35),
    "Broad, 5 segments": dict(length=110, width=110, relief=50, cephFrac=0.38, pygFrac=0.22, segCount=5, spineBase=0.1, genalSpine=0.1, widthMaxPos=0.465, widthHeadFront=0.607, widthThoraxFront=0.971, widthThoraxRear=0.651, widthTail=0.276),
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

def describe(P, m):
    """Name what the sliders made: a morphotype guess plus the descriptors a paleontologist would use."""
    n = int(P["segCount"]); ratio = P["pygFrac"] / P["cephFrac"]
    pyg = "micropygous" if ratio < 0.6 else ("isopygous" if ratio < 1.15 else "macropygous")
    sp = P["spineBase"]; ge = P["genalSpine"]
    spiny = "spiny" if (sp > 0.45 or ge > 0.6) else ("smooth-margined" if (sp < 0.12 and ge < 0.15) else "moderately spined")
    eyes = {0: "blind", 1: "holochroal-eyed", 2: "schizochroal-eyed"}[int(P["eyeType"])]
    eff = "effaced, " if P["effacement"] > 0.5 else ""
    if P["brim"] >= 0.5: fam, order = "harpetid-like", "Harpetida"
    elif P["glabBlend"] > 0.5 and int(P["eyeType"]) == 2 and sp < 0.3: fam, order = "phacopid-like", "Phacopida"
    elif n >= 12 and ge > 0.6 and P["pygFrac"] < 0.12: fam, order = "olenellid-like", "Redlichiida"
    elif n <= 3: fam, order = "agnostid-like", "Agnostida"
    else: fam, order = "Olenoides-like", "Corynexochida"
    roll = {"none": "cannot enroll", "partial": f"partial enroller ({m['free_curl_deg']:.0f}° of curl)", "complete": "complete enroller"}[m["enroll_class"]]
    name = f"{spiny.capitalize()}, {eff}{pyg} {fam} trilobite"
    sentence = (f"{order}-type body plan · {n} thoracic segments · {eyes} · {P['length']:.0f} mm long · {roll}")
    return dict(name=name, order=order, sentence=sentence)

CACHE, LOCK = {}, threading.Lock()

def build(P, mode):
    k = schema.param_hash(P) + ("s" if mode == "segment" else "a")
    with LOCK:
        if k in CACHE: return CACHE[k]
        t0 = time.time(); folder = os.path.join(OUT, k); os.makedirs(folder, exist_ok=True)
        if mode != "segment":
            import mega, skin
            pieces, axes, MP = mega.build_mega(P)
            names = ["head"] + [f"seg{i}" for i in range(len(pieces) - 2)] + ["tail"]
            meas = skin.measure(pieces, axes, MP["maxAngle"]); meas.update(instrument="sculpt-1.1", print_valid=True, violations=[], params=schema.param_hash(P))
            blobs = {n: p.export(file_type="stl") for n, p in zip(names, pieces)}
            CACHE[k] = dict(key=k, names=names, urls=[f"/api/mesh/{k}/{n}.stl" for n in names], offsets=[], axes=[dict(y=a["y"], zh=a["zh"]) for a in axes],
                            hinge_z=axes[0]["zh"], maxAngle=MP["maxAngle"], seconds=round(time.time() - t0, 1), measure=meas, blobs=blobs,
                            pieces=pieces, axes_full=axes, label=describe(P, meas))
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
        if path.startswith("/api/download/"):
            key = path.split("/")[-1].replace(".zip", ""); entry = CACHE.get(key)
            if not entry: return self.send_json(dict(error="build not in cache — press Build again"), 404)
            import zipfile, io as _io, trimesh as _tm
            buf = _io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for n, b in entry["blobs"].items(): z.writestr(f"parts/{n}.stl", b)
                if entry.get("pieces"):
                    import skin as _sk
                    z.writestr("animal_flat.stl", _tm.util.concatenate(_sk.posed(entry["pieces"], entry["axes_full"], 0.0, entry["maxAngle"])).export(file_type="stl"))
                    e = entry["measure"]["e_max"]
                    if e > 0.02: z.writestr(f"animal_enrolled_{int(e*100)}pct.stl", _tm.util.concatenate(_sk.posed(entry["pieces"], entry["axes_full"], e, entry["maxAngle"])).export(file_type="stl"))
                z.writestr("measurement.json", json.dumps(dict(label=entry.get("label"), measure=entry["measure"]), indent=1))
                z.writestr("README.txt", "Print parts/*.stl individually (segments x N, head, tail); pin with 1.75 mm filament through the side bores.\nanimal_flat.stl is the assembled reference. See measurement.json for the enrollment verdict.")
            data = buf.getvalue()
            self.send_response(200); self.send_header("Content-Type", "application/zip"); self.send_header("Content-Disposition", f'attachment; filename="trilobite_{key}.zip"')
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
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
                return self.send_json(dict({k: v for k, v in b.items() if k not in ("blobs", "pieces", "axes_full")}, derived=derived(P)))
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
