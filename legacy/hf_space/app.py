"""
app.py — Trilobite Morphospace Generator on Hugging Face Spaces (Gradio).

Sliders for every knob → instant readout + printability warnings → "Build" runs build123d
(60–120 s on the free CPU) → interactive 3D model. The enrollment slider re-poses the built
parts without rebuilding. "Check enrollment" runs the collision test. STLs come as a zip.
"""
import os, io, json, time, hashlib, zipfile, tempfile, threading, contextlib
import gradio as gr
from build123d import Compound, Color, export_gltf, export_stl
import trilobite as T

# ---------------- knobs (key, label, min, max, step, group)
KNOBS = [
    ("length","Body length (mm)",60,250,1,"Body"), ("width","Body width (mm)",30,140,1,"Body"),
    ("relief","Relief / dorsal height (mm)",6,40,0.5,"Body"), ("segCount","Thoracic segments",2,14,1,"Body"),
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
KEYS = [k[0] for k in KNOBS]
INT_KEYS = {"segCount", "nKnuckles"}
PRESETS = {
    "Reference (Weaver STL)": {},
    "Slender & smooth": dict(width=48, taper=0.95, pleuralSpine=0.0, pygSpine=0.3, marginHeight=0.15, segCount=8),
    "Broad, macropygous": dict(width=80, cephFrac=0.26, pygFrac=0.30, pygWidth=1.0, pygSpine=0.0, segCount=5),
    "Spiny": dict(pleuralSpine=0.8, spineSweep=55, pygSpine=1.4, pygSplay=30, eyeSize=0.12),
    "Many segments, tight roll": dict(segCount=12, maxAngle=12, taper=0.97, overlap=0.4),
    "Phacopid (big eyes, blunt)": dict(eyeSize=0.16, eyePos=0.55, glabInflate=1.3, pleuralSpine=0.15, pygSpine=0.0, cephParallel=0.3),
}

def to_P(*vals):
    P = dict(T.P)
    for k, v in zip(KEYS, vals):
        P[k] = int(round(v)) if k in INT_KEYS else float(v)
    if P["nKnuckles"] % 2 == 0: P["nKnuckles"] += 1
    return P

def derived_md(*vals):
    P = to_P(*vals)
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
    md = (f"**pitch {d:.1f} mm · {joints} joints × {P['maxAngle']:g}° = total curl {total:.0f}°** · {pyg}  \n"
          f"head {P['cephFrac']*P['length']:.0f} · thorax {d*P['segCount']:.0f} · tail shield {P['pygFrac']*P['length']:.0f} mm · last segment {2*T.seg_halfwidth(P, P['segCount']-1):.0f} mm wide  \n"
          f"hinge z {zh:.1f} · width {Wh:.1f} · knuckle {knuckle:.1f} mm · pin wall {pin_wall:.1f} mm")
    if warns: md += "\n\n" + "\n".join(f"⚠️ {w}" for w in warns)
    else: md += "\n\n✅ no printability warnings"
    return md

# ---------------- build cache
CACHE, LOCK = {}, threading.Lock()
WORK = tempfile.mkdtemp(prefix="trilobite_")
COLORS = {"head": Color(0.86, 0.72, 0.42), "tail": Color(0.86, 0.72, 0.42)}

def key_of(P): return hashlib.md5(json.dumps(P, sort_keys=True).encode()).hexdigest()[:10]

def get_parts(P):
    k = key_of(P)
    with LOCK:
        if k not in CACHE:
            parts = T.parts_list(P)
            names = ["head"] + [f"seg{i}" for i in range(P["segCount"])] + ["tail"]
            for n, p in zip(names, parts):
                p.color = COLORS.get(n, Color(0.85, 0.77, 0.55))
            CACHE[k] = (parts, names)
            if len(CACHE) > 12: CACHE.pop(next(iter(CACHE)))
        return k, CACHE[k]

def pose_glb(P, enroll, k, parts):
    path = os.path.join(WORK, f"{k}_{int(enroll*1000)}.glb")
    if not os.path.exists(path):
        export_gltf(Compound(T.assemble(P, enroll, parts)), path, binary=True)
    return path

def build(enroll, *vals, progress=gr.Progress()):
    P = to_P(*vals)
    progress(0.05, desc="building parts (60–120 s on this CPU)…")
    t0 = time.time()
    k, (parts, names) = get_parts(P)
    progress(0.8, desc="posing + exporting…")
    glb = pose_glb(P, enroll, k, parts)
    zpath = os.path.join(WORK, f"{k}_stl.zip")
    if not os.path.exists(zpath):
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for n, p in zip(names, parts):
                f = os.path.join(WORK, f"{k}_{n}.stl"); export_stl(p, f); z.write(f, f"{n}.stl")
            f = os.path.join(WORK, f"{k}_enrolled.stl"); export_stl(Compound(T.assemble(P, 1.0, parts)), f); z.write(f, "animal_enrolled.stl")
    return glb, zpath, k, f"built in {time.time()-t0:.0f} s · {len(parts)} parts · enroll {enroll:.2f}"

def repose(enroll, k, *vals):
    if not k or k not in CACHE: return gr.update(), "build first"
    P = to_P(*vals)
    if key_of(P) != k: return gr.update(), "sliders changed since the last build — press Build"
    parts, _ = CACHE[k]
    return pose_glb(P, enroll, k, parts), f"enroll {enroll:.2f} → {enroll*P['maxAngle']:.1f}° per joint"

def check(*vals, progress=gr.Progress()):
    P = to_P(*vals)
    progress(0.1, desc="building (if needed) and testing 5 enroll values…")
    k, (parts, _) = get_parts(P)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.check_enrollment(P, parts=parts)
    return "```\n" + buf.getvalue() + "```"

def dump(*vals):
    P = to_P(*vals)
    return "```python\nP = dict(" + ", ".join(f"{k}={P[k]!r}" for k in P) + ")\n```"

def preset_values(name):
    P = dict(T.P); P.update(PRESETS[name])
    return [P[k] for k in KEYS]

# ---------------- UI
CSS = """
.gradio-container{max-width:1500px !important}
#model3d{min-height:560px}
"""
with gr.Blocks(css=CSS, title="Trilobite Morphospace Generator") as demo:
    gr.Markdown("## Trilobite Morphospace Generator\nEvery slider is one axis of the body plan. The readout is instant; **Build** runs the CAD (≈1–2 min); the enrollment slider then rolls the built animal without rebuilding. Joints are pin hinges with ventral stops; the collision check proves them.")
    with gr.Row():
        with gr.Column(scale=1, min_width=340):
            preset = gr.Dropdown(list(PRESETS), value="Reference (Weaver STL)", label="Preset")
            sliders = []
            group = None; acc = None
            for key, label, lo, hi, step, g in KNOBS:
                if g != group:
                    if acc is not None: acc.__exit__(None, None, None)
                    acc = gr.Accordion(g, open=(g == "Body")); acc.__enter__(); group = g
                sliders.append(gr.Slider(lo, hi, value=T.P[key], step=step, label=label))
            if acc is not None: acc.__exit__(None, None, None)
        with gr.Column(scale=2):
            readout = gr.Markdown(derived_md(*[T.P[k] for k in KEYS]))
            model = gr.Model3D(label="Built animal (drag to orbit)", elem_id="model3d", clear_color=(0.106, 0.125, 0.149, 1.0))
            enroll = gr.Slider(0, 1, value=0.0, step=0.01, label="Enrollment (0 = flat, 1 = at the stops)")
            status = gr.Markdown("not built yet")
            with gr.Row():
                build_btn = gr.Button("Build", variant="primary")
                check_btn = gr.Button("Check enrollment (collision test)")
                dump_btn = gr.Button("Show P as Python")
            stl_zip = gr.File(label="STLs (zip: every part + enrolled pose)")
            report = gr.Markdown("")
    key_state = gr.State("")

    for s in sliders:
        s.change(derived_md, sliders, readout, show_progress="hidden")
    preset.change(preset_values, preset, sliders)
    build_btn.click(build, [enroll] + sliders, [model, stl_zip, key_state, status])
    enroll.release(repose, [enroll, key_state] + sliders, [model, status], show_progress="hidden")
    check_btn.click(check, sliders, report)
    dump_btn.click(dump, sliders, report)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
