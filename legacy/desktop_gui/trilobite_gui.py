"""
trilobite_gui.py — the knobs.

A control panel for trilobite.py: every parameter as a slider AND a number box, a live readout of
the derived quantities and printability warnings (instant, no CAD), presets, and buttons that
rebuild the model into the OCP CAD Viewer, animate the roll, run the collision check, and export STLs.

Run:  python trilobite_gui.py      (with the OCP CAD Viewer panel open in VS Code)
"""
import threading, math, os
import tkinter as tk
from tkinter import ttk, messagebox
import trilobite as T

# ---------------------------------------------------------------- knob definitions
# (key, label, min, max, step, group)
KNOBS = [
    ("length",       "Body length (mm)",              60,   250,  1,     "Body"),
    ("width",        "Body width (mm)",               30,   140,  1,     "Body"),
    ("relief",       "Relief / dorsal height (mm)",   6,    40,   0.5,   "Body"),
    ("segCount",     "Thoracic segments",             2,    16,   1,     "Body"),
    ("taper",        "Taper per segment",             0.80, 1.00, 0.005, "Body"),
    ("axisFrac",     "Axial ring width fraction",     0.20, 0.50, 0.01,  "Body"),
    ("axisRise",     "Axial ring rise",               0.00, 0.40, 0.01,  "Body"),
    ("marginHeight", "Pleural margin height",         0.00, 0.60, 0.01,  "Body"),
    ("overlap",      "Shingle overlap (x pitch)",     0.10, 0.80, 0.01,  "Body"),
    ("cephFrac",     "Head length fraction",          0.18, 0.45, 0.005, "Head"),
    ("cephParallel", "Head: parallel rear fraction",  0.00, 0.80, 0.01,  "Head"),
    ("glabInflate",  "Glabella inflation",            0.70, 1.60, 0.01,  "Head"),
    ("eyeSize",      "Eye size",                      0.00, 0.20, 0.005, "Head"),
    ("eyePos",       "Eye position (rear→front)",     0.10, 0.90, 0.01,  "Head"),
    ("pygFrac",      "Tail shield length fraction",   0.05, 0.40, 0.005, "Tail"),
    ("pygWidth",     "Tail width (x last segment)",   0.50, 1.10, 0.01,  "Tail"),
    ("pygSpine",     "Tail spine length (x shield)",  0.00, 2.00, 0.05,  "Tail"),
    ("pygSplay",     "Tail spine splay (deg)",        0,    45,   1,     "Tail"),
    ("pleuralSpine", "Pleural spine length",          0.00, 1.00, 0.01,  "Spines"),
    ("spineSweep",   "Pleural spine sweep (deg)",     0,    80,   1,     "Spines"),
    ("maxAngle",     "Stop angle per joint (deg)",    4,    40,   0.5,   "Hinge"),
    ("clearance",    "Joint clearance (mm)",          0.15, 0.60, 0.01,  "Hinge"),
    ("wall",         "Shell wall (mm)",               1.2,  4.0,  0.1,   "Hinge"),
    ("barrelR",      "Knuckle radius (mm)",           1.8,  4.0,  0.1,   "Hinge"),
    ("nKnuckles",    "Knuckles (odd)",                3,    7,    2,     "Hinge"),
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

# ---------------------------------------------------------------- derived quantities & warnings
def derived(P):
    d = T.pitch(P)
    joints = P["segCount"] + 1
    total = joints * P["maxAngle"]
    zh, Wh = T.hinge_z(P), T.hinge_width(P)
    kw = Wh / P["nKnuckles"]
    knuckle_len = kw - P["clearance"]
    pin_wall = P["barrelR"] - P["boreDia"] / 2
    ratio = P["pygFrac"] / P["cephFrac"]
    pyg_class = "micropygous" if ratio < 0.6 else ("isopygous" if ratio < 1.15 else "macropygous")
    warns = []
    if P["wall"] < 1.5: warns.append("wall < 1.5 mm — fragile on FDM")
    if pin_wall < 1.2: warns.append(f"only {pin_wall:.1f} mm around the pin bore — raise knuckle radius")
    if knuckle_len < 3.0: warns.append(f"knuckles {knuckle_len:.1f} mm long — too small; fewer segments, wider axis, or fewer knuckles")
    if zh < P["barrelR"] + 3: warns.append("hinge axis too low — raise relief or lower knuckle radius")
    if d < 8: warns.append(f"segment pitch {d:.1f} mm — very short; fewer segments or longer body")
    if total > 360: warns.append(f"total curl {total:.0f}° > 360° — tail would pass the head; lower stop angle")
    if P["marginHeight"] * P["relief"] < P["wall"] + 0.5 and P["pleuralSpine"] > 0:
        warns.append("margin rim thinner than the spine base — spines will be weak")
    lines = [
        f"pitch {d:.1f} mm  |  {joints} joints × {P['maxAngle']:.1f}° = total curl {total:.0f}°"
        + ("  (partial)" if total < 300 else "  (near ball)" if total <= 360 else ""),
        f"hinge axis z = {zh:.1f} mm  |  hinge width {Wh:.1f} mm  |  knuckle {knuckle_len:.1f} mm  |  pin wall {pin_wall:.1f} mm",
        f"head {P['cephFrac']*P['length']:.0f} mm, thorax {d*P['segCount']:.0f} mm, tail shield {P['pygFrac']*P['length']:.0f} mm  →  {pyg_class}",
        f"last segment width {2*T.seg_halfwidth(P, P['segCount']-1):.0f} mm (taper {P['taper']**P['segCount']:.2f} overall)",
    ]
    return lines, warns

# ---------------------------------------------------------------- GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title("Trilobite Morphospace — parametric controls")
        root.configure(bg="#20262e")
        self.P = dict(T.P)
        self.vars = {}
        self.parts = None
        self.busy = False

        style = ttk.Style(root)
        try: style.theme_use("clam")
        except tk.TclError: pass
        style.configure(".", background="#20262e", foreground="#e8e2d4", fieldbackground="#1b2026")
        style.configure("TLabel", background="#20262e", foreground="#e8e2d4")
        style.configure("Group.TLabel", foreground="#4fc4d6", font=("Segoe UI", 9, "bold"))
        style.configure("Warn.TLabel", foreground="#f0a35a")
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", background="#4fc4d6", foreground="#08222a")

        left = ttk.Frame(root, padding=10); left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(root, padding=10); right.grid(row=0, column=1, sticky="nsew")
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

        # presets
        ttk.Label(left, text="Preset", style="Group.TLabel").grid(row=0, column=0, sticky="w")
        self.preset = ttk.Combobox(left, values=list(PRESETS), state="readonly", width=32)
        self.preset.set("Reference (Weaver STL)")
        self.preset.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 6))
        self.preset.bind("<<ComboboxSelected>>", self.apply_preset)

        # knobs, grouped
        r = 1; last_group = None
        for key, label, lo, hi, step, group in KNOBS:
            if group != last_group:
                ttk.Label(left, text=group, style="Group.TLabel").grid(row=r, column=0, sticky="w", pady=(8, 0)); r += 1
                last_group = group
            var = tk.DoubleVar(value=self.P[key]); self.vars[key] = var
            ttk.Label(left, text=label, width=30).grid(row=r, column=0, sticky="w")
            s = ttk.Scale(left, from_=lo, to=hi, variable=var, orient="horizontal", length=220,
                          command=lambda v, k=key, st=step: self.on_slide(k, st))
            s.grid(row=r, column=1, sticky="ew", padx=6)
            e = ttk.Entry(left, width=8); e.insert(0, self.fmt(key, self.P[key]))
            e.grid(row=r, column=2); e.bind("<Return>", lambda ev, k=key, en=e: self.on_entry(k, en))
            e.bind("<FocusOut>", lambda ev, k=key, en=e: self.on_entry(k, en))
            setattr(self, f"entry_{key}", e)
            r += 1

        # readout + actions
        ttk.Label(right, text="Live readout (no CAD, instant)", style="Group.TLabel").grid(row=0, column=0, sticky="w")
        self.readout = tk.Text(right, width=64, height=7, bg="#1b2026", fg="#e8e2d4", relief="flat", font=("Consolas", 9))
        self.readout.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        self.warnbox = tk.Text(right, width=64, height=6, bg="#1b2026", fg="#f0a35a", relief="flat", font=("Consolas", 9))
        self.warnbox.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        btns = ttk.Frame(right); btns.grid(row=3, column=0, sticky="ew")
        ttk.Button(btns, text="Rebuild + show animal", command=lambda: self.run(self.do_rebuild)).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btns, text="Preview one segment (fast)", command=lambda: self.run(self.do_preview)).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btns, text="Animate roll", command=lambda: self.run(self.do_animate)).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btns, text="Check enrollment (collisions)", command=lambda: self.run(self.do_check)).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btns, text="Export STLs", command=lambda: self.run(self.do_export)).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btns, text="Print P (copy into trilobite.py)", command=self.print_P).grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        btns.columnconfigure(0, weight=1); btns.columnconfigure(1, weight=1)

        self.status = ttk.Label(right, text="ready"); self.status.grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.log = tk.Text(right, width=64, height=12, bg="#1b2026", fg="#9aa3ac", relief="flat", font=("Consolas", 9))
        self.log.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        self.update_readout()

    # ---- helpers
    def fmt(self, key, v):
        return str(int(round(v))) if key in INT_KEYS else f"{v:g}"

    def set_param(self, key, v, step):
        v = round(v / step) * step
        if key in INT_KEYS:
            v = int(round(v))
            if key == "nKnuckles" and v % 2 == 0: v += 1
        self.P[key] = v
        self.vars[key].set(v)
        e = getattr(self, f"entry_{key}"); e.delete(0, "end"); e.insert(0, self.fmt(key, v))
        self.update_readout()

    def on_slide(self, key, step):
        self.set_param(key, self.vars[key].get(), step)

    def on_entry(self, key, entry):
        try: v = float(entry.get())
        except ValueError: return
        step = next(k[4] for k in KNOBS if k[0] == key)
        self.set_param(key, v, step)

    def apply_preset(self, _=None):
        self.P = dict(T.P); self.P.update(PRESETS[self.preset.get()])
        for key, label, lo, hi, step, group in KNOBS:
            self.set_param(key, self.P[key], step)

    def update_readout(self):
        lines, warns = derived(self.P)
        self.readout.delete("1.0", "end"); self.readout.insert("end", "\n".join(lines))
        self.warnbox.delete("1.0", "end"); self.warnbox.insert("end", "\n".join("⚠ " + w for w in warns) or "no printability warnings")
        self.parts = None                       # geometry is stale

    def logln(self, s):
        self.log.insert("end", s + "\n"); self.log.see("end")

    def print_P(self):
        self.logln("P = dict(" + ", ".join(f"{k}={self.fmt(k, v) if k in INT_KEYS else round(v, 4)}" for k, v in self.P.items()) + ")")

    # ---- long-running actions run in a thread so the sliders stay responsive
    def run(self, fn):
        if self.busy:
            self.logln("busy — wait for the current build to finish"); return
        self.busy = True
        def go():
            try: fn()
            except Exception as ex: self.root.after(0, lambda: self.logln(f"error: {ex}"))
            finally:
                self.busy = False; self.root.after(0, lambda: self.status.config(text="ready"))
        threading.Thread(target=go, daemon=True).start()

    def viewer(self):
        from ocp_vscode import show, set_port
        set_port(T.VIEWER_PORT)
        return show

    def ensure_parts(self):
        if self.parts is None:
            self.root.after(0, lambda: self.status.config(text="building whole animal (~30–60 s)…"))
            self.parts = T.parts_list(self.P)
        return self.parts

    def do_preview(self):
        self.root.after(0, lambda: self.status.config(text="building one segment…"))
        seg = T.build_segment(self.P, 2)
        self.viewer()(seg)
        self.root.after(0, lambda: self.logln(f"segment: valid={seg.is_valid} size={tuple(round(v,1) for v in seg.bounding_box().size)}"))

    def do_rebuild(self):
        parts = self.ensure_parts()
        self.viewer()(*T.assemble(self.P, 0.0, parts))
        self.root.after(0, lambda: self.logln("animal shown (flat pose)"))

    def do_animate(self):
        parts = self.ensure_parts()
        self.root.after(0, lambda: self.status.config(text="sending animation…"))
        T.animate_enrollment(self.P, parts)
        self.root.after(0, lambda: self.logln("animation sent — use the viewer's play/scrub bar"))

    def do_check(self):
        parts = self.ensure_parts()
        self.root.after(0, lambda: self.status.config(text="checking collisions at 5 enroll values…"))
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.check_enrollment(self.P, parts=parts)
        self.root.after(0, lambda: self.logln(buf.getvalue().rstrip()))

    def do_export(self):
        parts = self.ensure_parts()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl")
        os.makedirs(out, exist_ok=True)
        T.export_stl(parts[0], os.path.join(out, "cephalon.stl"))
        T.export_stl(parts[-1], os.path.join(out, "pygidium.stl"))
        for i in range(self.P["segCount"]):
            T.export_stl(parts[1 + i], os.path.join(out, f"segment{i}.stl"))
        T.export_stl(T.Compound(T.assemble(self.P, 1.0, parts)), os.path.join(out, "animal_enrolled.stl"))
        self.root.after(0, lambda: self.logln(f"exported {self.P['segCount'] + 3} STLs to {out}"))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
