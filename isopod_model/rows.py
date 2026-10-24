"""
isopod_model/rows.py (10 Oct 2026): the body as ONE LIST OF ROWS, head to tail tip, for the body editor.

A row is a tergite: its own length (pitch), width, pleural trajectory (sweep, curve, tip), height and ring. The
thorax rows and the pygidial rings are the same thing; the pygidium is the trailing rows that have fused. Rows fuse
by a rule, not by hand: a row too short for the isopod joint at the print scale has no joint on either side.

    ROW_KEYS                      the keys a row has, with what they mean
    rows_from_P(P)                the preset's rows (segCount thorax rows + pygRings pygidial rows), every edit "off"
    apply(rows, edits)            {"rows": {"3": {"len": 9.5}}, ...} on top
    groups(rows, scale, min_mm)   the fusion: [[0], [1], ..., [9, 10, 11, 12]] (row indices per printed piece)
    tergite(row, nxt, last)       outline(u, v), zfun(x, y) of one row, in the row's frame (y = 0 its front joint
                                  plane): a smooth slab, no front drop, no per-row hump, the plate over the next
                                  row 'overlap' of a pitch long. Print model only.
    group_plan(rows, g, ...)      one plan for a fused group: the rows' slabs end to end, furrows between
    plans(P, spec)                [(name, mod, S, offset)] in body.py's shape, plus the pivots and the print scale
    diagram(P, spec)              the drawing for the editor: outline polylines per row, groups, pivots, handles

TWO MODELS (spec "model"):
  "thorax" (default, 10 Oct 2026 evening): the rows drive the GENERATOR'S OWN segment and tail plans (anatomy/thorax.py,
      anatomy/tail.py) -- the surfaces the isopod build already prints. Each thoracic row becomes thorax.plan(P_i, i)
      with P_i = the preset overridden by the row (its pitch, half-width, axis, relief, margin, ring arch, fulcrum,
      slope, sweep, spine); the pygidial rows together become tail.plan(P_t) (their total length, the first row's
      width and relief, one rib per row). An unedited preset gives the preset's own plans, so the solid is the one
      the site already generates. Rows too short for the joint fuse into one piece (the generator's surfaces end to
      end). Keys the generator's segment has no use for (tip_w, rear_round) are ignored in this model.
  "slab": the tergite slab (below), the first body editor's surface.

Nothing here touches anatomy/, joints/, instrument.py or mesh.py: the pin pipeline is untouched. Defaults are taken
from the same schema parameters thorax.plan and tail.plan read, so an animal with no edits is the preset.
"""
import math
import numpy as np
from anatomy.common import pitch, smoothstep, plateau, trough
from anatomy import thorax as THORAX, tail as TAIL
import fields as _F
from fields import seg_halfwidth, pleural_spine_field, furrow_amp, tail_halfwidth, landmarks

# The thorax model hands thorax.plan / tail.plan a P with two private keys: "_w" (this row's half-width) and "_spines"
# (per-segment spine fractions). Those modules read the width and spine fields through their own module globals, so
# the fields are wrapped here to honour the keys; without the keys they are the fields themselves (bit-identical).
def _seg_halfwidth(P, i): return float(P["_w"]) if "_w" in P else _F.seg_halfwidth(P, i)
def _tail_halfwidth(P): return float(P["_w"]) if "_w" in P else _F.tail_halfwidth(P)
def _spine_field(P): return P["_spines"] if "_spines" in P else _F.pleural_spine_field(P)
THORAX.seg_halfwidth = _seg_halfwidth; THORAX.pleural_spine_field = _spine_field; TAIL.tail_halfwidth = _tail_halfwidth

ROW_KEYS = dict(
    len=("mm", "the row's pitch: front joint plane to rear joint plane"),
    width=("mm", "half-width at the pleural edge (where the blade ends and the spine, if any, starts)"),
    axis=("frac", "axial ring half-width / width"),
    relief=("mm", "height of the ring's top over the bed"),
    margin=("frac", "height at the pleural edge / relief"),
    sweep=("deg", "angle the pleura leaves the axis at: 0 straight across, + swept back"),
    curve=("deg", "extra bend over the blade's length: + curls back toward the tail (thorax model: added to the pleural spine's angle)"),
    tip=("mm", "pleural spine past the pleural edge (0 = none)"),
    tip_w=("frac", "spine width at its root / the row's length (slab model only)"),
    arch=("frac", "ring arch across the axis / relief"),
    fulcrum=("frac", "where the pleura bends down, / (width - axis)"),
    slope=("frac", "how far the pleura drops from the fulcrum to the edge / relief"),
    rear_round=("frac", "the rear margin's roundness (pygidial end): 0 straight across, 1 a half-ellipse (slab model only)"),
    # 24 Oct 2026: the row's own axial spine (the studio's tip and mid dots in ROWS SIDE). ax_own 0 = the preset's axial spine
    # (P.axialSpine / Pitch / Bend / Grad) as on every row; 1 = this row's ax_len / ax_pitch / ax_bend instead.
    ax_own=("0/1", "1 = this row's axial spine is its own (ax_len, ax_pitch, ax_bend); 0 = the preset's"),
    ax_len=("mm", "this row's axial spine length (with ax_own; 0 = none)"),
    ax_pitch=("deg", "how steeply it leaves the ring (with ax_own)"),
    ax_bend=("deg", "its arc, root to tip (with ax_own; + recurves the tip back)"),
    kind=("", "'thorax' or 'pyg' (which block the preset put it in; a label, not a rule)"),
)
ROW_EXTRA = dict(ax_own=0.0, ax_len=0.0, ax_pitch=60.0, ax_bend=0.0)   # optional keys: a row without them gets these (saved bodies predate them)
BODY_KEYS = dict(
    model=("", "'thorax': the generator's own segments and tail, as the site already prints them (default); 'slab': the tergite slab"),
    thorax_rows=("n", "number of thoracic rows (the preset's segCount); rows are re-spaced over the thorax length"),
    pyg_rows=("n", "number of pygidial rows (ribs); re-spaced over the pygidium length"),
    thorax_len=("mm", "length of the thorax (all thoracic rows together)"),
    pyg_len=("mm", "length of the pygidium (all pygidial rows together)"),
    term_spine=("mm", "a terminal spine from the pygidium's tip (0 = none)"),
    min_joint_mm=("mm", "the joint minimum at print: a row shorter than this fuses"),
    print_scale=("x", "print mm per model mm (default: 18 mm per median thoracic row)"),
)
MIN_JOINT_MM = 17.0     # the isopod's own pieces are 17.5 mm pitch at 1:1 print and carry a socket half and a ball half
                        # each; a printed piece shorter than this cannot (the kit halves would meet inside it).
OVERLAP = 0.45          # the plate over the next row, / the next row's pitch (the isopod's 8 / 18)


def rows_from_P(P, spec=None, model="thorax"):
    """The preset as rows. Thorax rows from thorax.plan's scalars; pygidial rows from tail.plan's ellipse. The body
    keys of `spec` (thorax_rows, pyg_rows, thorax_len, pyg_len) change how many rows there are and over what length
    BEFORE any per-row edit: the preset's width and spine fields are re-sampled over the new rows."""
    spec = spec or {}
    d0 = pitch(P); h = float(P["relief"]); F = furrow_amp(P)
    for k, (lo, hi) in BODY_RANGE.items():
        if spec.get(k) is not None: _num(spec[k], k, lo, hi, "body")
    n = int(spec.get("thorax_rows") or P["segCount"]); Lt = float(spec.get("thorax_len") or d0 * int(P["segCount"])); d = Lt / n
    from fields import width_curve
    wc = width_curve(P); L = landmarks(P)
    spines0 = pleural_spine_field(P); n0 = int(P["segCount"])
    rows = []
    for i in range(n):
        s_i = L["s_h"] + (i + 0.5) / n * (L["s_t"] - L["s_h"])           # the preset's width field, re-sampled
        w = float(wc(s_i)); a = P["axisFrac"] * w
        spine = float(np.interp((i + 0.5) / n, [(k + 0.5) / n0 for k in range(n0)], spines0)) if n0 > 1 else spines0[0]
        sweep = math.degrees(math.atan2(P["tipSweep"] * d, max(w - a, 1e-6)))
        rows.append(dict(len=d, width=w, axis=P["axisFrac"], relief=h, margin=P["marginHeight"], sweep=sweep, curve=0.0,
                         tip=spine * w, tip_w=0.55, arch=P["ringArch"], fulcrum=P["fulcrum"], slope=P["pleuralSlope"],
                         rear_round=0.0, kind="thorax"))
    # pygidium: pygRings rows across pygFrac x length; widths on the tail's ellipse; the last carries the margin
    # the pygidial rows: at least pygRings of them, and at most 0.55 of a thoracic pitch each, so that at the preset's
    # own print scale they are below the joint minimum and the pygidium comes out as ONE fused piece (a real
    # pygidium's ribs are far closer than its thoracic segments; the schema's pygRings only counts the drawn ribs)
    Lp = float(spec.get("pyg_len") or P["pygFrac"] * P["length"])
    m = int(spec.get("pyg_rows") or (max(int(P["pygRings"]), 1) if model == "thorax" else max(int(P["pygRings"]), int(math.ceil(Lp / (0.55 * d0))), 1)))
    wp = tail_halfwidth(P); hp = h * P["tailRelief"]
    dp = Lp / m
    for j in range(m):
        yc = (j + 0.5) * dp; w = max(wp * math.sqrt(max(1 - (yc / Lp) ** 2, 0.0)), 1.5)
        rows.append(dict(len=dp, width=w, axis=P["axisFrac"], relief=hp * (1 - 0.35 * j / m), margin=P["marginHeight"],
                         sweep=rows[-1]["sweep"] if rows else 20.0, curve=0.0, tip=0.0, tip_w=0.55, arch=P["ringArch"] * 0.6,
                         fulcrum=P["fulcrum"], slope=P["pleuralSlope"], rear_round=(1.0 if j == m - 1 else 0.0), kind="pyg"))
    return rows


def boring(P=None, n_thorax=9, n_pyg=6):
    """The editor's starting animal: a plain trilobite. Uniform thoracic rows (the default pitch and relief), a gentle
    even taper, pleurae swept 15 deg, no spines, and a short pygidium of n_pyg fused rows ending in a half-ellipse."""
    import schema
    P = P or schema.defaults(); d = pitch(P); h = float(P["relief"]); w0 = 0.5 * P["width"]
    rows = []
    for i in range(n_thorax):
        rows.append(dict(len=d, width=w0 * (1 - 0.35 * i / max(n_thorax - 1, 1)), axis=0.3, relief=h, margin=0.16, sweep=15.0, curve=0.0,
                         tip=0.0, tip_w=0.55, arch=0.15, fulcrum=0.55, slope=0.7, rear_round=0.0, kind="thorax"))
    wp = rows[-1]["width"] * 0.95; dp = 0.5 * d
    for j in range(n_pyg):
        t = (j + 0.5) / n_pyg
        rows.append(dict(len=dp, width=max(wp * math.sqrt(max(1 - t * t, 0.0)), 2.0), axis=0.3, relief=h * (1 - 0.3 * t), margin=0.16, sweep=15.0,
                         curve=0.0, tip=0.0, tip_w=0.55, arch=0.1, fulcrum=0.55, slope=0.7, rear_round=(1.0 if j == n_pyg - 1 else 0.0), kind="pyg"))
    return rows


# the range every row / body number must lie in (model mm, deg, fractions): what the surfaces are drawn for, and a
# ceiling on what a build may cost. Outside it the spec is refused (400), never built.
ROW_RANGE = dict(len=(0.2, 500.0), width=(0.5, 400.0), axis=(0.05, 0.9), relief=(0.5, 200.0), margin=(0.0, 1.0), sweep=(-85.0, 85.0),
                 curve=(-85.0, 85.0), tip=(0.0, 400.0), tip_w=(0.02, 5.0), arch=(0.0, 2.0), fulcrum=(0.02, 0.98), slope=(0.0, 3.0), rear_round=(0.0, 1.0),
                 ax_own=(0.0, 1.0), ax_len=(0.0, 500.0), ax_pitch=(0.0, 90.0), ax_bend=(-180.0, 180.0))
BODY_RANGE = dict(thorax_rows=(1, 80), pyg_rows=(0, 80), thorax_len=(0.5, 2000.0), pyg_len=(0.0, 2000.0), term_spine=(0.0, 500.0),
                  min_joint_mm=(1.0, 100.0), print_scale=(0.1, 20.0))
MAX_ROWS = 80


def _num(v, k, lo, hi, where):
    try: v = float(v)
    except (TypeError, ValueError): raise ValueError(f"{where}: {k} is not a number")
    if not math.isfinite(v): raise ValueError(f"{where}: {k} is not finite")
    if not lo <= v <= hi: raise ValueError(f"{where}: {k} = {v:g} is outside {lo:g}..{hi:g}")
    return v


def check_row(row, where="row"):
    """A full row: kinds and every number inside ROW_RANGE."""
    if row.get("kind") not in ("thorax", "pyg"): raise ValueError(f"{where}: kind must be 'thorax' or 'pyg'")
    for k, v in ROW_EXTRA.items(): row.setdefault(k, v)
    for k, (lo, hi) in ROW_RANGE.items(): row[k] = _num(row.get(k), k, lo, hi, where)
    return row


def validate_list(lst):
    """An explicit row list (the editor's +/-): every entry a full row (missing keys filled from the row before it, or
    the boring thoracic row), kinds 'thorax' / 'pyg', every number finite and inside ROW_RANGE, at most MAX_ROWS."""
    if not isinstance(lst, list) or not lst: raise ValueError("the row list is empty")
    if len(lst) > MAX_ROWS: raise ValueError(f"{len(lst)} rows: at most {MAX_ROWS}")
    base = boring()[0]; out = []
    for i, r in enumerate(lst):
        if not isinstance(r, dict): raise ValueError(f"row {i} is not a row")
        bad = set(r) - set(ROW_KEYS)
        if bad: raise KeyError(f"row {i}: unknown row keys {sorted(bad)} (have {sorted(ROW_KEYS)})")
        prev = out[-1] if out else base
        row = dict(prev); row.update(ROW_EXTRA)                       # a row's OWN spine is never inherited from the row before (only its shape keys are)
        row.update({k: (str(v) if k == "kind" else v) for k, v in r.items()})
        out.append(check_row(row, f"row {i}"))
    seen_pyg = False                                               # kinds must be contiguous: all thorax, then all pyg
    for i, r in enumerate(out):
        if r["kind"] == "pyg": seen_pyg = True
        elif seen_pyg: raise ValueError(f"row {i}: a thoracic row cannot come after a pygidial row (order the rows thorax..., then pyg...)")
    return out


def apply(rows, edits=None):
    """edits: {"rows": {"<i>": {key: value}}, "list": [full rows] (the editor's explicit list: REPLACES rows and
    ignores thorax_rows / pyg_rows / thorax_len / pyg_len), + the BODY_KEYS}. Unknown keys raise KeyError."""
    edits = edits or {}
    bad = set(edits) - set(BODY_KEYS) - {"rows", "list"}
    if bad: raise KeyError(f"unknown body keys {sorted(bad)} (have {sorted(BODY_KEYS)}, 'rows' and 'list')")
    if edits.get("model") not in (None, "thorax", "slab"): raise ValueError("model must be 'thorax' or 'slab'")
    for k, (lo, hi) in BODY_RANGE.items():
        if edits.get(k) is not None: _num(edits[k], k, lo, hi, "body")
    if edits.get("list") is not None: rows = validate_list(edits["list"])
    out = [dict(r) for r in rows]
    if len(out) > MAX_ROWS: raise ValueError(f"{len(out)} rows: at most {MAX_ROWS}")
    for k, e in (edits.get("rows") or {}).items():
        try: i = int(k)
        except (TypeError, ValueError): raise KeyError(f"row '{k}': not a row index")
        if not 0 <= i < len(out): raise KeyError(f"row {i}: the body has rows 0..{len(out) - 1}")
        if not isinstance(e, dict): raise ValueError(f"row {i}: edits must be an object")
        bad = set(e) - set(ROW_KEYS)
        if bad: raise KeyError(f"unknown row keys {sorted(bad)} (have {sorted(ROW_KEYS)})")
        for kk, v in e.items(): out[i][kk] = (str(v) if kk == "kind" else v)
        check_row(out[i], f"row {i}")
    return out


def groups(rows, scale, min_mm=MIN_JOINT_MM):
    """The fusion rule. A row shorter than min_mm at the print scale is 'small': small rows fuse with their small
    neighbours into one piece, and a piece still shorter than min_mm fuses into the piece behind it (ahead for the
    last). A joint sits at every remaining boundary. Returns [[row indices], ...] in order."""
    small = [r["len"] * scale < min_mm for r in rows]
    gs = []
    for i, s in enumerate(small):
        if gs and s and small[gs[-1][-1]]: gs[-1].append(i)
        else: gs.append([i])
    def length(g): return sum(rows[i]["len"] for i in g) * scale
    changed = True
    while changed and len(gs) > 1:
        changed = False
        for k, g in enumerate(gs):
            if length(g) < min_mm:
                j = k + 1 if k + 1 < len(gs) else k - 1
                a, b = (k, j) if k < j else (j, k)
                gs[a:b + 1] = [gs[a] + gs[b]]; changed = True; break
    return gs


KIT_HALF_W, KIT_H = 15.0, 13.5   # the isopod's middle strip at print: half-width, and the height of ball + socket (asset mm)
PRINT_MAX_MM = 700.0             # longest print side allowed (the export refuses 800; the bed is smaller than either)


def problems(rows, scale, term_spine=0.0):
    """Why this body cannot be BUILT, as messages (empty = fine). The joint kit is absolute (print mm) and sits inside
    every thoracic row: at this print scale the rows must be wider than its strip and taller than its socket. And
    the whole print must fit a bed. The editor draws such a body and shows the message; the build refuses it.
    11 Oct 2026: a 2 mm wide row crashed in the head, a 0.5 mm high one let the kit through its plate at every
    joint, 1.5 mm rows printed a metre long."""
    th = [r for r in rows if r["kind"] == "thorax"] or rows; out = []
    w_min = min(r["width"] for r in th); h_min = min(r["relief"] for r in th)
    if w_min * scale < KIT_HALF_W: out.append(f"the joint kit is {KIT_HALF_W:g} mm wide at print; the narrowest thoracic row is {w_min * scale:.1f} mm (raise print_scale or widen the row)")
    if h_min * scale < KIT_H: out.append(f"the joint kit is {KIT_H:g} mm tall at print; the lowest thoracic row is {h_min * scale:.1f} mm (raise print_scale or the row's relief)")
    L = (sum(r["len"] for r in rows) + max(term_spine, 0.0)) * scale; W = 2 * max(r["width"] + r["tip"] for r in rows) * scale
    if max(L, W) > PRINT_MAX_MM: out.append(f"the print would be {L:.0f} x {W:.0f} mm; at most {PRINT_MAX_MM:g} (lower print_scale, the rows or the terminal spine)")
    return out


def fits(rows, scale, term_spine=0.0):
    """problems() as a ValueError (the build path)."""
    p = problems(rows, scale, term_spine)
    if p: raise ValueError("; ".join(p))


def default_scale(rows):
    """18 mm (the isopod's pitch) per median thorax row: the print scale an unedited preset gets, as before."""
    th = [r["len"] for r in rows if r["kind"] == "thorax"] or [r["len"] for r in rows]
    return 18.0 / float(np.median(th))


# ---------------------------------------------------------------- the tergite slab
def _edges_fn(row, y_front=0.0, nxt=None):
    """(yf, yr)(x): the row's front and rear edges at |x|, in the row's frame. The rear edge carries the plate over
    the next row: OVERLAP of the NEXT row's pitch (its own, for the last). A pleural spine of length `tip` continues
    past the pleural edge, tapering to a point."""
    d = row["len"]; w = row["width"]; a = row["axis"] * w
    ovl = OVERLAP * (nxt["len"] if nxt else d)
    sw, cv = math.radians(row["sweep"]), math.radians(row["curve"])
    tip = max(row["tip"], 0.0); tw = max(row["tip_w"], 0.05) * d
    X_tip = w + tip * math.cos(sw + cv)
    def back(ax):                                                  # how far back the pleura has swept at |x|
        q = np.clip((ax - a) / max(w - a, 1e-6), 0, 1)
        return np.tan(sw) * q * (w - a) + np.tan(cv) * q * q * (w - a) * 0.5
    def edges(x):
        ax = np.abs(np.asarray(x, float)); b = back(ax)
        yf = y_front + b; yr = y_front + d + ovl + b
        if tip > 0.05:                                             # the spine: a tapering chord beyond the pleural edge
            p = np.clip((ax - w) / max(X_tip - w, 1e-6), 0, 1); bt = np.tan(sw + cv) * np.maximum(ax - w, 0)
            yc = 0.5 * (yf + yr) + bt; half = 0.5 * tw * (1 - p) ** 0.8 + 0.35 * p
            yf = np.where(ax > w, yc - half, yf); yr = np.where(ax > w, yc + half, yr)
        if row["rear_round"] > 0.01:                               # the pygidial margin: the rear edge an ellipse
            rr = row["rear_round"] * 0.6 * w
            yr = np.where(ax <= w, yr - rr + rr * np.sqrt(np.clip(1 - (ax / max(w, 1e-6)) ** 2, 0, 1)), yr)
        return yf, yr
    return edges, X_tip


def tergite(row, nxt=None, y_front=0.0):
    """One row's slab: outline(u, v) and zfun(x, y) in its frame. The top is a transverse vault (ring plateau + pleural
    drop), rising slightly to the rear so the plate over the next row wedges up as the isopod's does. No front drop,
    no hump along the row: that was the pin shingle."""
    d = row["len"]; w = row["width"]; a = row["axis"] * w; h = row["relief"]
    margin = row["margin"] * h; edges, X_tip = _edges_fn(row, y_front, nxt)
    def outline(u, v):
        x = u * X_tip; yf, yr = edges(np.array([abs(x)]))
        return x, float(yf[0] + v * (yr[0] - yf[0]))
    def zfun(x, y):
        ax = np.abs(np.asarray(x, float)); y = np.asarray(y, float)
        q = np.clip((ax - a) / max(w - a, 1e-6), 0, 1)
        f = max(min(row["fulcrum"], 0.95), 0.05)
        inner = 1 - 0.25 * (np.minimum(q / f, 1.0)) ** 2            # the inner run: nearly flat
        drop = row["slope"] * np.clip((q - f) / (1 - f), 0, 1) ** 1.4
        side = h * inner * (1 - drop)
        z = np.maximum(side, margin)
        z += row["arch"] * h * plateau(ax, a, 0.6 * max(a, 1.0) + 0.5) * 0.6
        z += 0.06 * h * np.clip((y - y_front) / max(d, 1e-6), 0, 1.6)   # rises to the rear: the plate's wedge
        if row["tip"] > 0.05:                                      # the spine: a low ridge tapering to the tip
            p = np.clip((ax - w) / max(X_tip - w, 1e-6), 0, 1)
            z = np.where(ax > w, np.maximum(margin * (1 - 0.5 * p), 2.6), z)
        return z
    return dict(outline=outline, zfun=zfun, edges=edges, X_tip=X_tip, d=d, w=w, a=a, h=h, margin=margin, rise=0.0,
                ovl=OVERLAP * (nxt["len"] if nxt else d))


class _Mod:
    """What body.py expects of a slab part's module: ornaments(P, S) (none on slabs) and a name. The thorax model's
    parts carry anatomy.thorax / anatomy.tail themselves, so their spines are the preset's."""
    __name__ = "rows"
    @staticmethod
    def ornaments(P, S): return []


def _shift(T, y0):
    """T (edges, zfun, outline in its own frame, y = 0 its front) moved to y0 in the group frame."""
    if abs(y0) < 1e-12: return T
    S = dict(T)
    S["edges"] = lambda x: tuple(e + y0 for e in T["edges"](x))
    S["zfun"] = lambda x, y: T["zfun"](x, np.asarray(y, float) - y0)
    S["outline"] = lambda u, v: (lambda xy: (xy[0], xy[1] + y0))(T["outline"](u, v))
    return S


def group_plan(rows, g, F=1.0, maker=None, mod=None):
    """One plan for a fused group: the rows' surfaces end to end in the FIRST row's frame, the group's plate over the
    next row from its last row. maker(i, n, is_last_in_group) -> the row's plan in its own frame (default: the slab).
    A single-row group is the row's own plan, untouched."""
    maker = maker or (lambda i, n, last: tergite(rows[i], (rows[i + 1] if i + 1 < len(rows) else None) if last else rows[g[n + 1]], 0.0))
    y0 = 0.0; parts = []
    for n, i in enumerate(g):
        T = maker(i, n, n == len(g) - 1); parts.append((y0, _shift(T, y0))); y0 += rows[i]["len"]
    if len(g) == 1:
        S = dict(parts[0][1]); S["rows"] = list(g); return S
    first, last = parts[0][1], parts[-1][1]
    X_tip = max(p[1]["X_tip"] for p in parts)
    bounds = [p[0] for p in parts[1:]]                             # the internal joint planes, now furrows
    def edges(x):
        """The group's outline tapers with its rows: at |x| the front edge is the FIRST row that reaches |x|, the rear
        edge the LAST row that does (a fused pygidium narrows to its tip instead of keeping the first rib's width)."""
        ax = np.abs(np.asarray(x, float)); yf = np.full(ax.shape, np.nan); yr = np.full(ax.shape, np.nan)
        for y0_, T in parts:
            reach = ax <= T["X_tip"] + 1e-9; f, r = T["edges"](ax)
            yf = np.where(reach & np.isnan(yf), f, yf); yr = np.where(reach, r, yr)
        yf = np.where(np.isnan(yf), first["edges"](ax)[0], yf); yr = np.where(np.isnan(yr), yf + 0.5, yr)
        return yf, np.maximum(yr, yf + 0.5)
    def outline(u, v):
        x = u * X_tip; yf, yr = edges(np.array([abs(x)]))
        return x, float(yf[0] + v * (yr[0] - yf[0]))
    def zfun(x, y):
        y = np.asarray(y, float); z = np.zeros_like(np.asarray(x, float) + y)
        for (ya, T), (yb, _) in zip(parts, parts[1:] + [(np.inf, None)]):
            sel = (y >= ya) & (y < yb);
            if np.any(sel): z = np.where(sel, T["zfun"](x, y), z)
        for yb in bounds: z = z - 0.55 * F * trough(y - yb, 0.9)   # the furrow between fused rows (slab; 0 for the thorax model)
        return z
    d = sum(rows[i]["len"] for i in g)
    return dict(outline=outline, zfun=zfun, edges=edges, X_tip=X_tip, d=d, w=max(p[1]["w"] for p in parts), a=first["a"],
                h=max(p[1]["h"] for p in parts), margin=first["margin"], rise=first.get("rise", 0.0), ovl=last["ovl"], t=first.get("t", 2.0), rows=list(g))


# ---------------------------------------------------------------- the thorax model: the generator's own plans
def _P_row(P, row, i, inner=False, n_rows=None):
    """The preset overridden by one thoracic row, for thorax.plan(P_i, i). inner: a row inside a fused group -- its
    front drop (the lip under the plate ahead) becomes a shallow furrow instead."""
    Pi = dict(P); n = int(P["segCount"]); fr = max(1.0 - P["cephFrac"] - P["pygFrac"], 1e-6)
    Pi["length"] = row["len"] * n / fr                                 # pitch(Pi) == the row's length
    w = max(row["width"], 1.0); a = row["axis"] * w
    Pi["_w"] = w; Pi["axisFrac"] = row["axis"]; Pi["relief"] = row["relief"]; Pi["marginHeight"] = row["margin"]
    Pi["ringArch"] = row["arch"]; Pi["fulcrum"] = min(max(row["fulcrum"], 0.05), 0.95); Pi["pleuralSlope"] = row["slope"]
    Pi["tipSweep"] = math.tan(math.radians(max(min(row["sweep"], 80.0), -80.0))) * max(w - a, 1e-6) / row["len"]
    sp = np.zeros(max(n, (n_rows or 0), i + 1)); sp[i] = max(row["tip"], 0.0) / w; Pi["_spines"] = sp
    Pi["spineSweep"] = P["spineSweep"] + row["curve"]
    if float(row.get("ax_own", 0.0)) >= 0.5:                               # the row's own axial spine (24 Oct 2026)
        Pi["axialSpine"] = float(row.get("ax_len", 0.0)) / max(float(row["relief"]), 1e-6); Pi["axialGrad"] = 0.0
        Pi["axialPitch"] = float(row.get("ax_pitch", 60.0)); Pi["axialBend"] = float(row.get("ax_bend", 0.0))
    if inner: Pi["clearance"] = 0.0; Pi["wall"] = min(P["wall"], 0.6)
    return Pi


def thorax_row(P, row, i, inner=False, n_rows=None):
    """thorax.plan for one row: the generator's segment surface with the row's numbers."""
    return THORAX.plan(_P_row(P, row, i, inner, n_rows), i)


def tail_plan(P, pyg, d_last):
    """tail.plan for the pygidial rows together: their total length, the first row's width, relief and axis, one rib
    per row; the segment pitch it reads (its overlap under the last segment's plate) is the last thoracic row's."""
    Pt = dict(P); n = int(P["segCount"]); Lp = sum(r["len"] for r in pyg)
    Pt["length"] = (d_last * n + Lp) / max(1.0 - P["cephFrac"], 1e-6); Pt["pygFrac"] = Lp / Pt["length"]
    # the first row's width is the ellipse at the row's centre (rows_from_P); the tail's own half-width is at y = 0
    yc = 0.5 * pyg[0]["len"]; Pt["_w"] = max(pyg[0]["width"], 1.5) / max(math.sqrt(max(1 - (yc / max(Lp, 1e-6)) ** 2, 0.05)), 1e-6)
    Pt["pygRings"] = max(len(pyg), 1)
    Pt["tailRelief"] = pyg[0]["relief"] / max(P["relief"], 1e-6); Pt["axisFrac"] = pyg[0]["axis"]
    S = TAIL.plan(Pt); S = dict(S); S["d"] = Lp; S["Lp_rows"] = [r["len"] for r in pyg]
    return S


def split_rows(rows):
    """(thorax rows, pygidial rows): the thorax is every row before the first 'pyg'; from there on it is the tail."""
    k = next((i for i, r in enumerate(rows) if r["kind"] == "pyg"), len(rows))
    return rows[:k], rows[k:]


def plans(P, spec=None, strict=True):
    """The body as body.py plans: [(name, mod, S, offset)] for every printed piece (the head is body.py's), the
    offsets, the print scale and the groups. spec: {"model": "thorax"|"slab", "list": [...], "rows": {...edits...},
    "min_joint_mm": .., "print_scale": ..}. strict: refuse (ValueError) a body the kit does not fit or that would not
    fit a bed -- the build path; the editor's drawing passes strict=False and shows `problems` instead."""
    spec = spec or {}; model = spec.get("model") or "thorax"
    rows = apply(rows_from_P(P, spec, model), spec)
    scale = float(spec.get("print_scale") or default_scale(rows))
    min_mm = float(spec.get("min_joint_mm") or MIN_JOINT_MM)
    term = float(spec.get("term_spine") or 0.0)
    if strict: fits(rows, scale, term)
    F = furrow_amp(P)
    out = []; y = 0.0; offsets = []
    if model == "slab":
        gs = groups(rows, scale, min_mm)
        for n, g in enumerate(gs):
            S = group_plan(rows, g, F); name = "tail" if n == len(gs) - 1 else f"seg{n}"
            out.append((name, _Mod, S, y)); offsets.append(y); y += S["d"]
    else:
        th, pg = split_rows(rows)
        if not th: raise ValueError("the thorax model needs at least one thoracic row before the pygidium")
        gs = groups(th, scale, min_mm)
        for n, g in enumerate(gs):
            maker = lambda i, k, last, g=g: thorax_row(P, th[i], i, inner=(k > 0), n_rows=len(th))
            S = group_plan(th, g, 0.0, maker=maker); name = f"seg{n}" if (pg or n < len(gs) - 1) else "tail"
            out.append((name, THORAX, S, y)); offsets.append(y); y += S["d"]
        if pg:
            S = tail_plan(P, pg, th[-1]["len"]); out.append(("tail", TAIL, S, y)); offsets.append(y); y += S["d"]
            gs = gs + [list(range(len(th), len(rows)))]
    return dict(rows=rows, groups=gs, scale=scale, min_joint_mm=min_mm, parts=out, offsets=offsets, length=y, model=model, problems=problems(rows, scale, term),
                term_spine=float(spec.get("term_spine") or 0.0), mode=("list" if spec.get("list") is not None else "preset"))


def _tail_band(S, y_tail, row, y0, last):
    """A pygidial row drawn as a band of the tail's outline: the tail's margin (right half, world y) clipped between
    the row's front y0 and rear y0 + len; the last row ends on the margin itself. Gives `poly` (the closed polygon,
    right half) besides edges()/X_tip for the handles."""
    us = np.linspace(0, 1, 241); pts = np.array([S["outline"](u, 1.0) for u in us])   # the margin, right half (any shape)
    pts = pts[pts[:, 0] >= -1e-9]; xm, ym = pts[:, 0], pts[:, 1] + y_tail
    def xmax(y):
        """the outermost x where the margin polyline crosses height y"""
        best = 0.0
        for (xa, ya), (xb, yb) in zip(zip(xm, ym), zip(xm[1:], ym[1:])):
            if (ya - y) * (yb - y) <= 0 and ya != yb: best = max(best, xa + (xb - xa) * (y - ya) / (yb - ya))
        return float(best)
    y1 = y0 + row["len"]; X_tip = max(xmax(y0), xmax(y1), 1.0)
    inside = (ym > y0) & (ym < y1) if not last else (ym > y0)
    mid = np.c_[xm, ym][inside][::-1]                                   # the margin runs tip -> front corner; walk it front -> tip
    if last: poly = np.r_[[[0.0, y0], [xmax(y0), y0]], mid if len(mid) else np.zeros((0, 2)), [[0.0, float(ym[0])]]]
    else: poly = np.r_[[[0.0, y0], [xmax(y0), y0]], mid if len(mid) else np.zeros((0, 2)), [[xmax(y1), y1], [0.0, y1]]]
    X_tip = float(max(X_tip, poly[:, 0].max()))
    def edges(x):
        ax = np.abs(np.asarray(x, float)); yf = np.full(ax.shape, y0, float)
        yr = np.full(ax.shape, (float(ym.max()) if last else y1), float)
        return yf, yr
    return dict(edges=edges, X_tip=X_tip, poly=poly, zfun=lambda x, y: S["zfun"](x, np.asarray(y, float) - y_tail), w=X_tip, a=row["axis"] * X_tip, h=row["relief"])


def diagram(P, spec=None, n=60):
    """For the editor: per row the outline polyline (right half, world mm, y = 0 at the first row's front),
    per group its piece outline, the joints, and handles (rear, tip, mid, top per row). Plus the head's half-width
    and length so the page can draw the head from its own diagram."""
    R = plans(P, spec, strict=False); rows = R["rows"]; scale = R["scale"]; model = R["model"]
    th, pg = split_rows(rows) if model == "thorax" else (rows, [])
    tail_S = R["parts"][-1][2] if (model == "thorax" and pg) else None; y_tail = R["offsets"][-1] if tail_S is not None else None
    y0 = 0.0; drawn = []
    for i, r in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if model == "thorax" and i < len(th):
            T = _shift(thorax_row(P, r, i, n_rows=len(th)), y0)
        elif model == "thorax":                                            # a pygidial row: a band of the tail's ellipse
            T = _tail_band(tail_S, y_tail, r, y0, nxt is None)
        else:
            T = tergite(r, nxt, y0)
        xs = np.linspace(0, T["X_tip"], n); yf, yr = T["edges"](xs)
        poly = T["poly"] if "poly" in T else np.r_[np.c_[xs, yf], np.c_[xs, yr][::-1]]
        ztop = float(T["zfun"](np.zeros(1), np.array([y0 + 0.5 * r["len"]]))[0])
        w = r["width"]; e_w = T["edges"](np.array([w])); e_m = T["edges"](np.array([0.5 * (r["axis"] * w + w)]))
        H = dict(rear=[0.0, y0 + r["len"]], tip=[float(T["X_tip"]), float(0.5 * (T["edges"](np.array([T["X_tip"]]))[0][0] + T["edges"](np.array([T["X_tip"]]))[1][0]))],
                 edge=[w, float(0.5 * (e_w[0][0] + e_w[1][0]))], mid=[float(0.5 * (r["axis"] * w + w)), float(0.5 * (e_m[0][0] + e_m[1][0]))],
                 top=[y0 + 0.5 * r["len"], ztop])
        drawn.append(dict(i=i, kind=r["kind"], y0=y0, y1=y0 + r["len"], print_len=r["len"] * scale,
                          small=r["len"] * scale < R["min_joint_mm"], outline=np.round(poly, 3).tolist(), handles=H, row=r))
        y0 += r["len"]
    pieces = [dict(rows=g, y0=R["offsets"][k], y1=R["offsets"][k] + R["parts"][k][2]["d"], name=R["parts"][k][0]) for k, g in enumerate(R["groups"])]
    SOLID = model == "thorax"                                              # the isopod build: solid to the bed, a plate only behind the rear joint plane
    joints = [R["offsets"][k] for k in range(1, len(R["offsets"]))]
    # the elevation: the top along the axis, every row
    ys = np.linspace(0, R["length"], 240); prof = np.zeros_like(ys)
    for name, mod, S, off in R["parts"]:
        sel = (ys >= off) & (ys <= off + S["d"]); prof[sel] = S["zfun"](np.zeros(sel.sum()), ys[sel] - off)
    # the side view in section on the axis (10 Oct 2026): every PIECE as its slab -- top from its front joint plane to
    # the end of its plate over the next piece (OVERLAP of the next row), bottom WALL below, the front lowered under
    # the piece ahead's plate by GAP_Z (the lip), the column down to the bed around each joint, the pivot at the
    # isopod ball's height (7.5 mm print). The same numbers body.py builds from; the lip is the plan's rule, not the
    # swept one, so it is the rest pose.
    WALL, GAP_Z, PIV_Z = 2.0, 0.5, 7.5 / scale
    side = []; prev = None
    for k, (name, mod, S, off) in enumerate(R["parts"]):
        nxt_len = R["parts"][k + 1][2]["d"] if k + 1 < len(R["parts"]) else S["d"]
        ovl = S.get("ovl", 0.0)
        if SOLID and "edges" in S: y_end = off + float(S["edges"](np.array([0.0]))[1][0]) if k + 1 < len(R["parts"]) else off + S["d"]   # the plate's real end
        else: y_end = off + S["d"] + (ovl if k + 1 < len(R["parts"]) else 0.0)
        yy = np.linspace(off, y_end, max(24, int((y_end - off) / 0.8)))
        top = S["zfun"](np.zeros(len(yy)), yy - off)
        if prev is not None:                                           # the lip: under the piece ahead's plate
            under = yy <= prev["y_end"] + 1e-6
            if np.any(under):
                pb = np.interp(yy[under], prev["y"], prev["bottom"]); top = top.copy(); top[under] = np.minimum(top[under], pb - GAP_Z)
        bottom = np.maximum(top - WALL, 0.0)
        if SOLID: bottom = np.where(yy > off + S["d"] + 1e-6, bottom, 0.0)   # solid to the bed ahead of the rear joint plane
        piece = dict(y=yy.tolist(), top=np.round(top, 3).tolist(), bottom=np.round(bottom, 3).tolist(), y0=off, y_end=y_end,
                     rows=R["groups"][k], column=[off + 0.5 * (0.0), off + S["d"]])
        side.append(dict(k=k, rows=R["groups"][k], y0=round(off, 3), y1=round(off + S["d"], 3), y_end=round(y_end, 3),
                         top=np.round(np.c_[yy, top], 3).tolist(), bottom=np.round(np.c_[yy, bottom], 3).tolist()))
        prev = dict(y=yy, bottom=bottom, y_end=y_end)
    pivots = [[round(R["offsets"][k], 3), round(PIV_Z, 3)] for k in range(1, len(R["offsets"]))]
    return dict(rows=drawn, pieces=pieces, joints=joints, scale=scale, min_joint_mm=R["min_joint_mm"], length=R["length"], term_spine=R["term_spine"],
                side=side, pivots=pivots, wall=WALL, gap_z=GAP_Z, model=model, problems=R["problems"],
                body_keys={k: dict(unit=u, doc=dd) for k, (u, dd) in BODY_KEYS.items()},
                mode=R["mode"], list=[{k: (v if k == "kind" else round(float(v), 4)) for k, v in r.items()} for r in rows],
                body=dict(model=model, thorax_rows=sum(1 for r in rows if r["kind"] == "thorax"), pyg_rows=sum(1 for r in rows if r["kind"] == "pyg"),
                          thorax_len=round(sum(r["len"] for r in rows if r["kind"] == "thorax"), 3), pyg_len=round(sum(r["len"] for r in rows if r["kind"] == "pyg"), 3),
                          term_spine=R["term_spine"], min_joint_mm=R["min_joint_mm"], print_scale=round(scale, 4)),
                profile=np.round(np.c_[ys, prof], 3).tolist(), keys={k: dict(unit=u, doc=dd) for k, (u, dd) in ROW_KEYS.items()},
                print_size_mm=dict(length=round(R["length"] * scale, 1), width=round(2 * max(r["width"] + r["tip"] for r in rows) * scale, 1)))

# ---------------------------------------------------------------- the macro projection (11 Oct 2026)
def macro(P, spec=None):
    """The schema parameters that ROUGHLY reproduce this body on the opening page's macro model (the sliders, the pin
    model, the instrument): row counts and lengths, the width field at its four landmarks, the medians of the per-row
    shape numbers, the pleural-spine field fitted as base + gradient (+ one macropleural bump), the tail's relief and
    width. Returns {schema key: value} to lay over P. A projection, not an inverse: the rows can hold more than the
    macro model can (per-row widths and sweeps become one taper and one sweep)."""
    import schema
    R = plans(P, spec, strict=False); rows = R["rows"]; th, pg = split_rows(rows) if R["model"] == "thorax" else ([r for r in rows if r["kind"] == "thorax"], [r for r in rows if r["kind"] == "pyg"])
    if not th: return {}
    med = lambda k: float(np.median([r[k] for r in th]))
    out = {}
    n = len(th); Lt = sum(r["len"] for r in th); Lp = sum(r["len"] for r in pg) if pg else P["pygFrac"] * P["length"]
    Lc = P["cephFrac"] * P["length"]                                      # the head keeps its length
    out["segCount"] = n; out["length"] = Lc + Lt + Lp; out["cephFrac"] = Lc / out["length"]; out["pygFrac"] = Lp / out["length"]
    if pg: out["pygRings"] = len(pg)
    # the width field. The rows see the thorax only: the max width stays where the preset has it (in the head) unless
    # a thoracic row is wider than that or the widest row is mid-thorax; the shoulder and the rear are the first row
    # and the width at the thorax's end (the last two rows extrapolated half a row), as fractions of the max.
    ws = [r["width"] for r in th]; i_max = int(np.argmax(ws)); W0 = 0.5 * P["width"]
    s_h = out["cephFrac"]; s_t = 1 - out["pygFrac"]
    if i_max > 0 and ws[i_max] > ws[0] + 1e-6: W = ws[i_max]; out["widthMaxPos"] = s_h + (i_max + 0.5) / n * (s_t - s_h)
    elif ws[0] > W0: W = ws[0] / P["widthThoraxFront"]                       # the thorax outgrew the preset's max: the head grows with it
    else: W = W0
    out["width"] = 2 * W
    w_rear = ws[-1] + 0.5 * (ws[-1] - ws[-2]) if n > 1 else ws[-1]
    out["widthThoraxFront"] = ws[0] / W; out["widthThoraxRear"] = max(w_rear, 0.5) / W
    if pg:
        yc = 0.5 * pg[0]["len"]; wp = max(pg[0]["width"], 1.0) / max(math.sqrt(max(1 - (yc / max(Lp, 1e-6)) ** 2, 0.05)), 1e-6)   # the tail's ellipse at y = 0
        out["pygWidth"] = wp / max(out["widthThoraxRear"] * W, 1e-6)
        out["tailRelief"] = pg[0]["relief"] / max(med("relief"), 1e-6)
    # the shape numbers: medians over the thorax
    out["relief"] = med("relief"); out["axisFrac"] = med("axis"); out["marginHeight"] = med("margin"); out["ringArch"] = med("arch")
    out["fulcrum"] = med("fulcrum"); out["pleuralSlope"] = med("slope")
    out["tipSweep"] = float(np.median([math.tan(math.radians(r["sweep"])) * max(r["width"] - r["axis"] * r["width"], 1e-6) / r["len"] for r in th]))
    # the pleural spines: tip / half-width per row = base + grad * (i / (n-1)) + one bump
    f = np.array([max(r["tip"], 0.0) / max(r["width"], 1e-6) for r in th]); x = np.arange(n) / max(n - 1, 1)
    if f.max() > 1e-6:
        A = np.c_[np.ones(n), x]; base, grad = np.linalg.lstsq(A, f, rcond=None)[0]
        resid = f - (base + grad * x); j = int(np.argmax(resid))
        if resid[j] > 0.1:                                                   # one row sticks out: the macropleural segment
            keep = np.arange(n) != j; base, grad = np.linalg.lstsq(A[keep], f[keep], rcond=None)[0] if keep.sum() >= 2 else (base, grad)
            out["macroIndex"] = j; out["macroAmp"] = float(f[j] - (base + grad * x[j]))
        else: out["macroIndex"] = -1
        out["spineBase"] = float(max(base, 0.0)); out["spineGrad"] = float(grad)
    else: out["spineBase"] = 0.0; out["spineGrad"] = 0.0; out["macroIndex"] = -1
    if R["term_spine"] > 0.05 and Lp > 0: out["termSpine"] = R["term_spine"] / Lp
    Q = schema.coerce(out, base=P); return {k: Q[k] for k in out if k in Q}   # the overrides only, clamped to the schema's ranges

