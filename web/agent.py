"""
web/agent.py — a way in for programs (an AI agent, a script): describe an animal in one request, get it built on the
pin model and measured, as plain JSON. Everything here goes through the same code the page uses (schema.coerce_report,
instrument.read, /api/build), so an animal made here is the animal the page would make, with the same hash.

    GET  /llms.txt                 the guide, as text: what the site is and how to use these endpoints
    GET  /api/agent                the same as JSON, with the presets and characters
    GET  /api/agent/params         the parameter table (key, range, default, meaning); ?group= or ?q= narrows it
    GET  /api/agent/make?...       make one animal: preset=, chars=a,b, any parameter key=value, build=1, wait=<s>
    POST /api/agent/make           the same as JSON: {"preset": "...", "chars": [...], "set": {key: value}, "build": true}
    GET  /api/agent/animal/<hash>  an animal already asked for: its status and reading
    GET  /api/agent/log            operator view (token-gated): every make request this server has seen, as JSON or a live HTML table

A measurement takes about a minute and the server runs one at a time, so `make` waits up to `wait` seconds and
otherwise answers {"status": "running"}; ask again with the same URL (it is the same animal, so nothing restarts).
TRILO_AGENT=0 removes these routes. A token, if set (TRILO_AGENT_TOKEN, else web/agent_token.txt), must be given to
`make` (?token= or Authorization: Bearer).
"""
import os, json, time, base64, threading, urllib.parse
from html import escape as _esc
from flask import Blueprint, request, jsonify, Response
import recorder as RECORDER

MAX_WAITING = 6            # animals queued or running at once; more are refused (429) rather than piled up
DEFAULT_WAIT, MAX_WAIT = 90, 110
JOBS = {}                  # param hash -> dict(state, P, notes, reading, error, t0, t1, build)
# presets that do not measure on the pin model as shipped (the instrument returns "invalid: unsane parts"); isopod_model/README.md, 28 Sep 2026
PIN_INVALID_PRESETS = ("phacopida", "ptychopariida", "redlichiida")
FIELDS = {
    "theta_joint_deg": "largest uniform flexion per joint before closure or collision (degrees). null when not valid",
    "total_deg": "theta_joint_deg times the number of joints",
    "joints": "number of joints = segCount + 1",
    "s_tail": "where the tail margin ends in the head's frame, in head lengths: |s_tail| <= 0.05 margins meet (sphaeroidal), > 0 tucked inside the head, < 0 past the head's front. Only meaningful when limited_by is closed",
    "closure_gap_mm": "head-to-tail gap at the stop (mm); under 3 mm counts as closed", "gap_over_L": "that gap divided by body length",
    "stopped_by": "the first pairs of parts to interfere just past the stop, each [part index, part index, overlap growth in mm3]; index 0 is the head, 1..segCount the thorax segments front to back, the last is the tail. For a closed animal this is the head-tail contact that closure runs into",
    "unsane_parts": "parts that did not build as one sane body at rest; any entry makes the animal invalid", "reason": "why an invalid animal is invalid",
    "bound_deg": "the per-joint bevel the measurement build used: 45, or 35 / 25 when a wider wedge would have cut a segment's tips off. The sweep cannot go past it", "bevel_built_deg": "same, as actually built",
    "print_valid": "whether this pin animal meets the lab's print minimums (pitch >= 8 mm, knuckle >= 3 mm, wall >= 1.5 mm). Independent of 'valid': it is about printing, not measuring", "violations": "which print minimums it misses",
    "measure_timed_out": "true if the sweep ran out of time (then nothing above is a measurement)", "seconds": "time the sweep took",
    "valid (top level)": "true only when limited_by is closed or anatomy",
    "hash (top level)": "10 characters naming the parameter set. Same parameters, same hash, same reading. build and fill do not change it",
    "characters (top level)": "the named characters this animal's parameters express", "chars_applied (top level)": "the chars= you sent in this request",
    "changed_from_preset (top level)": "parameters whose values differ from the preset you named, after chars and your settings",
    "notes (top level)": "what the server did to your input: clamps, rounding, settings that changed nothing",
    "links.view": "the same animal on the page (relative to the site root). links.stl appears after build=1",
}
_JL = threading.Lock()

LIMITED = {"closed": "the animal closed: the head-tail gap fell under 3 mm before anything collided",
           "anatomy": "two parts collided before the animal closed (see stopped_by for which)",
           "bound": "the ruler ran out at 45 degrees per joint before the animal stopped: censored, not a measurement",
           "invalid": "the geometry was not sane at rest (see reason / unsane_parts): no measurement"}
CLASSES = {"sphaeroidal": "closed with the tail margin meeting the head margin",
           "double": "closed with the tail tucked inside the head margin (s_tail > 0)",
           "spiral": "closed with the tail carried past the head margin (s_tail < 0)",
           "discoidal": "closed on a short curl with the tail overlapping the head",
           "open": "did not close: stopped by its own anatomy",
           "censored": "no class: the ruler's bound was reached or the animal was invalid"}

def make_blueprint(app, schema, I, presets, measure, lock):
    """presets() -> {name: params}; measure(P, notes, source) -> reading dict (the page's own measurement path)."""
    bp = Blueprint("agent", __name__)
    _TOKEN_FILE = os.path.join(os.path.dirname(__file__), "agent_token.txt")
    def _on(): return os.environ.get("TRILO_AGENT", "1") not in ("0", "false", "off", "")
    def _token():
        # the token for `make`: TRILO_AGENT_TOKEN wins; otherwise the token file (TRILO_AGENT_TOKEN_FILE, else
        # web/agent_token.txt) — git-ignored, so the phrase stays out of the repo. No env and no file: no token.
        tok = os.environ.get("TRILO_AGENT_TOKEN")
        if tok: return tok
        try:
            with open(os.environ.get("TRILO_AGENT_TOKEN_FILE") or _TOKEN_FILE, encoding="utf-8") as f: tok = f.read().strip()
            return tok or None
        except OSError:
            return None
    @bp.before_request
    def _gate():
        if not _on(): return jsonify(error="the agent interface is switched off on this server"), 404

    def _defaults(): return schema.table_defaults()
    def _view_url(P):
        d = {k: v for k, v in P.items() if k in _defaults() and v != _defaults()[k]}
        return "/?joint=pin&p=" + urllib.parse.quote(base64.b64encode(json.dumps(d).encode("utf-8")).decode("ascii"), safe="")
    def _compact(r):
        keep = ("limited_by", "enroll_class", "theta_joint_deg", "total_deg", "s_tail", "closure_gap_mm", "gap_over_L", "joints",
                "stopped_by", "unsane_parts", "reason", "print_valid", "violations", "bevel_built_deg", "bound_deg", "measure_timed_out", "seconds")
        return {k: r.get(k) for k in keep if k in r}
    def _sentence(r):
        lb, cl = r.get("limited_by"), r.get("enroll_class")
        if lb == "closed": return f"Closes ({cl}) at {r.get('theta_joint_deg')} degrees per joint, {r.get('total_deg')} in total; {CLASSES.get(cl, '')}."
        if lb == "anatomy":
            sb = r.get("stopped_by") or []; n = int(r.get("joints") or 0) + 1
            nm = lambda i: "head" if i == 0 else "tail" if i == n - 1 else f"seg{i - 1}"
            who = f"{nm(sb[0][0])} and {nm(sb[0][1])}" if sb else "two parts"
            return f"Does not close: stops at {r.get('theta_joint_deg')} degrees per joint, {r.get('total_deg')} in total, when {who} meet (gap left {r.get('closure_gap_mm')} mm)."
        if lb == "bound": return f"Censored: reached the ruler's bound ({r.get('bound_deg')} degrees per joint) without stopping. Not a measurement."
        return f"Invalid: {r.get('reason') or 'the parts were not sane at rest'}. No measurement."
    def _answer(h, wait_url=None, req=None):
        j = JOBS[h]; P = j["P"]; req = req or {}
        out = dict(status=j["state"], hash=h, schema=schema.SCHEMA_VERSION, instrument=I.INSTRUMENT_VERSION, model="pin", **dict(dict(notes=j["notes"]), **req), characters=schema.characters(P),
                   links=dict(view=_view_url(P), animal=f"/api/agent/animal/{h}"))
        if j["state"] == "done":
            r = j["reading"]; out.update(reading=_compact(r), summary=_sentence(r), measured_in_s=round(j["t1"] - j["t0"], 1),
                                         valid=r.get("limited_by") in ("closed", "anatomy"))
            if j.get("build"): out["build"] = j["build"]; out["links"].update(stl=j["build"].get("stl"))
        elif j["state"] == "error": out.update(error=j["error"])
        else:
            ahead = sum(1 for k, x in JOBS.items() if x["state"] in ("queued", "running") and x["t0"] < j["t0"])
            out.update(ask_again=wait_url or f"/api/agent/animal/{h}", ahead_in_queue=ahead,
                       hint="A measurement takes about a minute and the server runs one at a time. Ask again in 20-30 s; the same request never starts a second job.")
        return out

    def _run(h, build, fill):
        j = JOBS[h]
        try:
            j["state"] = "running"
            if j.get("reading") is None: j["reading"] = measure(j["P"], j["notes"], "agent")
            if build:
                with app.test_client() as c:                               # the page's own print build, pin joint
                    m = c.post("/api/build", json=dict(P=j["P"], joint="pin", fill=fill)).get_json()
                j["build"] = dict(key=m.get("key"), stl=f"/api/stl/{m.get('key')}", fill_mm=m.get("fill_mm"), seconds=m.get("build_seconds"),
                                  parts=[dict(name=p.get("name"), watertight=p.get("watertight"), bodies=p.get("bodies"), error=p.get("error")) for p in m.get("parts", [])],
                                  print=m.get("print"), contents="one STL holding every part, laid out flat at rest in print millimetres; typically 10-15 MB",
                                  note="The pin model is a thin shell made for measuring. fill=<mm> (up to 3) thickens it for printing; the page's separate print joints are not offered here.")
            j["t1"] = time.time(); j["state"] = "done"
        except Exception as ex:
            j["t1"] = time.time(); j["error"] = str(ex)[:300]; j["state"] = "error"

    def _spec():
        """The request as (preset, chars, set, build, fill, wait), from JSON or the query string."""
        b = request.get_json(silent=True) or {} if request.method == "POST" else {}
        q = request.args
        preset = b.get("preset") or q.get("preset") or schema.DEFAULT_PRESET
        chars = b.get("chars") if isinstance(b.get("chars"), list) else [c for c in (q.get("chars") or "").split(",") if c]
        st = dict(b.get("set") or {})
        for k in q:
            if k in schema.BY_KEY: st[k] = q.get(k)
        unknown = [k for k in q if k not in schema.BY_KEY and k not in ("preset", "chars", "build", "fill", "wait", "token", "measure")] + [k for k in (b.get("set") or {}) if k not in schema.BY_KEY]
        truthy = lambda v: str(v).lower() in ("1", "true", "yes", "on")
        build = truthy(b.get("build", q.get("build", "0")))
        try: fill = max(0.0, min(3.0, float(b.get("fill", q.get("fill", 0)) or 0)))
        except (TypeError, ValueError): fill = 0.0
        try: wait = max(0.0, min(MAX_WAIT, float(b.get("wait", q.get("wait", DEFAULT_WAIT)))))
        except (TypeError, ValueError): wait = DEFAULT_WAIT
        return preset, chars, st, unknown, build, fill, wait

    def _tok_ok():
        tok = _token()
        return not tok or request.args.get("token") == tok or request.headers.get("Authorization") == "Bearer " + tok
    def _rec(outcome, code, spec=None, **extra):
        # one raw line for this agent request, whatever its outcome (web/recorder.py -> agent_log.jsonl + the live page)
        try:
            info = spec if spec is not None else {k: v for k, v in request.args.items() if k != "token"}
            RECORDER.log_event(dict(source="agent", method=request.method, outcome=outcome, code=code, request=info, **extra))
        except Exception:
            pass

    @bp.route("/api/agent/make", methods=["GET", "POST"])
    def make():
        if not _tok_ok():
            _rec("unauthorized", 401)
            return jsonify(error="this server needs a token for /api/agent/make (?token=... or Authorization: Bearer ...)"), 401
        preset, chars, st, unknown, build, fill, wait = _spec()
        spec = dict(preset=preset, chars=chars, set=st, build=build)
        PR = presets()
        if preset not in PR:
            _rec("unknown_preset", 400, spec=spec); return jsonify(error=f"unknown preset {preset!r}", presets=sorted(PR)), 400
        bad = [c for c in chars if c not in schema.CHARACTERS]
        if bad:
            _rec("unknown_chars", 400, spec=spec, bad=bad); return jsonify(error=f"unknown character(s) {bad}", characters=sorted(schema.CHARACTERS)), 400
        if unknown:
            _rec("unknown_params", 400, spec=spec, unknown=unknown); return jsonify(error=f"unknown parameter(s) {unknown}", hint="GET /api/agent/params lists every key; ?q=word searches them"), 400
        base = dict(PR[preset])
        for c in chars: base.update(schema.CHARACTERS[c]["set"])
        base.update(st)
        P, notes = schema.coerce_report(base, base=schema.table_defaults())
        h = schema.param_hash(P); P0 = schema.coerce(PR[preset])
        notes = list(notes) + [f"{k}={st[k]}: already the value in preset {preset}, so this changed nothing" for k in st if k in P0 and P.get(k) == P0.get(k)]
        if preset in PIN_INVALID_PRESETS: notes.append(f"preset {preset} does not measure on the pin model as shipped (unsane parts); expect 'invalid' unless your changes fix it")
        req = dict(preset=preset, chars_applied=chars, changed_from_preset={k: P[k] for k in P if P[k] != P0.get(k)}, notes=notes)
        with _JL:
            j = JOBS.get(h)
            if j is None or j["state"] == "error" or (build and j["state"] == "done" and not j.get("build")):
                if sum(1 for x in JOBS.values() if x["state"] in ("queued", "running")) >= MAX_WAITING:
                    _rec("busy", 429, spec=spec, hash=h); return jsonify(error="the server is busy: too many animals waiting. Try again in a few minutes."), 429
                j = JOBS[h] = dict(state="queued", P=P, notes=notes, reading=(j or {}).get("reading"), error=None, t0=time.time(), t1=None, build=None)
                threading.Thread(target=_run, args=(h, build, fill), daemon=True).start()
        t_end = time.time() + wait
        while JOBS[h]["state"] in ("queued", "running") and time.time() < t_end: time.sleep(0.5)
        state = JOBS[h]["state"]; code = {"done": 200, "error": 500}.get(state, 202)
        _rec({"done": "done", "error": "error"}.get(state, "running"), code, spec=spec, hash=h,
             summary=(_sentence(JOBS[h]["reading"]) if state == "done" and JOBS[h].get("reading") else None))
        return jsonify(_answer(h, wait_url=request.full_path.rstrip("?"), req=req)), code

    @bp.get("/api/agent/animal/<h>")
    def animal(h):
        if h not in JOBS: return jsonify(error="no animal with that hash has been asked for since the server started", hint="GET /api/agent/make?... makes one"), 404
        return jsonify(_answer(h)), {"done": 200, "error": 500}.get(JOBS[h]["state"], 202)

    @bp.get("/api/agent/log")
    def agent_log():
        """The raw list: every make request this server has seen, newest first. Token-gated (operator view).
        JSON by default; a browser (or ?format=html) gets a small table that refreshes itself."""
        if not _tok_ok():
            return jsonify(error="this endpoint needs the agent token (?token=... or Authorization: Bearer ...)"), 401
        try: limit = max(1, min(2000, int(request.args.get("limit", 200))))
        except (TypeError, ValueError): limit = 200
        events = RECORDER.read_log(limit)
        wants_html = request.args.get("format") == "html" or "text/html" in (request.headers.get("Accept") or "")
        if wants_html: return Response(_log_html(events), mimetype="text/html; charset=utf-8")
        return jsonify(count=len(events), events=list(reversed(events)))      # newest first

    def _log_html(events):
        def row(e):
            req = e.get("request") or {}
            st = req.get("set") if isinstance(req.get("set"), dict) else {}
            bits = ", ".join(f"{k}={v}" for k, v in list(st.items())[:6]) + (" …" if len(st) > 6 else "")
            chars = ",".join(req.get("chars") or []) if isinstance(req.get("chars"), list) else ""
            return ("<tr><td>{t}</td><td class=o>{o}</td><td>{c}</td><td>{p}</td><td>{ch}</td>"
                    "<td>{h}</td><td>{s}</td><td>{sm}</td></tr>").format(
                t=_esc(e.get("time", "")), o=_esc(str(e.get("outcome", ""))), c=_esc(str(e.get("code", ""))),
                p=_esc(str(req.get("preset", ""))), ch=_esc(chars), h=_esc(str(e.get("hash", "") or "")),
                s=_esc(bits), sm=_esc(str(e.get("summary") or "")))
        body = "".join(row(e) for e in reversed(events))      # newest first
        return ("<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=10>"
                "<title>agent log</title>"
                "<style>body{font:13px/1.45 system-ui,Segoe UI,sans-serif;margin:1.5rem;color:#222}"
                "h1{font-size:1.2rem;margin:0 0 .2rem}p{color:#666;margin:.2rem 0 1rem}"
                "table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #e3e3e3;padding:.3rem .55rem;text-align:left;vertical-align:top}"
                "th{background:#f5f5f5;position:sticky;top:0}td.o{font-weight:600}"
                "tr:hover{background:#fafafa}</style>"
                f"<h1>Agent requests</h1><p>{len(events)} most recent, newest first · refreshes every 10&nbsp;s</p>"
                "<table><tr><th>time (UTC)</th><th>outcome</th><th>code</th><th>preset</th><th>chars</th>"
                "<th>hash</th><th>params set</th><th>result</th></tr>"
                + (body or "<tr><td colspan=8 style='color:#999'>no requests yet</td></tr>") + "</table>")

    @bp.get("/api/agent/preset/<name>")
    def preset_values(name):
        PR = presets()
        if name not in PR: return jsonify(error=f"unknown preset {name!r}", presets=sorted(PR)), 404
        P = schema.coerce(PR[name]); D = schema.table_defaults()
        return jsonify(preset=name, hash=schema.param_hash(P), characters=schema.characters(P), measures_on_pin=name not in PIN_INVALID_PRESETS,
                       differs_from_default={k: P[k] for k in P if k in D and P[k] != D[k]}, params=P)

    @bp.get("/api/agent/params")
    def params():
        g, q = request.args.get("group"), (request.args.get("q") or "").lower()
        rows = [dict(key=p.key, label=p.label, group=p.group, default=p.default, lo=p.lo, hi=p.hi, step=p.step, unit=p.unit, kind=p.kind, doc=p.doc)
                for p in schema.PARAMS if (not g or p.group == g) and (not q or q in (p.key + " " + p.label + " " + p.doc).lower())]
        return jsonify(schema=schema.SCHEMA_VERSION, groups=schema.GROUPS, count=len(rows), params=rows,
                       note="Values outside lo..hi are clamped and the clamp is reported in 'notes'. kind 'expr' parameters are formulas in s (0 at the head, 1 at the tail) and are best left alone. "
                            "'default' is the bare table default, not any preset's value: GET /api/agent/preset/<name> for those. ?q= matches anywhere in the key, label or description.")

    def _guide():
        PR = presets(); orders = sorted(n for n in PR if n.endswith("ida"))
        return dict(
            what="A parametric trilobite generator with an enrollment instrument. You describe an animal with about a hundred numbers; "
                 "the server builds it as a chain of articulated plates on a pin hinge and measures how far it can curl before it closes or collides.",
            how=["1. Pick a starting animal: preset=<name> (an order, e.g. proetida). GET /api/agent/preset/<name> shows its values, so you know what you are changing.",
                 "2. Optionally apply named characters: chars=isopygous,genalSpines.",
                 "3. Optionally set parameters by key: segCount=11&pygFrac=0.3 (GET /api/agent/params for the table).",
                 "4. GET /api/agent/make?preset=...&... and read 'status'. If it is 'running', ask again with the same URL after 20-30 s.",
                 "5. When status is 'done', 'reading' is the measurement and 'summary' says it in a sentence. links.view opens the same animal on the page."],
            example="/api/agent/make?preset=proetida&segCount=10&chars=isopygous",
            example_post='POST /api/agent/make  {"preset": "proetida", "chars": ["isopygous"], "set": {"segCount": 10, "pygFrac": 0.3}, "build": false}  (chars is a list, set is an object of parameter key: value)',
            endpoints={"/api/agent/make": "make and measure one animal (GET query or POST JSON {preset, chars, set, build, fill, wait})",
                       "/api/agent/params": "the parameter table; ?group= or ?q= to narrow",
                       "/api/agent/preset/<name>": "a starting animal's parameter values, its characters, and whether it measures on the pin model",
                       "/api/agent/animal/<hash>": "status and reading (and the build, if one was made) of an animal already asked for",
                       "/api/agent": "this guide as JSON", "/llms.txt": "this guide as text"},
            options=dict(build="1 = also build the pin model's parts; the answer gains links.stl, one file with every part laid flat (10-15 MB)",
                         fill="mm of print-only shell thickening for build=1 (0-3, default 0). It changes the STL, never the measurement or the hash",
                         wait=f"seconds to wait for the answer before replying 'running' (default {DEFAULT_WAIT}, max {MAX_WAIT})"),
            presets=orders, other_presets=sorted(set(PR) - set(orders)), default_preset=schema.DEFAULT_PRESET,
            preset_notes=dict(not_measurable_on_pin=list(PIN_INVALID_PRESETS), textured="the default animal: a generic trilobite with the sculpt on",
                              reference="the same body, smooth", harpetid_phacopid="older built-in sketches; prefer the orders ending in -ida"),
            fields=FIELDS,
            characters={k: c["name"] for k, c in schema.CHARACTERS.items()},
            reading=dict(limited_by=LIMITED, enroll_class=CLASSES),
            rules=["One job runs at a time and each takes about a minute: do not send many different animals at once (more than "
                   f"{MAX_WAITING} waiting are refused).", "The same parameters always give the same hash and the same reading, so repeats are free.",
                   "Out-of-range values are clamped, unknown keys are refused: check 'notes' in the answer.",
                   "Only the pin model is measured. It is an idealised hinge used as the ruler, not the printed joint.",
                   "All links in answers are relative to the site root.",
                   "Each distinct animal measured is kept by the lab as research data (hash, parameters, reading). Nothing about the caller is stored."],
            versions=dict(schema=schema.SCHEMA_VERSION, instrument=I.INSTRUMENT_VERSION))

    @bp.get("/api/agent")
    def guide(): return jsonify(_guide())

    @bp.get("/llms.txt")
    @bp.get("/agents.txt")
    def llms():
        g = _guide(); L = ["# Trilobite Morphospace: generator and enrollment instrument", "", "> " + g["what"], "",
                           "This file is for programs and AI agents. Everything below is plain HTTP and JSON; no login, no JavaScript.", "", "## How to make an animal", ""]
        L += g["how"] + ["", "Example: GET " + g["example"], "Example: " + g["example_post"], "", "## Endpoints", ""] + [f"- {k}: {v}" for k, v in g["endpoints"].items()]
        L += ["", "## Options for /api/agent/make", ""] + [f"- {k}: {v}" for k, v in g["options"].items()]
        L += ["", "## Starting animals (preset=)", "", ", ".join(g["presets"]) + ". Also: " + ", ".join(g["other_presets"]) + f". Default: {g['default_preset']}.",
              "", f"Not measurable on the pin model as shipped (they come back invalid): {', '.join(PIN_INVALID_PRESETS)}. textured = {g['preset_notes']['textured']}; reference = {g['preset_notes']['reference']}; harpetid and phacopid are {g['preset_notes']['harpetid_phacopid']}.",
              "", "## Characters (chars=)", ""] + [f"- {k}: {v}" for k, v in g["characters"].items()]
        L += ["", "## Reading the answer", ""] + [f"- {k}: {v}" for k, v in FIELDS.items()] + ["- limited_by:"] + [f"  - {k}: {v}" for k, v in LIMITED.items()] + ["- enroll_class:"] + [f"  - {k}: {v}" for k, v in CLASSES.items()]
        L += ["", "## Rules", ""] + [f"- {r}" for r in g["rules"]] + ["", f"Schema {g['versions']['schema']}, instrument {g['versions']['instrument']}.", ""]
        return Response("\n".join(L), mimetype="text/plain; charset=utf-8")
    return bp
