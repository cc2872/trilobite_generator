"""
web/recorder.py — keep every distinct animal that gets measured on the site, and send it to Claire.

When someone measures enrollment (/api/measure), record(P, reading) stores one line of text:
the parameter hash, the full parameter table, and the instrument's reading. One line per distinct
animal: the key is the parameter hash plus the schema and instrument versions, so the same animal
measured again is not stored twice, and a new instrument version is.

There are two sources, both append-only JSONL in the same folder:
  - web/records/readings.jsonl — the clean list, one line per DISTINCT measured animal (record(), kind "reading").
  - web/records/agent_log.jsonl — the raw list, one line per agent REQUEST, no dedup (log_event(), kind "agent_request"):
    every /api/agent/make call, including repeats, rejected ones (bad preset/params, 401, 429) and errors.

Each new line from either source goes two places, in this order:
  1. The JSONL file on the server (TRILO_RECORD_DIR moves the folder). Always.
  2. If TRILO_RECORD_URL is set, the line is POSTed there as JSON (Authorization: Bearer $TRILO_RECORD_TOKEN
     when that is set; "secret": $TRILO_RECORD_SECRET in the body when that is set). The "kind" field says which
     list it belongs to so a receiver can route it (e.g. two tabs of one Sheet). Lines that could not be sent are
     retried with the next record and at start-up (web/records/sent.txt lists the keys that arrived).

The raw log is also served live at GET /api/agent/log (web/agent.py), gated by the agent token.
Nothing about the visitor is stored: no address, no browser, no cookie. Recording never slows or fails a
measurement: the send runs on its own thread and every error is swallowed into web/records/errors.log.
TRILO_RECORD=0 turns the whole thing off.  docs/recording.md has the set-up for the receiving end.
"""
import os, json, time, threading, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCK = threading.Lock()
_SEQ = 0                   # per-process counter so each raw event gets a unique key even within the same second

def _dir():
    d = os.environ.get("TRILO_RECORD_DIR") or os.path.join(ROOT, "web", "records"); os.makedirs(d, exist_ok=True); return d
def _on(): return os.environ.get("TRILO_RECORD", "1") not in ("0", "false", "off", "")
def _keys(name):
    try:
        with open(os.path.join(_dir(), name), encoding="utf-8") as f:
            return [json.loads(l)["key"] if name.endswith(".jsonl") else l.strip() for l in f if l.strip()]
    except OSError: return []
def _log(msg):
    try:
        with open(os.path.join(_dir(), "errors.log"), "a", encoding="utf-8") as f: f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg + "\n")
    except OSError: pass

def entry(P, reading, schema_version, instrument_version, param_hash, source="site"):
    """The stored line for one measured animal. The reading is kept whole except for the bulky kinematics/trace."""
    r = {k: v for k, v in reading.items() if k not in ("kinematics", "scan_trace", "schema_notes")}
    line = (f"{param_hash}  schema {schema_version}  instrument {instrument_version}  "
            f"{r.get('limited_by', '?')} / {r.get('enroll_class', '?')}  theta/joint {r.get('theta_joint_deg', '?')}  s_tail {r.get('s_tail', '?')}")
    return dict(kind="reading", key=f"{param_hash}@{schema_version}@{instrument_version}", hash=param_hash, schema=schema_version, instrument=instrument_version,
                time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source=source, text=line, params=P, reading=r)

def _post(e):
    url = os.environ.get("TRILO_RECORD_URL")
    if not url: return False
    sec = os.environ.get("TRILO_RECORD_SECRET")                    # for receivers that cannot read headers (Google Apps Script): the secret in the body
    body = dict(e, secret=sec) if sec else e
    req = urllib.request.Request(url, data=json.dumps(body, default=str).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
    tok = os.environ.get("TRILO_RECORD_TOKEN")
    if tok: req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=15) as resp: return 200 <= resp.status < 300

def flush():
    """Send every stored line that has not arrived yet. Returns how many were sent now."""
    if not _on() or not os.environ.get("TRILO_RECORD_URL"): return 0
    n = 0
    with _LOCK:
        sent = set(_keys("sent.txt"))
        lines = []
        for name in ("readings.jsonl", "agent_log.jsonl"):          # the clean list first, then the raw request log
            try: lines += [json.loads(l) for l in open(os.path.join(_dir(), name), encoding="utf-8") if l.strip()]
            except OSError: pass
        for e in lines:
            if e["key"] in sent: continue
            try:
                if _post(e):
                    with open(os.path.join(_dir(), "sent.txt"), "a", encoding="utf-8") as f: f.write(e["key"] + "\n")
                    sent.add(e["key"]); n += 1
                else: _log(f"send refused {e['key']}"); break
            except Exception as ex: _log(f"send failed {e['key']}: {ex!r}"); break       # receiver down: stop, try again next time
    return n

def record(P, reading, schema_version, instrument_version, param_hash, source="site", wait=False):
    """Store this animal if it is new. Returns True if a line was written. Never raises."""
    try:
        if not _on(): return False
        e = entry(P, reading, schema_version, instrument_version, param_hash, source)
        with _LOCK:
            if e["key"] in set(_keys("readings.jsonl")): return False
            with open(os.path.join(_dir(), "readings.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(e, default=str, sort_keys=True) + "\n")
        t = threading.Thread(target=flush, daemon=True); t.start()
        if wait: t.join()
        return True
    except Exception as ex:
        _log(f"record failed: {ex!r}"); return False

def log_event(fields):
    """Store one raw agent request (no dedup) and stream it on. `fields` is a dict (source, outcome, request, ...).
    Returns the event key, or None when recording is off or something went wrong. Never raises."""
    global _SEQ
    try:
        if not _on(): return None
        with _LOCK:
            _SEQ += 1
            e = dict(kind="agent_request", key=f"log-{int(time.time() * 1000)}-{_SEQ}",
                     time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
            with open(os.path.join(_dir(), "agent_log.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(e, default=str, sort_keys=True) + "\n")
        threading.Thread(target=flush, daemon=True).start()
        return e["key"]
    except Exception as ex:
        _log(f"log_event failed: {ex!r}"); return None

def read_log(limit=200):
    """The most recent raw agent events, oldest first. Returns a list (empty if the log does not exist). Never raises."""
    try:
        with open(os.path.join(_dir(), "agent_log.jsonl"), encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        return [json.loads(l) for l in lines[-max(1, int(limit)):]]
    except (OSError, ValueError, TypeError):
        return []
