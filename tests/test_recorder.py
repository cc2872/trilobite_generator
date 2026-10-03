"""web/recorder.py: one stored line per distinct animal, sent on once, never in the way of a measurement."""
import os, sys, json, threading, http.server
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "web"))
import recorder as R

READING = dict(valid=True, limited_by="closed", enroll_class="double", theta_joint_deg=22.81, s_tail=0.258, kinematics={"big": [1] * 50})
def _lines(d): return [json.loads(l) for l in open(os.path.join(d, "readings.jsonl"), encoding="utf-8")]

class _Sink(http.server.BaseHTTPRequestHandler):
    got, status = [], 200
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if _Sink.status == 200: _Sink.got.append((self.headers.get("Authorization"), json.loads(body)))
        self.send_response(_Sink.status); self.end_headers()
    def log_message(self, *a): pass

def _server():
    s = http.server.HTTPServer(("127.0.0.1", 0), _Sink); threading.Thread(target=s.serve_forever, daemon=True).start(); return s

def test_one_line_per_distinct_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("TRILO_RECORD", "1"); monkeypatch.setenv("TRILO_RECORD_DIR", str(tmp_path)); monkeypatch.delenv("TRILO_RECORD_URL", raising=False)
    assert R.record({"segCount": 9}, READING, "6.2", "2.1", "aaaaaaaaaa", wait=True)
    assert not R.record({"segCount": 9}, READING, "6.2", "2.1", "aaaaaaaaaa", wait=True)        # same animal again: nothing new
    assert R.record({"segCount": 10}, READING, "6.2", "2.1", "bbbbbbbbbb", wait=True)
    assert R.record({"segCount": 9}, READING, "6.2", "2.2", "aaaaaaaaaa", wait=True)            # a new instrument version is a new record
    L = _lines(tmp_path); assert [e["hash"] for e in L] == ["aaaaaaaaaa", "bbbbbbbbbb", "aaaaaaaaaa"]
    e = L[0]; assert e["params"] == {"segCount": 9} and e["reading"]["theta_joint_deg"] == 22.81 and "kinematics" not in e["reading"]
    assert "aaaaaaaaaa" in e["text"] and "double" in e["text"]
    assert not any(k in json.dumps(e).lower() for k in ("remote_addr", "user-agent", "cookie"))   # nothing about the visitor

def test_sent_once_with_token_and_retried_after_failure(tmp_path, monkeypatch):
    srv = _server(); _Sink.got, _Sink.status = [], 500
    monkeypatch.setenv("TRILO_RECORD", "1"); monkeypatch.setenv("TRILO_RECORD_DIR", str(tmp_path)); monkeypatch.setenv("TRILO_RECORD_URL", f"http://127.0.0.1:{srv.server_port}/hook")
    monkeypatch.setenv("TRILO_RECORD_TOKEN", "s3cret")
    assert R.record({"a": 1}, READING, "6.2", "2.1", "cccccccccc", wait=True)                    # receiver refuses: kept, not lost
    assert _Sink.got == [] and len(_lines(tmp_path)) == 1
    _Sink.status = 200
    assert R.record({"a": 2}, READING, "6.2", "2.1", "dddddddddd", wait=True)                    # the next record takes the stuck one with it
    assert [g[1]["hash"] for g in _Sink.got] == ["cccccccccc", "dddddddddd"] and _Sink.got[0][0] == "Bearer s3cret"
    assert R.flush() == 0 and len(_Sink.got) == 2                                               # nothing is sent twice
    srv.shutdown()

def test_off_switch_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TRILO_RECORD_DIR", str(tmp_path)); monkeypatch.setenv("TRILO_RECORD", "0")
    assert not R.record({"a": 1}, READING, "6.2", "2.1", "eeeeeeeeee", wait=True) and not os.path.exists(os.path.join(tmp_path, "readings.jsonl"))
    monkeypatch.setenv("TRILO_RECORD", "1"); monkeypatch.setenv("TRILO_RECORD_URL", "http://127.0.0.1:9/nowhere")
    assert R.record({"a": 1}, READING, "6.2", "2.1", "eeeeeeeeee", wait=True)                    # unreachable receiver: stored, no exception
    assert len(_lines(tmp_path)) == 1
