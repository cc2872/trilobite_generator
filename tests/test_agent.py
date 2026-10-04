"""web/agent.py: the way in for programs. The instrument is stubbed here (the real one is covered elsewhere and takes a minute)."""
import os, sys, json, time, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "web")); sys.path.insert(0, ROOT)
import schema

@pytest.fixture()
def client(monkeypatch):
    import app as A, agent as AG
    calls = []
    def fake_read(P):
        calls.append(schema.param_hash(P)); time.sleep(0.05)
        return dict(limited_by="closed", enroll_class="double", theta_joint_deg=22.81, total_deg=228.1, s_tail=0.258, closure_gap_mm=2.1, stopped_by=[], unsane_parts=[], reason=""), {}
    monkeypatch.setattr(A.I, "read", fake_read); AG.JOBS.clear(); A.MEASURED.clear()
    monkeypatch.delenv("TRILO_AGENT_TOKEN", raising=False); monkeypatch.setenv("TRILO_AGENT", "1")
    monkeypatch.setenv("TRILO_AGENT_TOKEN_FILE", os.path.join(ROOT, "tests", "_no_token_file"))   # ignore any real web/agent_token.txt
    c = A.app.test_client(); c.calls = calls; return c

def test_guide_is_readable_without_javascript(client):
    t = client.get("/llms.txt"); assert t.status_code == 200 and t.mimetype == "text/plain"
    body = t.get_data(as_text=True); assert "/api/agent/make" in body and "proetida" in body and "isopygous" in body and "theta_joint_deg" in body
    g = client.get("/api/agent").get_json(); assert "proetida" in g["presets"] and g["versions"]["instrument"] == "2.1"
    assert b"/llms.txt" in client.get("/").data                                                  # the page points programs at it

def test_params_table_and_search(client):
    r = client.get("/api/agent/params").get_json(); assert r["count"] == len(schema.PARAMS)
    seg = client.get("/api/agent/params?q=segCount").get_json()["params"]; assert any(p["key"] == "segCount" and p["lo"] < p["hi"] for p in seg)

def test_make_matches_the_page_and_is_idempotent(client):
    r = client.get("/api/agent/make?preset=proetida&segCount=10&chars=isopygous"); j = r.get_json()
    assert r.status_code == 200 and j["status"] == "done" and j["valid"] and j["reading"]["enroll_class"] == "double"
    P = schema.coerce(dict(json.load(open(os.path.join(ROOT, "presets", "proetida.json")))["params"], **schema.CHARACTERS["isopygous"]["set"], segCount=10))
    assert j["hash"] == schema.param_hash(P)                                                     # the same animal the page would make
    assert j["changed_from_preset"]["segCount"] == 10 and "isopygous" in j["characters"] and j["links"]["view"].startswith("/?joint=pin&p=")
    again = client.post("/api/agent/make", json=dict(preset="proetida", chars=["isopygous"], set=dict(segCount=10))).get_json()
    assert again["hash"] == j["hash"] and client.calls.count(j["hash"]) == 1                     # asked twice, measured once
    assert client.get(f"/api/agent/animal/{j['hash']}").get_json()["summary"].startswith("Closes (double)")

def test_each_answer_describes_its_own_request(client):
    a = client.get("/api/agent/make?preset=phacopida&segCount=11&chars=schizochroalEyes").get_json()      # all three are already phacopida's values
    assert a["changed_from_preset"] == {} and any("changed nothing" in n for n in a["notes"]) and any("does not measure on the pin" in n for n in a["notes"])
    b = client.get("/api/agent/make?preset=phacopida").get_json()
    assert b["hash"] == a["hash"] and b["chars_applied"] == [] and a["chars_applied"] == ["schizochroalEyes"]
    p = client.get("/api/agent/preset/phacopida").get_json(); assert p["params"]["segCount"] == 11 and p["measures_on_pin"] is False and p["hash"] == a["hash"]

def test_running_then_done(client):
    r = client.get("/api/agent/make?preset=asaphida&wait=0"); j = r.get_json()
    assert r.status_code == 202 and j["status"] in ("queued", "running") and "ask_again" in j
    for _ in range(40):
        time.sleep(0.05); k = client.get(j["links"]["animal"])
        if k.status_code == 200: break
    assert k.get_json()["status"] == "done"

def test_bad_requests_are_explained(client):
    assert client.get("/api/agent/make?preset=nope").status_code == 400
    assert "unknown parameter" in client.get("/api/agent/make?preset=proetida&segcount=9").get_json()["error"]
    assert "unknown character" in client.get("/api/agent/make?preset=proetida&chars=wings").get_json()["error"]
    j = client.get("/api/agent/make?preset=proetida&segCount=999").get_json(); assert any("segCount" in n for n in j["notes"])   # clamped, and said so

def test_token_and_off_switch(client, monkeypatch):
    monkeypatch.setenv("TRILO_AGENT_TOKEN", "t0k")
    assert client.get("/api/agent/make?preset=proetida").status_code == 401
    assert client.get("/api/agent/make?preset=proetida&token=t0k").status_code == 200
    assert client.get("/api/agent/make?preset=proetida", headers={"Authorization": "Bearer t0k"}).status_code == 200
    assert client.get("/api/agent/log").status_code == 401                                        # the log is gated by the same token
    assert client.get("/api/agent/log?token=t0k").status_code == 200
    monkeypatch.setenv("TRILO_AGENT", "0"); assert client.get("/llms.txt").status_code == 404

def test_log_keeps_every_request(client, tmp_path, monkeypatch):
    monkeypatch.setenv("TRILO_RECORD", "1"); monkeypatch.setenv("TRILO_RECORD_DIR", str(tmp_path)); monkeypatch.delenv("TRILO_RECORD_URL", raising=False)
    client.get("/api/agent/make?preset=proetida&segCount=10")                                     # done
    client.get("/api/agent/make?preset=nope")                                                     # rejected (400)
    client.get("/api/agent/make?preset=proetida&segCount=10")                                     # same animal again: still logged (no dedup)
    log = client.get("/api/agent/log").get_json(); outs = [e["outcome"] for e in log["events"]]
    assert log["count"] >= 3 and outs.count("unknown_preset") == 1 and outs.count("done") >= 2     # repeats and rejections both kept
    assert log["events"][0]["time"] >= log["events"][-1]["time"]                                   # newest first in the JSON
    assert "<table" in client.get("/api/agent/log?format=html").get_data(as_text=True)
