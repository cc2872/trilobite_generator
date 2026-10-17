"""The site password (web/app.py, 17 Oct 2026): with TRILO_SITE_PASSWORD set, pages redirect to /login and the API answers 401
until the cookie is there; the agent token stands in on /api/agent/* and /llms.txt; the cache routes refuse traversal."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"))
import pytest


@pytest.fixture
def client(monkeypatch):
    import app as A
    monkeypatch.setattr(A, "SITE_PASSWORD", "sesame"); monkeypatch.setenv("TRILO_AGENT_TOKEN", "agt")
    A.app.config["TESTING"] = True
    with A.app.test_client() as c: yield c


def test_pages_and_api_are_gated_until_login(client):
    r = client.get("/"); assert r.status_code == 302 and r.headers["Location"].startswith("/login")
    assert client.get("/api/schema").status_code == 401
    assert client.get("/files/abc/def").status_code == 401
    assert client.get("/login").status_code == 200
    assert client.post("/login", data={"password": "wrong"}).status_code == 401
    r = client.post("/login", data={"password": "sesame", "next": "/body"}); assert r.status_code == 302 and r.headers["Location"].endswith("/body")
    assert client.get("/").status_code == 200 and client.get("/api/schema").status_code == 200
    assert client.get("/logout").status_code == 302 and client.get("/").status_code == 302


def test_agent_token_stands_in_for_programs(client):
    assert client.get("/llms.txt").status_code == 302
    assert client.get("/llms.txt?token=agt").status_code == 200
    assert client.get("/llms.txt", headers={"Authorization": "Bearer agt"}).status_code == 200


def test_login_next_stays_on_site(client):
    r = client.post("/login", data={"password": "sesame", "next": "https://evil.example/"}); assert r.headers["Location"] == "/"


def test_cache_routes_refuse_traversal_and_unknown_keys(client):
    client.post("/login", data={"password": "sesame"})
    assert client.get("/files/../agent_token.txt").status_code == 404
    assert client.get("/files/..%2fagent_token.txt").status_code == 404
    assert client.get("/api/stl/nonexistent").status_code == 404
    assert client.get("/api/stl/..%2f..%2fweb").status_code == 404


def test_open_when_no_password_is_set(monkeypatch):
    import app as A
    monkeypatch.setattr(A, "SITE_PASSWORD", "")
    with A.app.test_client() as c:
        assert c.get("/").status_code == 200 and c.get("/login").status_code == 302
