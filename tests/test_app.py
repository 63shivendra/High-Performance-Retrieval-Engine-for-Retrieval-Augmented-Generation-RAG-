"""Tests for the web demo: query normalisation, snippets and the HTTP API."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import make_snippet, normalize_boolean  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("machine learning", '"machine" AND "learning"'),
    ("apple and not fruit", '"apple" AND NOT "fruit"'),
    ("history of the roman empire", '"history" AND "roman" AND "empire"'),
    ("PHRASE united states of america", 'PHRASE "united" "states" "america"'),
    ('PHRASE "united states" AND war', 'PHRASE "united states" AND "war"'),
    ("(war OR peace) russia", '( "war" OR "peace" ) AND "russia"'),
    ('"cat" OR "dog"', '"cat" OR "dog"'),
    ("the OR cat", '"cat"'),
    ("the", ""),
])
def test_normalize_boolean(raw, expected):
    assert normalize_boolean(raw) == expected


def test_snippet_highlights_stemmed_matches():
    text = "Intro text. " * 40 + "Researchers studied running engines and the engine ran well."
    segs = make_snippet(text, ["engin", "run"])
    hits = [s["text"] for s in segs if s["hit"]]
    assert hits == ["running", "engines", "engine"]
    assert segs[0]["text"] == "… "


@pytest.fixture(scope="module")
def client():
    corpus = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_data", "corpus.jsonl.gz")
    if not os.path.exists(corpus):
        pytest.skip("demo corpus not present")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_api_stats(client):
    s = client.get("/api/stats").json()
    assert s["documents"] > 1000 and s["vocabulary"] > 10000


@pytest.mark.parametrize("model", ["boolean", "tf", "tfidf"])
@pytest.mark.parametrize("mode", ["taat", "daat"])
def test_api_search(client, model, mode):
    r = client.get("/api/search", params={"q": "quantum physics", "model": model, "mode": mode, "k": 5})
    assert r.status_code == 200
    d = r.json()
    assert 0 < len(d["results"]) <= 5
    assert d["latency_ms"] >= 0
    assert all(r["title"] for r in d["results"])


def test_api_taat_daat_same_ranking(client):
    q = {"q": "roman empire history", "model": "tfidf", "k": 10}
    taat = [r["doc_id"] for r in client.get("/api/search", params={**q, "mode": "taat"}).json()["results"]]
    daat = [r["doc_id"] for r in client.get("/api/search", params={**q, "mode": "daat"}).json()["results"]]
    assert taat == daat


def test_api_validation(client):
    assert client.get("/api/search", params={"q": "x", "model": "nope"}).status_code == 400
    assert client.get("/api/search", params={"q": ""}).status_code == 422


def test_api_compare_and_benchmarks(client):
    rows = client.get("/api/compare", params={"q": "solar energy"}).json()["rows"]
    assert len(rows) == 6
    bench = client.get("/api/benchmarks").json()
    assert any(b["id"].endswith("COLD") for b in bench)


def test_pages_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/plots/plot_ES_comparison.png").status_code == 200
