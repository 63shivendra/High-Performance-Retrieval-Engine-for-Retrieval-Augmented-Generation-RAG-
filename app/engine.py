"""
DemoEngine — wraps the SelfIndex indexers (src/) for the live web demo.

On startup it loads a small bundled Wikipedia sample, builds the Boolean (x=1),
TF (x=2) and TF-IDF (x=3) indexes in memory, and exposes search / compare helpers
that time *only* the engine's own query call.
"""
from __future__ import annotations

import gc
import glob
import gzip
import json
import os
import re
import statistics
import sys
import time
from array import array
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocessor import preprocess_text, stemmer, stop_words  # noqa: E402
from src.self_indexer import SelfIndexer  # noqa: E402
from src.self_indexer_x2 import SelfIndexer_x2  # noqa: E402
from src.self_indexer_x3 import SelfIndexer_x3  # noqa: E402

CORPUS = os.path.join(ROOT, "demo_data", "corpus.jsonl.gz")
TOKENS = os.path.join(ROOT, "demo_data", "tokens.jsonl.gz")

MODELS = ("boolean", "tf", "tfidf")
MODES = ("taat", "daat")
OPERATORS = {"AND", "OR", "NOT", "PHRASE"}
MAX_QUERY_LEN = 300


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def prepare_tokens() -> None:
    """Pre-tokenize the corpus once (run at Docker build time)."""
    t0 = time.perf_counter()
    n = 0
    with gzip.open(CORPUS, "rt", encoding="utf-8") as src, \
            gzip.open(TOKENS, "wt", encoding="utf-8") as dst:
        for line in src:
            doc = json.loads(line)
            dst.write(json.dumps([doc["doc_id"], preprocess_text(doc["content"])]) + "\n")
            n += 1
    print(f"Tokenized {n} documents in {time.perf_counter() - t0:.1f}s -> {TOKENS}")


def _iter_docs(meta: Dict[str, Dict]):
    """Stream documents with their cached tokens, one at a time (keeps peak memory low)."""
    if not os.path.exists(TOKENS):
        prepare_tokens()
    with gzip.open(CORPUS, "rt", encoding="utf-8") as cf, \
            gzip.open(TOKENS, "rt", encoding="utf-8") as tf:
        for c_line, t_line in zip(cf, tf):
            doc = json.loads(c_line)
            doc_id, tokens = json.loads(t_line)
            assert doc_id == doc["doc_id"], "tokens cache out of sync — delete it and rebuild"
            meta[doc_id] = {"title": doc["title"], "url": doc["url"], "content": doc["content"]}
            # intern so each distinct term is stored once, not once per occurrence
            doc["tokens"] = [sys.intern(t) for t in tokens]
            yield doc


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()"]+')


def normalize_boolean(query: str) -> str:
    """
    Make free-form input valid for the Boolean parser, which expects quoted terms:
      machine learning          -> "machine" AND "learning"
      apple and not fruit       -> "apple" AND NOT "fruit"
      PHRASE united states      -> PHRASE "united" "states"
    Already-quoted queries pass through with implicit ANDs inserted where needed.
    """
    out: List[str] = []
    in_phrase = False
    for tok in _TOKEN_RE.findall(query):
        up = tok.upper()
        if up in OPERATORS:
            in_phrase = up == "PHRASE"
            if out and up in ("NOT", "PHRASE") and out[-1] not in OPERATORS | {"("}:
                out.append("AND")
            out.append(up)
            continue
        if tok == "(":
            in_phrase = False
            if out and out[-1] not in OPERATORS | {"("}:
                out.append("AND")
            out.append(tok)
            continue
        if tok == ")":
            in_phrase = False
            out.append(tok)
            continue
        # Stopwords were removed at index time, so a stopword term matches nothing.
        # Drop them inside phrases and wherever they'd only be joined by an implicit AND
        # (but keep them after an explicit operator so the expression stays valid).
        explicit_op = bool(out) and out[-1] in OPERATORS
        words = [w for w in tok.strip('"').split()
                 if preprocess_text(w) or (explicit_op and not in_phrase)]
        if not words:
            continue
        if in_phrase:
            out.append(f'"{" ".join(words)}"')
            continue
        for w in words:
            if out and out[-1] not in OPERATORS | {"("}:
                out.append("AND")
            out.append(f'"{w}"')

    # tidy up binary operators left without an operand (e.g. "the OR cat")
    binary = {"AND", "OR"}
    cleaned: List[str] = []
    for tok in out:
        if tok in binary and (not cleaned or cleaned[-1] in OPERATORS | {"("}):
            continue
        if tok == ")" and cleaned and cleaned[-1] in binary:
            cleaned.pop()
        cleaned.append(tok)
    while cleaned and cleaned[-1] in OPERATORS:
        cleaned.pop()
    return " ".join(cleaned)


def query_stems(query: str) -> List[str]:
    cleaned = re.sub(r"\b(AND|OR|NOT|PHRASE)\b", " ", query)
    return list(dict.fromkeys(preprocess_text(cleaned)))


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return stemmer.stem(word)


_WORD_RE = re.compile(r"[A-Za-z]+")


def make_snippet(text: str, stems: List[str], width: int = 260) -> List[Dict]:
    """Return the best text window as [{text, hit}] segments (rendered safely client-side)."""
    text = re.sub(r"\s+", " ", text)
    targets = set(stems)
    prefixes = {s[:3] for s in stems}
    hits = []
    for m in _WORD_RE.finditer(text):
        w = m.group().lower()
        if len(w) > 2 and w[:3] in prefixes and w not in stop_words and _stem(w) in targets:
            hits.append((m.start(), m.end()))

    start = 0
    if hits:
        # anchor the window on the hit followed by the most other hits
        reach = width * 2 // 3
        anchor = max(hits, key=lambda h: sum(1 for o in hits if h[0] <= o[0] < h[0] + reach))
        start = max(0, anchor[0] - width // 3)
    end = min(len(text), start + width)
    if start > 0:
        start = text.find(" ", start) + 1 or start
    if end < len(text):
        end = text.rfind(" ", start, end) if text.rfind(" ", start, end) > start else end

    segments: List[Dict] = []
    if start > 0:
        segments.append({"text": "… ", "hit": False})
    cursor = start
    for s, e in hits:
        if s < start or e > end:
            continue
        if s > cursor:
            segments.append({"text": text[cursor:s], "hit": False})
        segments.append({"text": text[s:e], "hit": True})
        cursor = e
    if cursor < end:
        segments.append({"text": text[cursor:end], "hit": False})
    if end < len(text):
        segments.append({"text": " …", "hit": False})
    return segments


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class DemoEngine:
    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.meta: Dict[str, Dict] = {}
        self.build_seconds: Dict[str, float] = {}

        # TF-IDF (x=3) is built by the real indexer from a streamed corpus.
        self.tfidf = SelfIndexer_x3()
        self.tfidf._save_index = lambda *_: None  # in-memory only for the demo
        t = time.perf_counter()
        self.tfidf.create_index("demo", _iter_docs(self.meta))
        self.build_seconds["tfidf"] = time.perf_counter() - t

        # Pack position lists into compact C int arrays (~8x smaller than lists of ints).
        # Query code only iterates / tests membership, which arrays support.
        index = self.tfidf.inverted_index
        for postings in index.values():
            for p in postings:
                p[2] = array("I", p[2])
        self.num_tokens = sum(p[1] for postings in index.values() for p in postings)

        # Fit the demo in a 512 MB container by sharing postings between models:
        #  * TF (x=2) uses the identical layout [doc_id, tf, positions] -> share outright.
        #  * Boolean (x=1) uses [doc_id, positions] -> new thin lists, same position arrays.
        self.tf = SelfIndexer_x2()
        self.tf.inverted_index = index
        self.tf.documents = self.tfidf.documents
        self.build_seconds["tf"] = 0.0

        t = time.perf_counter()
        self.boolean = SelfIndexer()
        self.boolean.inverted_index = defaultdict(
            list, {term: [[p[0], p[2]] for p in postings] for term, postings in index.items()})
        self.boolean.documents = self.tfidf.documents
        self.build_seconds["boolean"] = time.perf_counter() - t
        gc.collect()

        self.vocab_size = len(self.tfidf.inverted_index)
        self.num_postings = sum(len(p) for p in self.tfidf.inverted_index.values())
        self.startup_seconds = time.perf_counter() - t0
        self.benchmarks = load_benchmarks()

    # -- core -------------------------------------------------------------
    def _run(self, model: str, mode: str, query: str, k: int) -> List[str]:
        if model == "boolean":
            return self.boolean.query_daat(query) if mode == "daat" else self.boolean.query(query)
        idx = self.tf if model == "tf" else self.tfidf
        return idx.query(query, mode=mode.upper(), top_k=k)

    def _scores(self, model: str, stems: List[str], doc_ids: List[str]) -> Dict[str, float]:
        if model == "boolean":
            return {}
        wanted = set(doc_ids)
        scores = dict.fromkeys(doc_ids, 0.0)
        for term in stems:
            weight = 1.0 if model == "tf" else self.tfidf.idf_scores.get(term, 0.0)
            for posting in self.tfidf.inverted_index.get(term, ()):
                if posting[0] in wanted:
                    scores[posting[0]] += posting[1] * weight
        return scores

    def search(self, query: str, model: str, mode: str, k: int = 10) -> Dict:
        query = query[:MAX_QUERY_LEN]
        executed = normalize_boolean(query) if model == "boolean" else query
        stems = query_stems(executed)

        t = time.perf_counter()
        doc_ids = self._run(model, mode, executed, k)
        latency_ms = (time.perf_counter() - t) * 1000

        total = len(doc_ids)
        shown = doc_ids[:k]
        scores = self._scores(model, stems, shown)
        results = []
        for rank, doc_id in enumerate(shown, 1):
            m = self.meta.get(doc_id, {})
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": m.get("title", doc_id),
                "url": m.get("url"),
                "score": round(scores[doc_id], 3) if doc_id in scores else None,
                "snippet": make_snippet(m.get("content", ""), stems),
            })
        return {
            "query": query,
            "executed_query": executed,
            "terms": stems,
            "model": model,
            "mode": mode,
            "latency_ms": round(latency_ms, 3),
            "total_matches": total if model == "boolean" else None,
            "results": results,
        }

    def compare(self, query: str, k: int = 10, runs: int = 7) -> List[Dict]:
        query = query[:MAX_QUERY_LEN]
        rows = []
        for model in MODELS:
            executed = normalize_boolean(query) if model == "boolean" else query
            for mode in MODES:
                timings = []
                for _ in range(runs):
                    t = time.perf_counter()
                    res = self._run(model, mode, executed, k)
                    timings.append((time.perf_counter() - t) * 1000)
                rows.append({
                    "model": model,
                    "mode": mode,
                    "median_ms": round(statistics.median(timings), 3),
                    "min_ms": round(min(timings), 3),
                    "results": len(res),
                    "top": [self.meta[d]["title"] for d in res[:3] if d in self.meta],
                })
        return rows

    def stats(self) -> Dict:
        return {
            "documents": len(self.meta),
            "tokens": self.num_tokens,
            "vocabulary": self.vocab_size,
            "postings": self.num_postings,
            "build_seconds": {k: round(v, 2) for k, v in self.build_seconds.items()},
            "startup_seconds": round(self.startup_seconds, 2),
            "corpus": "English Wikipedia (20231101.en), first ~4K articles, truncated to 3.5K chars",
        }


def load_benchmarks() -> List[Dict]:
    """Read the full-scale (100K docs, 256 queries) evaluation results from results/."""
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "eval_*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        lat = d.get("artifact_A_latency")
        if not lat or "before_optimization" in path or "_OLD_" in path:
            continue
        name = os.path.basename(path)[5:-5]
        system = "Elasticsearch" if name.startswith("esindex") else "SelfIndex"
        rows.append({
            "id": name,
            "system": system,
            "p50_ms": round(lat.get("p50_ms", 0), 2),
            "p95_ms": round(lat.get("p95_ms", 0), 2),
            "p99_ms": round(lat.get("p99_ms", 0), 2),
            "qps": round(d.get("artifact_B_throughput", {}).get("queries_per_second", 0)),
            "disk_mb": round(d.get("artifact_C_memory", {}).get("disk_mb", 0)),
        })
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare_tokens()
