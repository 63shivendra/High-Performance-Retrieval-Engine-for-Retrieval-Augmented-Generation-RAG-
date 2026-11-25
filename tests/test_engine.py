"""
Pytest suite for the SelfIndex retrieval engine.

Covers:
  * Boolean retrieval (x=1): AND / OR / NOT / PHRASE / parentheses / precedence
  * TAAT vs DAAT equivalence
  * TF ranking (x=2) and TF-IDF ranking (x=3)
  * Elias and Zlib postings compression round-trips
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compression import EliasCompressor, ZlibCompressor  # noqa: E402
from src.self_indexer import SelfIndexer  # noqa: E402
from src.self_indexer_x2 import SelfIndexer_x2  # noqa: E402
from src.self_indexer_x3 import SelfIndexer_x3  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_index_dir(tmp_path, monkeypatch):
    """create_index() persists to ./indices — keep that inside a temp dir."""
    monkeypatch.chdir(tmp_path)


def build(cls, docs):
    indexer = cls()
    indexer.create_index("test", docs)
    return indexer


# --------------------------------------------------------------------------
# Boolean retrieval (x=1)
# --------------------------------------------------------------------------
FRUIT_DOCS = [
    {"doc_id": "doc1", "content": "apple banana", "title": "Doc 1"},
    {"doc_id": "doc2", "content": "apple orange", "title": "Doc 2"},
    {"doc_id": "doc3", "content": "banana orange", "title": "Doc 3"},
    {"doc_id": "doc4", "content": "grape apple", "title": "Doc 4"},
    {"doc_id": "doc5", "content": "grape banana orange", "title": "Doc 5"},
]

BOOLEAN_CASES = [
    ('"apple"', ["doc1", "doc2", "doc4"]),
    ('"mango"', []),
    ('"apple" AND "banana"', ["doc1"]),
    ('"banana" AND "orange"', ["doc3", "doc5"]),
    ('"apple" OR "grape"', ["doc1", "doc2", "doc4", "doc5"]),
    ('NOT "apple"', ["doc3", "doc5"]),
    ('"apple" AND NOT "banana"', ["doc2", "doc4"]),
    ('("apple" AND "banana") OR "grape"', ["doc1", "doc4", "doc5"]),
    ('"apple" AND ("banana" OR "orange")', ["doc1", "doc2"]),
    ('("apple" AND "banana") OR ("orange" AND NOT "grape")', ["doc1", "doc2", "doc3"]),
    # precedence: NOT > AND > OR
    ('"apple" OR "banana" AND "orange"', ["doc1", "doc2", "doc3", "doc4", "doc5"]),
    ('"apple" AND "banana" OR "grape"', ["doc1", "doc4", "doc5"]),
    # case-insensitive operators and terms
    ('"Apple" and "BANANA"', ["doc1"]),
]


@pytest.fixture
def boolean_index():
    return build(SelfIndexer, FRUIT_DOCS)


@pytest.mark.parametrize("query,expected", BOOLEAN_CASES)
def test_boolean_taat(boolean_index, query, expected):
    assert sorted(boolean_index.query(query)) == expected


@pytest.mark.parametrize("query,expected", BOOLEAN_CASES)
def test_boolean_daat_matches_taat(boolean_index, query, expected):
    assert sorted(boolean_index.query_daat(query)) == expected


def test_boolean_malformed_queries_return_empty(boolean_index):
    assert boolean_index.query('("apple" AND "banana"') == []
    assert boolean_index.query('"apple" "banana"') == []
    assert boolean_index.query("") == []


def test_phrase_query_respects_word_order():
    docs = [
        {"doc_id": "d1", "content": "the united states of america", "title": "t"},
        {"doc_id": "d2", "content": "states that are united", "title": "t"},
    ]
    idx = build(SelfIndexer, docs)
    assert idx.query('PHRASE "united states"') == ["d1"]
    assert idx.query('PHRASE "united" "states"') == ["d1"]
    assert sorted(idx.query('"united" AND "states"')) == ["d1", "d2"]


def test_skip_pointer_index_gives_same_results():
    plain = build(SelfIndexer, FRUIT_DOCS)
    skip = SelfIndexer(optim="Skipping")
    skip.create_index("test", FRUIT_DOCS)
    for query, _ in BOOLEAN_CASES:
        assert sorted(skip.query(query)) == sorted(plain.query(query)), query


# --------------------------------------------------------------------------
# Ranked retrieval (x=2 TF, x=3 TF-IDF)
# --------------------------------------------------------------------------
RANK_DOCS = [
    {"doc_id": "doc1", "title": "Python Guide", "content": "python python python programming language"},
    {"doc_id": "doc2", "title": "Java Guide", "content": "java programming language"},
    {"doc_id": "doc3", "title": "Python Basics", "content": "python easy learn"},
]


@pytest.mark.parametrize("mode", ["TAAT", "DAAT"])
def test_tf_ranking_orders_by_term_frequency(mode):
    idx = build(SelfIndexer_x2, RANK_DOCS)
    assert idx.query("python", mode=mode) == ["doc1", "doc3"]
    assert idx.query("python programming", mode=mode)[0] == "doc1"
    assert len(idx.query("python programming language", mode=mode, top_k=2)) == 2
    assert idx.query("nonexistentterm", mode=mode) == []


def test_tfidf_idf_values():
    docs = [
        {"doc_id": f"d{i}", "title": "t", "content": text}
        for i, text in enumerate(
            ["common rare", "common word", "common other", "unrelated thing"]
        )
    ]
    idx = build(SelfIndexer_x3, docs)
    assert idx.num_documents == 4
    assert idx.idf_scores["common"] == pytest.approx(math.log(4 / 3))
    assert idx.idf_scores["rare"] == pytest.approx(math.log(4 / 1))


@pytest.mark.parametrize("mode", ["TAAT", "DAAT"])
def test_tfidf_prefers_rare_terms(mode):
    docs = [
        {"doc_id": "a", "title": "t", "content": "common common common filler"},
        {"doc_id": "b", "title": "t", "content": "common rare"},
        {"doc_id": "c", "title": "t", "content": "common thing"},
        {"doc_id": "d", "title": "t", "content": "common stuff"},
    ]
    idx = build(SelfIndexer_x3, docs)
    # "common" appears everywhere (idf=0), so the rare-term doc must win.
    assert idx.query("common rare", mode=mode)[0] == "b"


def test_tfidf_taat_and_daat_agree():
    idx = build(SelfIndexer_x3, RANK_DOCS)
    for q in ["python", "programming language", "java python"]:
        assert set(idx.query(q, mode="TAAT")) == set(idx.query(q, mode="DAAT"))


# --------------------------------------------------------------------------
# Compression
# --------------------------------------------------------------------------
POSTINGS_INDEX = {
    "appl": [["wiki_1", 3, [0, 5, 9]], ["wiki_7", 1, [2]], ["wiki_30", 2, [1, 40]]],
    "banana": [["wiki_2", 1, [3]]],
}


@pytest.mark.parametrize("compressor", [EliasCompressor, ZlibCompressor])
def test_compression_round_trip(compressor):
    compressed = compressor.compress_inverted_index(POSTINGS_INDEX)
    restored = compressor.decompress_inverted_index(compressed)
    for term, postings in POSTINGS_INDEX.items():
        got = {p[0]: list(p[-1]) for p in restored[term]}
        want = {p[0]: p[-1] for p in postings}
        assert got == want, term
