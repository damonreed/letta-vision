from letta.services.recall.hybrid_search import (
    FILE_CONTENTS_SEARCH_MAX_CHARS,
    HybridHit,
    fuse_rrf,
    truncate_at_word_boundary,
)


def test_fuse_rrf_boosts_overlap():
    vector = [
        HybridHit(layer="archival", handle="p1", score=1.0, reasons=["vector"], snippet="alpha"),
    ]
    lexical = [
        HybridHit(layer="archival", handle="p1", score=0.9, reasons=["lexical"], snippet="alpha"),
        HybridHit(layer="archival", handle="p2", score=0.8, reasons=["lexical"], snippet="beta"),
    ]
    fused = fuse_rrf(vector, lexical, limit=5)
    assert fused[0].handle == "p1"
    assert fused[0].score > fused[1].score
    assert "vector" in fused[0].reasons
    assert "lexical" in fused[0].reasons


def test_fuse_rrf_respects_limit():
    hits = [HybridHit(layer="file", handle=f"p{i}", score=1.0 / (i + 1), reasons=["vector"]) for i in range(10)]
    fused = fuse_rrf(hits, [], limit=3)
    assert len(fused) == 3


def test_truncate_at_word_boundary():
    text = "word " * 300
    out = truncate_at_word_boundary(text, FILE_CONTENTS_SEARCH_MAX_CHARS)
    assert len(out) <= FILE_CONTENTS_SEARCH_MAX_CHARS + 1
    assert out.endswith("…")


def test_truncate_short_text_unchanged():
    assert truncate_at_word_boundary("hello world", 100) == "hello world"
