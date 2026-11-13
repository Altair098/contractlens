from contractlens.chunking import chunk_pages, chunk_text
from contractlens.index import HybridIndex, TfidfEmbedder


def test_chunk_offsets_match_source():
    text = "Clause one is here. " * 200
    chunks = chunk_text("d", text, size=300, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert text[c.start:c.end] == c.text
    assert chunks[-1].end == len(text)


def test_chunk_pages_tracks_page_numbers():
    full, chunks = chunk_pages("d", ["page one text " * 10, "page two text " * 10], size=100, overlap=20)
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        assert full[c.start:c.end] == c.text


def test_search_filters_by_document(index):
    hits = index.search("governing law", top_k=3, doc_id="docB")
    assert hits and all(h.chunk.doc_id == "docB" for h in hits)
    assert "England" in hits[0].chunk.text


def test_index_roundtrip(index, settings):
    loaded = HybridIndex.load(settings.index_dir, TfidfEmbedder())
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.search("assignment consent", top_k=1, doc_id="docA")[0].chunk.chunk_id == \
        index.search("assignment consent", top_k=1, doc_id="docA")[0].chunk.chunk_id
