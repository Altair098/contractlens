import argparse
from pathlib import Path

from contractlens.chunking import chunk_pages, chunk_text
from contractlens.config import get_settings
from contractlens.data import load_eval_set
from contractlens.index import HybridIndex, get_embedder
from contractlens.parsing import get_parser


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Chunk contracts and build the hybrid index.")
    parser.add_argument("--data-dir", default=str(settings.data_dir))
    parser.add_argument("--index-dir", default=str(settings.index_dir))
    parser.add_argument("--embedder", default=settings.embedder)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--pdf-dir", help="Also index every PDF in this folder (doc_id = file stem).")
    args = parser.parse_args()

    chunks = []
    contracts_file = Path(args.data_dir) / "contracts.jsonl"
    if contracts_file.exists():
        contracts, _ = load_eval_set(args.data_dir)
        for c in contracts:
            chunks.extend(chunk_text(c.doc_id, c.text, args.chunk_size, args.chunk_overlap))

    if args.pdf_dir:
        pdf_parser = get_parser("local", ocr=False)
        for pdf in sorted(Path(args.pdf_dir).glob("*.pdf")):
            _, pdf_chunks = chunk_pages(pdf.stem, pdf_parser.parse(pdf), args.chunk_size, args.chunk_overlap)
            chunks.extend(pdf_chunks)

    if not chunks:
        raise SystemExit("nothing to index: run scripts/prepare_data.py or pass --pdf-dir")

    index = HybridIndex(get_embedder(args.embedder))
    index.build(chunks)
    index.save(args.index_dir)
    print(f"indexed {len(chunks)} chunks from {len(index.doc_ids())} documents -> {args.index_dir}")


if __name__ == "__main__":
    main()
