import argparse
from pathlib import Path

from contractlens.data import DEFAULT_CATEGORIES, build_eval_set, download_cuad, parse_cuad, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CUAD and build the evaluation subset.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-contracts", type=int, default=10)
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    raw_path = download_cuad(args.data_dir)
    contracts, items = parse_cuad(raw_path)
    contracts, items = build_eval_set(contracts, items, args.n_contracts, args.categories, args.seed)

    out = Path(args.data_dir)
    save_jsonl([c.__dict__ for c in contracts], out / "contracts.jsonl")
    save_jsonl([i.to_dict() for i in items], out / "eval_set.jsonl")
    answerable = sum(i.answerable for i in items)
    print(f"contracts={len(contracts)} questions={len(items)} answerable={answerable} unanswerable={len(items) - answerable}")


if __name__ == "__main__":
    main()
