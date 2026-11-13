import argparse
import json

from contractlens.pipeline import ContractLens


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about an indexed contract.")
    parser.add_argument("question")
    parser.add_argument("--contract-id")
    args = parser.parse_args()

    lens = ContractLens.from_settings()
    if args.contract_id and args.contract_id not in lens.index.doc_ids():
        raise SystemExit(f"unknown contract id {args.contract_id}; available: {', '.join(lens.index.doc_ids())}")
    result = lens.ask_sync(args.question, args.contract_id)
    print(json.dumps({k: result[k] for k in ("decision", "answer", "quotes", "citations", "confidence", "escalated", "latency_ms")}, indent=2))


if __name__ == "__main__":
    main()
