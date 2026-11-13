from __future__ import annotations

import io
import json
import random
import re
import urllib.request
import zipfile
from dataclasses import dataclass, asdict, field
from pathlib import Path

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

DEFAULT_CATEGORIES = [
    "Governing Law",
    "Expiration Date",
    "Renewal Term",
    "Termination For Convenience",
    "Cap On Liability",
    "Non-Compete",
    "Exclusivity",
    "Anti-Assignment",
    "Audit Rights",
    "Insurance",
]

_CATEGORY_RE = re.compile(r'related to "(?P<cat>[^"]+)"')
_DETAILS_RE = re.compile(r"Details:\s*(?P<desc>.+)$")


@dataclass
class Contract:
    doc_id: str
    title: str
    text: str


@dataclass
class EvalItem:
    qid: str
    doc_id: str
    category: str
    question: str
    answers: list[str] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.answers)

    def to_dict(self) -> dict:
        return asdict(self)


def download_cuad(data_dir: str | Path) -> Path:
    raw_dir = Path(data_dir) / "raw"
    target = raw_dir / "test.json"
    if target.exists():
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CUAD_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extract("test.json", raw_dir)
    return target


def make_question(category: str, description: str) -> str:
    return f"What does this contract say about {category.lower()}? Specifically: {description.strip().rstrip('.')}."


def parse_cuad(path: str | Path) -> tuple[list[Contract], list[EvalItem]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    contracts: list[Contract] = []
    items: list[EvalItem] = []
    for i, entry in enumerate(data):
        doc_id = f"doc{i:03d}"
        paragraph = entry["paragraphs"][0]
        context = paragraph["context"]
        contracts.append(Contract(doc_id=doc_id, title=entry["title"], text=context))
        for qa in paragraph["qas"]:
            cat_match = _CATEGORY_RE.search(qa["question"])
            if not cat_match:
                continue
            category = cat_match.group("cat")
            desc_match = _DETAILS_RE.search(qa["question"])
            description = desc_match.group("desc") if desc_match else category
            answers = [] if qa.get("is_impossible") else qa["answers"]
            spans = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in answers]
            items.append(
                EvalItem(
                    qid=f"{doc_id}::{category}",
                    doc_id=doc_id,
                    category=category,
                    question=make_question(category, description),
                    answers=[a["text"] for a in answers],
                    spans=spans,
                )
            )
    return contracts, items


def build_eval_set(
    contracts: list[Contract],
    items: list[EvalItem],
    n_contracts: int = 10,
    categories: list[str] | None = None,
    seed: int = 7,
) -> tuple[list[Contract], list[EvalItem]]:
    categories = categories or DEFAULT_CATEGORIES
    rng = random.Random(seed)
    chosen = rng.sample(contracts, min(n_contracts, len(contracts)))
    chosen_ids = {c.doc_id for c in chosen}
    subset = [it for it in items if it.doc_id in chosen_ids and it.category in categories]
    return chosen, subset


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_set(data_dir: str | Path) -> tuple[list[Contract], list[EvalItem]]:
    data_dir = Path(data_dir)
    contracts = [Contract(**r) for r in load_jsonl(data_dir / "contracts.jsonl")]
    items = []
    for r in load_jsonl(data_dir / "eval_set.jsonl"):
        r["spans"] = [tuple(s) for s in r["spans"]]
        items.append(EvalItem(**r))
    return contracts, items
