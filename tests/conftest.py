import pytest

from contractlens.chunking import chunk_text
from contractlens.config import get_settings
from contractlens.data import Contract, EvalItem
from contractlens.index import HybridIndex, TfidfEmbedder

CONTRACT_A = """MASTER SERVICES AGREEMENT

1. Term. This Agreement commences on January 1, 2020 and expires on December 31, 2022 unless terminated earlier.

2. Governing Law. This Agreement shall be governed by and construed in accordance with the laws of the State of New York.

3. Limitation of Liability. In no event shall either party's aggregate liability exceed the fees paid in the twelve months preceding the claim.

4. Assignment. Neither party may assign this Agreement without the prior written consent of the other party.
"""

CONTRACT_B = """SUPPLY AGREEMENT

1. Supply. Supplier shall deliver the products listed in Exhibit A at the prices stated therein.

2. Governing Law. This Agreement is governed by the laws of England and Wales.

3. Payment. Buyer shall pay all invoices within thirty days of receipt.
"""


@pytest.fixture
def contracts() -> list[Contract]:
    return [Contract("docA", "MSA", CONTRACT_A), Contract("docB", "Supply", CONTRACT_B)]


@pytest.fixture
def eval_items() -> list[EvalItem]:
    law = "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."
    start = CONTRACT_A.index(law)
    return [
        EvalItem("docA::Governing Law", "docA", "Governing Law",
                 "What does this contract say about governing law? Specifically: Which state's laws govern the agreement.",
                 [law], [(start, start + len(law))]),
        EvalItem("docB::Non-Compete", "docB", "Non-Compete",
                 "What does this contract say about non-compete? Specifically: Restriction on competing with the counterparty.",
                 [], []),
    ]


@pytest.fixture
def settings(tmp_path):
    return get_settings(
        model="fake",
        escalation_model="",
        embedder="tfidf",
        nli_model="lexical",
        index_dir=tmp_path / "index",
        log_dir=tmp_path / "logs",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
        confidence_threshold=0.6,
        use_verifier=True,
    )


@pytest.fixture
def index(contracts, settings) -> HybridIndex:
    chunks = []
    for c in contracts:
        chunks.extend(chunk_text(c.doc_id, c.text, settings.chunk_size, settings.chunk_overlap))
    idx = HybridIndex(TfidfEmbedder())
    idx.build(chunks)
    idx.save(settings.index_dir)
    return idx
