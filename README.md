# ContractLens

Grounded, self-verifying contract review agents built with **Google ADK** and **Gemma**, evaluated on real commercial contracts and tracked with **MLflow**.

ContractLens answers questions about a contract ("Can either party terminate for convenience?", "Is liability capped?") with verbatim, cited clauses. A verifier agent checks every answer before it is returned. Answers that fail verification are escalated to a stronger model or withheld, so the system says "insufficient evidence" instead of hallucinating.

## Architecture

```
            ┌──────────────── ADK root agent: ContractLensAgent ────────────────┐
question ──►│ retrieve ──► answerer (LlmAgent) ──► verifier ──► decision        │──► answer + citations
            │ hybrid       Gemma 3 4B via        quote check     ≥ τ  answered  │    + confidence
            │ BM25+dense   Ollama, JSON mode     NLI entailment  < τ  escalate ─┼──► escalator (Gemma 12B / Gemini)
            │ RRF fusion                         Q-evidence rel.      or abstain│
            └───────────────────────────────────────────────────────────────────┘
                     │                                      │
          contracts ─┘ PDF parser (PyMuPDF | Document AI)   └─► MLflow traces, metrics ─► BigQuery
```

| Component | Implementation |
|---|---|
| Agent framework | Google ADK: custom `BaseAgent` orchestrator with `LlmAgent` sub-agents, session state, `adk web` UI |
| LLMs | Gemma 3 4B (answerer) and Gemma 3 12B (escalation) via Ollama; any Gemini model via `CL_MODEL=gemini-2.5-flash` |
| Retrieval | Hybrid BM25 + `bge-small-en-v1.5` dense retrieval, reciprocal rank fusion, per-contract filtering |
| Grounding | Prompted for verbatim quotes, quotes fuzzy-matched against retrieved text, page/offset citations |
| Verification | Weighted confidence from quote support, NLI entailment (`nli-deberta-v3-xsmall`), question–evidence relevance |
| Evaluation | CUAD lawyer-labelled spans: retrieval hit rate, token F1, coverage, abstention accuracy, false-answer rate, latency |
| Experiment tracking | MLflow runs (params, metrics, prompt versions, per-question artifacts), MLflow tracing, threshold-sweep curves |
| Serving | FastAPI (`/ask`, `/contracts` PDF upload, `/health`), Docker image compatible with Cloud Run |
| Analytics | Evaluation results and request logs batch-loaded into BigQuery (free sandbox), SQL in `sql/analysis.sql` |

## Results

Evaluation set: 10 CUAD contracts × 10 clause categories = 100 questions (46 answerable, 54 with no relevant clause).

| Run | Accuracy | Retrieval hit | Answer F1 | Coverage | False-answer rate | Avg latency |
|---|---|---|---|---|---|---|
| gemma4b-v1-baseline | | | | | | |
| gemma4b-v2-prompt | | | | | | |
| gemma4b-v2-verifier | | | | | | |
| gemma4b-v2-verifier-escalate12b | | | | | | |

`python scripts/compare_runs.py` prints this table from MLflow. `python scripts/threshold_sweep.py results/<run>.csv` plots accuracy, coverage and false-answer rate across verifier thresholds without re-running the model.

## Quickstart

```bash
pip install -e ".[dev]"

ollama pull gemma3:4b
export OLLAMA_CONTEXT_LENGTH=8192   # set before `ollama serve`

python scripts/prepare_data.py --n-contracts 10
python scripts/build_index.py
python scripts/ask.py "Is there a cap on liability?" --contract-id doc041

python scripts/run_eval.py --run-name gemma4b-v2-verifier --trace
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Without a GPU, `notebooks/kaggle_eval.ipynb` runs all experiments on a free Kaggle or Colab T4.

### Agent UI

```bash
adk web agents
```

### API

```bash
uvicorn service.app:app --port 8080
curl -X POST localhost:8080/ask -H "Content-Type: application/json" \
  -d '{"question": "Which law governs this agreement?", "contract_id": "doc041"}'
curl -X POST localhost:8080/contracts -F "file=@my_contract.pdf"
```

### Docker

```bash
docker build -t contractlens .
docker run -p 8080:8080 -e OLLAMA_API_BASE=http://host.docker.internal:11434 contractlens
```

### BigQuery

```bash
pip install -e ".[gcp]"
gcloud auth application-default login
python scripts/export_bigquery.py --project <your-project-id>
```

## Configuration

All settings are environment variables (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `CL_MODEL` | `ollama_chat/gemma3:4b` | Answerer model (`fake` runs offline) |
| `CL_ESCALATION_MODEL` | empty | Second model for low-confidence answers |
| `CL_EMBEDDER` | `st:BAAI/bge-small-en-v1.5` | Dense embedder (`tfidf` runs offline) |
| `CL_NLI_MODEL` | `cross-encoder/nli-deberta-v3-xsmall` | Entailment model (`lexical` runs offline) |
| `CL_PROMPT_VERSION` | `v2` | Prompt template version |
| `CL_TOP_K` | `4` | Retrieved chunks per question |
| `CL_CONFIDENCE_THRESHOLD` | `0.6` | Verifier acceptance threshold |

## Design decisions

- **Retrieval runs outside the LLM.** Small open models have unreliable tool calling, so the orchestrator retrieves deterministically and passes the excerpts through ADK session state. The answer agent only has to read and quote.
- **The verifier checks three things.** Faithful quotes alone do not catch an answer that quotes an irrelevant clause, so confidence combines quote fidelity, entailment and question–evidence relevance.
- **Escalation, not a bigger model everywhere.** Only answers below the threshold go to the larger model, which keeps cost and latency close to the small model's.
- **Evaluation uses CUAD unanswerable questions.** More than half of the questions have no relevant clause, so the false-answer rate measures hallucination directly.

## GCP deployment path

The project runs entirely on local and free resources. The GCP integrations are behind interfaces:

| Local (default) | GCP |
|---|---|
| `LocalPdfParser` (PyMuPDF) | `DocumentAIParser` in `parsing.py` |
| Ollama Gemma | `CL_MODEL=gemini-2.5-flash` (native ADK Gemini) |
| `uvicorn` / Docker | `deploy/cloud_run.sh`, or `adk deploy cloud_run` / `adk deploy agent_engine` for the agent |
| Local MLflow | BigQuery sandbox analytics (works now, no billing needed) |

## Project structure

```
src/contractlens/   config, data (CUAD), parsing, chunking, index, prompts, models,
                    verifier, agents (ADK), pipeline, evaluation, tracking, bq
agents/             ADK entry point for `adk web` / `adk run`
scripts/            prepare_data, build_index, ask, run_eval, threshold_sweep, compare_runs, export_bigquery
service/            FastAPI app
sql/                BigQuery analysis queries
notebooks/          Kaggle / Colab evaluation notebook
tests/              offline tests (fake model, TF-IDF, lexical NLI)
```

## Data

[CUAD v1](https://github.com/TheAtticusProject/cuad) (Contract Understanding Atticus Dataset), CC BY 4.0: 510 commercial contracts with expert annotations across 41 clause categories.
