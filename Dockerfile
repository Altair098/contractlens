FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    CL_INDEX_DIR=/app/data/index \
    CL_LOG_DIR=/app/logs \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[gcp]"

COPY service ./service
COPY data/index ./data/index

RUN python -c "from contractlens.config import get_settings as g; from contractlens.index import get_embedder; from contractlens.verifier import NliScorer; s=g(); get_embedder(s.embedder); NliScorer(s.nli_model)"

EXPOSE 8080
CMD exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT}
