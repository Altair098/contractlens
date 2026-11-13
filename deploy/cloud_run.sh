#!/usr/bin/env bash
# Deploy the API to Cloud Run. Requires a billing-enabled GCP project (the $300 free trial works).
# The model backend must be reachable from Cloud Run: set CL_MODEL to a Gemini model or point
# OLLAMA_API_BASE at a hosted Ollama endpoint.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-contractlens}"

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "CL_MODEL=${CL_MODEL:-gemini-2.5-flash},CL_CONFIDENCE_THRESHOLD=${CL_CONFIDENCE_THRESHOLD:-0.6}" \
  --set-secrets "GOOGLE_API_KEY=${GOOGLE_API_KEY_SECRET:-gemini-api-key}:latest" \
  --no-allow-unauthenticated
