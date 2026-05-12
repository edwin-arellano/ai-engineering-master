#!/usr/bin/env bash
# Extended thinking en Anthropic Claude Haiku 4.5 con budget de 2000 tokens.
# Sólo aplica si LLM_PROVIDER=anthropic. En OpenAI gpt-4o-mini este parámetro se ignora.
# OJO: los thinking_tokens cuentan como tokens de salida → coste mayor.
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., thinking_budget: 2000, max_tokens: 6000}' fixtures/transcriptions/long_transcription.txt)" \
  | jq .
