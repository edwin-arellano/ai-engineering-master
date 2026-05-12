#!/usr/bin/env bash
# Output del LLM en JSON con schema documentado. El evaluator parsea el JSON
# y aplica los mismos 7 chequeos (con adaptaciones por formato).
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., output_format: "json"}' fixtures/transcriptions/short_transcription.txt)" \
  | jq .
