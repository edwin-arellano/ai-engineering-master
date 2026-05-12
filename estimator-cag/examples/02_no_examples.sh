#!/usr/bin/env bash
# num_examples=0: sin contexto CAG. Score esperado muy bajo (~0.1-0.3).
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., num_examples: 0}' fixtures/transcriptions/short_transcription.txt)" \
  | jq .
