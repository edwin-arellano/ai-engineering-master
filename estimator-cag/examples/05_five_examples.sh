#!/usr/bin/env bash
# num_examples=5: saturación de contexto. Score puede bajar respecto a 3 por overfitting.
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., num_examples: 5}' fixtures/transcriptions/short_transcription.txt)" \
  | jq .
