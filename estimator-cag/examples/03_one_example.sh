#!/usr/bin/env bash
# num_examples=1: score sube notablemente respecto a 0 ejemplos (~0.6).
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., num_examples: 1}' fixtures/transcriptions/short_transcription.txt)" \
  | jq .
