#!/usr/bin/env bash
# num_examples=3: el "punto dulce" según la sesión en vivo (score ~0.87).
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., num_examples: 3}' fixtures/transcriptions/short_transcription.txt)" \
  | jq .
