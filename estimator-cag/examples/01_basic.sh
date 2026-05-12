#!/usr/bin/env bash
# Llamada básica: transcripción corta sin opciones extra (toma todos los defaults).
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: .}' fixtures/transcriptions/short_transcription.txt)" \
  | jq .
