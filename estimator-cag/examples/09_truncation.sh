#!/usr/bin/env bash
# Forzar truncado bajando max_tokens. La evaluación debe detectar finish_reason != stop/end_turn.
# Score esperado: bajo (al menos finish_reason_ok=false, posiblemente totales faltantes).
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., max_tokens: 200}' fixtures/transcriptions/short_transcription.txt)" \
  | jq .
