#!/usr/bin/env bash
# Preprocesado inline_cleaning sobre la transcripción larga (ruidosa).
# Se le instruye al modelo, dentro del system prompt, a ignorar mentalmente el ruido.
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., preprocessing: "inline_cleaning"}' fixtures/transcriptions/long_transcription.txt)" \
  | jq .
