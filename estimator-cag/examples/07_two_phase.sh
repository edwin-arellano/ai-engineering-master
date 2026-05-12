#!/usr/bin/env bash
# Preprocesado two_phase: primera llamada extrae requisitos limpios, segunda llamada estima.
# Mayor latencia y más tokens (dos llamadas) pero la estimación parte de un input ya destilado.
# Ver en la respuesta el campo `extracted_requirements`.
set -euo pipefail
cd "$(dirname "$0")/.."
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: ., preprocessing: "two_phase"}' fixtures/transcriptions/long_transcription.txt)" \
  | jq .
