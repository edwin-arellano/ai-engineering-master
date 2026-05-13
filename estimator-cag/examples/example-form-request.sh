#!/usr/bin/env bash
# Ejemplo de uso del endpoint /api/v1/estimate post-pre-session-04.
# El request es un formulario tipado: description + 3 enums.
set -euo pipefail

curl -sS -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Mobile app with login, chat and push notifications for a fitness tracker product targeting iOS and Android.",
    "project_type": "mobile_app",
    "detail_level": "detailed",
    "output_format": "phases_table"
  }' | jq .
