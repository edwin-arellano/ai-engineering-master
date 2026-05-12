#!/usr/bin/env bash
# Verifica que el servicio está arriba y que jq está instalado.
set -euo pipefail

if ! command -v jq &>/dev/null; then
  echo "❌ jq no está instalado. Instálalo con tu gestor de paquetes." >&2
  exit 1
fi

if ! curl -s -f http://localhost:8000/health > /dev/null; then
  echo "❌ El servicio no responde en http://localhost:8000/health" >&2
  echo "   Arráncalo con: cd estimator-cag && docker compose up --build" >&2
  exit 1
fi

echo "✅ Todo listo. Puedes ejecutar cualquiera de los scripts examples/0X_*.sh"
