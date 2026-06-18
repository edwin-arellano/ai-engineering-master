"""Cliente del trace del diagnóstico pre-S09. NO es código de producto: solo ejecuta las
llamadas de la sección 2 del ejercicio contra el servicio tal como está (S08) y captura
las respuestas crudas. Único Python permitido por el ejercicio.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import httpx

from app.generation.rag.embedding.embedder import LiteLLMEmbedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", default="examples/transcripts/02_ambiguous.txt")
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="examples/transcripts/trace_02_ambiguous.out.txt")
    args = parser.parse_args()

    text = Path(args.transcript).read_text(encoding="utf-8")
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    # --- Paso 1 del trace: embedding de la transcripción completa ---
    emit("=" * 80)
    emit(f"[1] EMBEDDING de {args.transcript}  (chars={len(text)})")
    emit("-" * 80)
    vector = LiteLLMEmbedder().embed_one(text)
    norm = math.sqrt(sum(x * x for x in vector))
    emit(f"  dim          = {len(vector)}")
    emit(f"  L2 norm      = {norm:.6f}")
    emit(f"  first comp.  = {vector[0]:.6f}")
    emit(f"  last comp.   = {vector[-1]:.6f}")
    emit("  (observación: vector único 'promedio' de un texto largo y ambiguo)")

    # --- Paso 2 del trace: búsqueda semántica top-k ---
    emit("")
    emit("=" * 80)
    emit(f"[2] POST {args.backend}/search  {{query: <transcripción>, k: {args.k}}}")
    emit("-" * 80)
    with httpx.Client(base_url=args.backend, timeout=60.0) as client:
        resp = client.post("/search", json={"query": text, "k": args.k})
        resp.raise_for_status()
        body = resp.json()

    emit(f"  search_time_ms = {body.get('search_time_ms')}")
    for i, hit in enumerate(body.get("results", []), start=1):
        md = hit.get("metadata", {}) or {}
        emit("")
        emit(f"  --- hit {i} ---")
        emit(f"    chunk_id    = {hit.get('chunk_id')}   document_id = {hit.get('document_id')}")
        emit(f"    distance    = {hit.get('distance')}")
        emit(f"    chunk_type  = {hit.get('chunk_type')}")
        emit(f"    budget_id   = {md.get('budget_id')}   sector = {md.get('sector') or md.get('client_sector')}   tech = {md.get('main_technology')}")
        content = (hit.get("content") or "").replace("\n", " ")
        emit(f"    content[:160] = {content[:160]}")

    emit("")
    emit("=" * 80)
    emit("[raw JSON]")
    emit(json.dumps(body, ensure_ascii=False, indent=2))

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nEscrito {args.out}")


if __name__ == "__main__":
    main()
