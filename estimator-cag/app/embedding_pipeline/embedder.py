"""Embedder vía LiteLLM (portabilidad: text-embedding-3-small hoy, voyage-3 mañana
cambiando un string). Batching para no serializar peticiones; retry exponencial
simple ante rate limit; coste estimado por tokens reales devueltos por la API.
"""

from __future__ import annotations

import time

import litellm
import structlog

from app.config import get_settings
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

logger = structlog.get_logger(__name__)

# Precio de text-embedding-3-small en USD por 1M de tokens de entrada.
# PLACEHOLDER: verificar contra la tarifa oficial de OpenAI (cambia con el tiempo).
EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.02

_MAX_RETRIES = 3
_RETRY_WAITS = (1.0, 2.0, 4.0)  # esperas en segundos: 1s, 2s, 4s


def _extract_embedding(item: object) -> list[float]:
    """Extrae el vector de un item de ``response.data``.

    Según la versión de LiteLLM, cada item puede ser un dict (``item["embedding"]``)
    o un objeto (``item.embedding``). Se cubren ambos para no acoplar el pipeline
    a una versión concreta.
    """
    if isinstance(item, dict):
        return item["embedding"]
    return item.embedding


class LiteLLMEmbedder:
    def __init__(self, model: str | None = None, batch_size: int | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.embeddings_model
        self._batch_size = batch_size or settings.embedding_batch_size
        # stats del último embed_many (instancia por request en el router → sin estado compartido)
        self.last_run_total_tokens: int = 0
        self.last_run_cost_usd: float = 0.0

    def embed_one(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.last_run_total_tokens = 0
        self.last_run_cost_usd = 0.0
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._embed_batch([c.text for c in batch], accumulate=True)
            for chunk, vector in zip(batch, vectors):
                embedded.append(EmbeddedChunk(**chunk.model_dump(), embedding=vector))
        return embedded

    def _embed_batch(
        self, texts: list[str], *, accumulate: bool = False
    ) -> list[list[float]]:
        for attempt in range(_MAX_RETRIES):
            try:
                started = time.perf_counter()
                response = litellm.embedding(model=self._model, input=texts)
                latency_ms = (time.perf_counter() - started) * 1000
                # response.data: lista con un item por input; cada item trae "embedding"
                vectors = [_extract_embedding(item) for item in response.data]
                usage = getattr(response, "usage", None)
                tokens = 0
                if usage is not None:
                    tokens = (
                        getattr(usage, "total_tokens", None)
                        or getattr(usage, "prompt_tokens", 0)
                        or 0
                    )
                if accumulate:
                    self.last_run_total_tokens += tokens
                    self.last_run_cost_usd += (
                        tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS
                    )
                logger.info(
                    "embed.batch_completed",
                    model=self._model,
                    chunks=len(texts),
                    tokens=tokens,
                    latency_ms=round(latency_ms, 1),
                )
                return vectors
            except litellm.RateLimitError:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_WAITS[attempt]
                    logger.warning(
                        "embed.rate_limited", attempt=attempt + 1, wait_seconds=wait
                    )
                    time.sleep(wait)
                    continue
                raise  # agotados los reintentos
        # inalcanzable, pero satisface al type checker
        raise RuntimeError("embed_batch: estado inalcanzable")
