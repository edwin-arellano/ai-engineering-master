"""Contrato canónico del subsistema de ingesta.

Es el puente (DTO/interfaz) entre la fase de ingesta y la futura fase de
vectorización. Todo parser, sea cual sea el formato de origen, termina
produciendo instancias de Document a través de los normalizers. El downstream
(chunking, embedding, retrieval — sesiones 7+) opera EXCLUSIVAMENTE sobre Document.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Metadatos que viajan con cada documento por todo el pipeline.

    Los tres primeros campos vienen del catálogo y son obligatorios para
    cualquier documento, sea cual sea el formato. El resto los rellena el
    parser cuando el formato lo permite. La trazabilidad por construcción
    (source_name + source_location + lineage) es lo que después permite citar
    fuentes en la respuesta final.
    """

    model_config = ConfigDict(extra="forbid")

    # --- catálogo (obligatorios) ---
    source_name: str  # coincide con una entrada de data_catalog.yaml
    source_location: str  # path físico o URL de origen
    ingested_at: datetime

    # --- catálogo (propagados, opcionales) ---
    source_version: Optional[str] = None
    lineage_upstream: Optional[str] = None
    access_restrictions: Optional[str] = None
    contains_pii: bool = False

    # --- parser (opcionales, según formato) ---
    document_id: str
    document_title: Optional[str] = None
    document_created_at: Optional[datetime] = None
    document_author: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None

    # válvula de escape consciente: metadatos específicos de un parser que no
    # encajan en el schema canónico (p. ej. speaker/timestamp de transcripción).
    extra: dict = Field(default_factory=dict)


class Document(BaseModel):
    """Salida canónica del subsistema de ingesta."""

    model_config = ConfigDict(extra="forbid")

    content: str
    metadata: DocumentMetadata
