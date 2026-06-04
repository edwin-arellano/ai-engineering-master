"""Capa de acceso físico. Un loader resuelve "cómo llego al fichero" y entrega
bytes. NO entiende formato. Separar origen de contenido evita triplicar parsers
por cada sitio donde vive un formato.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FileRef:
    path: str
    format: str  # extensión sin punto: "json", "txt", "xlsx"


class Loader(Protocol):
    def list_files(self, location: str) -> list[FileRef]: ...
    def read(self, ref: FileRef) -> bytes: ...
