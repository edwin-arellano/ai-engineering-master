from __future__ import annotations

from pathlib import Path

from app.ingest.loaders.base import FileRef


class FilesystemLoader:
    """Loader para fuentes en disco local."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root

    def list_files(self, location: str) -> list[FileRef]:
        base = self._root / location
        return [
            FileRef(path=str(f), format=f.suffix.lower().lstrip("."))
            for f in sorted(base.rglob("*"))
            if f.is_file() and f.suffix
        ]

    def read(self, ref: FileRef) -> bytes:
        return Path(ref.path).read_bytes()
