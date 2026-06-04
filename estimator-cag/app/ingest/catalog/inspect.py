"""Inspección factual de fuentes en sistema de ficheros.

Recoge folder facts (conteo, tamaño, mtime, formatos) sin emitir ningún
juicio. Los campos subjetivos (owner, sensibilidad, decisión) los aporta
después el evaluador LLM o el revisor humano.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FilesystemSourceFacts:
    name: str
    path: str  # relativo a la raíz del proyecto
    file_count: int
    total_size_mb: float
    latest_modified: datetime
    observed_lag_days: int
    formats_detected: list[str]
    # muestreo ESTRUCTURAL (no valores): claves de un JSON, columnas de un xlsx,
    # flags de formato de un txt. Es lo único que ve el evaluador LLM.
    structural_sample: dict = field(default_factory=dict)


def _sample_structure(files: list[Path], formats: set[str]) -> dict:
    """Muestrea estructura sin exponer valores sensibles."""
    sample: dict = {}
    if "json" in formats:
        first_json = next((f for f in files if f.suffix.lower() == ".json"), None)
        if first_json is not None:
            try:
                data = json.loads(first_json.read_text(encoding="utf-8"))
                keys = sorted(data.keys()) if isinstance(data, dict) else []
                sample["json_top_level_keys"] = keys
            except Exception:  # noqa: BLE001 — muestreo best-effort
                sample["json_top_level_keys"] = []
    if "txt" in formats:
        first_txt = next((f for f in files if f.suffix.lower() == ".txt"), None)
        if first_txt is not None:
            head = first_txt.read_text(encoding="utf-8", errors="replace")[:500]
            # detecta el patrón [hh:mm:ss] Speaker: ... sin exponer el contenido
            import re

            sample["txt_has_speaker_tags"] = bool(
                re.search(r"\[\d{2}:\d{2}:\d{2}\]\s+\w+:", head)
            )
    if "xlsx" in formats:
        first_xlsx = next((f for f in files if f.suffix.lower() == ".xlsx"), None)
        if first_xlsx is not None:
            try:
                import openpyxl

                wb = openpyxl.load_workbook(first_xlsx, read_only=True)
                ws = wb.active
                header = [c.value for c in next(ws.iter_rows(max_row=1))]
                sample["xlsx_columns"] = [str(h) for h in header if h is not None]
                wb.close()
            except Exception:  # noqa: BLE001
                sample["xlsx_columns"] = []
    return sample


def inspect_filesystem_source(
    name: str, root: Path, *, project_root: Path
) -> FilesystemSourceFacts:
    files = [f for f in root.rglob("*") if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in files)
    latest_ts = max((f.stat().st_mtime for f in files), default=0.0)
    latest_modified = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
    lag_days = (datetime.now(timezone.utc) - latest_modified).days if latest_ts else 0
    formats = {f.suffix.lower().lstrip(".") for f in files if f.suffix}

    return FilesystemSourceFacts(
        name=name,
        path=str(root.relative_to(project_root)),
        file_count=len(files),
        total_size_mb=round(total_bytes / (1024 * 1024), 4),
        latest_modified=latest_modified,
        observed_lag_days=lag_days,
        formats_detected=sorted(formats),
        structural_sample=_sample_structure(files, formats),
    )
