"""Carga y valida data_catalog.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.ingest.catalog.models import DataCatalog


def load_catalog(path: Path) -> DataCatalog:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DataCatalog(**raw)
