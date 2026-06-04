from __future__ import annotations

import io

import pandas as pd

from app.ingest.parsers.base import ParsedSource


class XlsxParser:
    supported_formats = {"xlsx"}

    def parse(self, content: bytes, source_hint: str) -> ParsedSource:
        # tabla principal de la primera hoja; estructura compleja queda fuera (S6-04)
        df = pd.read_excel(io.BytesIO(content), sheet_name=0, engine="openpyxl")
        return ParsedSource(kind="tabular", dataframe=df)
