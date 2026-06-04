from __future__ import annotations

import json

import pandas as pd

from app.ingest.parsers.base import ParsedSource


class JsonBudgetParser:
    supported_formats = {"json"}

    def parse(self, content: bytes, source_hint: str) -> ParsedSource:
        data = json.loads(content.decode("utf-8"))
        records = data if isinstance(data, list) else [data]
        df = pd.DataFrame(records)
        return ParsedSource(kind="tabular", dataframe=df)
