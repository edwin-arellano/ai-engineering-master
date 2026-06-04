"""Parsers: contra la representación intermedia (no Document todavía)."""

from __future__ import annotations

import io

import openpyxl

from app.ingest.parsers.json_parser import JsonBudgetParser
from app.ingest.parsers.txt_parser import TxtTranscriptParser
from app.ingest.parsers.xlsx_parser import XlsxParser


def test_json_parser_to_dataframe():
    content = b'[{"budget_id": "B1", "total": 10}, {"budget_id": "B2", "total": 20}]'
    parsed = JsonBudgetParser().parse(content, "b.json")
    assert parsed.kind == "tabular"
    assert list(parsed.dataframe.columns) == ["budget_id", "total"]
    assert len(parsed.dataframe) == 2


def test_json_parser_single_object():
    parsed = JsonBudgetParser().parse(b'{"budget_id": "B1"}', "b.json")
    assert len(parsed.dataframe) == 1


def test_txt_parser_with_and_without_speaker():
    content = b"[00:00:01] Antonio: hola\nlinea legacy sin tag"
    parsed = TxtTranscriptParser().parse(content, "t.txt")
    assert parsed.kind == "text_turns"
    assert parsed.records[0] == {
        "timestamp": "00:00:01",
        "speaker": "Antonio",
        "text": "hola",
    }
    assert parsed.records[1] == {
        "timestamp": None,
        "speaker": None,
        "text": "linea legacy sin tag",
    }


def test_xlsx_parser_to_dataframe():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["role", "rate"])
    ws.append(["backend", 75])
    buffer = io.BytesIO()
    wb.save(buffer)

    parsed = XlsxParser().parse(buffer.getvalue(), "r.xlsx")
    assert parsed.kind == "tabular"
    assert list(parsed.dataframe.columns) == ["role", "rate"]
    assert len(parsed.dataframe) == 1
