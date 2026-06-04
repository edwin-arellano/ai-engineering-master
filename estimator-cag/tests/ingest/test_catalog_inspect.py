"""La inspección produce hechos y un muestreo estructural SIN valores crudos."""

from __future__ import annotations

import json

from app.ingest.catalog.inspect import inspect_filesystem_source


def test_facts_and_structural_sample_no_values(tmp_path):
    source_dir = tmp_path / "budgets"
    source_dir.mkdir()
    payload = {"budget_id": "BUDGET-2024-0001", "client_name": "Acme Secreta"}
    (source_dir / "b1.json").write_text(json.dumps(payload), encoding="utf-8")

    facts = inspect_filesystem_source("budgets", source_dir, project_root=tmp_path)

    assert facts.name == "budgets"
    assert facts.path == "budgets"
    assert facts.file_count == 1
    assert facts.formats_detected == ["json"]
    # solo claves, nunca valores
    assert facts.structural_sample["json_top_level_keys"] == [
        "budget_id",
        "client_name",
    ]
    serialized = json.dumps(facts.structural_sample)
    assert "Acme Secreta" not in serialized
    assert "BUDGET-2024-0001" not in serialized


def test_txt_speaker_tag_flag(tmp_path):
    source_dir = tmp_path / "transcripts"
    source_dir.mkdir()
    (source_dir / "t.txt").write_text("[00:00:01] Antonio: hola", encoding="utf-8")

    facts = inspect_filesystem_source("transcripts", source_dir, project_root=tmp_path)
    assert facts.structural_sample["txt_has_speaker_tags"] is True
