from __future__ import annotations

import re

from app.ingest.parsers.base import ParsedSource

_TURN_RE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2})\]\s+(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)


class TxtTranscriptParser:
    supported_formats = {"txt"}

    def parse(self, content: bytes, source_hint: str) -> ParsedSource:
        text = content.decode("utf-8", errors="replace")
        records: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _TURN_RE.match(line)
            if m:
                records.append(
                    {
                        "timestamp": m.group("ts"),
                        "speaker": m.group("speaker").strip(),
                        "text": m.group("text").strip(),
                    }
                )
            else:
                # legacy sin tags de speaker: turno sin atribuir
                records.append({"timestamp": None, "speaker": None, "text": line})
        return ParsedSource(kind="text_turns", records=records)
