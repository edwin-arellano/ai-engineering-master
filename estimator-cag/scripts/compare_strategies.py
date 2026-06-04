"""Compara estrategias de chunking sobre data/budgets_sample.json.

  uv run python -m scripts.compare_strategies \
      --strategies structural,recursive,sentence_window,hierarchical \
      --query "OAuth 2.0 authentication backend for fintech"
  # estrategias LLM (coste): --strategies semantic,propositional,contextual_retrieval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.generation.rag.chunking.registry import LLM_BASED
from app.generation.rag.comparison import compare_strategies
from app.generation.rag.schemas import Budget

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "structural,recursive,sentence_window,semantic,hierarchical"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=DEFAULT)
    ap.add_argument("--query", default=None)
    args = ap.parse_args()
    names = [n.strip() for n in args.strategies.split(",") if n.strip()]
    budgets = [
        Budget(**b)
        for b in json.loads((ROOT / "data" / "budgets_sample.json").read_text())
    ]
    wrapper = None
    if any(n in LLM_BASED for n in names):
        from app.foundations.config import get_settings
        from app.foundations.llm_wrapper import LLMWrapper

        wrapper = LLMWrapper(get_settings())
    reports = compare_strategies(budgets, names, query=args.query, wrapper=wrapper)
    print(
        f"{'strategy':<22}{'chunks':>7}{'orphans':>8}{'min':>5}{'p50':>7}"
        f"{'p95':>7}{'max':>6}{'ms':>8}  top_scores"
    )
    for r in reports:
        print(
            f"{r.name:<22}{r.num_chunks:>7}{r.orphan_count:>8}{r.min_tokens:>5}"
            f"{r.p50_tokens:>7.0f}{r.p95_tokens:>7.0f}{r.max_tokens:>6}"
            f"{r.latency_ms:>8.0f}  {r.top_scores}"
        )


if __name__ == "__main__":
    main()
