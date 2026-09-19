#!/usr/bin/env python3
"""Emit the pipeline summary report (writes data/REPORT.md)."""
import pathlib

REPORT = r"""# Analysis pipeline summary

CPU-only re-analysis of public per-seed evaluation assets; no training.

## Inputs (see DOWNLOAD.md for locations and checksums)
- DataDecide per-seed/per-step evaluation tables, macro tables, 11-domain perplexity tables
- Signal & Noise official `random_seeds` runs (10 init arms + 9 data-order arms + dense-eval run)
- PolyPythias public lm-eval JSON files (5 sizes x 10 seeds)

## Stages
- `00_parse_dd.py` — tidy the DataDecide tables into parquet
- `01_t1_sn_replication.py` — T1: checkpoint-proxy replication on Signal & Noise arms
- `02_t1_datadecide.py` — T1: cross-scale transfer + Monte-Carlo ceiling
- `03_t2_source_equiv.py` — T2: source-equivalence bands (F intervals + bootstrap)
- `04_t3_decision_replay.py` — T3: decision replay and top-k regret
- `05_t1_ppl.py` — T1: perplexity readout
- `06_figures.py`, `09_aggregates.py` — aggregates and figures
- `07_parse_polypythias.py` — parse PolyPythias JSONs (with label-config consistency checks)
- `08_polypythias_analysis.py` — PolyPythias re-analysis

## Outputs
- `data/analysis/` (parquet analysis layer), `data/tables/` (CSVs behind the paper's tables)
"""

out = pathlib.Path("data/REPORT.md")
out.write_text(REPORT, encoding="utf-8")
print("wrote", out, len(REPORT), "chars")
