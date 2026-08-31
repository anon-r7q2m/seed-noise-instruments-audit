# Reproduction package — Works Where Calibrated, Fails Where Misused

Anonymous reproduction package for the ICLR 2027 submission.

**One click:** `bash reproduce.sh` runs every gate: each number in the
abstract/introduction/Sections 5/8 recomputed from the released assets
(`check_numbers.py`, 30 checks), the Table 1 reference implementation reproduced digit for
digit (`rule3_repro.py`), and the revision-round sensitivity analyses (finite-sample
Monte-Carlo calibration of the sample-complexity result; the four-piece attribution
sensitivity; the 4M aggregation sensitivity; the equivalence-band sensitivity and the
60M→90M jump stratification). Pure CPU, numpy/pandas/scipy only, a few minutes.

## Layout

- `verify-scripts/` — the number gates and the sensitivity analyses (entry points)
- `code/` — the full pipeline from raw assets to the analysis layer (`00`–`10`) plus the
  hierarchical model (`enhancement3/hme.py` and its table generator)
- `data/analysis/` — the derived analysis layer (parquet; every number reproduces here)
- `data/tables/` — the CSVs behind the paper's tables
- `expected/` — archived outputs of the hierarchical model for diffing
- `exclusions/polypythias_excluded_1095.csv` — the machine-readable list of the 1,095
  label–config-mismatched files (per-file sha256)
- `tex_snapshot/` — the submission's LaTeX source as reviewed (what the number gates check)
- `DOWNLOAD.md` — where the raw assets come from, with pull dates and sha256
- `checksums.sha256` — checksums of the shipped data files

## Environment

python3 + numpy/pandas/scipy (+ pyarrow for parquet). No GPU, no network needed at run time.
All script paths are injected via `NFT_*` environment variables set by `reproduce.sh`;
nothing needs editing.

## Rebuilding the analysis layer from raw assets (optional)

Fetch the raw assets per `DOWNLOAD.md`, then run `code/00_parse_dd.py` … `code/10_emit_report.py`
in order (each script documents its inputs/outputs at the top); the gates in
`reproduce.sh` should then pass on your rebuilt layer identically.
