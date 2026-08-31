#!/usr/bin/env bash
# reproduce.sh — one-click reproduction of every gated number in the paper.
# Pure CPU; needs only python3 + numpy/pandas/scipy (parquet I/O needs pyarrow or fastparquet).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export NFT_R="$HERE/data/analysis"
export NFT_T2="$HERE/data/tables/t2_source_equivalence.csv"
export NFT_DD="$NFT_R/dd_tidy.parquet"
export NFT_HME_SRC="$HERE/code/enhancement3"
export NFT_TEX="$HERE/tex_snapshot"
export NFT_STAGE1="$HERE/data/stage_results"
export NFT_STAGE2="$HERE/data/stage_results"
PY="${PYTHON:-python3}"
"$PY" -c "import numpy, pandas, scipy" || { echo "need numpy/pandas/scipy (and pyarrow for parquet)"; exit 1; }

cd "$HERE/verify-scripts"
echo "== gate 1: every number in abstract/intro/S5/S7 recomputed from raw assets =="
"$PY" check_numbers.py
echo "== gate 2: Table 1 (tab:snr) rule-3 reference implementation, digit-for-digit =="
"$PY" rule3_repro.py
echo "== revision-4 sensitivity analyses (results also archived as .json/.out.txt) =="
"$PY" r4_prop1_mc_calibration.py
"$PY" r4_attribution_sensitivity.py
"$PY" r4_4m_aggregation_sensitivity.py
"$PY" r4_band_jump_sensitivity.py
echo
echo "ALL GATES PASSED"
