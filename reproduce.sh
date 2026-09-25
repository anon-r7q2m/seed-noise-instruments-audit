#!/usr/bin/env bash
# reproduce.sh — one-click reproduction of every gated number in the paper.
# Pure CPU; needs only python3 + numpy/pandas/scipy (parquet I/O needs pyarrow or fastparquet).
# Runtime ~20 min (the joint-bootstrap cell construction dominates; all steps deterministic).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export NFT_R="$HERE/data/analysis"
export NFT_T2="$HERE/data/tables/t2_source_equivalence.csv"
export NFT_DD="$NFT_R/dd_tidy.parquet"
export NFT_HME_SRC="$HERE/code/enhancement3"
export NFT_TEX="$HERE/tex_snapshot"
export NFT_STAGE3_BUNDLE="$HERE/data/self_run_controls/grid20m_full_evals.json"
export NFT_STAGE2_RUNS="$HERE/data/self_run_controls"
export NFT_SN="$HERE/data/analysis/random_seeds.parquet"
export NFT_STAGE1="$HERE/data/self_run_controls"
export NFT_STAGE2="$HERE/data/self_run_controls"
export NFT_PPL="$HERE/data/analysis/dd_ppl.parquet"
PY="${PYTHON:-python3}"
"$PY" -c "import numpy, pandas, scipy" || { echo "need numpy/pandas/scipy (and pyarrow for parquet)"; exit 1; }

cd "$HERE/verify-scripts"
echo "== step 0: regenerate every intermediate json from the raw assets =="
# (these regenerate the intermediate JSONs in place; nothing in the package is pre-baked)
"$PY" rep_separation.py
"$PY" regen_intermediates.py
"$PY" perfect_proxy_ceiling.py
"$PY" chance_task_ablation.py
"$PY" repetition_check.py
"$PY" decidable_share.py
"$PY" sn_headtohead.py
"$PY" seed_budget_curve.py
"$PY" bpb_replay.py
"$PY" joint_bootstrap.py
"$PY" pooled_decidable.py
"$PY" extra_robustness.py
echo "== gate 1: every number in abstract/intro/S5/S7 recomputed from raw assets =="
"$PY" check_numbers.py
echo "== gate 2: Table 1 (tab:snr) rule-3 reference implementation, digit-for-digit =="
"$PY" rule3_repro.py
echo "== sensitivity analyses (results also archived as .json/.out.txt) =="
"$PY" prop1_mc_calibration.py
"$PY" attribution_sensitivity.py
"$PY" 4m_aggregation_sensitivity.py
"$PY" band_jump_sensitivity.py
echo "== conditional-arm check (App H 20M crossed grid) =="
"$PY" conditional_arms.py
"$PY" cv_f_check.py
"$PY" hme_projection.py
"$PY" sn_exact_protocol.py
"$PY" hme_full_correction.py
"$PY" snr_ablation.py
"$PY" apph_proxy_boot.py
"$PY" crossed_grid.py
echo
echo "ALL GATES PASSED"
