# Raw-asset download manifest

The package ships the derived analysis layer in full (every number reproduces from it).
To rebuild that layer from the raw public assets, fetch the following (pull date
2026-08-13; sha256 in `raw_checksums.sha256`):

| file | source (HuggingFace) | path / split |
|---|---|---|
| dd_eval_0..3.parquet | `allenai/DataDecide-eval-results` | `train-0000{0..3}-of-00004` |
| dd_macro_avg.parquet | same | `macro_avg-00000-of-00001` |
| dd_ppl.parquet | `allenai/DataDecide-ppl-results` | train |
| sn_random_seeds.parquet | `allenai/signal-and-noise` | `random_seeds` |
| sn_core.parquet | same | `core` (archived, unused) |
| sn_datadecide_intermediate.parquet | same | DataDecide intermediate re-evaluation |
| pp_json/ (8,474 JSONs) | `EleutherAI/polypythias-evals` | lm-eval JSON tree |

PolyPythias tree check: 8,474 files; the 1,095 label–config-mismatched files (2024-08-15
batch) are excluded per `exclusions/polypythias_excluded_1095.csv` (per-file sha256).
Declared defects inherited by all analyses are documented in Appendix A of the paper.
