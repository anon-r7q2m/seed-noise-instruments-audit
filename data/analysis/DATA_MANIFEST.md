# Data manifest (anonymous reproduction package)

All quantities in the paper are recomputed from the derived analysis layer in
`data/analysis/` (parquet), which this package ships in full. The raw assets are public:
DataDecide eval/ppl results (allenai/DataDecide-eval-results, allenai/DataDecide-ppl-results),
Signal & Noise random_seeds (allenai/signal-and-noise), PolyPythias evals
(EleutherAI/polypythias-evals). Pull dates, sizes and sha256 of the raw downloads are in
`DOWNLOAD.md`. Declared asset defects (budget truncation at 750M, bundled seed semantics,
misaligned S&N step grids, PolyPythias coverage gaps) are documented in the paper's
Appendix A and inherited by all analyses.
