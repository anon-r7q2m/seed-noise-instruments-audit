# Figure scripts (revision-11 visual system)

Exact sources of every data figure in the submission (`tex_snapshot/`).
Each script recomputes its numbers from the package's own `data/` assets and
**asserts the paper-quoted values in-line** before writing the PDF
(the assert messages are the audit trail).

## Run

```
python3 make_fig2a_snr.py         # Fig 1a  (SNR cliff)
python3 make_fig2bc_inversion.py  # Fig 1b/c (macro inversion + bpb control);
                                  #   also writes out/fig2bc_inversion_data.csv
python3 make_appd_alluvial.py     # App D alluvial (reads the CSV above: run fig2bc first)
python3 make_appb1_tau_ceiling.py # App B row ...
python3 make_appb2_underread.py
python3 make_appb3_ppl_reversal.py
python3 make_appc_forest.py
python3 make_appc_noise_spread.py # App C noise x spread plane (Fig 4)
python3 make_appd_replay.py
python3 make_appe_atlas.py
python3 make_appe_pp.py
```

Outputs land in `out/` and should equal `tex_snapshot/figures/*.pdf`
(verified textually at packaging time).

## Paths

Data paths default to this repository's `data/` directory (relative to the
script location). Every path is overridable via the package-wide `NFT_*`
environment variables (`NFT_R` = analysis dir, `NFT_TABLES` = tables dir,
`NFT_T2` = the source-equivalence CSV, `NFT_PPL` = the ppl-cells parquet),
same convention as `verify-scripts/`.

## Fonts

Figures are designed in Noto Sans CJK SC (Type-3 embedding). If
`/usr/share/fonts/google-noto-cjk/` is absent the scripts fall back to the
matplotlib default font; all numbers, geometry and layout logic are
unaffected.

Pure CPU; numpy/pandas/matplotlib only. Runtime: seconds per figure.
