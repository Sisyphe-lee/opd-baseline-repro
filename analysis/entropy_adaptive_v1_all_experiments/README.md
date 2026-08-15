# Adaptive v1 canonical aggregate analysis

This directory is the canonical three-row comparison for the matched 4-GPU
Adaptive v1 experiments. It contains Vanilla OPD, TCOD F2B, the thresholds in
ascending order, the two original step-80 branches adjacent to `tau=0.100`, and
the two cosine-annealed step-80 branches on the appended third row.

The 8-GPU `tau=0.125` run and the older `tau=0.175` reference are intentionally
excluded from the main panel because their layouts are not matched to the 4-GPU
threshold sweep. Their original analysis directories remain unchanged.

## Updating with future experiments

Treat this directory as append-only analysis: add the completed experiment to
`manifest.json` at the scientifically appropriate position, then rerun:

```bash
.venv_tcod/bin/python analysis/plot_adaptive_v1_all_experiments.py
```

The command regenerates both PNGs, both combined CSVs, `plot_summary.json`, and
`provenance.json`. Do not replace an existing source entry unless its underlying
analysis was corrected and the replacement is documented in provenance.
