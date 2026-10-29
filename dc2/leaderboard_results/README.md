# Leaderboard baselines

The `results_<model>.json` (and `_per_bins.jsonl.gz`) files here are the baselines every DC2
leaderboard is built with: each evaluation run copies its results here, and the next runs show
them beside their own models.

The GloNet results that were here (2024) had been computed before the scoring fixes of the
`dc2-optimisation` branch (along-track SSH anomaly scored against an absolute height, bins
averaged instead of pooled, Argo without its standard mode, L2 along-track references): they were
removed. Re-evaluate GloNet (`python dc2/evaluate.py`) and GLO4ENS
(`python scripts/evaluate_glo4ens.py`) to put the baselines back.
