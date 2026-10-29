# Leaderboard baselines

The `results_<model>.json` (and `_per_bins.jsonl.gz`) files here are the baselines every DC2
leaderboard is built with: each evaluation run copies its results here, and the next runs show
them beside their own models. The `_per_bins.jsonl.gz` files are large and not committed: the maps
are published through `docs/source/_extra/leaderboard/map_data.tar.gz`.

The GloNet results that were here (2024) had been computed before the scoring fixes of the
`dc2-optimisation` branch (along-track SSH anomaly scored against an absolute height, bins
averaged instead of pooled, Argo without its standard mode, L2 along-track references): they were
removed. Re-evaluate GloNet and GLO4ENS to put the baselines back.

## Re-evaluating the baselines on Datarmor

On the Ifremer centre's Slurm cluster (`ssh access.hpc.ifremer.fr`), whose compute nodes reach
EDITO and CloudFerro through the centre's proxy; everything on `/work` (the scratch is purged after
ten days). One environment holds both xESMF (the GLO4ENS conversion) and dctools:

```bash
git clone -b dc2-optimisation https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics.git /work/$USER/dc2
cd /work/$USER/dc2 && mkdir -p logs
conda create -y -n dc2 -c conda-forge python=3.12 esmpy xesmf
conda activate dc2
pip install -e .                 # (in `srun -p cpu --pty bash` if the access node has no network)
export DC2_PYTHON=$(which python)
```

The SST reference (`SST_fields`) is read through the Copernicus Marine API: a Copernicus Marine
account is needed, its credentials saved once in `~/.copernicusmarine/` (`copernicusmarine login`,
in the `dc2` environment).

A test first (one cycle, two members, its pages outside the clone), to see the time per cycle in
`/work/$USER/g4test/logs/`:

```bash
DC2_CYCLES=20240104 DC2_MEMBERS=1-2 DC2_WORK=/work/$USER/g4test \
  DC2_LEADERBOARD_ROOT=/work/$USER/g4test/site sbatch --export=ALL scripts/datarmor/glo4ens.sbatch
```

Then the two baselines (they can run together), over 2024 (GloNet's period: the leaderboard
compares the models over their common period):

```bash
sbatch --export=ALL scripts/datarmor/glonet.sbatch
sbatch --export=ALL scripts/datarmor/glo4ens.sbatch
```

A job cut by its walltime resumes when submitted again (from its last batch, its last cycle). Once
both have ended, `glonet.sbatch` submitted once more only consolidates and renders the site with
the two models: `dc2/leaderboard_results/` and `docs/source/_extra/leaderboard/`
(`map_data.tar.gz` included) of the clone.

They are then brought back to the clone that is committed from:

```bash
rsync -av access.hpc.ifremer.fr:/work/<user>/dc2/dc2/leaderboard_results/ dc2/leaderboard_results/
rsync -av --exclude map_data/ \
  access.hpc.ifremer.fr:/work/<user>/dc2/docs/source/_extra/leaderboard/ docs/source/_extra/leaderboard/
git add dc2/leaderboard_results docs/source/_extra/leaderboard     # map_data.tar.gz is in git-lfs
```

A look before publishing: `tar xzf map_data.tar.gz && python -m http.server` in
`docs/source/_extra/leaderboard/`. The site is published by `.github/workflows/docs.yml`, on a push
to `main` only (it replaces the `map_data.tar.gz` release asset Read the Docs downloads).
