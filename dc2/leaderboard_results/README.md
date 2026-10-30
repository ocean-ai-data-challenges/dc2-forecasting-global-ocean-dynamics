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

**Before anything else, the edited references must be on EDITO**: the L3 along-track stores
(`DC2/ZARR/L3/<mission>/`, built by `scripts/build_alongtrack_l3.py`) and the drifter stores
(`DC2/ZARR/Drifters15m/`, `scripts/build_drifters_15m.py`), which `dc2_edito.yaml` reads. They
need write access to the `project-ppr-ocean-ia` bucket, from the machine they were built on, e.g.:

```bash
aws s3 sync --endpoint-url https://minio.dive.edito.eu <l3 dir>/ s3://project-ppr-ocean-ia/DC2/ZARR/L3/ --exclude "logs/*"
aws s3 sync --endpoint-url https://minio.dive.edito.eu <drifters dir>/ s3://project-ppr-ocean-ia/DC2/ZARR/Drifters15m/
```

On the Ifremer centre's Slurm cluster (`ssh access.hpc.ifremer.fr`), whose compute nodes reach
EDITO and CloudFerro through the centre's proxy; everything on `/work` (the scratch is purged after
ten days). One environment holds both xESMF (the GLO4ENS conversion) and dctools:

The login shell is probably csh: the conda module is loaded there, then bash (every command here
is bash), where `conda` is an alias csh does not pass on. Conda is the centre's module
(<https://doc.hpc.ifremer.fr/compute/application/conda/>: no `conda init`, no conda of one's own;
Anaconda's channels are blocked, conda-forge only). The environments go on `/work`, not in the
default `~/conda-env` of the small `/home`. Poetry has an environment of its own (the system's
`python3` has no pip), only its command on the PATH: its Python (the latest) must not shadow the
environment's. The installation needs the network (`srun -p cpu --pty bash` if the access node has
none):

```bash
module add conda                                   # in csh
bash
eval "$("$CONDA_EXE" shell.bash hook)"
# once
conda create -y -p /work/$USER/envs/poetry -c conda-forge --override-channels poetry
mkdir -p ~/bin && ln -sf /work/$USER/envs/poetry/bin/poetry ~/bin/poetry
git clone -b dc2-optimisation https://github.com/ocean-ai-data-challenges/dc2-forecasting-global-ocean-dynamics.git /work/$USER/dc2
conda create -y -p /work/$USER/envs/dc2 -c conda-forge --override-channels python=3.12 esmpy pyinterp=2025.11.0
rm -f /work/$USER/envs/dc2/lib/python3.12/site-packages/pyinterp-*.dist-info/direct_url.json
```

Then, in every session (`module add conda` and `bash` in csh first):

```bash
source /work/$USER/dc2/scripts/datarmor/env.sh    # conda env, Poetry, DC2_PYTHON, LD_PRELOAD, cd
VIRTUAL_ENV=$CONDA_PREFIX POETRY_VIRTUALENVS_CREATE=false poetry install   # after each git pull
python -c "import dctools, dctools_core, oceanbench, dcleaderboard, xesmf; print('ok')"
```

ESMF (for xESMF) and pyinterp (Boost, no wheel on PyPI: Poetry would build it and fail) come from
conda-forge, pyinterp at the version of `poetry.lock`. Poetry takes a conda package for one of
another source (its `direct_url.json`, the conda build's directory) and reinstalls it from PyPI:
harmless for the others, a failed build for pyinterp; without that file, Poetry leaves it alone.
`VIRTUAL_ENV` makes Poetry install into the conda environment (it would take its own Python
otherwise). ESMF needs the environment's libstdc++, the system's (`/lib64`) being too old
(`CXXABI_1.3.15' not found`): hence `LD_PRELOAD`, which `env.sh` sets; the jobs set it
themselves. The jobs call `$DC2_PYTHON`, passed by `--export=ALL`: no conda activation in them.
Poetry installs from `poetry.lock`: dctools, oceanbench, dcleaderboard and xESMF are in its `custom`
group, which `pip install -e .` would leave out. The lock pins a commit of each `dc2-optimisation`
branch: after a push to one of them, `poetry update --lock <package>` (and the lock committed)
before `poetry install`, or the run keeps the former commit.

The SST reference (`SST_fields`) is read through the Copernicus Marine API: a Copernicus Marine
account is needed, its credentials saved once in `~/.copernicusmarine/` (`copernicusmarine login`,
in the `dc2` environment).

A test first (one cycle, two members, its pages outside the clone), to see the time per cycle in
`/work/$USER/g4test/logs/`:

```bash
DC2_CYCLES=20240104 DC2_MEMBERS=1-2 DC2_WORK=/work/$USER/g4test \
  DC2_LEADERBOARD_ROOT=/work/$USER/g4test/site sbatch --export=ALL scripts/datarmor/glo4ens.sbatch
```

Then the two baselines (they can run together), each over its track: GloNet over 2024, GLO4ENS
over 2023-2024 (the reference of that view; the 2024 view keeps its 2024 cycles):

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
