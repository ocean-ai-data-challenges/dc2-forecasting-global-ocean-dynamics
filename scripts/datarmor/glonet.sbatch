#!/bin/bash
#SBATCH --job-name=dc2-glonet
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#
# DC2's deterministic baseline, GloNet, scored over its period (2024) on the Ifremer centre's Slurm
# cluster (access.hpc.ifremer.fr: its compute nodes reach EDITO's S3 through the centre's proxy).
# Submitted from the root of a dc2 clone; the baselines and the site are written into that clone
# (dc2/leaderboard_results/, docs/source/_extra/leaderboard/).
#
#     sbatch scripts/datarmor/glonet.sbatch
#     sbatch --export=ALL,DC2_PYTHON=$HOME/miniforge3/envs/dc2/bin/python scripts/datarmor/glonet.sbatch
#
# A job cut by the walltime resumes from its last saved batch when submitted again. Submitted once
# more after the GLO4ENS job has ended, it only consolidates and renders the site with both models.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PY="${DC2_PYTHON:-python}"
WORK="${DC2_WORK:-/work/$USER/dc2_glonet}"
FIRST="${DC2_FIRST:-2024-01-01}"     # GloNet's forecasts cover 2024
mkdir -p "$WORK" logs
sed "s/^start_time:.*/start_time: \"$FIRST\"/" dc2/config/dc2_edito.yaml > "$WORK/glonet.yaml"
"$PY" dc2/evaluate.py -d "$WORK" -c "$WORK/glonet.yaml" -l "$WORK/dc2.log"
