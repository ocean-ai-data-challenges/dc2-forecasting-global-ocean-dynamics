#!/bin/bash
#SBATCH --job-name=dc2-glo4ens
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#
# DC2's probabilistic baseline, GLO4ENS, scored cycle by cycle from its CloudFerro bucket
# (scripts/evaluate_glo4ens.py: one converted cycle on disk at a time, deleted once scored) on the
# Ifremer centre's Slurm cluster. Submitted from the root of a dc2 clone whose environment has both
# xESMF (esmpy) and dctools.
#
#     sbatch scripts/datarmor/glo4ens.sbatch
#     sbatch --export=ALL,DC2_FIRST=2023-01-01 scripts/datarmor/glo4ens.sbatch   # 2023 too
#     # a test: one cycle, two members, its pages and baselines outside the clone
#     export DC2_CYCLES=20240104 DC2_MEMBERS=1-2 DC2_WORK=/work/$USER/g4test
#     export DC2_LEADERBOARD_ROOT=/work/$USER/g4test/site
#     sbatch --export=ALL scripts/datarmor/glo4ens.sbatch
#
# DC2_CYCLES takes dates separated by spaces ("20240104 20240111"). DC2_LEADERBOARD_ROOT sends the
# baselines and the pages elsewhere than the clone (a test leaves the clone's baselines alone).
# A job cut by the walltime resumes at the cycle it stopped at when submitted again (cycles_done/ in
# the work directory). The work directory is on /work, not the scratch (purged after ten days).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PY="${DC2_PYTHON:-python}"
WORK="${DC2_WORK:-/work/$USER/glo4ens}"
mkdir -p "$WORK" logs
args=(--work "$WORK" --members "${DC2_MEMBERS:-1-50}" --workers "${DC2_WORKERS:-4}")
if [[ -n "${DC2_CYCLES:-}" ]]; then
    args+=(--cycles $DC2_CYCLES)
else
    args+=(--first "${DC2_FIRST:-2024-01-01}" --last "${DC2_LAST:-2024-12-31}")
fi
CONFIG="${DC2_CONFIG:-dc2/config/dc2_edito.yaml}"
if [[ -n "${DC2_LEADERBOARD_ROOT:-}" ]]; then
    { cat "$CONFIG"; echo; echo "leaderboard_root: \"$DC2_LEADERBOARD_ROOT\""; } > "$WORK/config.yaml"
    CONFIG="$WORK/config.yaml"
fi
args+=(--config "$CONFIG")
"$PY" scripts/evaluate_glo4ens.py "${args[@]}"
