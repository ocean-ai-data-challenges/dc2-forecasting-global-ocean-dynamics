#!/bin/bash
#SBATCH --job-name=dc2-glo4ens
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#
# DC2's probabilistic baseline, GLO4ENS, scored cycle by cycle from its CloudFerro bucket
# (scripts/evaluate_glo4ens.py: one converted cycle on disk at a time, deleted once scored) on the
# Ifremer centre's Slurm cluster. Submitted from the root of a dc2 clone whose environment has both
# xESMF (esmpy) and dctools.
#
#     sbatch scripts/datarmor/glo4ens.sbatch                                     # 2023-2024
#     sbatch --export=ALL,DC2_FIRST=2024-01-01 scripts/datarmor/glo4ens.sbatch   # 2024 only
#     # a test: one cycle, two members, its pages and baselines outside the clone
#     export DC2_CYCLES=20240104 DC2_MEMBERS=1-2 DC2_WORK=/work/$USER/g4test
#     export DC2_LEADERBOARD_ROOT=/work/$USER/g4test/site
#     sbatch --export=ALL scripts/datarmor/glo4ens.sbatch
#
#     # the references whose tasks failed, in the cycles done
#     sbatch --export=ALL,DC2_REDO_ERRORS=1 scripts/datarmor/glo4ens.sbatch
#     # references named, in cycles done (a fix that adds tasks)
#     export DC2_CYCLES="20230511 20230706" DC2_REDO=swot
#     sbatch --export=ALL scripts/datarmor/glo4ens.sbatch
#
# GLO4ENS is the reference of the 2023-2024 track: its cycles cover both years by default.
# DC2_CYCLES takes dates separated by spaces ("20240104 20240111"). DC2_LEADERBOARD_ROOT sends the
# baselines and the pages elsewhere than the clone (a test leaves the clone's baselines alone).
# A job cut by the walltime resumes at the cycle it stopped at when submitted again (cycles_done/ in
# the work directory). The work directory is on /work, not the scratch (purged after ten days).
# A job also stops at a cycle with failed tasks (exit 1), which the next one takes up again: chain
# the jobs with singleton (one at a time, however the previous one ended), not afterok, which a
# walltime or a failed cycle breaks (DependencyNeverSatisfied):
#
#     foreach i (1 2 3 4)
#         sbatch --dependency=singleton scripts/datarmor/glo4ens.sbatch
#     end
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PY="${DC2_PYTHON:-python}"
# The environment's libstdc++, not the system's (/lib64, too old for ESMF: CXXABI_1.3.15 not found)
STDCXX="$(dirname "$(dirname "$(readlink -f "$(command -v "$PY")")")")/lib/libstdc++.so.6"
[[ -f "$STDCXX" ]] && export LD_PRELOAD="$STDCXX${LD_PRELOAD:+:$LD_PRELOAD}"
WORK="${DC2_WORK:-/work/$USER/glo4ens}"
mkdir -p "$WORK" logs
args=(--work "$WORK" --members "${DC2_MEMBERS:-1-50}" --workers "${DC2_WORKERS:-4}")
[[ -n "${DC2_REDO_ERRORS:-}" ]] && args+=(--redo-errors)
[[ -n "${DC2_REDO:-}" ]] && args+=(--redo $DC2_REDO)
# dctools stops a task whose data take more than 90 s to compute, thinking of a stalled S3 read; the
# mean of 50 members of a 3-D field read from the local disk takes longer (GLORYS tasks)
export DCTOOLS_S3_COMPUTE_TIMEOUT="${DCTOOLS_S3_COMPUTE_TIMEOUT:-1800}"
if [[ -n "${DC2_CYCLES:-}" ]]; then
    args+=(--cycles $DC2_CYCLES)
else
    args+=(--first "${DC2_FIRST:-2023-01-01}" --last "${DC2_LAST:-2024-12-31}")
fi
# EDITO's profile, its parallelism sized for this node share (dc2/config/dc2_datarmor.yaml)
CONFIG="$WORK/config.yaml"
settings=()
[[ -n "${DC2_LEADERBOARD_ROOT:-}" ]] && settings+=(leaderboard_root="$DC2_LEADERBOARD_ROOT")
"$PY" scripts/datarmor/site_config.py "${DC2_SITE_CONFIG:-dc2/config/dc2_datarmor.yaml}" \
    "$CONFIG" "${settings[@]}"
args+=(--config "$CONFIG")
"$PY" scripts/evaluate_glo4ens.py "${args[@]}"
