# DC2's environment for an interactive Datarmor session, in bash. From the csh login shell:
#     module add conda
#     bash
#     source scripts/datarmor/env.sh          # from the clone; DC2_ENV for another environment
# Activates the conda environment, puts Poetry on the PATH, exports DC2_PYTHON (read by the jobs,
# passed by sbatch --export=ALL) and LD_PRELOAD (ESMF needs the environment's libstdc++), goes to
# the clone. Sourced, not run: no set -e, so an error does not close the shell.
if [[ -z "${CONDA_EXE:-}" ]]; then
    echo "env.sh: conda not found; run 'module add conda' in csh, then bash" >&2
    return 1
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate "${DC2_ENV:-/work/$USER/envs/dc2}" || return 1
export PATH="$HOME/bin:$PATH"
export DC2_PYTHON="$CONDA_PREFIX/bin/python"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
cd "$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")" && mkdir -p logs
echo "env.sh: $CONDA_PREFIX, in $PWD"
