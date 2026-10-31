#!/bin/bash
#SBATCH --job-name=pilot-ready
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/logs/%x_%j.out
#SBATCH --error=/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/logs/%x_%j.err
#SBATCH --chdir=/home/b5ac/freddieyu.b5ac/project_2

set -euo pipefail
: "${PILOT_STAGE:?}"
: "${PILOT_CHILD_RUN:?}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp/${USER}/pairpilot_pyc}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${PYTHONPYCACHEPREFIX}" /projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/logs
echo "START host=$(hostname) job=${SLURM_JOB_ID} stage=${PILOT_STAGE} child=${PILOT_CHILD_RUN} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec /projects/b5ac/project_2/envs/core_semmed_py312/bin/python -u \
  /home/b5ac/freddieyu.b5ac/project_2/03_integrated_analysis/pilot_study_v1/src/pairpilot/readiness.py \
  "${PILOT_STAGE}"
