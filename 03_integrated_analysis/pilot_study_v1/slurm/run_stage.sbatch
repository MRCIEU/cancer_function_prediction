#!/bin/bash
#SBATCH --job-name=pilot-v1
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/logs/%x_%j.out
#SBATCH --error=/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/logs/%x_%j.err
#SBATCH --chdir=/home/b5ac/freddieyu.b5ac/project_2

set -euo pipefail
: "${PILOT_STAGE:?}"
: "${PILOT_RUN_ID:?}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp/${USER}/pairpilot_pyc}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp/${USER}/pairpilot_mpl}"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}" \
  /projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/logs
echo "START host=$(hostname) job=${SLURM_JOB_ID} stage=${PILOT_STAGE} run=${PILOT_RUN_ID} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "${PILOT_STAGE}" == "gpu-smoke" ]]; then
  /projects/b5ac/envs/p2_vllm013/bin/python - <<'PY'
import torch
print("cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
print("torch", torch.__version__)
PY
  exit 0
fi
exec /projects/b5ac/project_2/envs/core_semmed_py312/bin/python -u \
  /home/b5ac/freddieyu.b5ac/project_2/03_integrated_analysis/pilot_study_v1/src/pairpilot/cli.py \
  "${PILOT_STAGE}" --run-id "${PILOT_RUN_ID}" ${PILOT_RESUME:+--resume}
