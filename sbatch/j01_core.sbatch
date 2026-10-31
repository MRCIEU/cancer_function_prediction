#!/bin/bash
#SBATCH --job-name=da-j01-core
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/b5ac/project_2/results/01_core_audit/logs/j01_%j.out
#SBATCH --error=/projects/b5ac/project_2/results/01_core_audit/logs/j01_%j.err
#SBATCH --chdir=/home/b5ac/freddieyu.b5ac/project_2

set -euo pipefail
echo "START_SHELL job=${SLURM_JOB_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) cpus=${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
exec /projects/b5ac/project_2/envs/core_semmed_py312/bin/python -u /home/b5ac/freddieyu.b5ac/project_2/01_core_audit/run_core_audit.py
