#!/bin/bash
#SBATCH --job-name=article_by_source
#SBATCH --account=brics.b5ac
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work/logs/%x-%j.err

set -euo pipefail

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc

cd /home/b5ac/freddieyu.b5ac/project_2
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m src.article_links_by_source \
  --out-dir /home/b5ac/freddieyu.b5ac/projects/project_2/results/scientific_figures_figure6_by_source_20261117T1204Z \
  --workers 4
