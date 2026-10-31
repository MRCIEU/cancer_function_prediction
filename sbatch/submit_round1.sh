#!/bin/bash
# Submit the six analysis jobs. This script does not run the analysis itself.
set -euo pipefail

CODE_ROOT=/home/b5ac/freddieyu.b5ac/project_2
RESULTS=/projects/b5ac/project_2/results
LOGDIR=${RESULTS}/03_integrated_analysis/data_analysis_round1/logs

mkdir -p "${LOGDIR}" \
  "${RESULTS}/01_core_audit/logs" \
  "${RESULTS}/01_core_audit/tables" \
  "${RESULTS}/01_core_audit/report" \
  "${RESULTS}/02_semmeddb_validation/logs" \
  "${RESULTS}/02_semmeddb_validation/tables" \
  "${RESULTS}/02_semmeddb_validation/report" \
  "${RESULTS}/03_integrated_analysis/data_analysis_round1/tables" \
  "${RESULTS}/03_integrated_analysis/data_analysis_round1/derived" \
  "${RESULTS}/03_integrated_analysis/data_analysis_round1/figures" \
  "${RESULTS}/03_integrated_analysis/data_analysis_round1/cases" \
  "${RESULTS}/03_integrated_analysis/data_analysis_round1/logs"

CODE_FP=$(find "${CODE_ROOT}" -type f \( -name '*.py' -o -name '*.json' -o -name '*.sbatch' -o -name '*.sh' -o -name 'requirements.txt' \) ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')

J00=$(sbatch --parsable --export=ALL "${CODE_ROOT}/sbatch/j00_preflight.sbatch")
J00=${J00%%;*}
J01=$(sbatch --parsable --dependency=afterok:${J00} --export=ALL "${CODE_ROOT}/sbatch/j01_core.sbatch")
J01=${J01%%;*}
J02=$(sbatch --parsable --dependency=afterok:${J00} --export=ALL "${CODE_ROOT}/sbatch/j02_semmed.sbatch")
J02=${J02%%;*}
J03=$(sbatch --parsable --dependency=afterok:${J01}:${J02} --export=ALL "${CODE_ROOT}/sbatch/j03_align.sbatch")
J03=${J03%%;*}
J04=$(sbatch --parsable --dependency=afterok:${J03} --export=ALL "${CODE_ROOT}/sbatch/j04_joint.sbatch")
J04=${J04%%;*}
J05=$(sbatch --parsable --dependency=afterok:${J04} --export=ALL "${CODE_ROOT}/sbatch/j05_report.sbatch")
J05=${J05%%;*}

MANIFEST="${LOGDIR}/job_manifest.tsv"
{
  printf 'job_name\tjob_id\tdependency\tscript\tconfig\tcode_fingerprint\tstdout\tstderr\toutput_dir\n'
  printf 'J00\t%s\tnone\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${J00}" "${CODE_ROOT}/sbatch/j00_preflight.sbatch" "${CODE_ROOT}/config/analysis_config.json" "${CODE_FP}" \
    "${LOGDIR}/j00_${J00}.out" "${LOGDIR}/j00_${J00}.err" "${RESULTS}/03_integrated_analysis/data_analysis_round1/derived/j00"
  printf 'J01\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${J01}" "${J00}" "${CODE_ROOT}/sbatch/j01_core.sbatch" "${CODE_ROOT}/config/analysis_config.json" "${CODE_FP}" \
    "${RESULTS}/01_core_audit/logs/j01_${J01}.out" "${RESULTS}/01_core_audit/logs/j01_${J01}.err" "${RESULTS}/01_core_audit"
  printf 'J02\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${J02}" "${J00}" "${CODE_ROOT}/sbatch/j02_semmed.sbatch" "${CODE_ROOT}/config/analysis_config.json" "${CODE_FP}" \
    "${RESULTS}/02_semmeddb_validation/logs/j02_${J02}.out" "${RESULTS}/02_semmeddb_validation/logs/j02_${J02}.err" "${RESULTS}/02_semmeddb_validation"
  printf 'J03\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${J03}" "${J01}:${J02}" "${CODE_ROOT}/sbatch/j03_align.sbatch" "${CODE_ROOT}/config/analysis_config.json" "${CODE_FP}" \
    "${LOGDIR}/j03_${J03}.out" "${LOGDIR}/j03_${J03}.err" "${RESULTS}/03_integrated_analysis/data_analysis_round1/derived/j03"
  printf 'J04\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${J04}" "${J03}" "${CODE_ROOT}/sbatch/j04_joint.sbatch" "${CODE_ROOT}/config/analysis_config.json" "${CODE_FP}" \
    "${LOGDIR}/j04_${J04}.out" "${LOGDIR}/j04_${J04}.err" "${RESULTS}/03_integrated_analysis/data_analysis_round1/tables"
  printf 'J05\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${J05}" "${J04}" "${CODE_ROOT}/sbatch/j05_report.sbatch" "${CODE_ROOT}/config/analysis_config.json" "${CODE_FP}" \
    "${LOGDIR}/j05_${J05}.out" "${LOGDIR}/j05_${J05}.err" "${RESULTS}/03_integrated_analysis/data_analysis_round1"
} > "${MANIFEST}"

echo "J00=${J00} J01=${J01} J02=${J02} J03=${J03} J04=${J04} J05=${J05}"
echo "manifest=${MANIFEST}"
