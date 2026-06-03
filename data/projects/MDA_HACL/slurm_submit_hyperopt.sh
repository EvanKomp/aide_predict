#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --account=proteinml
#SBATCH --job-name=bnn_opt
#SBATCH --output=slurm_logs/hyperopt_%A_%a.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --time=24:00:00            # 24h per task; bump if 100 trials don't finish
#SBATCH --cpus-per-task=30
#SBATCH --qos=high
#SBATCH --array=0-4                # one task per metric (see METRICS[] below)

# ─────────────────────────────────────────────────────────────────────────────
# Re-run new_opt_05 hyperopt for all 5 metrics on the full substrate-split CV.
# Scope is "all-substrates" (no formaldehyde-only shortcut). Each metric runs
# as its own array task, independently, on its own GPU.
#
# Submit:    sbatch data/projects/MDA_HACL/slurm_submit_hyperopt.sh
# Monitor:   squeue -u $USER
# Stop one:  scancel <jobid>_<arrayidx>
# Stop all:  scancel <jobid>
#
# Each task writes to:
#   results/new_opt_05/all_substrates_<metric>/
#       best_hyperparams.json
#       study_results.json
#       working_range.json
#       narrowed_config_suggestion.yaml
#       (+ optuna SQLite storage)
#
# After all 5 finish, re-run the substrate-split CV + scoring via the existing
# data/projects/MDA_HACL/slurm_submit.sh (which reads each study's
# best_hyperparams.json).
# ─────────────────────────────────────────────────────────────────────────────

module load conda
module load cuda
conda activate aidep

PROJECT_DIR=/kfs2/projects/proteinml/repos/aide_predict/data/projects/MDA_HACL
cd "$PROJECT_DIR" || exit 1
mkdir -p slurm_logs

# Five metrics to re-optimise (all now per-substrate-correct after the
# aggregation-semantics bugfix). The "overall" names (spearman_rho,
# top3_recovery, ndcg) used to mean pooled-across-substrates; they now mean
# within-substrate then mean-across-substrates.
METRICS=(
    "ndcg"
    "spearman_rho"
    "top3_recovery"
    "per_position_ndcg_mean"
    "per_position_top3_recovery_mean"
)

METRIC="${METRICS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$METRIC" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no matching metric"
    exit 2
fi

STUDY_NAME="all_substrates_${METRIC}"
OUT_DIR="results/new_opt_05/${STUDY_NAME}"

echo "============================================================"
echo "task ${SLURM_ARRAY_TASK_ID}: ${METRIC}"
echo "  scope:      all-substrates (full substrate-split CV per trial)"
echo "  output:     ${OUT_DIR}"
echo "  trials:     from config.yaml cv.n_hyperopt_trials (default 100)"
echo "  device:     cuda:0"
echo "  started:    $(date)"
echo "============================================================"

# NOTE: NO --fresh. We want to resume any partial study.sqlite from an earlier
# timed-out run. new_opt_05.py uses load_if_exists=True and only runs
# (n_trials_total - n_completed) more trials, so resubmission picks up where
# the wall-clock killed the previous job. Pass --fresh manually if you ever
# need to start from scratch.
python code/round3/new_opt_05.py \
    --scope all-substrates \
    --objective-metric "${METRIC}" \
    --output-dir "${OUT_DIR}" \
    --device cuda:0

echo "============================================================"
echo "task ${SLURM_ARRAY_TASK_ID} (${METRIC}) finished: $(date)"
echo "============================================================"
