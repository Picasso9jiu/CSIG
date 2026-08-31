#!/usr/bin/env bash
set -euo pipefail

# Full inference path for M244. This is slower than run_m244_fast.sh because
# it regenerates M233 scores from the released M10/M26/M111 checkpoints.
# It uses only event statistics for domain routing, never video names/labels.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing test/}"
TEST_ROOT="${TEST_ROOT:-$DATA_ROOT/test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/m244_from_checkpoints}"

if [[ ! -d "$TEST_ROOT" ]]; then
  echo "Missing test directory: $TEST_ROOT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite existing OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
WORK_ROOT="$OUTPUT_ROOT/work"
GROUP_ROOT="$WORK_ROOT/groups"
PRED_ROOT="$WORK_ROOT/predictions"
RAW_ROOT="$WORK_ROOT/raw_scores"
mkdir -p "$WORK_ROOT" "$PRED_ROOT" "$RAW_ROOT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the EV39 Conda environment before running this script." >&2
  exit 2
fi
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

python "$PROJECT_DIR/scripts/split_m244_domains.py" \
  --test-root "$TEST_ROOT" \
  --output-root "$GROUP_ROOT" \
  --copy

COMMON_ARGS=(
  --config "$PROJECT_DIR/configs/evisseg_evuav.yaml" --set
  TEST.challenge_dataset_mode=test
  TEST.prediction_threshold=0.7226
  TEMPORAL_FRAME.temporal_frame_enabled=false
  TEMPORAL_MEMORY.temporal_memory_enabled=true
  TEMPORAL_MEMORY.temporal_memory_model_path="$PROJECT_DIR/checkpoints/m26_targetflow_m20e3_epoch_003_seed53.pt"
  TEMPORAL_MEMORY.temporal_memory_secondary_model_path="$PROJECT_DIR/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
  TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_enabled=true
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_model_path="$PROJECT_DIR/checkpoints/m111_phase_specialist_seed72_73_76_average.pt"
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_event_count_cutoff=30000
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_weight=0.25
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_offset=25
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0
  TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8
  INFERENCE_TTA.p41_temporal_phase_enabled=true
  INFERENCE_TTA.p41_temporal_phase_offset=25
  INFERENCE_TTA.p41_temporal_phase_original_weight=0.75
  INFERENCE_TTA.p41_temporal_phase_min_event_count=30000
  POSTPROCESS.p0_enabled=true
  POSTPROCESS.p0_spatial_radius=2
  POSTPROCESS.p0_temporal_bin_size=50
  POSTPROCESS.p0_temporal_radius_bins=1
  POSTPROCESS.p0_min_cluster_events=3
  POSTPROCESS.p0_min_duration_bins=5
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true
  POSTPROCESS.p0c_retain_min_score=0.95
  POSTPROCESS.p0b_enabled=false
  POSTPROCESS.p18_score_track_recovery_enabled=true
  POSTPROCESS.p18_event_count_cutoff=1
  POSTPROCESS.p18_max_event_count=0
  POSTPROCESS.p18_candidate_floor=0.53
  POSTPROCESS.p18_spatial_radius=5
  POSTPROCESS.p18_temporal_bin_size=50
  POSTPROCESS.p18_max_link_distance=8.0
  POSTPROCESS.p18_max_gap_bins=1
  POSTPROCESS.p18_min_track_bins=4
  POSTPROCESS.p18_restore_mode=best
  POSTPROCESS.p6_density_threshold_enabled=true
  POSTPROCESS.p6_event_count_cutoff=30000
  POSTPROCESS.p6_low_density_threshold=0.718
  POSTPROCESS.p6_polarity_domain_enabled=true
  POSTPROCESS.p6_middle_event_count_cutoff=200000
  POSTPROCESS.p6_middle_density_threshold=0.728
  POSTPROCESS.p6_high_polarity_minority_cutoff=0.20
  POSTPROCESS.p6_high_imbalanced_threshold=0.722
  POSTPROCESS.p6_high_balanced_threshold=0.724
  POSTPROCESS.p32_track_quality_bonus_enabled=true
  POSTPROCESS.p32_candidate_floor=0.60
  POSTPROCESS.p32_spatial_radius=2
  POSTPROCESS.p32_temporal_bin_size=50
  POSTPROCESS.p32_max_link_distance=8.0
  POSTPROCESS.p32_max_gap_bins=2
  POSTPROCESS.p32_min_track_bins=4
  POSTPROCESS.p32_min_seed_components=3
  POSTPROCESS.p32_bonus=0.010
  POSTPROCESS.p32_max_score_cap=0.97
  POSTPROCESS.p32_max_motion_residual=2.0
  POSTPROCESS.p32_velocity_history_bins=2
  POSTPROCESS.m124_background_verifier_enabled=true
)

run_group() {
  local domain="$1"
  local threshold="$2"
  local input_root="$GROUP_ROOT/$domain"
  local output_dir="$PRED_ROOT/$domain"
  local artifact="$WORK_ROOT/m124_${domain}_${threshold}.pkl.gz"
  if [[ ! -d "$input_root/test" ]] || ! compgen -G "$input_root/test/test_*.npz" > /dev/null; then
    return
  fi
  python "$PROJECT_DIR/scripts/prepare_m124_artifact.py" \
    --source "$PROJECT_DIR/artifacts/m124_background_verifier_threshold065.pkl.gz" \
    --destination "$artifact" \
    --threshold "$threshold"
  mkdir -p "$output_dir"
  if [[ "$domain" == "h2_extreme" ]]; then
    export EVSOD_DEBUG_RAW_DIR="$RAW_ROOT"
  else
    unset EVSOD_DEBUG_RAW_DIR || true
  fi
  python -u "$PROJECT_DIR/submit_challenge2.py" "${COMMON_ARGS[@]}" \
    DATA.root="$input_root" \
    TEST.challenge_output_dir="$output_dir" \
    POSTPROCESS.m124_background_verifier_model_path="$artifact" \
    POSTPROCESS.m124_background_verifier_threshold="$threshold"
}

run_group low_middle 0.94
run_group h1 0.62
run_group h2_other 0.65
run_group h2_extreme 0.63
unset EVSOD_DEBUG_RAW_DIR || true

MERGE_ARGS=(--test-root "$TEST_ROOT" --output-dir "$OUTPUT_ROOT/m233_predictions" --zip "$OUTPUT_ROOT/m233_base_submission.zip")
for domain in low_middle h1 h2_other h2_extreme; do
  if [[ -d "$PRED_ROOT/$domain" ]]; then
    MERGE_ARGS+=(--input-dir "$PRED_ROOT/$domain")
  fi
done
python "$PROJECT_DIR/scripts/merge_submission_dirs.py" "${MERGE_ARGS[@]}"

python "$PROJECT_DIR/scripts/audit_m243_raw_scores.py" \
  --test-root "$TEST_ROOT" \
  --raw-dir "$RAW_ROOT" \
  --base-zip "$OUTPUT_ROOT/m233_base_submission.zip" \
  --output "$WORK_ROOT/m243_report.json"

python "$PROJECT_DIR/scripts/rebuild_m244.py" \
  --test-root "$TEST_ROOT" \
  --base "$OUTPUT_ROOT/m233_base_submission.zip" \
  --report "$WORK_ROOT/m243_report.json" \
  --output "$OUTPUT_ROOT/m244_empty_top4_raw_test.zip"

python "$PROJECT_DIR/scripts/verify_submission.py" \
  --zip "$OUTPUT_ROOT/m244_empty_top4_raw_test.zip" \
  --test-root "$TEST_ROOT"

echo "M244 checkpoint inference output: $OUTPUT_ROOT/m244_empty_top4_raw_test.zip"
