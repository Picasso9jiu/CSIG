#!/usr/bin/env bash
set -euo pipefail

# Versioned release training chain: M4 -> M13 -> M15 -> M20 -> M26.
# DATA_ROOT must be the official directory containing train/ and val/.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing train/ and val/}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/outputs/retrain_m26}"

cd "$PROJECT_DIR"
M4_CKPT="$PROJECT_DIR/checkpoints/m4_dacc_m5_best_loss_seed42.pt"
M13_ROOT="$RUN_ROOT/m13_dense_views4_ft30_seed42"
M15_ROOT="$RUN_ROOT/m15_e3_low_lr_seed43"
M20_ROOT="$RUN_ROOT/m20_attn_dense_views8_e12_seed48"
M26_ROOT="$RUN_ROOT/m26_targetflow_m20e3_dense8_e12_seed53"

python -u train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=42 TRAIN.epochs=30 TRAIN.batch_size=1 TRAIN.lr=0.00002 \
  TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.000001 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M13_ROOT" \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M4_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=4 \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_weight=0.05 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_margin_logit=1.0 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_min_points=3 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_warmup_epochs=3

M13_E3="$(find "$M13_ROOT/runs" -type f -name 'epoch_003_seed42.pt' -print -quit)"
test -n "$M13_E3"

python -u train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=43 TRAIN.epochs=8 TRAIN.batch_size=1 TRAIN.lr=0.000003 \
  TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.0000003 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M15_ROOT" \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M13_E3" \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=4 \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=false

M15_E8="$(find "$M15_ROOT/runs" -type f -name 'epoch_008_seed43.pt' -print -quit)"
test -n "$M15_E8"

python -u train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=48 TRAIN.epochs=12 TRAIN.batch_size=1 TRAIN.lr=0.000001 \
  TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.0000001 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M20_ROOT" \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M15_E8" \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=8 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled=false \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=false

M20_E3="$(find "$M20_ROOT/runs" -type f -name 'epoch_003_seed48.pt' -print -quit)"
test -n "$M20_E3"

python -u train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=53 TRAIN.epochs=12 TRAIN.batch_size=1 TRAIN.lr=0.000001 \
  TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.0000001 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M26_ROOT" \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M20_E3" \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=8 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_attention_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_loss_weight=0.01 \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_lr_multiplier=4.0 \
  TEMPORAL_MEMORY.temporal_memory_advection_max_flow=2.0 \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_weight=0.5 \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_huber_delta=1.0

M26_E3="$(find "$M26_ROOT/runs" -type f -name 'epoch_003_seed53.pt' -print -quit)"
test -n "$M26_E3"
echo "Rebuilt M26 checkpoint: $M26_E3"
