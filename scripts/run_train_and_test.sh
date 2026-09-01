#!/usr/bin/env bash
set -euo pipefail
# One-click release entry point.  It trains the reproducible M26 chain and
# then runs the released inference/submission pipeline.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:?Set TRAIN_DATA_ROOT}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:?Set TEST_DATA_ROOT}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/outputs/retrain_m26_one_click}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/final_test_one_click}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate EV39 before running this script." >&2
  exit 2
fi
if [[ ! -d "$TRAIN_DATA_ROOT/train" || ! -d "$TRAIN_DATA_ROOT/val" ]]; then
  echo "TRAIN_DATA_ROOT must contain train/ and val/: $TRAIN_DATA_ROOT" >&2
  exit 2
fi
if [[ ! -d "$TEST_DATA_ROOT/test" ]]; then
  echo "TEST_DATA_ROOT must contain test/: $TEST_DATA_ROOT" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite RUN_ROOT: $RUN_ROOT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
if ! python -c "import torch, HAIS_OP; import spconv.pytorch; assert torch.cuda.is_available()"; then
  echo "CUDA, HAIS_OP or spconv is unavailable; run the complete EV39 setup first." >&2
  exit 2
fi

echo "[1/2] training M4 -> M13 -> M15 -> M20 -> M26"
DATA_ROOT="$TRAIN_DATA_ROOT" RUN_ROOT="$RUN_ROOT" bash "$PROJECT_DIR/scripts/retrain_m26_chain.sh"
M26_RUN_ROOT="$RUN_ROOT/m26_targetflow_m20e3_dense8_e12_seed53/runs"
mapfile -t M26_MATCHES < <(find "$M26_RUN_ROOT" -type f -name 'epoch_003_seed53.pt' -print | sort)
if [[ "${#M26_MATCHES[@]}" -ne 1 ]]; then
  echo "Expected exactly one M26 epoch_003 checkpoint under $M26_RUN_ROOT; found ${#M26_MATCHES[@]}" >&2
  printf '%s\n' "${M26_MATCHES[@]}" >&2
  exit 2
fi
M26_CHECKPOINT="${M26_MATCHES[0]}"
echo "[2/2] full checkpoint inference and M244 submission build"
DATA_ROOT="$TEST_DATA_ROOT" TEST_ROOT="$TEST_DATA_ROOT/test" OUTPUT_ROOT="$OUTPUT_ROOT" M26_CHECKPOINT="$M26_CHECKPOINT" bash "$PROJECT_DIR/scripts/run_m244_from_checkpoints.sh"
echo "Training and test completed."
echo "M26 checkpoint: $M26_CHECKPOINT"
echo "M244 package: $OUTPUT_ROOT/m244_empty_top4_raw_test.zip"
