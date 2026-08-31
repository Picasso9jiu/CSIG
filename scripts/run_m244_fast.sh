#!/usr/bin/env bash
set -euo pipefail

# Fast, CPU-only reconstruction of the official M244 package. It does not
# rerun the network and never reads labels. DATA_ROOT must contain test/.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing test/}"
TEST_ROOT="${TEST_ROOT:-$DATA_ROOT/test}"
OUTPUT_ZIP="${OUTPUT_ZIP:-$PROJECT_DIR/outputs/m244_empty_top4_raw_test.zip}"

python "$PROJECT_DIR/scripts/rebuild_m244.py" \
  --test-root "$TEST_ROOT" \
  --output "$OUTPUT_ZIP"

python "$PROJECT_DIR/scripts/verify_submission.py" \
  --zip "$OUTPUT_ZIP" \
  --test-root "$TEST_ROOT" \
  --expected-positive 87706

echo "M244 package: $OUTPUT_ZIP"
