#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:?epochs argument is required}"
INPUT_WEIGHT_BITS="${2:?input_weight_bits is required}"
WEIGHT_BITS="${3:?weight_bits is required}"
ACTIVATION_BITS="${4:?activation_bits is required}"
TRIAL_NAME="${5:?trial name is required}"
CHECKPOINT_DIR="${6:?checkpoint directory is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -u "${SCRIPT_DIR}/CPU/hpo_katib_run.py" \
  --epochs "${EPOCHS}" \
  --input_weight_bits "${INPUT_WEIGHT_BITS}" \
  --weight_bits "${WEIGHT_BITS}" \
  --activation_bits "${ACTIVATION_BITS}" \
  --trial-name "${TRIAL_NAME}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"
