#!/usr/bin/env bash
set -euo pipefail

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_ASYNC_ERROR_HANDLING=1

# Use thresholds computed during training by default
USE_TRAINVAL_THRESHOLDS=false

# ======= DATA =======
REAL_TRAIN="data/real/real_train_codes_notes.pt"
REAL_VAL="data/real/real_val_codes_notes.pt"
REAL_TEST="data/real/real_test_codes_notes.pt"

EVAL_PAIRS="${REAL_TEST}"
MI_MEMBERS="${REAL_TRAIN}"
MI_NONMEMBERS="${REAL_TEST}"

# ======= ROOTS =======
ROOT_CLASSIFIERS="artifacts/classifiers"
ROOT_EVAL="artifacts/eval"

# ======= RUNTIME =======
BATCH=64
MAXLEN=512
MIBATCH=48
VERBOSITY=1
GPU="1"

run_py () {
  CUDA_VISIBLE_DEVICES=${GPU} python "$@"
}

# Helper function to determine appropriate classification thresholds
get_thresholds_path () {
  local CKPT_DIR="$1"   # .../best_model
  local THR_PATH=""
  if $USE_TRAINVAL_THRESHOLDS; then
    THR_PATH="${CKPT_DIR}/optimal_thresholds_trainval.npy"
    if [[ ! -f "${THR_PATH}" ]]; then
      >&2 echo "🧮 Computing train+val thresholds for: ${CKPT_DIR}"
      run_py 04_b_find_trainval_optimal_thresholds.py \
        --ckpt "${CKPT_DIR}" \
        --train_pairs "${REAL_TRAIN}" \
        --val_pairs   "${REAL_VAL}" \
        --batch ${BATCH} \
        --max_len ${MAXLEN} > /dev/null 2>&1
    fi
  else
    THR_PATH="${CKPT_DIR}/optimal_thresholds.npy"
    >&2 echo "↪︎ Using training-time thresholds: ${THR_PATH}"
  fi
  printf '%s\n' "${THR_PATH}"
}

# Helper function to evaluate a single classifier checkpoint
eval_ckpt () {
  local CKPT_DIR="$1"  # path to .../best_model
  local OUT_DIR="$2"

  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "⏭️  Skip (missing): ${CKPT_DIR}"
    return 0
  fi

  mkdir -p "${OUT_DIR}"
  echo "🔎 Evaluating: ${CKPT_DIR} → ${OUT_DIR}"

  local THRESH_PATH
  THRESH_PATH="$(get_thresholds_path "${CKPT_DIR}")"

  run_py 04_a_eval_classifiers.py \
    --ckpt "${CKPT_DIR}" \
    --pairs "${EVAL_PAIRS}" \
    --thresholds_path "${THRESH_PATH}" \
    --mi_member_pairs "${MI_MEMBERS}" \
    --mi_nonmember_pairs "${MI_NONMEMBERS}" \
    --batch ${BATCH} \
    --max_len ${MAXLEN} \
    --mi_batch ${MIBATCH} \
    --verbosity ${VERBOSITY} \
    --out_dir "${OUT_DIR}"
}

# Evaluate all trained 1B parameter classifiers
eval_ckpt "${ROOT_CLASSIFIERS}/LoRA-No-DP_1B/best_model" \
          "${ROOT_EVAL}/LoRA-No-DP_1B"

for EPS in 2 4 6; do
  eval_ckpt "${ROOT_CLASSIFIERS}/DP-Small_1B/eps${EPS}/best_model" \
            "${ROOT_EVAL}/DP-Small_1B/eps${EPS}"
done

for EPS in 2 4 6; do
  eval_ckpt "${ROOT_CLASSIFIERS}/DP-Synthetic_1B/eps${EPS}/best_model" \
            "${ROOT_EVAL}/DP-Synthetic_1B/eps${EPS}"
done

for EPS in 2 4 6; do
  eval_ckpt "${ROOT_CLASSIFIERS}/DP-Distil_student_1B/eps${EPS}/best_model" \
            "${ROOT_EVAL}/DP-Distil_student_1B/eps${EPS}"
done

# Evaluate DP-Distil teacher models (3B parameters)
for HALF in 1 2 3; do
  eval_ckpt "${ROOT_CLASSIFIERS}/DP-Distil_teacher_3B/eps${HALF}/best_model" \
            "${ROOT_EVAL}/DP-Distil_teacher_3B/eps$((HALF*2))"
done

echo "✅ All evaluations complete → ${ROOT_EVAL}"
