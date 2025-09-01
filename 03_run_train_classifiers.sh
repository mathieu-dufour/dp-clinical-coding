#!/usr/bin/env bash
set -euo pipefail

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_ASYNC_ERROR_HANDLING=1

# ========= Models =========
MODEL_1B="meta-llama/Llama-3.2-1B"
MODEL_3B="meta-llama/Llama-3.2-3B"

# ========= Data =========
REAL_TRAIN="data/real/real_train_codes_notes.pt"
REAL_VAL="data/real/real_val_codes_notes.pt"

SYN_TRAIN_SYN () { echo "data/synthetic/DP-Synthetic/eps${1}/synth_train_codes_notes_cleaned.pt"; }
SYN_VAL_SYN   () { echo "data/synthetic/DP-Synthetic/eps${1}/synth_val_codes_notes_cleaned.pt"; }

SYN_TRAIN_DIS () { echo "data/synthetic/DP-Distil/eps${1}/synth_train_codes_notes_cleaned.pt"; }
SYN_VAL_DIS   () { echo "data/synthetic/DP-Distil/eps${1}/synth_val_codes_notes_cleaned.pt"; }

# ========= Common training settings =========
EPOCHS=18
PATIENCE=3
MAXLEN=512
NUM_WORKERS=8

CLIP_1B=1.0
CLIP_3B=0.7

DELTA=1e-5
DELTA_DISTIL=5e-6

# LoRA ranks & alpha
R_1B=4
LORA_ALPHA_1B=16
R_3B=8
LORA_ALPHA_3B=32

# Batch sizes
BATCH_1B=56
BATCH_1B_DP=28
BATCH_3B_DP=10

# GPU selection
GPUS="${GPUS:-0}"

run_ddp () {
  CUDA_VISIBLE_DEVICES=${GPUS} torchrun --standalone --nproc_per_node=1 "$@"
}

mkdir -p artifacts/classifiers

echo "=== LoRA–No–DP (1B, real) ==="
OUT="artifacts/classifiers/LoRA-No-DP_1B"
mkdir -p "${OUT}"
run_ddp 03_a_train_classifier.py \
  --model_name "${MODEL_1B}" \
  --train_pairs "${REAL_TRAIN}" \
  --val_pairs "${REAL_VAL}" \
  --threshold_val_pairs "${REAL_VAL}" \
  --out_dir "${OUT}" \
  --epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --batch_size ${BATCH_1B} \
  --max_seq_len ${MAXLEN} \
  --lora_r ${R_1B} \
  --lora_alpha ${LORA_ALPHA_1B} \
  --lr 5e-4 \
  --scheduler cosine \
  --warmup_ratio 0.1 \
  --max_grad_norm ${CLIP_1B} \
  --pipeline_type LoRA-No-DP \
  --use_flash_attention \
  --num_workers ${NUM_WORKERS}

for EPS in 2 4 6; do
  echo "=== ε=${EPS} loop ==="

  echo "--- DP-Small (1B, DP; real) ---"
  OUT="artifacts/classifiers/DP-Small_1B/eps${EPS}"
  mkdir -p "${OUT}"
  run_ddp 03_a_train_classifier.py \
    --model_name "${MODEL_1B}" \
    --train_pairs "${REAL_TRAIN}" \
    --val_pairs "${REAL_VAL}" \
    --threshold_val_pairs "${REAL_VAL}" \
    --out_dir "${OUT}" \
    --with_dp \
    --epsilon "${EPS}" \
    --delta "${DELTA}" \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --batch_size ${BATCH_1B_DP} \
    --max_seq_len ${MAXLEN} \
    --lora_r ${R_1B} \
    --lora_alpha ${LORA_ALPHA_1B} \
    --lr 1.5e-3 \
    --max_grad_norm ${CLIP_1B} \
    --pipeline_type DP-Small \
    --num_workers ${NUM_WORKERS}

  echo "--- DP-Synthetic (1B, non-DP; synthetic train + synthetic val; thresholds on real val) ---"
  OUT="artifacts/classifiers/DP-Synthetic_1B/eps${EPS}"
  mkdir -p "${OUT}"
  run_ddp 03_a_train_classifier.py \
    --model_name "${MODEL_1B}" \
    --train_pairs "$(SYN_TRAIN_SYN ${EPS})" \
    --val_pairs   "$(SYN_VAL_SYN   ${EPS})" \
    --threshold_val_pairs "${REAL_VAL}" \
    --out_dir "${OUT}" \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --batch_size ${BATCH_1B} \
    --max_seq_len ${MAXLEN} \
    --lora_r ${R_1B} \
    --lora_alpha ${LORA_ALPHA_1B} \
    --lr 5e-4 \
    --scheduler cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm ${CLIP_1B} \
    --pipeline_type DP-Synthetic \
    --use_flash_attention \
    --num_workers ${NUM_WORKERS}

  # DP-Distil: Use half privacy budget for generator and teacher models
  HALF_EPS=$((EPS / 2))
  echo "--- DP-Distil: HALF_EPS=${HALF_EPS} (teacher) ---"
  OUT_TEACH="artifacts/classifiers/DP-Distil_teacher_3B/eps${HALF_EPS}"
  mkdir -p "${OUT_TEACH}"
  run_ddp 03_a_train_classifier.py \
    --model_name "${MODEL_3B}" \
    --train_pairs "${REAL_TRAIN}" \
    --val_pairs   "${REAL_VAL}" \
    --threshold_val_pairs "${REAL_VAL}" \
    --out_dir "${OUT_TEACH}" \
    --with_dp \
    --epsilon ${HALF_EPS} \
    --delta ${DELTA_DISTIL} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --batch_size ${BATCH_3B_DP} \
    --max_seq_len ${MAXLEN} \
    --lora_r ${R_3B} \
    --lora_alpha ${LORA_ALPHA_3B} \
    --lr 3e-4 \
    --max_grad_norm ${CLIP_3B} \
    --pipeline_type DP-Distil \
    --num_workers ${NUM_WORKERS}

  # Export teacher model predictions for knowledge distillation
  TEACH_CKPT="${OUT_TEACH}/best_model"
  LOGITS_TRAIN="data/synthetic/DP-Distil/eps${HALF_EPS}/teacher_train_logits.pt"
  LOGITS_VAL="data/synthetic/DP-Distil/eps${HALF_EPS}/teacher_val_logits.pt"
  run_ddp 03_b_export_distil_teacher_logits.py \
    --teacher_ckpt "${TEACH_CKPT}" \
    --pairs_pt "$(SYN_TRAIN_DIS ${HALF_EPS})" \
    --out_path "${LOGITS_TRAIN}" \
    --batch_size 16 \
    --max_seq_len ${MAXLEN} \
    --save_logits
  run_ddp 03_b_export_distil_teacher_logits.py \
    --teacher_ckpt "${TEACH_CKPT}" \
    --pairs_pt "$(SYN_VAL_DIS ${HALF_EPS})" \
    --out_path "${LOGITS_VAL}" \
    --batch_size 16 \
    --max_seq_len ${MAXLEN} \
    --save_logits

  echo "--- DP-Distil student (1B; synthetic train/val + KD logits; thresholds on real val) ---"
  OUT_STUD="artifacts/classifiers/DP-Distil_student_1B/eps${EPS}"
  mkdir -p "${OUT_STUD}"
  run_ddp 03_a_train_classifier.py \
    --model_name "${MODEL_1B}" \
    --train_pairs "$(SYN_TRAIN_DIS ${HALF_EPS})" \
    --val_pairs   "$(SYN_VAL_DIS   ${HALF_EPS})" \
    --threshold_val_pairs "${REAL_VAL}" \
    --out_dir "${OUT_STUD}" \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --batch_size ${BATCH_1B} \
    --max_seq_len ${MAXLEN} \
    --lora_r ${R_1B} \
    --lora_alpha ${LORA_ALPHA_1B} \
    --lr 1e-3 \
    --pipeline_type DP-Distil \
    --use_flash_attention \
    --num_workers ${NUM_WORKERS} \
    --distil_train "${LOGITS_TRAIN}" \
    --distil_val   "${LOGITS_VAL}" \
    --distil_source logits \
    --distil_alpha 0.0
done

echo "✅ All classifier trainings complete → artifacts/classifiers"
