#!/usr/bin/env bash
set -euo pipefail

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_ASYNC_ERROR_HANDLING=1

# ======= MODELS =======
GEN_3B="meta-llama/Llama-3.2-3B"

# ======= DATA =======
REAL_TRAIN_PT="data/real/real_train_codes_notes.pt"
REAL_VAL_PT="data/real/real_val_codes_notes.pt"

# ======= TRAIN HP =======
EPOCHS=18
BATCH_SIZE=14
ACCUM_STEPS=1
MAX_SEQ_LEN=512
DELTA_SYN=1e-5
DELTA_DISTIL=5e-6
LR=2e-4
MAX_GRAD_NORM=1.0
PATIENCE=3

# ======= GEN HP =======
GEN_BATCH=320
MAX_NEW_TOKENS=512
TEMPERATURE=0.8
TOP_P=0.9
SEED=123

# ======= GPUs =======
GPUS="${GPUS:-0}"

run_torch () {
  CUDA_VISIBLE_DEVICES="$1" torchrun --standalone --nproc_per_node=1 "$@"
}

# Generate and clean synthetic data for given model and output directory
gen_and_clean () {
  local BEST_MODEL_DIR="$1"
  local OUT_DIR="$2"
  mkdir -p "${OUT_DIR}"

  echo "→ Generating synthetic TRAIN into ${OUT_DIR}"
  CUDA_VISIBLE_DEVICES=${GPUS} torchrun --standalone --nproc_per_node=1 02_b_generate_synthetic.py \
    --model_dir "${BEST_MODEL_DIR}" \
    --real_train_pairs_pt "${REAL_TRAIN_PT}" \
    --gen_batch ${GEN_BATCH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --seed ${SEED} \
    --out_pt "${OUT_DIR}/synth_train_codes_notes.pt"

  echo "→ Generating synthetic VAL into ${OUT_DIR}"
  CUDA_VISIBLE_DEVICES=${GPUS} torchrun --standalone --nproc_per_node=1 02_b_generate_synthetic.py \
    --model_dir "${BEST_MODEL_DIR}" \
    --real_train_pairs_pt "${REAL_VAL_PT}" \
    --gen_batch ${GEN_BATCH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --seed ${SEED} \
    --out_pt "${OUT_DIR}/synth_val_codes_notes.pt"

  echo "→ Cleaning synthetic (train & val)"
  python utils/clean_synth_notes.py --input_pt "${OUT_DIR}/synth_train_codes_notes.pt" --output_pt "${OUT_DIR}/synth_train_codes_notes_cleaned.pt"
  python utils/clean_synth_notes.py --input_pt "${OUT_DIR}/synth_val_codes_notes.pt"   --output_pt "${OUT_DIR}/synth_val_codes_notes_cleaned.pt"
}

# Train DP-SYNTHETIC generators with privacy budget ε ∈ {2,4,6}
for EPS in 2 4 6; do
  OUT_DIR="artifacts/generators/DP-Synthetic_generator_3b/eps${EPS}"
  mkdir -p "${OUT_DIR}"
  echo -e "\n===== DP-SYNTHETIC GEN: ε=${EPS} ====="
  export CUDA_VISIBLE_DEVICES=${GPUS}
  torchrun --standalone --nproc_per_node=1 02_a_train_generative.py \
    --gpus 1 \
    --model_name "${GEN_3B}" \
    --train_pairs_pt "${REAL_TRAIN_PT}" \
    --val_pairs_pt   "${REAL_VAL_PT}" \
    --out_dir "${OUT_DIR}" \
    --epsilon ${EPS} --delta ${DELTA_SYN} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --accum_steps ${ACCUM_STEPS} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --lr ${LR} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --patience ${PATIENCE}

  gen_and_clean "${OUT_DIR}/best_model" "data/synthetic/DP-Synthetic/eps${EPS}"
done

# Train DP-DISTIL generators with half privacy budget ε/2 ∈ {1,2,3}
for HALF_EPS in 1 2 3; do
  OUT_DIR="artifacts/generators/DP-Distil_generator_3B/eps${HALF_EPS}"
  mkdir -p "${OUT_DIR}"
  echo -e "\n===== DP-DISTIL GEN: ε/2=${HALF_EPS} ====="
  export CUDA_VISIBLE_DEVICES=${GPUS}
  torchrun --standalone --nproc_per_node=1 02_a_train_generative.py \
    --gpus 1 \
    --model_name "${GEN_3B}" \
    --train_pairs_pt "${REAL_TRAIN_PT}" \
    --val_pairs_pt   "${REAL_VAL_PT}" \
    --out_dir "${OUT_DIR}" \
    --epsilon ${HALF_EPS} --delta ${DELTA_DISTIL} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --accum_steps ${ACCUM_STEPS} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --lr ${LR} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --patience ${PATIENCE}

  gen_and_clean "${OUT_DIR}/best_model" "data/synthetic/DP-Distil/eps${HALF_EPS}"
done

echo -e "\nALL DONE – generators under artifacts/generators/, synthetic under data/synthetic/"
