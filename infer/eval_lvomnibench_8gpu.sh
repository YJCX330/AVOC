#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_NUM_FRAMES=320

DATA_PATH="./data/lvomnibench_formatted_question.json"
BASE_MODEL_PATH="./ckpt/checkpoint-AVOC"
MODEL_PATH="./ckpt/checkpoint-AVOC"
RESULTS_DIR="./results"
LOG_DIR=${RESULTS_DIR}

# Compression override for eval.py: true / false / empty; empty uses checkpoint config.json.
COMPRESSION_ENABLED="true"
COMPRESSION_TOPK=10240
COMPRESSION_VIDEO_AUDIO_TOKEN_RATIO="2"  # e.g. 2.0 = 2/3 video, 1/3 audio; empty = use default balance bias
COMPRESSION_DIVERSITY_LAMBDA="0.15"
COMPRESSION_MMR_WINDOW="3"           # ±N blocks penalise each other in MMR; empty = use checkpoint default (3). Only effective when COMPRESSION_DIVERSITY_LAMBDA > 0.

COMP_ARGS=()
if [[ "${COMPRESSION_ENABLED}" == "true" ]]; then
    COMP_ARGS+=(--compression)
elif [[ "${COMPRESSION_ENABLED}" == "false" ]]; then
    COMP_ARGS+=(--no-compression)
fi
[[ -n "${COMPRESSION_TOPK}" ]] && COMP_ARGS+=(--compression-topk "${COMPRESSION_TOPK}")
[[ -n "${COMPRESSION_VIDEO_AUDIO_TOKEN_RATIO}" ]] && COMP_ARGS+=(--compression-video-audio-token-ratio "${COMPRESSION_VIDEO_AUDIO_TOKEN_RATIO}")
[[ -n "${COMPRESSION_DIVERSITY_LAMBDA}" ]] && COMP_ARGS+=(--compression-diversity-lambda "${COMPRESSION_DIVERSITY_LAMBDA}")
[[ -n "${COMPRESSION_MMR_WINDOW}" ]] && COMP_ARGS+=(--compression-mmr-window "${COMPRESSION_MMR_WINDOW}")

GPU_LIST=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_GPUS=${#GPU_LIST[@]}
# lvomnibench.json has 1014 items; ceil(1014/8)=127 so the last GPU gets [889,1016) -> 125 samples.
SAMPLES_PER_GPU=127

mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  gpu_id="${GPU_LIST[$i]}"
  start_idx=$((i * SAMPLES_PER_GPU))
  end_idx=$((start_idx + SAMPLES_PER_GPU))
  output_path="${RESULTS_DIR}/lvomnibench_${start_idx}_${end_idx}.json"
  log_path="${LOG_DIR}/eval_lvomnibench_gpu${gpu_id}_${start_idx}_${end_idx}.log"

  echo "Launching GPU ${gpu_id}: [${start_idx}, ${end_idx}) -> ${output_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python3 infer/eval.py \
    --base-model-path "${BASE_MODEL_PATH}" \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-path "${output_path}" \
    --start-idx "${start_idx}" \
    --end-idx "${end_idx}" \
    "${COMP_ARGS[@]}" > "${log_path}" 2>&1 &
done

echo "All jobs started (${NUM_GPUS} GPUs × up to ${SAMPLES_PER_GPU} samples; 1014 total)."
echo "Logs: ${LOG_DIR}"
