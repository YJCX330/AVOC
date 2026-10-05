#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# ---- paths ----
MODEL="./ckpt/stage1"
DATA="./data/finevideo.json" 
OUTPUT_DIR="./ckpt/stage2"
EVAL_DATA=""

# ---- training hyper-parameters ----
MODEL_MAX_LENGTH=40000                  
MAX_SLICE_NUMS=1                       # image slice count (1 = 64 tokens/frame)
MAX_NUM_FRAMES=320                     # max video frames to sample
LR=5e-6
COMPRESSION_LR=5e-5
EPOCHS=1
MAX_STEPS=-1                           # -1 = no limit; train for EPOCHS. >0 overrides EPOCHS and stops at that step.
BATCH_SIZE=1                           # per-device batch size (keep 1 for large models)
GRAD_ACCUM=2                          # effective batch = GPUS * BATCH_SIZE * GRAD_ACCUM

# ---- what to train ----
TUNE_VISION=false                       # train vision encoder
TUNE_LLM=true                          # train LLM backbone
TUNE_AUDIO=false                        # train audio encoder
USE_LORA=false                         # set true for LoRA fine-tuning

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

EVAL_ARGS=""
if [ -n "$EVAL_DATA" ]; then
    EVAL_ARGS="--eval_data_path $EVAL_DATA --do_eval --evaluation_strategy steps --eval_steps 500"
fi

torchrun $DISTRIBUTED_ARGS train/finetune_av.py \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA" \
    $EVAL_ARGS \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision $TUNE_VISION \
    --tune_llm $TUNE_LLM \
    --tune_audio $TUNE_AUDIO \
    --use_lora $USE_LORA \
    --model_max_length $MODEL_MAX_LENGTH \
    --max_slice_nums $MAX_SLICE_NUMS \
    --max_num_frames $MAX_NUM_FRAMES \
    --num_train_epochs $EPOCHS \
    --max_steps $MAX_STEPS \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR" \
    --logging_strategy "steps" \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --deepspeed train/zero2.json \
    --report_to "none" \
    --compression_enabled true \
    --compression_topk_ratio 1 \
    --compression_topk_ratio_end 0.1 \
    --tune_compression true \
    --compression_debug_log false \
    --compression_lr $COMPRESSION_LR
