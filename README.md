# AVOC

This repository contains the AVOC checkpoint, training code, and inference scripts built on top of [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5). All commands below assume that you run them from the repository root.


Download the checkpoint from https://huggingface.co/mxxxxxxxxxxxxxxxxx/AVOC and place it at:
```text
./ckpt/checkpoint-AVOC
```


## Environment Setup

```bash
conda create -n avoc python=3.10 -y
conda activate avoc
pip install -r requirement.txt
```


## Checkpoints

For inference with the released model, use:

```text
./ckpt/checkpoint-AVOC
```

For training from the base model, place the MiniCPM-o base checkpoint at:

```text
./ckpt/minicpm-o-4.5
```


## Training Data Format

Example:
```json
[
  {
    "video": "./data/videos/example.mp4",
    "use_audio": true,
    "conversations": [
      {
        "role": "user",
        "content": "Describe the video."
      },
      {
        "role": "assistant",
        "content": "The video shows a person cooking in a kitchen."
      }
    ]
  }
]
```

## Inference Data Format

Example:
```json
[
  {
    "id": "video_001_0000",
    "video": "./data/LVOmniBench/videos/video_001.mp4",
    "audio": "./data/LVOmniBench/audios/video_001.wav",
    "question": "What is the person doing?\nA. Running\nB. Cooking\nC. Sleeping\nD. Driving",
    "conversations": [
      {
        "from": "human",
        "value": "What is the person doing?\nA. Running\nB. Cooking\nC. Sleeping\nD. Driving"
      },
      {
        "from": "gpt",
        "value": "B"
      }
    ]
  }
]
```

## Training

### Stage 1

Stage 1 fine-tunes from the MiniCPM-o base checkpoint without token compression:

```bash
bash train/run_train_stage1.sh
```

Before running, edit these variables in `train/run_train_stage1.sh` if needed:

```bash
MODEL="./ckpt/minicpm-o-4.5"
DATA="./data/mix_avsd_how2_finevideo_chronus_longvila.json"
OUTPUT_DIR="./ckpt/stage1"
```

### Stage 2

Stage 2 starts from the Stage-1 checkpoint and trains with token compression enabled:

```bash
bash train/run_train_stage2.sh
```

Before running, edit these variables in `train/run_train_stage2.sh` if needed:

```bash
MODEL="./ckpt/stage1"
DATA="./data/finevideo.json"
OUTPUT_DIR="./ckpt/stage2"
```

Important Stage-2 settings:

```bash
--compression_enabled true
--compression_topk_ratio 1
--compression_topk_ratio_end 0.1
--tune_compression true
--compression_lr 5e-5
```

`compression_topk_ratio_end` enables training-time random ratio sampling between the start and end ratios. Set it to the same value as `compression_topk_ratio`, or remove it, if you do not want this behavior.

## Inference

### Evaluation

Run:

```bash
mkdir -p ./results
python3 infer/eval.py \
  --model-path ./ckpt/checkpoint-AVOC \
  --base-model-path ./ckpt/checkpoint-AVOC \
  --data-path ./data/lvomnibench_formatted_question.json \
  --output-path ./results/lvomnibench.json \
  --compression \
  --compression-topk 10240 \
  --compression-video-audio-token-ratio 2 \
  --compression-diversity-lambda 0.15 \
  --compression-mmr-window 3
```
Or refer to the evaluation script `infer/eval_lvomnibench_8gpu.sh`.
