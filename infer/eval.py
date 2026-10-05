import os
os.environ.setdefault("MAX_NUM_FRAMES", "320")
# Before first CUDA alloc; reduces fragmentation (see PyTorch CUDA memory notes).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def extract_option(text):
    text = text.strip()
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    return text.strip()[:1].upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="./ckpt/checkpoint-AVOC",
                        help="Checkpoint directory (weights).")
    parser.add_argument("--base-model-path", type=str, default=None,
                        help="Base model dir for dynamic modeling code and tokenizer/processor. "
                             "Defaults to --model-path when omitted (same as infer_my when base == ckpt).")
    parser.add_argument("--data-path", type=str,
                        default="./data/lvomnibench_formatted_question.json")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--no-audio", dest="use_audio", action="store_false", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--compression", dest="compression_enabled",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Enable token compression. Omit to use checkpoint config.")
    parser.add_argument("--compression-topk", type=int, default=None,
                        help="Fixed number of VA tokens to keep. Overrides --compression-topk-ratio if set.")
    parser.add_argument("--compression-topk-ratio", type=float, default=None,
                        help="Fraction of VA tokens to keep (0,1]. Used when --compression-topk is not set. "
                             "If both omitted, use checkpoint defaults.")
    parser.add_argument("--compression-video-audio-token-ratio", type=float, default=None,
                        help="Video:audio token ratio. E.g. 2.0 means 2/3 video, 1/3 audio in topk. "
                             "None = fall back to balance-bias heuristic.")
    parser.add_argument("--compression-diversity-lambda", type=float, default=None,
                        help="MMR diversity weight for inference. 0=off, 0.3=recommended. "
                             "Only affects inference-time token selection.")
    parser.add_argument("--compression-mmr-window", type=int, default=None,
                        help="Temporal block window for MMR. Only tokens within ±window blocks "
                             "penalise each other. Default 3.")
    parser.add_argument("--system-prompt", type=str, default="\nAnswer with the option's letter from the given choices directly.", 
                        help="System prompt appended after the question. "
                             "Excluded from compression text query (only the question is used).")
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx or 0
        end = args.end_idx or len(data)
        data = data[start:end]

    if args.output_path is None:
        suffix = "with_audio" if args.use_audio else "no_audio"
        if args.start_idx is not None or args.end_idx is not None:
            suffix += f"_{start}_{end}"
        args.output_path = os.path.join(
            os.path.dirname(args.data_path),
            f"omnivideobench_results_{suffix}.json"
        )

    existing_results = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            existing = json.load(f)
        existing_results = {r["id"]: r for r in existing}
        print(f"Loaded {len(existing_results)} existing results, resuming...")

    base_path = args.base_model_path or args.model_path
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if args.compression_enabled is not None:
        config.compression_enabled = args.compression_enabled
    if args.compression_topk is not None:
        config.compression_topk = args.compression_topk
    if args.compression_topk_ratio is not None:
        config.compression_topk_ratio = args.compression_topk_ratio
        if args.compression_topk is None:
            config.compression_topk = None
    if args.compression_video_audio_token_ratio is not None:
        config.compression_video_audio_token_ratio = args.compression_video_audio_token_ratio
    if args.compression_diversity_lambda is not None:
        config.compression_diversity_lambda = args.compression_diversity_lambda
    if args.compression_mmr_window is not None:
        config.compression_mmr_window = args.compression_mmr_window

    _has_comp = getattr(config, 'compression_enabled', False)
    if _has_comp:
        _topk_desc = (f"topk={config.compression_topk}"
                      if getattr(config, 'compression_topk', None) is not None
                      else f"topk_ratio={getattr(config, 'compression_topk_ratio', 0.5)}")
        _va_ratio = getattr(config, 'compression_video_audio_token_ratio', None)
        _ratio_desc = f", va_token_ratio={_va_ratio}" if _va_ratio is not None else ""
        _div_lam = getattr(config, 'compression_diversity_lambda', 0.0)
        _mmr_w = getattr(config, 'compression_mmr_window', 3)
        _div_desc = f", diversity_lambda={_div_lam}, mmr_window={_mmr_w}" if _div_lam > 0 else ""
        print(f"[compression] enabled, {_topk_desc}{_ratio_desc}{_div_desc}")
    else:
        print("[compression] disabled")

    config.init_vision = True
    config.init_audio = args.use_audio
    config.init_tts = False
    config._name_or_path = base_path

    MiniCPMO = get_class_from_dynamic_module(
        "modeling_minicpmo.MiniCPMO",
        base_path,
    )
    model = MiniCPMO.from_pretrained(
        args.model_path,
        config=config,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.config._name_or_path = base_path
    model.eval().cuda()

    results = list(existing_results.values())

    for item in tqdm(data, desc="Evaluating"):
        if item["id"] in existing_results:
            continue

        video_path = item["video"]
        question = item["question"]
        gt_answer = item["conversations"][1]["value"].strip()

        prompt_text = question + "\n" + args.system_prompt if args.system_prompt else question
        content = [
            {"type": "video_url", "video_url": {"url": video_path, "use_audio": args.use_audio}},
            {"type": "text", "text": prompt_text},
        ]
        msgs = [{"role": "user", "content": content}]

        chat_kwargs = dict(
            msgs=msgs,
            max_new_tokens=args.max_new_tokens,
            max_inp_length=40000,
            max_slice_nums=1,
            use_image_id=False,
            use_tts_template=False,
            generate_audio=False,
            enable_thinking=False,
            do_sample=False,
        )
        if args.system_prompt:
            chat_kwargs["text_query_override"] = question

        try:
            pred_raw = model.chat(**chat_kwargs)
        except Exception as e:
            print(f"[ERROR] {item['id']}: {e}")
            pred_raw = ""


        pred_option = extract_option(pred_raw)

        result = {
            "id": item["id"],
            "video": video_path,
            "question": question,
            "gt_answer": gt_answer,
            "pred_raw": pred_raw,
            "pred_option": pred_option,
        }
        results.append(result)
        print(result)

        if len(results) % 10 == 0:
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{len(results)}/{len(data)}] saved")

        torch.cuda.empty_cache()

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(results)} results saved to {args.output_path}")


if __name__ == "__main__":
    main()
