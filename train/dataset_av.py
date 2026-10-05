import copy
import json
import logging
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AudioVideoDataset(Dataset):
    """
    Dataset for audio/video/image → text supervised fine-tuning.

    Each sample in the JSON list should be a dict with optional media fields
    and a required conversations list:

        {
            "video": "/path/to/video.mp4",       # optional
            "audio": "/path/to/audio.wav",        # optional
            "image": "/path/to/image.jpg",        # optional (str or list[str])
            "use_audio": true,                    # use video audio track (default true)
            "conversations": [
                {"role": "user", "content": "describe the video"},
                {"role": "assistant", "content": "The video shows …"}
            ]
        }
    """

    def __init__(
        self,
        raw_data: list,
        processor,
        max_length: int = 2048,
        max_slice_nums: int = 1,
        use_image_id: bool = False,
        max_num_frames: int = 32,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.max_slice_nums = max_slice_nums
        self.use_image_id = use_image_id
        self.max_num_frames = max_num_frames

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        try:
            result = self._process_item(idx)
            result["sample_idx"] = idx
            return result
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    # ------------------------------------------------------------------
    # core processing
    # ------------------------------------------------------------------
    def _process_item(self, idx):
        item = self.raw_data[idx]

        images: List[Image.Image] = []
        audios: List[np.ndarray] = []
        audio_parts: List[int] = []
        content_parts: List[str] = []

        # --- video ---
        if "video" in item:
            os.environ["MAX_NUM_FRAMES"] = str(self.max_num_frames)
            from minicpmo.utils import get_video_frame_audio_segments

            use_audio = item.get("use_audio", True)
            video_frames, audio_segments, _ = get_video_frame_audio_segments(
                item["video"], use_audio=use_audio
            )
            for i, frame in enumerate(video_frames):
                images.append(frame)
                content_parts.append("<image>./</image>")
                if use_audio and audio_segments and i < len(audio_segments):
                    audios.append(audio_segments[i])
                    audio_parts.append(0)
                    # audio_parts.append(len(audio_parts))
                    content_parts.append("<audio>./</audio>")

        # --- standalone audio ---
        if "audio" in item:
            import librosa
            audio_data, _ = librosa.load(item["audio"], sr=16000)
            audios.append(audio_data)
            audio_parts.append(len(audio_parts))
            content_parts.append("<audio>./</audio>")

        # --- image(s) ---
        if "image" in item:
            img_field = item["image"]
            paths = [img_field] if isinstance(img_field, str) else img_field
            for p in paths:
                images.append(Image.open(p).convert("RGB"))
                content_parts.append("<image>./</image>")

        # --- build conversation messages with placeholders ---
        conversations = copy.deepcopy(item["conversations"])
        assert len(conversations) >= 2, "Need at least one user-assistant pair"
        assert conversations[0]["role"] == "user"

        user_question_text = conversations[0]["content"]

        if content_parts:
            conversations[0]["content"] = (
                "".join(content_parts) + conversations[0]["content"]
            )
        # from fpdb import ForkedPdb; ForkedPdb().set_trace()
        # use_tts_template = len(audios) > 0 #没用 加不上<|tts_bos|>
        prompt_text = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
            # use_tts_template=use_tts_template,#没用 加不上<|tts_bos|>
        )
        
        # --- call processor (batch_size=1) ---
        inputs = self.processor(
            [prompt_text],
            [images] if images else None,
            [audios] if audios else None,
            [audio_parts] if audios else None,
            max_slice_nums=self.max_slice_nums,
            use_image_id=self.use_image_id,
            return_tensors="pt",
            max_length=self.max_length,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length] 
        
        labels = self._build_labels(input_ids)
        position_ids = torch.arange(input_ids.size(0), dtype=torch.long)
        # from fpdb import ForkedPdb; ForkedPdb().set_trace()
        text_query_ids = self.tokenizer.encode(user_question_text, add_special_tokens=False)
        text_query = torch.tensor(text_query_ids, dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "text_query": text_query,
            "pixel_values": inputs.get("pixel_values", [[]])[0],
            "tgt_sizes": inputs.get("tgt_sizes", [[]])[0],
            "image_bound": (
                inputs["image_bound"][0]
                if inputs.get("image_bound") and len(inputs["image_bound"]) > 0
                else torch.zeros(0, 2, dtype=torch.long)
            ),
            "audio_bounds": (
                inputs["audio_bounds"][0]
                if inputs.get("audio_bounds") and len(inputs["audio_bounds"]) > 0
                else torch.zeros(0, 2, dtype=torch.long)
            ),
        }

        af = inputs.get("audio_features")
        al = inputs.get("audio_feature_lens")
        if af is not None and isinstance(af, torch.Tensor) and af.numel() > 0:
            result["audio_features"] = af
            result["audio_feature_lens"] = al[0] if al else []
        else:
            result["audio_features"] = []
            result["audio_feature_lens"] = []

        return result

    # ------------------------------------------------------------------
    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shifted labels: only predict assistant response tokens."""
        labels = torch.full_like(input_ids, -100)

        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        eos_id = self.tokenizer.eos_token_id

        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        im_end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0].tolist()

        for start_pos in im_start_positions:
            next_pos = start_pos + 1
            if next_pos >= len(input_ids):
                continue
            if input_ids[next_pos].item() != assistant_token_id:
                continue
            st = next_pos + 2
            for end_pos in im_end_positions:
                if end_pos > st:
                    labels[st - 1 : end_pos] = input_ids[st : end_pos + 1].clone()
                    labels[end_pos] = eos_id
                    break

        if torch.all(labels == -100):
            raise ValueError("No assistant response tokens found – cannot compute loss")

        return labels


# ======================================================================
# data collator
# ======================================================================
def av_data_collator(examples, padding_value=0, max_length=2048):
    """Collate samples from AudioVideoDataset for the trainer."""

    def _trim_pad(seqs, pad_val):
        return pad_sequence(
            [s[:max_length] for s in seqs], batch_first=True, padding_value=pad_val
        )

    input_ids = _trim_pad([ex["input_ids"] for ex in examples], padding_value)
    position_ids = _trim_pad([ex["position_ids"] for ex in examples], padding_value)
    labels = _trim_pad([ex["labels"] for ex in examples], -100)
    attention_mask = _trim_pad(
        [ex["attention_mask"] for ex in examples], False
    )

    pixel_values = [ex["pixel_values"] for ex in examples]
    tgt_sizes = [ex["tgt_sizes"] for ex in examples]
    image_bound = [ex["image_bound"] for ex in examples]
    audio_bounds = [ex["audio_bounds"] for ex in examples]
    text_query = [ex["text_query"] for ex in examples]

    # audio features: pad across items then stack
    all_feats = []
    all_lens = []
    for ex in examples:
        af = ex["audio_features"]
        al = ex["audio_feature_lens"]
        if isinstance(af, torch.Tensor) and af.numel() > 0:
            all_feats.append(af)
            all_lens.append(al)
        else:
            all_lens.append([])

    if all_feats:
        max_frames = max(f.shape[-1] for f in all_feats)
        padded = []
        for f in all_feats:
            if f.shape[-1] < max_frames:
                pad = torch.zeros(
                    f.shape[0], f.shape[1], max_frames - f.shape[-1],
                    dtype=f.dtype,
                )
                f = torch.cat([f, pad], dim=-1)
            padded.append(f)
        audio_features = torch.cat(padded, dim=0)
    else:
        audio_features = []

    sample_indices = [ex.get("sample_idx", -1) for ex in examples]

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "text_query": text_query,
        "pixel_values": pixel_values,
        "tgt_sizes": tgt_sizes,
        "image_bound": image_bound,
        "audio_features": audio_features,
        "audio_feature_lens": all_lens,
        "audio_bounds": audio_bounds,
        "sample_indices": sample_indices,
    }
