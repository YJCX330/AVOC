"""
UnifiedCompression: token compression.

When compression is enabled, tokens are selected with a weighted sum of
text-guided relevance and video/audio temporal-block importance score:

    score = alpha * text_score + (1 - alpha) * va_block_score

The module uses SubsetGumbelSampler for differentiable top-k during training.
"""
from typing import Optional, List, Tuple

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubsetGumbelSampler(nn.Module):
    """Differentiable top-k selection via Gumbel-Softmax straight-through estimator."""

    EPSILON = torch.finfo(torch.float32).tiny

    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def __call__(
        self, scores: torch.Tensor, hard: bool = True, temperature: float = 1.0, k: int = None
    ) -> torch.Tensor:
        k = k if k is not None else self.k
        scores = scores.float()

        seed = int.from_bytes(os.urandom(4), "big") & 0xFFFFFFFF
        generator = torch.Generator(device=scores.device).manual_seed(seed)
        u = torch.rand(scores.shape, device=scores.device, dtype=scores.dtype, generator=generator)
        u = u.clamp(1e-10, 1 - 1e-10)
        g = -torch.log(-torch.log(u))
        # print("g: ", g)

        scores = scores + g
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        eps = torch.tensor([self.EPSILON], device=scores.device, dtype=scores.dtype)
        for i in range(k):
            khot_mask = torch.max(1.0 - onehot_approx, eps)
            scores = scores + torch.log(khot_mask)
            onehot_approx = F.softmax(scores / temperature, dim=-1)
            khot = khot + onehot_approx

        if hard:
            ids = torch.topk(khot, k, dim=-1).indices
            khot_hard = torch.zeros_like(khot).scatter_(-1, ids, 1)
            ret = khot_hard - khot.detach() + khot
        else:
            ret = khot
        return ret


# ----------------------------------------------------------------------
# MMR step body — pure-tensor, no host sync.
# ----------------------------------------------------------------------

def _mmr_step_full(scores, max_sim, candidates, S_row_table, lam, step_is_first):
    """One greedy step using a precomputed (N, N) similarity table.

    In-place on `max_sim` and `candidates` to avoid per-step N-sized allocations.
    """
    if step_is_first:
        mmr = scores.clone()
    else:
        mmr = (1.0 - lam) * scores - lam * max_sim
    mmr.masked_fill_(~candidates, float('-inf'))
    best = mmr.argmax()
    sim_row = S_row_table.index_select(0, best.unsqueeze(0)).squeeze(0)
    torch.maximum(max_sim, sim_row, out=max_sim)
    candidates[best] = False
    return best, max_sim, candidates


def _mmr_step_sparse(scores, max_sim, candidates, S_sparse, neigh_idx, lam, step_is_first):
    """Greedy step using a (N, M) padded sparse similarity table.

    S_sparse[i, k] = sim(i, neigh_idx[i, k]); padded slots: idx=0, sim=-inf.
    In-place on `max_sim` and `candidates`.
    """
    if step_is_first:
        mmr = scores.clone()
    else:
        mmr = (1.0 - lam) * scores - lam * max_sim
    mmr.masked_fill_(~candidates, float('-inf'))
    best = mmr.argmax()
    sim_pad = S_sparse.index_select(0, best.unsqueeze(0)).squeeze(0)        # (M,)
    idx_pad = neigh_idx.index_select(0, best.unsqueeze(0)).squeeze(0).long()  # (M,)
    # amax: padded (-inf, idx=0) entries are no-ops on position 0.
    max_sim.scatter_reduce_(0, idx_pad, sim_pad.to(max_sim.dtype),
                            reduce='amax', include_self=True)
    candidates[best] = False
    return best, max_sim, candidates



class UnifiedCompression(nn.Module):
    """
    Unified video/audio token compression.

    Projection weights:
      tg_query_proj / tg_key_proj  - text-guided attention.
      va_query_proj / va_key_proj  - VA-block attention.
    """

    def __init__(
        self,
        hidden_size: int,
        topk_ratio: float,
        topk: Optional[int] = None,
        text_guided_weight: float = 0.5,
        gumbel_temperature: float = 1.0,
        video_audio_token_ratio: Optional[float] = None,
        diversity_lambda: float = 0.0,
        mmr_window: int = 3,
        debug_log: bool = False,
        debug_log_rank: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.topk = topk
        self.topk_ratio = topk_ratio
        self.text_guided_weight = text_guided_weight
        self.gumbel_temperature = gumbel_temperature
        self.video_audio_token_ratio = video_audio_token_ratio
        self.diversity_lambda = diversity_lambda
        self.mmr_window = mmr_window
        self.sampler = SubsetGumbelSampler()
        self.debug_log = debug_log
        self.debug_log_rank = debug_log_rank
        self._debug_step = 0

        self.tg_query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tg_key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.va_query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.va_key_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _text_scores(self, text_embeds, va_embeds):
        """Mean attention from all text tokens to each VA token."""
        Q = self.tg_query_proj(text_embeds)
        K = self.tg_key_proj(va_embeds)
        attn = torch.matmul(Q, K.t()) / math.sqrt(self.hidden_size)
        return attn.mean(dim=0)

    def _va_scores(self, va_embeds, block_sizes):
        """Per-token importance via bidirectional cross-attention inside each block."""
        N = va_embeds.shape[0]
        scores = torch.zeros(N, device=va_embeds.device, dtype=va_embeds.dtype)
        offset = 0
        for frame_size, audio_size in block_sizes:
            frame_end = offset + frame_size
            audio_end = frame_end + audio_size
            if frame_size > 0 and audio_size > 0:
                frame_tok = va_embeds[offset:frame_end]
                audio_tok = va_embeds[frame_end:audio_end]

                Q_v = self.va_query_proj(frame_tok)
                K_a = self.va_key_proj(audio_tok)
                attn_va = torch.matmul(Q_v, K_a.t()) / math.sqrt(self.hidden_size)
                scores[frame_end:audio_end] = attn_va.mean(dim=0)

                Q_a = self.va_query_proj(audio_tok)
                K_v = self.va_key_proj(frame_tok)
                attn_av = torch.matmul(Q_a, K_v.t()) / math.sqrt(self.hidden_size)
                scores[offset:frame_end] = attn_av.mean(dim=0)

            offset = audio_end
        return scores

    @staticmethod
    def _build_modality_indices(block_sizes):
        """Return (video_idx, audio_idx) flat index lists from block_sizes."""
        video_idx = []
        audio_idx = []
        offset = 0
        for frame_size, audio_size in block_sizes:
            if frame_size > 0:
                video_idx.extend(range(offset, offset + frame_size))
                offset += frame_size
            if audio_size > 0:
                audio_idx.extend(range(offset, offset + audio_size))
                offset += audio_size
        return video_idx, audio_idx

    def _zscore_normalize_per_modality(self, scores, block_sizes):
        """Z-score normalize scores independently for each modality."""
        video_idx, audio_idx = self._build_modality_indices(block_sizes)
        eps = 1e-8
        scores = scores.clone()
        for idx_list in (video_idx, audio_idx):
            if len(idx_list) < 2:
                continue
            idx = torch.tensor(idx_list, device=scores.device)
            vals = scores[idx]
            scores[idx] = (vals - vals.mean()) / (vals.std() + eps)
        return scores

    def _select_topk_by_modality(self, va_embeds, scores, block_sizes, ratio, return_indices=False):
        """Per-modality top-k selection with explicit video:audio token ratio."""
        orig_dtype = va_embeds.dtype
        N = va_embeds.shape[0]
        topk = min(self.topk, N) if self.topk is not None else max(1, int(self.topk_ratio * N))

        video_idx, audio_idx = self._build_modality_indices(block_sizes)
        n_v, n_a = len(video_idx), len(audio_idx)

        k_video = int(round(topk * ratio / (ratio + 1)))
        k_audio = topk - k_video
        k_video = min(k_video, n_v)
        k_audio = min(k_audio, n_a)
        if k_video + k_audio < topk:
            k_video = min(topk - k_audio, n_v)
        if k_video + k_audio < topk:
            k_audio = min(topk - k_video, n_a)

        print(f"[modality split] ratio={ratio}, topk={topk}, k_video={k_video}, k_audio={k_audio}")

        v_idx_t = torch.tensor(video_idx, device=scores.device, dtype=torch.long)
        a_idx_t = torch.tensor(audio_idx, device=scores.device, dtype=torch.long)

        v_scores = scores[v_idx_t]
        a_scores = scores[a_idx_t]

        if self.diversity_lambda > 0 and block_sizes is not None:
            v_block_sizes = [(f, 0) for f, _ in block_sizes if f > 0]
            a_block_sizes = [(a, 0) for _, a in block_sizes if a > 0]
            top_v_local = self._mmr_select(
                va_embeds[v_idx_t], v_scores, k_video,
                lam=self.diversity_lambda,
                block_sizes=v_block_sizes if v_block_sizes else None,
                window=self.mmr_window,
            )
            top_a_local = self._mmr_select(
                va_embeds[a_idx_t], a_scores, k_audio,
                lam=self.diversity_lambda,
                block_sizes=a_block_sizes if a_block_sizes else None,
                window=self.mmr_window,
            )
        else:
            top_v_local = torch.topk(v_scores, k_video, dim=-1).indices
            top_a_local = torch.topk(a_scores, k_audio, dim=-1).indices

        selected_v = v_idx_t[top_v_local]
        selected_a = a_idx_t[top_a_local]
        selected = torch.cat([selected_v, selected_a]).sort().values

        compressed = va_embeds[selected].to(orig_dtype)

        if return_indices:
            return compressed, selected
        return compressed

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _local_rank(self) -> int:
        try:
            import torch.distributed as dist
            return dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            return 0

    def _should_log(self) -> bool:
        return self.debug_log and self.training and (self._local_rank() == self.debug_log_rank)

    # ------------------------------------------------------------------
    # MMR diversity-aware selection (inference only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _mmr_select(self, va_embeds, scores, topk, lam, block_sizes=None, window=3):
        """Temporal-windowed Maximal Marginal Relevance greedy selection.
        """
        device = va_embeds.device
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        N = scores.shape[0]
        topk = min(topk, N)

        # Centered cosine embeddings.
        centered = va_embeds - va_embeds.mean(dim=0, keepdim=True)
        normed = F.normalize(centered, dim=-1)

        # ----- vectorized block_id / is_video -------------------------
        block_id = None
        is_video = None
        if block_sizes is not None and window is not None and len(block_sizes) > 0:
            block_lens_t = torch.tensor(
                [f + a for f, a in block_sizes], device=device, dtype=torch.long
            )
            video_lens_t = torch.tensor(
                [f for f, _ in block_sizes], device=device, dtype=torch.long
            )
            audio_lens_t = torch.tensor(
                [a for _, a in block_sizes], device=device, dtype=torch.long
            )
            B = block_lens_t.numel()
            block_id = torch.repeat_interleave(
                torch.arange(B, device=device, dtype=torch.long), block_lens_t
            )
            mod_pattern = torch.tensor(
                [True, False] * B, device=device, dtype=torch.bool
            )
            mod_lens = torch.empty(2 * B, device=device, dtype=torch.long)
            mod_lens[0::2] = video_lens_t
            mod_lens[1::2] = audio_lens_t
            is_video = torch.repeat_interleave(mod_pattern, mod_lens)
            assert is_video.numel() == N, (is_video.numel(), N, block_sizes)

        # ----- similarity table ---------------------------------------
        # If block_sizes + window are given, build a padded (N, M) sparse
        # table restricted to same-modality neighbors within ±window blocks.
        # Otherwise fall back to dense (N, N).
        use_sparse = (
            block_sizes is not None
            and window is not None
            and len(block_sizes) > 0
        )
        if use_sparse:
            B_blocks = len(block_sizes)
            video_lens_cpu = [f for f, _ in block_sizes]
            audio_lens_cpu = [a for _, a in block_sizes]
            block_lens_cpu = [f + a for f, a in block_sizes]
            block_starts_cpu = [0]
            for bl_ in block_lens_cpu:
                block_starts_cpu.append(block_starts_cpu[-1] + bl_)
            # Neighbor index lists per block (python-side; B << N).
            v_neigh_per_block = []
            a_neigh_per_block = []
            M_max = 0
            for b in range(B_blocks):
                bl = max(0, b - window)
                br = min(B_blocks - 1, b + window)
                v_list, a_list = [], []
                for k in range(bl, br + 1):
                    ks = block_starts_cpu[k]
                    fk = video_lens_cpu[k]
                    ak = audio_lens_cpu[k]
                    if fk > 0:
                        v_list.append(torch.arange(ks, ks + fk, device=device, dtype=torch.long))
                    if ak > 0:
                        a_list.append(torch.arange(ks + fk, ks + fk + ak, device=device, dtype=torch.long))
                v_neigh = torch.cat(v_list) if v_list else torch.empty(0, device=device, dtype=torch.long)
                a_neigh = torch.cat(a_list) if a_list else torch.empty(0, device=device, dtype=torch.long)
                v_neigh_per_block.append(v_neigh)
                a_neigh_per_block.append(a_neigh)
                M_max = max(M_max, v_neigh.numel(), a_neigh.numel())                      
            # Padded sparse table. Padding: idx=0 (any valid), sim=-inf.
            neigh_idx = torch.zeros(N, M_max, dtype=torch.int64, device=device)
            S_sparse = torch.full((N, M_max), float('-inf'),
                                  dtype=normed.dtype, device=device)

            for b in range(B_blocks):
                fb = video_lens_cpu[b]
                ab = audio_lens_cpu[b]
                bs = block_starts_cpu[b]
                v_neigh = v_neigh_per_block[b]
                a_neigh = a_neigh_per_block[b]

                if fb > 0 and v_neigh.numel() > 0:
                    V_b = normed[bs: bs + fb]                     # (fb, D)
                    S_v = V_b @ normed.index_select(0, v_neigh).t()  # (fb, Mv)
                    Mv = v_neigh.numel()
                    neigh_idx[bs: bs + fb, :Mv] = v_neigh.unsqueeze(0)
                    S_sparse[bs: bs + fb, :Mv] = S_v

                if ab > 0 and a_neigh.numel() > 0:
                    A_b = normed[bs + fb: bs + fb + ab]           # (ab, D)
                    S_a = A_b @ normed.index_select(0, a_neigh).t()  # (ab, Ma)
                    Ma = a_neigh.numel()
                    neigh_idx[bs + fb: bs + fb + ab, :Ma] = a_neigh.unsqueeze(0)
                    S_sparse[bs + fb: bs + fb + ab, :Ma] = S_a
            S_full = None  # marker: use sparse path below
        else:
            # Dense fallback (no windowing): keep original (N, N) path.
            S_full = normed @ normed.t()
            if is_video is not None:
                same_mod_mat = is_video.unsqueeze(0) == is_video.unsqueeze(1)
                S_full.masked_fill_(~same_mod_mat, float('-inf'))
                del same_mod_mat
            neigh_idx = None
            S_sparse = None

        # ----- preallocated selected --------------
        selected = torch.empty(topk, dtype=torch.long, device=device)
        max_sim = torch.full((N,), -1.0, device=device, dtype=scores.dtype)
        candidates = torch.ones(N, device=device, dtype=torch.bool)

        for step in range(topk):
            first = (step == 0)
            if use_sparse:
                best, max_sim, candidates = _mmr_step_sparse(
                    scores, max_sim, candidates, S_sparse, neigh_idx, lam, first
                )
            else:
                best, max_sim, candidates = _mmr_step_full(
                    scores, max_sim, candidates, S_full, lam, first
                )
            selected[step] = best

        return selected.sort().values

    # ------------------------------------------------------------------
    # Shared top-k selection
    # ------------------------------------------------------------------

    def _select_topk(self, va_embeds, raw_scores, block_sizes=None, return_indices=False):
        orig_dtype = va_embeds.dtype
        N = va_embeds.shape[0]
        if self.topk is not None:
            topk = min(self.topk, N)
        else:
            topk = max(1, int(self.topk_ratio * N))

        logits = F.log_softmax(raw_scores.float().unsqueeze(0), dim=-1)  # (1, N)
        do_log = self._should_log()
        print("topk: ", topk, " topk_ratio: ", self.topk_ratio)
        # print("logits: ", logits)
        # print(logits.mean(), logits.std(), logits.min(), logits.max())

        if self.training:
            khot = self.sampler(
                logits, hard=True, temperature=self.gumbel_temperature, k=topk
            )
            top_k_indices = torch.topk(khot, topk, dim=-1).indices
            top_k_indices = top_k_indices.sort(dim=-1).values
        else:
            khot = None
            if self.diversity_lambda > 0 and block_sizes is not None:
                top_k_indices = self._mmr_select(
                    va_embeds, logits.squeeze(0), topk,
                    lam=self.diversity_lambda,
                    block_sizes=block_sizes,
                    window=self.mmr_window,
                ).unsqueeze(0)
            else:
                top_k_indices = torch.topk(logits, topk, dim=-1).indices
                top_k_indices = top_k_indices.sort(dim=-1).values

        if block_sizes:
            bounds = []
            offset = 0
            for frame_size, audio_size in block_sizes:
                if frame_size > 0:
                    bounds.append((offset, offset + frame_size, "video"))
                    offset += frame_size
                if audio_size > 0:
                    bounds.append((offset, offset + audio_size, "audio"))
                    offset += audio_size
            score_cpu = logits.detach().float().reshape(-1).cpu()
            video_scores = []
            audio_scores = []
            for start, end, modality in bounds:
                if modality == "video":
                    video_scores.extend(score_cpu[start:end].tolist())
                else:
                    audio_scores.extend(score_cpu[start:end].tolist())
            if video_scores and audio_scores:
                print(
                    f"  video_score_mean={sum(video_scores)/len(video_scores):+.6f}  "
                    f"audio_score_mean={sum(audio_scores)/len(audio_scores):+.6f}"
                )
            else:
                print("  video_score_mean=NA  audio_score_mean=NA")
        else:
            print("  video_score_mean=NA  audio_score_mean=NA (block_sizes unavailable)")

        if do_log:
            self._debug_step += 1
            step = self._debug_step
            selected_ids = sorted(top_k_indices.squeeze(0).tolist())
            selected_set = set(selected_ids)

            print(f"\n{'='*60}")
            _mode = f"fixed_k={self.topk}" if self.topk is not None else f"ratio={self.topk_ratio}"
            print(f"[Compression debug] step={step}  N={N}  topk={topk} ({_mode})")
            print(f"  Selected token indices ({len(selected_ids)}): {selected_ids}")

            def _make_score_grad_hook(step_, sel_set_, N_, scores_ref):
                def _hook(grad: torch.Tensor):
                    g = grad.detach().cpu().float()
                    print(f"\n[Compression debug] step={step_}  d_loss/d_raw_scores ({N_} tokens):")
                    print(f"  {'idx':>6}  {'status':>8}  {'score_grad':>14}  {'score_val':>14}")
                    scores_cpu = scores_ref.detach().cpu().float()
                    for i in range(N_):
                        tag = "[SEL]" if i in sel_set_ else "     "
                        print(f"  {i:6d}  {tag:>8}  {g[i].item():+14.6f}  {scores_cpu[i].item():+14.6f}")
                return _hook

            def _make_compressed_grad_hook(step_, sel_ids_):
                def _hook(grad: torch.Tensor):
                    g = grad.detach().cpu().float()
                    grad_norms = g.norm(dim=-1)
                    print(f"\n[Compression debug] step={step_}  d_loss/d_compressed[i] L2-norm ({len(sel_ids_)} selected tokens):")
                    print(f"  {'rank':>6}  {'token_idx':>10}  {'grad_norm':>14}")
                    for rank_i, (tok_id, gn) in enumerate(zip(sel_ids_, grad_norms.tolist())):
                        print(f"  {rank_i:6d}  {tok_id:10d}  {gn:14.6f}")
                return _hook

            raw_scores.register_hook(_make_score_grad_hook(step, selected_set, N, raw_scores))

        D = va_embeds.shape[-1]
        va_unsq = va_embeds.unsqueeze(0)
        idx_exp = top_k_indices.unsqueeze(-1).expand(-1, -1, D)
        compressed = torch.gather(va_unsq, 1, idx_exp).squeeze(0)

        if khot is not None:
            selected_khot = torch.gather(khot, 1, top_k_indices).squeeze(0).unsqueeze(-1)
            compressed = compressed * selected_khot

        if do_log and compressed.requires_grad:
            compressed.register_hook(
                _make_compressed_grad_hook(self._debug_step, selected_ids)
            )
        compressed = compressed.to(orig_dtype)
        if return_indices:
            return compressed, top_k_indices.squeeze(0)
        return compressed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        video_audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor = None,
        block_sizes: list = None,
        return_indices: bool = False,
    ):
        """
        Args:
            video_audio_embeds : (num_va, hidden) flat multimodal token sequence.
            text_embeds        : (num_text, hidden) text tokens for relevance scoring.
            block_sizes        : list of (frame_size, audio_size) tuples — required
                                  for video/audio block importance scoring.
            return_indices     : if True, also return the selected token indices.
        Returns:
            compressed : (topk, hidden) selected tokens.
        """
        _empty_ret = lambda e: (e, torch.arange(e.shape[0], device=e.device)) if return_indices else e
        if video_audio_embeds.shape[0] == 0:
            return _empty_ret(video_audio_embeds)

        has_text = text_embeds is not None and text_embeds.shape[0] > 0
        has_both_modalities = (block_sizes is not None
                               and any(f > 0 for f, _ in block_sizes)
                               and any(a > 0 for _, a in block_sizes))

        if has_text and has_both_modalities:
            t = self._text_scores(text_embeds, video_audio_embeds)
            t = self._zscore_normalize_per_modality(t, block_sizes)
            v = self._va_scores(video_audio_embeds, block_sizes)
            v = self._zscore_normalize_per_modality(v, block_sizes)
            alpha = self.text_guided_weight
            scores = alpha * t + (1.0 - alpha) * v
        elif has_text:
            scores = self._text_scores(text_embeds, video_audio_embeds)
        else:
            return _empty_ret(video_audio_embeds)

        if not self.training and has_both_modalities and self.video_audio_token_ratio is not None:
            return self._select_topk_by_modality(
                video_audio_embeds, scores, block_sizes,
                ratio=self.video_audio_token_ratio,
                return_indices=return_indices,
            )
        return self._select_topk(video_audio_embeds, scores, block_sizes=block_sizes, return_indices=return_indices)


def build_compression_module(config):
    """Build a UnifiedCompression module from model config."""
    hidden_size = getattr(config, 'hidden_size', 4096)
    topk = getattr(config, 'compression_topk', None)
    topk_ratio = getattr(config, 'compression_topk_ratio', 0.5)
    text_guided_weight = getattr(config, 'compression_text_guided_weight', 0.5)
    gumbel_temperature = getattr(config, 'compression_gumbel_temperature', 1.0)
    video_audio_token_ratio = getattr(config, 'compression_video_audio_token_ratio', None)
    diversity_lambda = getattr(config, 'compression_diversity_lambda', 0.0)
    mmr_window = getattr(config, 'compression_mmr_window', 3)
    debug_log = getattr(config, 'compression_debug_log', False)
    return UnifiedCompression(
        hidden_size=hidden_size,
        topk_ratio=topk_ratio,
        topk=topk,
        text_guided_weight=text_guided_weight,
        gumbel_temperature=gumbel_temperature,
        video_audio_token_ratio=video_audio_token_ratio,
        diversity_lambda=diversity_lambda,
        mmr_window=mmr_window,
        debug_log=debug_log,
    )
