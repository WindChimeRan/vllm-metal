# SPDX-License-Identifier: Apache-2.0
"""Packed linear EAGLE3 using Metal paged attention.

Pair feature[t] with token[t+1], storing KV and applying RoPE at position t,
matching upstream EAGLE. The scheduler drops the last matching prefix block
before reuse because its draft KV can depend on the following token. Verified
rows are ingested again with actual target features, replacing recurrence state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import clear_context, prepare_unified
from vllm_metal.attention.impls.sdpa_wrapper import patch_sdpa_attention
from vllm_metal.v1.draft_model_proposer import (
    _DECODE_INGEST_MAX_TOKENS,
    DraftModelProposer,
)
from vllm_metal.v1.eagle3 import Eagle3Model
from vllm_metal.v1.proposer import ProposeContext


@dataclass
class _EaglePlan:
    req_id: str
    block_ids: list[int]
    start: int
    tokens: list[int]
    features: mx.array
    is_drafting: bool

    @property
    def end(self) -> int:
        return self.start + len(self.tokens)


class Eagle3Proposer(DraftModelProposer):
    """Reuse draft block ownership, with EAGLE-specific feature reconciliation."""

    _kv: MetalPagedKVCache

    @classmethod
    def build(
        cls,
        *,
        model: Eagle3Model,
        controller: Any,
        committed_num_blocks: int,
        scratch_reserve_blocks: int,
        block_size: int,
        dtype: mx.Dtype,
    ) -> Eagle3Proposer:
        proposer = cls(
            model=model,
            controller=controller,
            committed_num_blocks=committed_num_blocks,
            scratch_reserve_blocks=scratch_reserve_blocks,
            block_size=block_size,
            num_layers=1,
            extract_logits=lambda output: output[0],
        )
        attention = model.layers[0].self_attn
        proposer._kv = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=attention.n_kv_heads,
            head_dim=attention.head_dim,
            num_blocks=committed_num_blocks + scratch_reserve_blocks,
            block_size=block_size,
            dtype=dtype,
        )
        patch_sdpa_attention(model, proposer._kv, block_size)
        return proposer

    def needs_target_hidden_states(
        self, decode_segments: Any, *, has_final_prefill: bool
    ) -> bool:
        # EAGLE3 consumes auxiliary states, not the target's final hidden state.
        return False

    def _plan(
        self,
        *,
        req_id: str,
        blocks: list[int],
        target_start: int,
        tokens: list[int],
        features: mx.array,
        is_drafting: bool,
        k: int,
    ) -> _EaglePlan:
        # Shift tokens, leaving feature positions unchanged as in upstream EAGLE.
        start = target_start
        tokens = tokens[1:]
        assert len(tokens) == features.shape[0]
        owned = self._ensure_blocks(
            req_id,
            committed_group_block_ids=blocks,
            total_positions=start + len(tokens) + (max(k - 1, 0) if is_drafting else 0),
        )
        return _EaglePlan(req_id, owned, start, tokens, features, is_drafting)

    def _plans(self, ctx: ProposeContext, fused: mx.array) -> list[_EaglePlan]:
        assert self._committed_group_index is not None
        group = self._committed_group_index
        plans: list[_EaglePlan] = []
        for (req_id, state), segment, output in zip(
            ctx.decode_reqs, ctx.decode_segments, ctx.decode_token_ids, strict=True
        ):
            if not output:
                continue
            start, row, count = segment.cache_start_pos, segment.start_row, len(output)
            blocks = state.block_ids[group]
            plans.append(
                self._plan(
                    req_id=req_id,
                    blocks=blocks,
                    target_start=start,
                    tokens=[segment.input_token_ids[0], *output],
                    features=fused[row : row + count],
                    is_drafting=self._controller.can_draft_greedy(req_id, state),
                    k=ctx.num_speculative_tokens,
                )
            )

        for i, (prefill, mode) in enumerate(
            zip(ctx.prefill_reqs, ctx.prefill_result_modes, strict=True)
        ):
            row = ctx.cu_seqlens[ctx.num_decode_segments + i]
            start, count = prefill.start_pos, len(prefill.token_ids)
            end = start + count
            blocks = prefill.block_ids[group]
            state = ctx.request_states[prefill.req_id]
            if mode == "intermediate":
                # Recompute can replay committed outputs beyond the original
                # prompt. Sampling's full_prompt field excludes those tokens.
                next_token = state.token_ids[end]
            else:
                next_token = ctx.prefill_token_ids[i]
            plans.append(
                self._plan(
                    req_id=prefill.req_id,
                    blocks=blocks,
                    target_start=start,
                    tokens=[*prefill.token_ids, next_token],
                    features=fused[row : row + count],
                    is_drafting=mode != "intermediate"
                    and self._controller.can_draft_greedy(prefill.req_id, state),
                    k=ctx.num_speculative_tokens,
                )
            )

        return plans

    def _forward(self, plans: list[_EaglePlan]) -> tuple[mx.array, mx.array]:
        packed: list[int] = []
        last_rows: list[int] = []
        for plan in plans:
            packed.extend(plan.tokens)
            last_rows.append(len(packed) - 1)
        tokens = mx.array([packed], dtype=mx.int32)
        features = mx.concatenate([plan.features for plan in plans])[None]
        if max(len(plan.tokens) for plan in plans) <= _DECODE_INGEST_MAX_TOKENS:
            # Reuse draft-model SD's small-ingest decode path. Expanded rows
            # handle different acceptance lengths without padded query tokens.
            prepare_unified(
                [(p.block_ids, p.start, len(p.tokens)) for p in plans],
                [],
                self._block_size,
            )
        else:
            prepare_unified(
                [],
                [(p.block_ids, len(p.tokens), p.start) for p in plans],
                self._block_size,
            )
        try:
            normalized, recurrent = self._model(tokens, features)
        finally:
            clear_context()
        indices = mx.array(last_rows)
        return normalized[0, indices], recurrent[0, indices]

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        self.release_requests(ctx.finished_req_ids)
        self._prune_finished(ctx.request_states)
        assert ctx.target_aux_hidden_states
        fused = self._model.combine_hidden_states(
            mx.concatenate(ctx.target_aux_hidden_states, axis=-1)
        )
        plans = self._plans(ctx, fused)
        if not plans:
            return None
        normalized, recurrent = self._forward(plans)
        active = [i for i, plan in enumerate(plans) if plan.is_drafting]
        if not active or ctx.num_speculative_tokens <= 0:
            mx.eval(*self._kv.key_caches, *self._kv.value_caches)
            return None
        indices = mx.array(active)
        columns = [self._model.top_tokens(normalized[indices])]
        recurrent = recurrent[indices]
        drafting = [plans[i] for i in active]
        for depth in range(1, ctx.num_speculative_tokens):
            # Tokens stay on the GPU between recurrent draft steps.
            prepare_unified(
                [(p.block_ids, p.end + depth - 1) for p in drafting],
                [],
                self._block_size,
            )
            try:
                normalized, hidden = self._model(columns[-1][None], recurrent[None])
            finally:
                clear_context()
            recurrent = hidden[0]
            columns.append(self._model.top_tokens(normalized[0]))
        drafts = mx.stack(columns, axis=1)
        mx.eval(
            drafts,
            *self._kv.key_caches,
            *self._kv.value_caches,
        )
        return DraftTokenIds(
            req_ids=[p.req_id for p in drafting], draft_token_ids=drafts.tolist()
        )
