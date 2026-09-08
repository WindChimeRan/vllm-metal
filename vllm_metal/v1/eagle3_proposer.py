# SPDX-License-Identifier: Apache-2.0
"""Batched linear EAGLE3 with scheduler-owned, prefix-cacheable draft state.

Pair feature[t] with token[t+1], storing its KV at *token position* t+1.
Consequently a cached block depends only on its hashed token prefix. Physical
slot zero is unused and masked. A small sidecar retains feature[t] at each
target block boundary, allowing a prefix hit to reconstruct its first new pair
without replaying the target prefix. Recursive draft features are scratch;
every verified row is ingested again with the actual target features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.metal import get_ops
from vllm_metal.v1.draft_model_proposer import DraftModelProposer
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


class _PagedEagleBatch:
    """One ragged batch, gathered into MLX SDPA's dense batch layout.

    The paged backing stays scheduler owned. Padding neither writes KV nor
    contributes to attention. Metadata is shared by the head's Q/K/V paths.
    """

    def __init__(self, kv: MetalPagedKVCache, plans: list[_EaglePlan]):
        self.kv = kv
        self.lengths = [len(plan.tokens) for plan in plans]
        self.width = max(self.lengths)
        self.offset = mx.array([plan.start - 1 for plan in plans], dtype=mx.int32)
        ends = mx.array([plan.end for plan in plans], dtype=mx.int32)
        max_end = max(plan.end for plan in plans)
        table_width = (max_end + kv.block_size - 1) // kv.block_size
        table = mx.array(
            [
                p.block_ids[:table_width] + [0] * max(0, table_width - len(p.block_ids))
                for p in plans
            ],
            dtype=mx.int32,
        )
        positions = self.offset[:, None] + 1 + mx.arange(self.width)
        valid = mx.arange(self.width)[None, :] < mx.array(self.lengths)[:, None]
        block_indices = mx.minimum(positions // kv.block_size, table_width - 1)
        write_blocks = mx.take_along_axis(table, block_indices, axis=1)
        self.write_slots = (
            mx.where(
                valid, write_blocks * kv.block_size + positions % kv.block_size, -1
            )
            .reshape(-1)
            .astype(mx.int64)
        )
        keys = mx.arange(max_end)
        self.read_slots = (
            table[:, keys // kv.block_size] * kv.block_size + keys % kv.block_size
        )
        self.mask = (
            (keys[None, None, :] > 0)
            & (keys[None, None, :] < ends[:, None, None])
            & (keys[None, None, :] <= positions[:, :, None])
        )[:, None]

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        heads, dim = keys.shape[1], keys.shape[-1]
        key_cache, value_cache = get_ops().reshape_and_cache(
            mx.contiguous(
                keys.transpose(0, 2, 1, 3).reshape(-1, heads, dim).astype(self.kv.dtype)
            ),
            mx.contiguous(
                values.transpose(0, 2, 1, 3)
                .reshape(-1, heads, dim)
                .astype(self.kv.dtype)
            ),
            self.kv.key_caches[0],
            self.kv.value_caches[0],
            self.write_slots,
        )
        self.kv.replace_layer_cache(0, key_cache, value_cache)
        gathered_keys = key_cache.reshape(-1, heads, dim)[self.read_slots].transpose(
            0, 2, 1, 3
        )
        gathered_values = value_cache.reshape(-1, heads, dim)[
            self.read_slots
        ].transpose(0, 2, 1, 3)
        return gathered_keys, gathered_values


class Eagle3Proposer(DraftModelProposer):
    """Reuse draft block ownership, with EAGLE-specific feature reconciliation."""

    _kv: MetalPagedKVCache
    _boundary_features: mx.array

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
        args = model.config.layer
        proposer._kv = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=args.num_key_value_heads,
            head_dim=args.head_dim or args.hidden_size // args.num_attention_heads,
            num_blocks=committed_num_blocks + scratch_reserve_blocks,
            block_size=block_size,
            dtype=dtype,
        )
        proposer._boundary_features = mx.zeros(
            (committed_num_blocks, args.hidden_size), dtype=dtype
        )
        mx.eval(proposer._boundary_features)
        return proposer

    def needs_target_hidden_states(
        self, decode_segments: Any, *, has_final_prefill: bool
    ) -> bool:
        # Intermediate prefill and K=0 must populate the same cache as decode.
        return True

    def _plan(
        self,
        *,
        req_id: str,
        blocks: list[int],
        target_start: int,
        tokens: list[int],
        features: mx.array,
        current_boundaries: dict[int, mx.array],
        is_drafting: bool,
        k: int,
    ) -> _EaglePlan:
        start = target_start + 1
        known_end = self._draft_seq_lens.get(req_id, 0)
        if target_start and (known_end < start or target_start % self._block_size == 0):
            if target_start % self._block_size:
                raise RuntimeError("EAGLE3 prefix resumption must be block aligned")
            # The cached block's final target feature supplies the missing pair.
            boundary = blocks[target_start // self._block_size - 1]
            boundary_feature = (
                current_boundaries[boundary]
                if boundary in current_boundaries
                else self._boundary_features[boundary : boundary + 1]
            )
            features = mx.concatenate((boundary_feature, features), axis=0)
            start -= 1
            # The caller prepends the token at target_start for this case.
        elif len(tokens) > features.shape[0]:
            tokens = tokens[1:]
        if len(tokens) != features.shape[0]:
            raise RuntimeError("EAGLE3 token/feature alignment mismatch")
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
        boundary_ids: list[int] = []
        boundary_rows: list[int] = []

        def save_boundaries(
            start: int, length: int, row: int, blocks: list[int]
        ) -> None:
            for pos in range(
                start + (-start - 1) % self._block_size,
                start + length,
                self._block_size,
            ):
                boundary_ids.append(blocks[pos // self._block_size])
                boundary_rows.append(row + pos - start)

        # The scheduler can expose an earlier request's newly allocated prefix
        # to another request in the SAME forward. Resolve those dependencies
        # from the current target output, before consulting persistent state.
        # Reading the old sidecar and merely submitting its in-place update
        # later would leave the head dependent on stale/zero features.
        for (_, state), segment, output in zip(
            ctx.decode_reqs, ctx.decode_segments, ctx.decode_token_ids, strict=True
        ):
            save_boundaries(
                segment.cache_start_pos,
                len(output),
                segment.start_row,
                state.block_ids[group],
            )
        for i, prefill in enumerate(ctx.prefill_reqs):
            save_boundaries(
                prefill.start_pos,
                len(prefill.token_ids),
                ctx.cu_seqlens[ctx.num_decode_segments + i],
                prefill.block_ids[group],
            )
        rows_by_block = dict(zip(boundary_ids, boundary_rows, strict=True))
        current_boundaries = {
            block: fused[row : row + 1] for block, row in rows_by_block.items()
        }

        for (req_id, state), segment, output in zip(
            ctx.decode_reqs, ctx.decode_segments, ctx.decode_token_ids, strict=True
        ):
            if not output:
                continue
            start, row, count = segment.cache_start_pos, segment.start_row, len(output)
            blocks = state.block_ids[group]
            tokens = list(output)
            # At ordinary decode, prior ingestion already built the anchor KV.
            if start and (
                self._draft_seq_lens.get(req_id, 0) < start + 1
                or start % self._block_size == 0
            ):
                tokens = [segment.input_token_ids[0], *tokens]
            plans.append(
                self._plan(
                    req_id=req_id,
                    blocks=blocks,
                    target_start=start,
                    tokens=tokens,
                    features=fused[row : row + count],
                    current_boundaries=current_boundaries,
                    is_drafting=self._controller.can_draft_greedy(req_id, state),
                    k=ctx.num_speculative_tokens,
                )
            )

        for i, (prefill, mode) in enumerate(
            zip(ctx.prefill_reqs, ctx.prefill_result_modes, strict=True)
        ):
            if not prefill.token_ids:
                continue
            row = ctx.cu_seqlens[ctx.num_decode_segments + i]
            start, count = prefill.start_pos, len(prefill.token_ids)
            end = start + count
            blocks = prefill.block_ids[group]
            state = ctx.request_states.get(prefill.req_id)
            if mode == "intermediate":
                # Recompute can replay committed outputs beyond the original
                # prompt. Sampling's full_prompt field excludes those tokens.
                full = (
                    state.token_ids
                    if state is not None
                    else (prefill.full_prompt_token_ids or [])
                )
                if end >= len(full):
                    raise RuntimeError(
                        "EAGLE3 intermediate prefill is missing its next prompt token"
                    )
                next_token = full[end]
            else:
                next_token = ctx.prefill_token_ids[i]
            # Optional first token reconstructs a pair at a prefix-hit boundary.
            tokens = (
                [*prefill.token_ids, next_token]
                if start
                else [*prefill.token_ids[1:], next_token]
            )
            plans.append(
                self._plan(
                    req_id=prefill.req_id,
                    blocks=blocks,
                    target_start=start,
                    tokens=tokens,
                    features=fused[row : row + count],
                    current_boundaries=current_boundaries,
                    is_drafting=mode != "intermediate"
                    and state is not None
                    and self._controller.can_draft_greedy(prefill.req_id, state),
                    k=ctx.num_speculative_tokens,
                )
            )

        if boundary_rows:
            # Multiple requests can share a cached block; identical prefix
            # features are equivalent, but the scatter needs unique destinations.
            self._boundary_features = get_ops().gdn_state_scatter(
                self._boundary_features,
                fused[mx.array(list(rows_by_block.values()), dtype=mx.int32)].astype(
                    self._boundary_features.dtype
                ),
                mx.array(list(rows_by_block), dtype=mx.int32),
            )
        return plans

    def _forward(self, plans: list[_EaglePlan]) -> tuple[mx.array, mx.array]:
        batch = _PagedEagleBatch(self._kv, plans)
        tokens = mx.array(
            [p.tokens + [0] * (batch.width - len(p.tokens)) for p in plans],
            dtype=mx.int32,
        )
        features = mx.stack(
            [
                mx.pad(p.features, ((0, batch.width - len(p.tokens)), (0, 0)))
                for p in plans
            ]
        )
        normalized, recurrent = self._model(
            tokens, features, mask=batch.mask, cache=batch
        )
        rows, cols = mx.arange(len(plans)), mx.array(batch.lengths) - 1
        return normalized[rows, cols], recurrent[rows, cols]

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        if self._committed_group_index is None:
            raise RuntimeError("EAGLE3 draft cache has no scheduler group")
        self.release_requests(ctx.finished_req_ids)
        self._prune_finished(ctx.request_states)
        if ctx.target_hidden_states is None:
            raise RuntimeError("EAGLE3 requires target auxiliary hidden states")
        fused = self._model.combine_hidden_states(ctx.target_hidden_states)
        plans = self._plans(ctx, fused)
        if not plans:
            return None
        normalized, recurrent = self._forward(plans)
        for plan in plans:
            self._draft_seq_lens[plan.req_id] = plan.end
        active = [i for i, plan in enumerate(plans) if plan.is_drafting]
        if not active or ctx.num_speculative_tokens <= 0:
            mx.eval(
                *self._kv.key_caches, *self._kv.value_caches, self._boundary_features
            )
            return None
        indices = mx.array(active)
        columns = [self._model.top_tokens(normalized[indices])]
        recurrent = recurrent[indices]
        drafting = [plans[i] for i in active]
        for depth in range(1, ctx.num_speculative_tokens):
            # Tokens stay on the GPU between recurrent draft steps.
            metadata = [
                _EaglePlan(
                    p.req_id,
                    p.block_ids,
                    p.end + depth - 1,
                    [0],
                    recurrent[i : i + 1],
                    True,
                )
                for i, p in enumerate(drafting)
            ]
            batch = _PagedEagleBatch(self._kv, metadata)
            normalized, recurrent = self._model(
                columns[-1][:, None], recurrent[:, None], mask=batch.mask, cache=batch
            )
            recurrent = recurrent[:, 0]
            columns.append(self._model.top_tokens(normalized[:, 0]))
        drafts = mx.stack(columns, axis=1)
        mx.eval(
            drafts,
            *self._kv.key_caches,
            *self._kv.value_caches,
            self._boundary_features,
        )
        return DraftTokenIds(
            req_ids=[p.req_id for p in drafting], draft_token_ids=drafts.tolist()
        )
