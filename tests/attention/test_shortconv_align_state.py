# SPDX-License-Identifier: Apache-2.0
"""Align-mode (prefix caching) conv state for ShortConv hybrids.

Three layers of coverage, mirroring the GDN align tests:

- ``ShortConvStateCache``'s block-slab surface (zero/copy/copy_blocks and
  the scheduler layer layout with pool aliasing),
- ``AlignGDNStateManager`` driving a ShortConv cache (the manager is state-
  family-agnostic; these tests pin that),
- ``ShortConvPagedWrapper`` running under align-mode group slot mappings,
  including a block-boundary state migration mid-sequence — the wrapper
  output must stay identical to the unmodified mlx_lm module running the
  same tokens sequentially, which is exactly the lossless-restore guarantee
  prefix caching needs.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache

from tests.attention.test_shortconv_wrapper import (
    _assert_close,
    _make_module,
    _make_wrapper,
    HIDDEN,
    L_CACHE,
)
from vllm_metal.attention.caches.shortconv_cache import ShortConvStateCache
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    set_context,
)
from vllm_metal.attention.runtime.shortconv_hybrid import (
    ShortConvHybridPagedAttentionRuntime,
)
from vllm_metal.attention.state import AlignGDNStateManager

BLOCK = 4


@pytest.fixture(autouse=True)
def _cpu_device_and_clean_context():
    # Same rationale as test_shortconv_wrapper: pin the math-equivalence
    # comparison to the CPU stream so packed vs sequential runs cannot differ
    # by GPU kernel tiling choice.  The native row scatter still runs on its
    # own GPU stream, but it moves bytes without arithmetic.
    default_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(default_device)
    clear_context()


def _make_cache(
    *,
    num_layers: int = 2,
    num_blocks: int = 8,
    initial_blocks: int | None = None,
) -> ShortConvStateCache:
    return ShortConvStateCache(
        num_layers=num_layers,
        max_seqs=num_blocks,
        conv_kernel_dim=L_CACHE,
        conv_dim=4,
        initial_seqs=num_blocks if initial_blocks is None else initial_blocks,
        dtype=mx.float32,
    )


def _fill_slab(cache: ShortConvStateCache, layer: int, slab: int, value: float) -> None:
    rows = mx.full((1, cache.conv_kernel_dim - 1, cache.conv_dim), value, mx.float32)
    cache.write_conv_rows(layer, rows, mx.array([slab], dtype=mx.int32))
    mx.eval(cache.conv_states[layer])


def _slab(cache: ShortConvStateCache, layer: int, slab: int) -> np.ndarray:
    mx.eval(cache.conv_states[layer])
    return np.array(cache.conv_states[layer][slab])


class TestShortConvStateCacheAlignSurface:
    def test_zero_slots_clears_stale_bytes(self) -> None:
        cache = _make_cache()
        _fill_slab(cache, 0, 3, 7.0)

        cache.zero_slots([3], [0])

        assert np.all(_slab(cache, 0, 3) == 0)

    def test_copy_slots_preserves_the_source_checkpoint(self) -> None:
        cache = _make_cache()
        _fill_slab(cache, 0, 2, 5.0)

        cache.copy_slots([2], [6], [0])

        np.testing.assert_array_equal(_slab(cache, 0, 6), 5.0)
        np.testing.assert_array_equal(_slab(cache, 0, 2), 5.0)

    def test_copy_blocks_grows_and_copies_every_pool(self) -> None:
        cache = _make_cache(num_blocks=8, initial_blocks=3)
        _fill_slab(cache, 0, 1, 4.0)
        _fill_slab(cache, 1, 2, 9.0)

        cache.copy_blocks([(1, 5), (2, 6)])

        assert cache.allocated_seqs == 7
        np.testing.assert_array_equal(_slab(cache, 0, 5), 4.0)
        np.testing.assert_array_equal(_slab(cache, 1, 6), 9.0)

    def test_shared_pool_layout_aliases_and_rejects_collisions(self) -> None:
        cache = _make_cache(num_layers=2)
        cache.set_layer_layout([0, 1], [0, 0])

        assert cache.num_state_pools == 1
        assert cache.conv_states[0] is cache.conv_states[1]
        # A write through either layer must stay visible through both.
        _fill_slab(cache, 0, 2, 5.0)
        np.testing.assert_array_equal(_slab(cache, 1, 2), 5.0)

        with pytest.raises(ValueError, match="same"):
            _make_cache(num_layers=2).set_layer_layout([0, 0], [0, 0])


class TestAlignManagerOverShortConv:
    def _populate(self, manager, req_ids, tables, positions):
        ctx = PagedAttentionContext(slot_mapping=[])
        manager.populate_step_context(
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=tables,
            step_positions=positions,
        )
        return ctx

    def test_fresh_request_zeroes_its_state_block(self) -> None:
        cache = _make_cache()
        manager = AlignGDNStateManager(cache, BLOCK)
        _fill_slab(cache, 0, 3, 7.0)  # stale bytes from a previous block life

        ctx = self._populate(manager, ["req-A"], [[[3]]], [(0, 2)])

        assert ctx.gdn_group_slot_mappings == ([3],)
        assert np.all(_slab(cache, 0, 3) == 0)

    def test_boundary_crossing_copies_forward_and_keeps_checkpoint(self) -> None:
        cache = _make_cache()
        manager = AlignGDNStateManager(cache, BLOCK)
        _fill_slab(cache, 0, 2, 5.0)
        _fill_slab(cache, 1, 2, 5.0)

        # num_computed=4 (block boundary), decoding 1 token → block index 1.
        ctx = self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])

        assert ctx.gdn_group_slot_mappings == ([6],)
        for layer in (0, 1):
            np.testing.assert_array_equal(_slab(cache, layer, 2), 5.0)
            np.testing.assert_array_equal(_slab(cache, layer, 6), 5.0)

    def test_pool_materializes_lazily_by_high_water_block_id(self) -> None:
        cache = _make_cache(num_blocks=8, initial_blocks=2)
        manager = AlignGDNStateManager(cache, BLOCK)

        ctx = self._populate(manager, ["req-A"], [[[5]]], [(0, 2)])

        assert ctx.gdn_group_slot_mappings == ([5],)
        assert cache.allocated_seqs == 6
        assert np.all(_slab(cache, 0, 5) == 0)


class TestShortConvWrapperUnderAlign:
    def _set_align_context(
        self, seg_lens: list[int], group_slots: list[int]
    ) -> None:
        cu_seqlens = [0]
        for seg_len in seg_lens:
            cu_seqlens.append(cu_seqlens[-1] + seg_len)
        num_decode = 0
        for seg_len in seg_lens:
            if seg_len != 1:
                break
            num_decode += 1
        set_context(
            PagedAttentionContext(
                slot_mapping=[],
                cu_seqlens=cu_seqlens,
                gdn_group_slot_mappings=(list(group_slots),),
                num_decode_requests=num_decode,
            )
        )

    def test_block_migration_matches_sequential_reference(self) -> None:
        """Prefill on slab A, copy A→B at a block boundary, decode on B.

        This is the exact state motion align-mode prefix caching performs;
        the wrapper's output for every token must match mlx_lm running the
        same tokens through one contiguous ArraysCache session.
        """
        module = _make_module()
        wrapper, state_cache = _make_wrapper(module)

        prompt = mx.random.normal((1, BLOCK, HIDDEN), dtype=mx.float32)
        decode_tok = mx.random.normal((1, 1, HIDDEN), dtype=mx.float32)
        mx.eval(prompt, decode_tok)

        # Sequential reference: one uninterrupted mlx_lm session.
        ref_cache = ArraysCache(size=1)
        ref_prefill = module(prompt, cache=ref_cache)
        ref_decode = module(decode_tok, cache=ref_cache)
        mx.eval(ref_prefill, ref_decode)

        # Align-mode wrapper: prefill lands in block slab 2 ...
        self._set_align_context([BLOCK], [2])
        out_prefill = wrapper(prompt)
        clear_context()

        # ... the manager's copy-forward moves the checkpoint to slab 6 ...
        state_cache.copy_slots([2], [6], [0])

        # ... and the decode step runs against slab 6.  Slab 2 must stay a
        # bit-identical checkpoint (another request may restore from it).
        checkpoint_before = _slab(state_cache, 0, 2)
        self._set_align_context([1], [6])
        out_decode = wrapper(decode_tok)
        clear_context()

        _assert_close(out_prefill, ref_prefill)
        _assert_close(out_decode, ref_decode)
        np.testing.assert_array_equal(_slab(state_cache, 0, 2), checkpoint_before)

    def test_restored_checkpoint_continues_like_the_original_request(self) -> None:
        """Two requests share a prefix: B restores A's checkpoint slab.

        Request A prefills the shared prefix into slab 1 (its block).  A
        prefix hit admits request B with the same slab as its computed
        block; the manager copies it to B's next block (slab 4) and B runs
        only the suffix.  B's suffix output must match a from-scratch
        sequential run of prefix+suffix.
        """
        module = _make_module()
        wrapper, state_cache = _make_wrapper(module)

        prefix = mx.random.normal((1, BLOCK, HIDDEN), dtype=mx.float32)
        suffix = mx.random.normal((1, 2, HIDDEN), dtype=mx.float32)
        mx.eval(prefix, suffix)

        # Request A computes the shared prefix; checkpoint lands in slab 1.
        self._set_align_context([BLOCK], [1])
        wrapper(prefix)
        clear_context()

        # Request B hits the cached prefix: copy-forward 1→4, run suffix on 4.
        state_cache.copy_slots([1], [4], [0])
        self._set_align_context([2], [4])
        out_suffix = wrapper(suffix)
        clear_context()

        # Reference: the full sequence through one mlx_lm session.
        ref_cache = ArraysCache(size=1)
        module(prefix, cache=ref_cache)
        ref_suffix = module(suffix, cache=ref_cache)
        mx.eval(ref_suffix)

        _assert_close(out_suffix, ref_suffix)


class TestShortConvAlignRuntime:
    def test_adopts_shared_layout_before_materializing_state(self) -> None:
        runtime = ShortConvHybridPagedAttentionRuntime(
            layer_types=["conv", "full_attention", "conv", "full_attention"],
            max_num_seqs=2,
            num_kv_heads=1,
            head_dim=4,
            conv_kernel_dim=L_CACHE,
            conv_dim=4,
            block_size=BLOCK,
            dtype=mx.float32,
            mamba_cache_mode="align",
        )
        runtime.initialize(num_blocks=8)

        assert runtime.state_cache.allocated_seqs == 0
        runtime.adopt_scheduler_group(
            0,
            BLOCK,
            state_group_indices=(1, 2),
            layer_group_ordinals=[0, 1],
            layer_pool_ordinals=[0, 0],
        )
        runtime.state_cache.ensure_capacity(2)

        assert runtime.state_cache.num_state_pools == 1
        assert (
            runtime.state_cache.conv_states[0] is runtime.state_cache.conv_states[1]
        )
