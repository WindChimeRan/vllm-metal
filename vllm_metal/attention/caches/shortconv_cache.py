# SPDX-License-Identifier: Apache-2.0
"""Conv state cache for ShortConv (LFM2-style) layers.

LFM2/LFM2.5 conv layers keep a fixed-size causal-conv tail per slot: the
last ``conv_kernel_dim - 1`` rows of the gated ``B*x`` activation.

Layout per conv layer:
  - conv_state: [allocated_slots, conv_kernel_dim - 1, conv_dim]

What a slot means depends on the caching mode (the state managers decide):

- ``mamba_cache_mode="none"``: one slot per resident request for its
  lifetime, assigned by ``HybridGDNStateManager``.  ``max_seqs`` is the
  scheduler-visible request cap.
- ``mamba_cache_mode="align"``: slots are scheduler block ids
  (``AlignGDNStateManager``); a slab written at a block boundary is a
  prefix-cache checkpoint another request can restore from.  ``max_seqs``
  is the block-pool size.

Unlike ``GDNPagedStateCache`` there is no pending-compact handoff: the
ShortConv wrapper writes each request's updated tail straight into the
stable pool, so ``apply_pending_states`` is a no-op kept because the state
managers drain pending state before planning slot motion.

All row writes go through the native in-place row scatter.  MLX indexed
assignment (``pool[ids] = rows``) copies the whole pool whenever it is
aliased — and under align-mode layer striping several layers alias one
physical pool — so the scatter is load-bearing for aliasing, not just speed.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import mlx.core as mx


@functools.cache
def _native_row_scatter() -> Callable[..., mx.array]:
    """Return the required in-place row-scatter primitive."""
    from vllm_metal.metal import get_ops

    return get_ops().gdn_state_scatter


class ShortConvStateCache:
    """Per-layer MLX arrays for ShortConv causal-conv state."""

    def __init__(
        self,
        *,
        num_layers: int,
        max_seqs: int,
        conv_kernel_dim: int,
        conv_dim: int,
        initial_seqs: int | None = None,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for ShortConv state cache: {dtype}")
        if max_seqs < 0:
            raise ValueError("max_seqs must be non-negative")
        if conv_kernel_dim < 2:
            raise ValueError("conv_kernel_dim must be at least 2 to carry state")
        if initial_seqs is None:
            initial_seqs = max_seqs
        if initial_seqs < 0 or initial_seqs > max_seqs:
            raise ValueError(
                "initial_seqs must be between 0 and max_seqs "
                f"(got {initial_seqs}, max_seqs={max_seqs})"
            )

        self.num_layers = num_layers
        self.max_seqs = max_seqs
        self.allocated_seqs = initial_seqs
        self.conv_kernel_dim = conv_kernel_dim
        self.conv_dim = conv_dim
        self.dtype = dtype

        self.conv_states: list[mx.array] = [
            mx.zeros(self._conv_shape(initial_seqs), dtype=dtype)
            for _ in range(num_layers)
        ]
        # Scheduler layout, adopted via ``set_layer_layout``; until then every
        # layer is its own pool (none mode keeps this identity layout).
        self._layer_group_ordinals: list[int] = [0] * num_layers
        self._pool_siblings: list[list[int]] = [[i] for i in range(num_layers)]
        self._canonical_layers: list[int] = list(range(num_layers))
        self._eval_state_arrays()

    def _conv_shape(self, num_seqs: int) -> tuple[int, int, int]:
        return (num_seqs, self.conv_kernel_dim - 1, self.conv_dim)

    def _eval_state_arrays(self) -> None:
        arrays = [self.conv_states[i] for i in self._canonical_layers]
        if arrays:
            mx.eval(*arrays)

    @property
    def num_state_pools(self) -> int:
        """Number of distinct physical state pools under the adopted layout."""
        return len(self._canonical_layers)

    def store_conv_state(self, layer_idx: int, array: mx.array) -> None:
        """Store a layer's updated conv pool, keeping pool siblings aliased."""
        for sibling in self._pool_siblings[layer_idx]:
            self.conv_states[sibling] = array

    def write_conv_rows(self, layer_idx: int, rows: mx.array, ids: mx.array) -> None:
        """Write conv rows in place and rebind every sibling to the handle."""
        pool = self.conv_states[layer_idx]
        # MLX's indexed assignment converts the source implicitly and callers
        # rely on it; the primitive requires an exact match.  ``astype`` is a
        # no-op when the dtypes already agree, and O(rows) otherwise.
        self.store_conv_state(
            layer_idx, _native_row_scatter()(pool, rows.astype(pool.dtype), ids)
        )

    def ensure_capacity(self, num_seqs: int) -> None:
        """Grow stable state pools so slots ``[0, num_seqs)`` are valid."""
        if num_seqs < 0:
            raise ValueError("num_seqs must be non-negative")
        if num_seqs > self.max_seqs:
            raise RuntimeError(
                "ShortConv state cache requested more slots than max_num_seqs "
                f"({num_seqs} > {self.max_seqs})"
            )
        if num_seqs <= self.allocated_seqs:
            return

        old_allocated = self.allocated_seqs
        # Grow one pool at a time, releasing each old array before building the
        # next, so peak memory holds one extra pool rather than a full copy of
        # every layer's state.
        for layer_idx in self._canonical_layers:
            grown = mx.zeros(self._conv_shape(num_seqs), dtype=self.dtype)
            if old_allocated:
                grown[:old_allocated] = self.conv_states[layer_idx]
            mx.eval(grown)
            self.store_conv_state(layer_idx, grown)

        self.allocated_seqs = num_seqs

    def require_allocated_slots(self, slot_ids: list[int]) -> None:
        """Validate slots against both the scheduler cap and allocated rows."""
        if any(slot < 0 or slot >= self.max_seqs for slot in slot_ids):
            raise RuntimeError("ShortConv wrapper received out-of-range slot mapping")
        if any(slot >= self.allocated_seqs for slot in slot_ids):
            raise RuntimeError(
                "ShortConv wrapper received slot mapping beyond allocated state cache"
            )

    def reset_slot(self, slot: int) -> None:
        """Clear state for one allocated slot before it is reused."""
        self.zero_slots([slot], self._canonical_layers)

    def set_layer_layout(
        self, group_ordinals: list[int], pool_ordinals: list[int]
    ) -> None:
        """Adopt the scheduler's layer layout: group + physical pool per layer.

        ``group_ordinals[cache_idx]`` selects the block-table row addressing a
        layer's slabs; ``pool_ordinals[cache_idx]`` names its physical pool
        (vLLM's ``kv_cache_tensors.shared_by``: one pool per within-group
        position).  Layers sharing a pool must belong to different groups —
        their groups then own disjoint block ids, so slab rows never collide.
        Must be called before any state is written (pool arrays are rebuilt).
        """
        if not (len(group_ordinals) == len(pool_ordinals) == self.num_layers):
            raise ValueError(
                f"expected one group and one pool ordinal per conv layer "
                f"({self.num_layers}), got {len(group_ordinals)}/"
                f"{len(pool_ordinals)}"
            )
        pool_members: dict[int, list[int]] = {}
        for layer_idx, pool in enumerate(pool_ordinals):
            pool_members.setdefault(pool, []).append(layer_idx)
        for pool, members in pool_members.items():
            groups = [group_ordinals[i] for i in members]
            if len(set(groups)) != len(groups):
                raise ValueError(
                    f"state pool {pool} is shared by two layers of the same "
                    f"mamba cache group (layers {members}); their slab rows "
                    "would collide"
                )

        self._layer_group_ordinals = list(group_ordinals)
        self._pool_siblings = [pool_members[pool] for pool in pool_ordinals]
        self._canonical_layers = sorted(members[0] for members in pool_members.values())
        # Rebuild storage so pool siblings alias one array.  Pre-adopt state
        # is all zeros, so aliasing to the canonical member's array is exact.
        for members in pool_members.values():
            canonical = members[0]
            for layer_idx in members:
                self.conv_states[layer_idx] = self.conv_states[canonical]

    def layer_group_ordinal(self, cache_idx: int) -> int:
        """Return the mamba-cache-group ordinal for one conv layer."""
        return self._layer_group_ordinals[cache_idx]

    def layers_for_group_ordinal(self, ordinal: int) -> list[int]:
        """Return the cache indices of layers in one mamba cache group."""
        return [idx for idx, o in enumerate(self._layer_group_ordinals) if o == ordinal]

    def zero_slots(self, slot_ids: list[int], layer_indices: list[int]) -> None:
        """Zero state slabs for the given layers (batched, lazy).

        Align-mode slabs are addressed by scheduler block id, so a freshly
        allocated block may carry a previous request's bytes; fresh requests
        must start from zero state.
        """
        if not slot_ids or not layer_indices:
            return
        self.require_allocated_slots(slot_ids)
        ids = mx.array(slot_ids, dtype=mx.int32)
        zeros = mx.zeros(
            (len(slot_ids),) + self.conv_states[0].shape[1:], dtype=self.dtype
        )
        for layer_idx in layer_indices:
            self.write_conv_rows(layer_idx, zeros, ids)

    def copy_slots(
        self, src_ids: list[int], dst_ids: list[int], layer_indices: list[int]
    ) -> None:
        """Copy state slabs ``src → dst`` for the given layers (batched, lazy).

        Sources are left untouched — align-mode prefix caching relies on a
        checkpointed block's slab staying immutable once the request advances
        to its next block.
        """
        if not src_ids or not layer_indices:
            return
        self.require_allocated_slots(src_ids)
        self.require_allocated_slots(dst_ids)
        src = mx.array(src_ids, dtype=mx.int32)
        dst = mx.array(dst_ids, dtype=mx.int32)
        for layer_idx in layer_indices:
            # Gather first (its output is O(rows)), then write the rows back.
            # Reading the sources into their own array keeps the copy atomic
            # when one pair's destination is another pair's source.
            self.write_conv_rows(layer_idx, self.conv_states[layer_idx][src], dst)

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler copy-on-write operations to every physical pool."""
        if not block_copies:
            return
        src_ids, dst_ids = zip(*block_copies, strict=True)
        high_water = max(*src_ids, *dst_ids) + 1
        self.ensure_capacity(high_water)
        self.copy_slots(list(src_ids), list(dst_ids), self._canonical_layers)

    def apply_pending_states(self) -> None:
        """No-op: ShortConv writes state into the stable pool directly."""

    def updated_state_arrays(self) -> list[mx.array]:
        """Return the state arrays a forward pass mutates (for submission)."""
        return [self.conv_states[i] for i in self._canonical_layers]
