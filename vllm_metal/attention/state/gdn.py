# SPDX-License-Identifier: Apache-2.0
"""Hybrid GDN request-state lifecycle.

The hybrid paged runtime owns two different state systems:

- SDPA KV cache, indexed by scheduler block tables
- GDN recurrent state, one fixed-size slab per resident request

`HybridGDNStateManager` owns the second one.  The slab key is the request's
scheduler-assigned mamba block id (the first block of the first mamba KV cache
group), so state addressing shares provenance with the SDPA cache instead of
inventing a private request-to-slot allocator.  While ``mamba_cache_mode`` is
"none" each request holds exactly one block per mamba group for its whole
lifetime, so one slab per canonical block id is exactly one slab per request.

The manager still grows the recurrent cache on demand, resets reused slabs
before they are handed to a new request, and tracks when stable GDN arrays
must be materialized out of the lazy graph.

Ownership transfer: under async scheduling a finished request's mamba block
can be freed and reassigned to a new request before the runner's release for
the old request lands.  When a known block id arrives owned by a different
request, the slab is reset and re-owned in place; the stale owner's later
release then leaves the slab alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache

if TYPE_CHECKING:
    from vllm_metal.attention.context import PagedAttentionContext


class HybridGDNStateManager:
    """Own mamba-block-id-to-slab lifecycle for one hybrid runtime."""

    def __init__(self, state_cache: GDNPagedStateCache) -> None:
        self._state_cache = state_cache
        self._id_to_slab: dict[int, int] = {}
        self._id_owner: dict[int, str] = {}
        self._req_to_id: dict[str, int] = {}
        self._free_slabs: list[int] = []
        self._needs_materialize = False

    @property
    def request_slots(self) -> dict[str, int]:
        """Stable request-to-slab mapping for the active hybrid batch set."""
        return {
            req_id: self._id_to_slab[block_id]
            for req_id, block_id in self._req_to_id.items()
        }

    @property
    def free_slots(self) -> tuple[int, ...]:
        """Slabs available for reuse on later scheduler steps."""
        return tuple(self._free_slabs)

    @property
    def needs_materialize(self) -> bool:
        """Whether released slabs are waiting for state materialization."""
        return self._needs_materialize

    def populate_step_context(
        self,
        *,
        req_ids: list[str],
        ctx: PagedAttentionContext,
        state_block_ids: list[list[list[int]]],
        step_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        """Attach stable GDN slab ids to one forward-pass context.

        ``step_positions`` is align-mode's per-request
        ``(num_computed, num_scheduled)``; none mode keeps one slab per
        request for its whole lifetime, so it needs no step positions.
        Accepting and ignoring the argument keeps both managers on one
        signature, so the runtime delegates without dispatching on type.
        """
        del step_positions
        ctx.gdn_slot_mapping = self.assign_step_slots(req_ids, state_block_ids)

    @staticmethod
    def _canonical_block_id(
        req_id: str, block_ids_by_group: list[list[int]]
    ) -> int:
        """Return the request's slab key: first block of the first mamba group.

        Every mamba group holds the same request lifetime (blocks allocated at
        admission, freed together), so the first group's first block is a
        stable per-request key; the remaining groups' ids become relevant only
        when state moves into per-group scheduler blocks for prefix caching.
        """
        if not block_ids_by_group or not block_ids_by_group[0]:
            raise RuntimeError(
                f"GDN state manager requires mamba block ids for {req_id!r}; "
                "got an empty block table (mamba scheduler group not adopted?)"
            )
        return block_ids_by_group[0][0]

    def assign_step_slots(
        self,
        req_ids: list[str],
        state_block_ids: list[list[list[int]]],
    ) -> list[int]:
        """Plan one scheduler step's request-to-slab mapping atomically."""
        if len(state_block_ids) != len(req_ids):
            raise RuntimeError(
                "GDN state manager requires one mamba block table per request "
                f"(got {len(state_block_ids)} tables for {len(req_ids)} requests)"
            )

        step_slab_ids: list[int] = []
        planned_slab_by_id: dict[int, int] = {}
        planned_owner_by_id: dict[int, str] = {}
        # (block_id, slab, reuses_existing_slab)
        new_assignments: list[tuple[int, int, bool]] = []
        reusable_slabs = list(self._free_slabs)
        next_unallocated_slab = self._state_cache.allocated_seqs

        for req_id, block_ids_by_group in zip(req_ids, state_block_ids, strict=True):
            block_id = self._canonical_block_id(req_id, block_ids_by_group)

            known_id = self._req_to_id.get(req_id)
            if known_id is not None:
                if known_id != block_id:
                    raise RuntimeError(
                        f"GDN state manager: request {req_id!r} changed its "
                        f"mamba block id {known_id} -> {block_id} without a "
                        "release; scheduler and state pool are out of sync"
                    )
                step_slab_ids.append(self._id_to_slab[block_id])
                continue

            planned_slab = planned_slab_by_id.get(block_id)
            if planned_slab is not None:
                if planned_owner_by_id.get(block_id) != req_id:
                    raise RuntimeError(
                        f"GDN state manager: mamba block {block_id} assigned to "
                        f"two live requests in one step ({req_id!r} and "
                        f"{planned_owner_by_id.get(block_id)!r})"
                    )
                step_slab_ids.append(planned_slab)
                continue

            existing_slab = self._id_to_slab.get(block_id)
            if existing_slab is not None:
                # The pool reused this block id for a new request before the
                # old owner's release landed (async scheduling).  Take the
                # slab over in place; the stale owner's release becomes a
                # no-op via the ownership check.
                slab_id = existing_slab
                reuses_existing_slab = True
            elif reusable_slabs:
                slab_id = reusable_slabs.pop()
                reuses_existing_slab = True
            else:
                slab_id = next_unallocated_slab
                next_unallocated_slab += 1
                reuses_existing_slab = False

            planned_slab_by_id[block_id] = slab_id
            planned_owner_by_id[block_id] = req_id
            new_assignments.append((block_id, slab_id, reuses_existing_slab))
            step_slab_ids.append(slab_id)

        if not new_assignments:
            return step_slab_ids

        target_capacity = max(slab_id for _, slab_id, _ in new_assignments) + 1
        self._state_cache.ensure_capacity(target_capacity)

        # Reset reused state inside the forward-pass graph so the next request
        # starts from a clean slab without adding a separate synchronization
        # point to the release path.
        for _, slab_id, reuses_existing_slab in new_assignments:
            if reuses_existing_slab:
                self._state_cache.reset_slot(slab_id)

        for block_id, slab_id, _ in new_assignments:
            req_id = planned_owner_by_id[block_id]
            self._id_to_slab[block_id] = slab_id
            self._id_owner[block_id] = req_id
            self._req_to_id[req_id] = block_id
        self._free_slabs = reusable_slabs
        return step_slab_ids

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        """Append authoritative GDN state arrays that the forward mutates."""
        outputs.extend(self._state_cache.updated_state_arrays())

    def release_requests(self, req_ids: set[str]) -> None:
        """Release slabs for requests whose recurrent state is no longer valid."""
        freed_slabs: list[int] = []
        for req_id in req_ids:
            block_id = self._req_to_id.pop(req_id, None)
            if block_id is None:
                continue
            if self._id_owner.get(block_id) != req_id:
                # Ownership moved to a new request when the scheduler reused
                # this block id before our release ran; the slab stays live.
                continue
            freed_slabs.append(self._id_to_slab.pop(block_id))
            del self._id_owner[block_id]

        if not freed_slabs:
            return

        self._state_cache.apply_pending_states()
        self._needs_materialize = True
        self._free_slabs.extend(freed_slabs)

    def materialize_pending_state(self) -> None:
        """Force stable GDN arrays after a slab release updated them lazily."""
        if not self._needs_materialize:
            return

        self._state_cache.apply_pending_states()
        mx.eval(*self._state_cache.updated_state_arrays())
        self._needs_materialize = False
