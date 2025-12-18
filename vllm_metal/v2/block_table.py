# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible BlockTables class using PyTorch operations instead of Triton kernels."""

from collections.abc import Iterable

import torch
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer


class MetalBlockTables:
    """Metal-compatible BlockTables that uses PyTorch operations instead of Triton kernels.

    This is a drop-in replacement for vllm.v1.worker.gpu.block_table.BlockTables
    that works on Metal/MPS devices where Triton is not available.
    """

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.block_sizes = block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.device = device
        self.pin_memory = pin_memory

        self.num_kv_cache_groups = len(self.block_sizes)
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[torch.Tensor] = []
        for i in range(self.num_kv_cache_groups):
            block_size = self.block_sizes[i]
            max_num_blocks = cdiv(self.max_model_len, block_size)
            block_table = torch.zeros(
                self.max_num_reqs,
                max_num_blocks,
                dtype=torch.int32,
                device=self.device,
            )
            self.block_tables.append(block_table)

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(block_table) for block_table in self.block_tables
        ]

        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.num_blocks = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        # Misc buffers.
        self.req_indices = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.overwrite = self._make_buffer(self.max_num_reqs, dtype=torch.bool)
        self.cu_num_new_blocks = self._make_buffer(
            self.num_kv_cache_groups, self.max_num_reqs + 1, dtype=torch.int32
        )

        # These attributes are used by the original Triton code but we still need them
        # for compatibility with the rest of vLLM
        self.block_table_strides = torch.tensor(
            [b.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_table_ptrs = self._make_ptr_tensor(self.block_tables)
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def _make_buffer(self, *args, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *args, dtype=dtype, pin_memory=self.pin_memory, device=self.device
        )

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        ptrs_tensor_cpu = torch.tensor(
            [t.data_ptr() for t in x],
            dtype=torch.uint64,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        return ptrs_tensor_cpu.to(self.device, non_blocking=True)

    def append_block_ids(
        self,
        # [num_reqs]
        req_indices: list[int],
        # [num_kv_cache_groups, num_reqs + 1]
        cu_num_new_blocks: tuple[list[int], ...],
        # [num_kv_cache_groups, num_new_blocks]
        new_block_ids: tuple[list[int], ...],
        # [num_reqs]
        overwrite: list[bool],
    ) -> None:
        """Append new block IDs to the block tables.

        This is a PyTorch-based implementation that replaces the Triton kernel.
        """
        num_reqs = len(req_indices)

        for group_id in range(self.num_kv_cache_groups):
            group_cu_blocks = cu_num_new_blocks[group_id]
            group_new_blocks = new_block_ids[group_id]

            for batch_idx in range(num_reqs):
                req_idx = req_indices[batch_idx]
                do_overwrite = overwrite[batch_idx]

                start_idx = group_cu_blocks[batch_idx]
                end_idx = group_cu_blocks[batch_idx + 1]
                num_new_blocks = end_idx - start_idx

                if num_new_blocks == 0:
                    continue

                # Get destination start index
                if do_overwrite:
                    dst_start_idx = 0
                else:
                    dst_start_idx = int(self.num_blocks[group_id, req_idx].item())

                dst_end_idx = dst_start_idx + num_new_blocks

                # Update num_blocks
                self.num_blocks[group_id, req_idx] = dst_end_idx

                # Copy block IDs to the block table
                block_ids_to_copy = group_new_blocks[start_idx:end_idx]
                self.block_tables[group_id][req_idx, dst_start_idx:dst_end_idx] = (
                    torch.tensor(
                        block_ids_to_copy, dtype=torch.int32, device=self.device
                    )
                )

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Gather block tables for the given request indices.

        This is a PyTorch-based implementation that replaces the Triton kernel.
        """
        num_reqs = idx_mapping.shape[0]

        for group_id in range(self.num_kv_cache_groups):
            for batch_idx in range(num_reqs):
                req_idx = int(idx_mapping[batch_idx].item())
                num_blocks = int(self.num_blocks[group_id, req_idx].item())

                # Copy the block table row
                self.input_block_tables[group_id][batch_idx, :num_blocks] = (
                    self.block_tables[group_id][req_idx, :num_blocks]
                )

        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute slot mappings for the given positions.

        This is a PyTorch-based implementation that replaces the Triton kernel.
        """
        num_reqs = query_start_loc.shape[0] - 1
        num_tokens = positions.shape[0]

        # Initialize slot mappings to PAD_SLOT_ID
        self.slot_mappings.fill_(PAD_SLOT_ID)

        for group_id in range(self.num_kv_cache_groups):
            page_size = self.block_sizes[group_id]
            block_table = self.input_block_tables[group_id]

            for req_idx in range(num_reqs):
                start_idx = int(query_start_loc[req_idx].item())
                end_idx = int(query_start_loc[req_idx + 1].item())

                if start_idx >= end_idx:
                    continue

                # Get positions for this request
                req_positions = positions[start_idx:end_idx]

                # Compute block indices and offsets
                block_indices = req_positions // page_size
                block_offsets = req_positions % page_size

                # Get block numbers from block table
                # We need to handle this carefully to avoid index errors
                max_block_idx = block_table.shape[1]
                block_indices_clamped = torch.clamp(block_indices, 0, max_block_idx - 1)

                # Gather block numbers for this request
                block_numbers = torch.zeros_like(block_indices, dtype=torch.int64)
                for i, block_idx in enumerate(block_indices_clamped):
                    block_numbers[i] = block_table[req_idx, int(block_idx.item())]

                # Compute slot IDs
                slot_ids = block_numbers * page_size + block_offsets

                # Store in slot_mappings
                self.slot_mappings[group_id, start_idx:end_idx] = slot_ids

        return self.slot_mappings[:, :num_tokens]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]
