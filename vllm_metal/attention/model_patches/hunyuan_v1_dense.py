# SPDX-License-Identifier: Apache-2.0
"""Paged-attention contract for Hunyuan dense models."""

from vllm_metal.attention.model_patches.base import (
    AttentionContract,
    QKNormPlacement,
    register_attention_contract,
)


def register() -> None:
    register_attention_contract(
        "mlx_lm.models.hunyuan_v1_dense",
        AttentionContract(qk_norm_placement=QKNormPlacement.AFTER_ROPE),
    )
