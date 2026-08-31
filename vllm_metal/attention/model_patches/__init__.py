# SPDX-License-Identifier: Apache-2.0
"""Model-specific contracts for paged attention."""

from vllm_metal.attention.model_patches.base import (
    DEFAULT_ATTENTION_CONTRACT,
    AttentionContract,
    QKNormPlacement,
    register_attention_contract,
    resolve_attention_contract,
)
from vllm_metal.attention.model_patches.hunyuan_v1_dense import (
    register as _register_hunyuan_v1_dense,
)

_register_hunyuan_v1_dense()

__all__ = [
    "DEFAULT_ATTENTION_CONTRACT",
    "AttentionContract",
    "QKNormPlacement",
    "register_attention_contract",
    "resolve_attention_contract",
]
