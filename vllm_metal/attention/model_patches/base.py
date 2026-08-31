# SPDX-License-Identifier: Apache-2.0
"""Declarative model contracts for paged attention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class QKNormPlacement(Enum):
    """Placement of per-head Q/K normalization relative to RoPE."""

    BEFORE_ROPE = auto()
    AFTER_ROPE = auto()


@dataclass(frozen=True, slots=True)
class AttentionContract:
    """Architecture-specific behavior consumed by the paged SDPA path."""

    qk_norm_placement: QKNormPlacement = QKNormPlacement.BEFORE_ROPE


DEFAULT_ATTENTION_CONTRACT = AttentionContract()
_CONTRACTS: dict[str, AttentionContract] = {}


def register_attention_contract(module_name: str, contract: AttentionContract) -> None:
    """Register the contract for an MLX-LM attention module."""
    _CONTRACTS[module_name] = contract


def resolve_attention_contract(module: object) -> AttentionContract:
    """Resolve an attention contract, falling back to the standard order."""
    return _CONTRACTS.get(type(module).__module__, DEFAULT_ATTENTION_CONTRACT)
