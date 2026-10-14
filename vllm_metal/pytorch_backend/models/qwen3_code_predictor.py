# SPDX-License-Identifier: Apache-2.0
"""External implementations for Omni's native predictor operators."""

from functools import cache

from vllm_metal.pytorch_backend.normalization import rms_norm


@cache
def install():
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
        _RMSNorm,
        _RotaryEmbedding,
    )

    @_RMSNorm.register_oot
    class MPSPredictorRMSNorm(_RMSNorm):
        def forward_native(self, x):
            if x.device.type == "mps" and x.dtype == self.weight.dtype:
                return rms_norm(x, self.weight, self.variance_epsilon)
            return super().forward_native(x)

    @_RotaryEmbedding.register_oot
    class MPSPredictorRotaryEmbedding(_RotaryEmbedding):
        def forward_native(self, x, position_ids):
            if x.device.type == "mps":
                return self._lookup(x, position_ids)
            return super().forward_native(x, position_ids)
