# SPDX-License-Identifier: Apache-2.0
"""Experimental backend selection and unsupported-feature guards."""

from types import SimpleNamespace

import pytest
import torch
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

from vllm_metal.platform import MetalPlatform
from vllm_metal.pytorch_backend.worker import configure_mps


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3", head_dim=128),
            quantization=None,
            runner_type="generate",
            dtype=torch.float16,
        ),
        parallel_config=SimpleNamespace(world_size=1, data_parallel_size=1),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        additional_config={},
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        compilation_config=SimpleNamespace(),
    )


def test_mps_configuration():
    c = config()
    configure_mps(c)
    assert c.model_config.enforce_eager
    assert not c.scheduler_config.async_scheduling
    assert c.cache_config.block_size == 16
    assert c.model_config.model_impl == "transformers"


@pytest.mark.parametrize("flag", ["VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_HW_AGNOSTIC"])
def test_mps_rejects_disabled_required_interface(monkeypatch, flag):
    monkeypatch.setenv("VLLM_METAL_BACKEND", "mps")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    monkeypatch.setenv(flag, "0")
    with pytest.raises(ValueError, match=flag):
        configure_mps(config())


@pytest.mark.parametrize(
    "field", ["speculative_config", "lora_config", "kv_transfer_config"]
)
def test_unsupported_mps_features(field):
    c = config()
    setattr(c, field, object())
    with pytest.raises(NotImplementedError, match="Experimental MPS"):
        configure_mps(c)


@pytest.mark.parametrize("model_type,head_dim", [("llama", 128), ("qwen3", 64)])
def test_unsupported_model(model_type, head_dim):
    c = config()
    c.model_config.hf_text_config = SimpleNamespace(
        model_type=model_type, head_dim=head_dim
    )
    with pytest.raises(NotImplementedError, match="Experimental MPS"):
        configure_mps(c)


def test_mps_attention_selection(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_BACKEND", "mps")
    assert MetalPlatform.get_attn_backend_cls(None, None).endswith(
        "MPSAttentionBackend"
    )


def test_mps_prompt_logprobs_rejected_before_execution(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_BACKEND", "mps")
    with pytest.raises(VLLMValidationError, match="prompt_logprobs"):
        MetalPlatform.validate_request(None, SamplingParams(prompt_logprobs=1))


def test_mps_accepts_upstream_min_tokens_and_logit_bias(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_BACKEND", "mps")
    MetalPlatform.validate_request(
        None, SamplingParams(min_tokens=2, max_tokens=8, logit_bias={1: 1.0})
    )
