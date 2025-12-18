# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible async utilities using MPS instead of CUDA streams."""

import torch
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplerOutput,
)


class MetalAsyncOutput(AsyncModelRunnerOutput):
    """Metal-compatible AsyncOutput that uses synchronous copies.

    On Metal/MPS, we use synchronous copies instead of CUDA async streams.
    MPS has unified memory, so copies are relatively fast.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        copy_stream,  # Ignored on Metal
        copy_event,  # Ignored on Metal
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens

        # On Metal, we do synchronous copies (no async streams)
        # MPS uses unified memory so this is relatively efficient
        torch.mps.synchronize()  # Ensure GPU operations complete

        # Copy tensors to CPU synchronously
        self.sampled_token_ids = sampler_output.sampled_token_ids.to("cpu")

        if sampler_output.logprobs_tensors is not None:
            self.logprobs_tensors: LogprobsTensors | None = (
                sampler_output.logprobs_tensors.to_cpu_nonblocking()
            )
        else:
            self.logprobs_tensors = None

        self.num_sampled_tokens_cpu = num_sampled_tokens.to("cpu")

        self.prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}
        if self.model_runner_output.prompt_logprobs_dict:
            for k, v in self.model_runner_output.prompt_logprobs_dict.items():
                if v is not None:
                    self.prompt_logprobs_dict[k] = v.to_cpu_nonblocking()
                else:
                    self.prompt_logprobs_dict[k] = None

    def get_output(self) -> ModelRunnerOutput:
        # On Metal, copies are synchronous so no need to wait
        num_sampled_tokens_np = self.num_sampled_tokens_cpu.numpy()

        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_reqs = len(sampled_token_ids)
        for i in range(num_reqs):
            del sampled_token_ids[i][num_sampled_tokens_np[i] :]

        self.model_runner_output.sampled_token_ids = sampled_token_ids
        self.model_runner_output.logprobs_tensors = self.logprobs_tensors
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output
