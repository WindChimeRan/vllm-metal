# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible gumbel sampling using PyTorch instead of Triton kernels."""

import torch


def gumbel_sample(
    logits: torch.Tensor,  # [num_reqs, vocab_size]
    temperature: torch.Tensor,  # [num_reqs]
    seed: torch.Tensor,  # [num_reqs]
    pos: torch.Tensor,  # [num_reqs]
    apply_temperature: bool,
) -> torch.Tensor:
    """PyTorch implementation of gumbel_sample.

    Implements Gumbel-max trick for sampling from categorical distributions.
    For each request, adds Gumbel noise to logits and returns the argmax.
    """
    num_reqs, vocab_size = logits.shape
    device = logits.device

    # Work in float32 for numerical stability
    logits = logits.float()
    sampled = torch.empty(num_reqs, dtype=torch.int64, device=device)

    for i in range(num_reqs):
        temp = temperature[i].item()

        if temp == 0.0:
            # Greedy sampling - just take argmax
            sampled[i] = logits[i].argmax()
        else:
            # Apply temperature if requested
            req_logits = logits[i].clone()
            if apply_temperature:
                req_logits = req_logits / temp

            # Generate Gumbel noise using inverse CDF method
            # Gumbel(0, 1) = -log(-log(U)) where U ~ Uniform(0, 1)
            # Use seeded random for reproducibility
            # Create a generator seeded with the request's seed + position
            seed_val = int(seed[i].item())
            pos_val = int(pos[i].item())
            # Combine seed and position to get a unique seed for this sample
            combined_seed = (seed_val * 2654435761 + pos_val) & 0xFFFFFFFF

            # Generate uniform random numbers
            # Note: MPS doesn't support float64, so use float32 throughout
            generator = torch.Generator(device="cpu")
            generator.manual_seed(combined_seed)
            u = torch.rand(vocab_size, generator=generator, dtype=torch.float32)
            # Move to device and compute Gumbel noise
            u = u.to(device)
            # Add small epsilon to prevent log(0)
            gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)

            # Add Gumbel noise to logits and take argmax
            noisy_logits = req_logits + gumbel_noise
            sampled[i] = noisy_logits.argmax()

    return sampled
