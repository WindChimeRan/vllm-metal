# SPDX-License-Identifier: Apache-2.0
"""Quantize a self-contained EAGLE3 speculator's projections for MLX.

Token embeddings and normalization weights stay dense. This tool writes a
local checkpoint; it does not modify or upload the source model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--target", help="Target model; defaults to the checkpoint's verifier"
    )
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be an empty directory")

    import mlx.core as mx
    import mlx.nn as nn
    from huggingface_hub import hf_hub_download, snapshot_download
    from mlx.utils import tree_flatten

    from vllm_metal.v1.eagle3 import Eagle3Model

    source = Path(args.model)
    if not source.is_dir():
        source = Path(
            snapshot_download(
                args.model,
                revision=args.revision,
                allow_patterns=["config.json", "*.safetensors"],
            )
        )
    config = json.loads((source / "config.json").read_text())
    if config.get("speculators_model_type") != "eagle3":
        parser.error("Use a self-contained EAGLE3 speculators checkpoint")
    if config.get("quantization") is not None:
        parser.error("The source checkpoint is already quantized")
    target = args.target or config["speculators_config"]["verifier"]["name_or_path"]
    target_config = Path(target) / "config.json"
    if not target_config.is_file():
        target_config = Path(hf_hub_download(target, "config.json"))
    dtype = getattr(mx, args.dtype)
    model = Eagle3Model.load(str(source), json.loads(target_config.read_text()), dtype)
    nn.quantize(
        model,
        bits=args.bits,
        group_size=args.group_size,
        class_predicate=lambda _, module: isinstance(module, nn.Linear),
    )
    mx.eval(model.parameters())
    config["quantization"] = {
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": "affine",
    }
    config["dtype"] = args.dtype
    config["torch_dtype"] = args.dtype
    config["transformer_layer_config"]["dtype"] = args.dtype
    config["transformer_layer_config"]["torch_dtype"] = args.dtype
    config["conversion_source"] = {
        "model": args.model,
        "revision": args.revision or source.name,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output / "model.safetensors"),
        dict(tree_flatten(model.parameters())),
        metadata={"format": "mlx"},
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
