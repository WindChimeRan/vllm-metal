"""Render the two PR figures from the refreshed measurements."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
colors = ("#a9afb8", "#2563eb")


def bars(ax, before, after, unit, batch_label, decimals):
    positions = np.arange(3)
    for shift, values, color, label in (
        (-.18, before, colors[0], "Baseline"),
        (.18, after, colors[1], "Shared Metal sampler"),
    ):
        group = ax.bar(positions + shift, values, .35, color=color, label=label)
        ax.bar_label(group, fmt=f"%.{decimals}f", padding=4, fontsize=10)
    ax.set_xticks(positions, [f"{batch_label}={b}" for b in (1, 4, 8)])
    ax.set_ylim(0, max(before + after) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.18)
    ax.set_ylabel(unit)


def finish(fig, axes, title, subtitle, footer, name):
    fig.suptitle(title, fontsize=18, y=.98)
    fig.text(.5, .927, subtitle, ha="center", fontsize=11, color="#444b55")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .90), ncol=2, frameon=False)
    fig.text(.5, .035, footer, ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(left=.085, right=.98, top=.80, bottom=.145, hspace=.35, wspace=.24)
    fig.savefig(root / (name + ".png"), dpi=180, facecolor="white")
    fig.savefig(root / (name + ".pdf"), facecolor="white")
    plt.close(fig)


serving_path = root / "serving-results.json"
if not serving_path.exists():
    serving_path = root / "end-to-end.json"
serving = json.loads(serving_path.read_text())
comparison = []
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, backend in enumerate(("mlx", "mps")):
    for column, case in enumerate(("plain-seeded", "filtered-seeded")):
        pair = []
        for variant in ("baseline", "shared"):
            values = [next(x for x in serving if x["variant"] == f"{backend}-{variant}"
                           and x["case"] == case and x["concurrency"] == c)
                      for c in (1, 4, 8)]
            pair.append([x["mean_tpot_ms"] for x in values])
        for c, before, after in zip((1, 4, 8), *pair, strict=True):
            comparison.append(dict(backend=backend, case=case, concurrency=c,
                                   baseline_ms=before, shared_ms=after, speedup=before / after))
        ax = axes[row, column]
        bars(ax, *pair, "Time per output token (ms)", "C", 2)
        backend_label = "MLX" if backend == "mlx" else "PyTorch-MPS"
        case_label = "Unfiltered" if case == "plain-seeded" else "Top-k 50 / top-p 0.9"
        ax.set_title(f"{backend_label} · {case_label}", pad=10, fontsize=12)
finish(
    fig, axes, "Qwen3-0.6B serving latency", "Model Runner V2 in both arms · lower is better",
    "M5 Pro · 32 requests × 128 input / 128 output tokens · temperature 0.8, seed 42\n"
    "One run per setting after 8 warmups. Baseline = PR base (MLX), matched PyTorch-operator reference (MPS).",
    "end-to-end",
)

micro_path = root / "micro-results.json"
if not micro_path.exists():
    micro_path = root / "microbenchmark.json"
micro = json.loads(micro_path.read_text())
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, backend in enumerate(("mlx", "mps")):
    for column, vocab in enumerate((6562, 152064)):
        values = [next(x for x in micro if x["backend"] == backend
                       and x["vocab"] == vocab and x["batch"] == b) for b in (1, 4, 8)]
        ax = axes[row, column]
        bars(ax, [x["baseline_us"] for x in values], [x["shared_us"] for x in values],
             "Draw latency (µs)", "B", 0)
        backend_label = "MLX" if backend == "mlx" else "PyTorch-MPS"
        ax.set_title(f"{backend_label} · {vocab:,} vocabulary entries", pad=10, fontsize=12)
finish(
    fig, axes, "Sampling draw latency", "Speech-sized and text-sized inputs · lower is better",
    "M5 Pro · fresh FP32 logits on GPU · ≥0.5 s warmup + 200 draws · synchronized per draw\n"
    "Includes dispatch, allocation and reduction. Model execution, filters and key derivation are excluded.",
    "microbenchmark",
)
(root / "serving-comparison.json").write_text(json.dumps(comparison, indent=2))
print("Figures written")
