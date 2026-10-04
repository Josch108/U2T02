"""
Generate report-ready artifacts from completed U2T02 runs.

Usage:
    python -m src.make_report_artifacts --runs_dir runs --output_dir report/artifacts

The script never invents missing experimental results.
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


MAIN_RUNS = [
    "simcse_unsupervised_seed42",
    "simcse_supervised_seed42",
]

ABLATION_RUNS = [
    "simcse_unsupervised_same_dropout_seed42",
    "simcse_supervised_no_hard_negatives_seed42",
]

ALL_RUNS = MAIN_RUNS + ABLATION_RUNS


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_runs(runs_dir: str) -> Dict[str, Dict]:
    runs: Dict[str, Dict] = {}
    for run_name in ALL_RUNS:
        result_path = os.path.join(runs_dir, run_name, "run_results.json")
        if os.path.exists(result_path):
            runs[run_name] = load_json(result_path)
    return runs


def format_run_row(label: str, result: Dict) -> str:
    return (
        f"| {label} | {result['best_dev_spearman']:.2f} | "
        f"{result['test_spearman']:.2f} | {result['test_alignment']:.4f} | "
        f"{result['test_uniformity']:.4f} |"
    )


def format_baseline_row(label: str, result: Dict) -> str:
    required = ("dev_spearman", "test_spearman", "test_alignment", "test_uniformity")
    if not all(key in result for key in required):
        return f"| {label} | MISSING | MISSING | MISSING | MISSING |"
    return (
        f"| {label} | {result['dev_spearman']:.2f} | "
        f"{result['test_spearman']:.2f} | {result['test_alignment']:.4f} | "
        f"{result['test_uniformity']:.4f} |"
    )


def write_benchmark_markdown(
    runs: Dict[str, Dict],
    baselines: Dict[str, Dict],
    output_path: str,
) -> None:
    lines: List[str] = [
        "| Model | Dev Spearman | Test Spearman | Alignment | Uniformity |",
        "|---|---:|---:|---:|---:|",
    ]

    raw = baselines.get("raw_bert")
    sbert = baselines.get("sbert_2019")
    lines.append(
        format_baseline_row("Raw bert-base-uncased (mean)", raw)
        if raw
        else "| Raw bert-base-uncased (mean) | MISSING | MISSING | MISSING | MISSING |"
    )
    lines.append(
        format_baseline_row("SBERT-2019", sbert)
        if sbert
        else "| SBERT-2019 | MISSING | MISSING | MISSING | MISSING |"
    )

    unsup = runs.get("simcse_unsupervised_seed42")
    sup = runs.get("simcse_supervised_seed42")
    lines.append(
        format_run_row("SimCSE Unsupervised (ours)", unsup)
        if unsup
        else "| SimCSE Unsupervised (ours) | MISSING | MISSING | MISSING | MISSING |"
    )
    lines.append(
        format_run_row("SimCSE Supervised (ours)", sup)
        if sup
        else "| SimCSE Supervised (ours) | MISSING | MISSING | MISSING | MISSING |"
    )

    # Paper geometry is not reported under our exact estimator, so it is
    # deliberately shown as unavailable rather than fabricated.
    lines.append("| SimCSE Unsupervised (paper) | 82.50 | 76.85 | — | — |")
    lines.append("| SimCSE Supervised (paper) | 86.20 | 84.25 | — | — |")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def plot_similarity_distribution(result: Dict, output_path: str) -> None:
    distribution = result["similarity_distribution_by_human_score"]
    labels = list(distribution.keys())
    means = [distribution[label]["mean"] for label in labels]
    stds = [distribution[label]["std"] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(labels, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("STS-B human score bin")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine Similarity by Human Similarity Rating")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_ablation_summary(runs: Dict[str, Dict], output_path: str) -> None:
    pairs = [
        (
            "Unsupervised dropout ablation",
            "simcse_unsupervised_seed42",
            "simcse_unsupervised_same_dropout_seed42",
        ),
        (
            "Supervised hard-negative ablation",
            "simcse_supervised_seed42",
            "simcse_supervised_no_hard_negatives_seed42",
        ),
    ]

    lines = ["# Ablation deltas", ""]

    for title, control_name, ablation_name in pairs:
        lines.append(f"## {title}")
        control = runs.get(control_name)
        ablation = runs.get(ablation_name)

        if control is None or ablation is None:
            lines.append("MISSING: both completed runs are required.")
            lines.append("")
            continue

        dev_delta = ablation["best_dev_spearman"] - control["best_dev_spearman"]
        test_delta = ablation["test_spearman"] - control["test_spearman"]

        lines.extend(
            [
                f"- Control dev Spearman: {control['best_dev_spearman']:.2f}",
                f"- Ablation dev Spearman: {ablation['best_dev_spearman']:.2f}",
                f"- Dev delta (ablation - control): {dev_delta:+.2f}",
                f"- Control test Spearman: {control['test_spearman']:.2f}",
                f"- Ablation test Spearman: {ablation['test_spearman']:.2f}",
                f"- Test delta (ablation - control): {test_delta:+.2f}",
                "- Noise interpretation: compare these deltas with the expected ±1–3 "
                "point seed variation stated in the assignment; do not claim statistical "
                "significance from one seed.",
                "",
            ]
        )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--output_dir", default="report/artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = load_runs(args.runs_dir)

    baseline_path = os.path.join(args.output_dir, "baselines.json")
    baselines = load_json(baseline_path) if os.path.exists(baseline_path) else {}

    write_benchmark_markdown(
        runs,
        baselines,
        os.path.join(args.output_dir, "benchmark_generated.md"),
    )
    write_ablation_summary(
        runs,
        os.path.join(args.output_dir, "ablation_deltas_generated.md"),
    )

    for run_name, result in runs.items():
        plot_similarity_distribution(
            result,
            os.path.join(args.output_dir, f"{run_name}_similarity_distribution.png"),
        )

    missing = [run for run in ALL_RUNS if run not in runs]
    if missing:
        print("Missing runs:")
        for run in missing:
            print(f"  - {run}")
    else:
        print("All four required SimCSE runs were found.")

    if not baselines:
        print(
            "Baseline geometry is missing. Run: "
            "python -m src.verify_baselines --model both --include_test"
        )

    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
