"""
Generate report-ready artifacts from completed U2T02 runs.

Usage:
    python -m src.make_report_artifacts --runs_dir runs --output_dir report/artifacts

The script never invents missing results. If a required run is absent, the
benchmark marks it as MISSING.
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


EXPECTED_RUNS = [
    "simcse_unsupervised_seed42",
    "simcse_unsupervised_same_dropout_seed42",
    "simcse_supervised_seed42",
    "simcse_supervised_no_hard_negatives_seed42",
]


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_runs(runs_dir: str) -> Dict[str, Dict]:
    runs: Dict[str, Dict] = {}
    for run_name in EXPECTED_RUNS:
        result_path = os.path.join(runs_dir, run_name, "run_results.json")
        if os.path.exists(result_path):
            runs[run_name] = load_json(result_path)
    return runs


def write_benchmark_markdown(runs: Dict[str, Dict], output_path: str) -> None:
    labels = {
        "simcse_unsupervised_seed42": "SimCSE Unsupervised",
        "simcse_unsupervised_same_dropout_seed42": "Unsupervised: same dropout mask",
        "simcse_supervised_seed42": "SimCSE Supervised",
        "simcse_supervised_no_hard_negatives_seed42": "Supervised: hard negatives OFF",
    }

    lines: List[str] = [
        "| Model / run | Dev Spearman | Test Spearman | Alignment | Uniformity |",
        "|---|---:|---:|---:|---:|",
    ]

    for run_name in EXPECTED_RUNS:
        label = labels[run_name]
        result = runs.get(run_name)
        if result is None:
            lines.append(f"| {label} | MISSING | MISSING | MISSING | MISSING |")
            continue

        lines.append(
            "| {label} | {dev:.2f} | {test:.2f} | {align:.4f} | {uniform:.4f} |".format(
                label=label,
                dev=result["best_dev_spearman"],
                test=result["test_spearman"],
                align=result["test_alignment"],
                uniform=result["test_uniformity"],
            )
        )

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

    write_benchmark_markdown(
        runs,
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

    missing = [run for run in EXPECTED_RUNS if run not in runs]
    if missing:
        print("Missing runs:")
        for run in missing:
            print(f"  - {run}")
    else:
        print("All four required runs were found.")

    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
