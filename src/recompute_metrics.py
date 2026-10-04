"""
Recompute evaluation-only artifacts for completed U2T02 runs after fixing the
STS-B score scale (0-1 -> original 0-5).

This does NOT retrain models. It reloads each saved best_checkpoint, recomputes
STS-B test Spearman/alignment/uniformity, rating-bin distributions, and
retrieval examples, then updates run_results.json, retrieval_analysis.json,
and the global run registry.

Usage:
    python -m src.recompute_metrics --runs_dir runs
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from src.data_loader import load_stsb_data
from src.evaluate import (
    analyze_similarity_distribution,
    build_retrieval_analysis,
    evaluate_stsb,
)
from src.models import SimCSE


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    _, test_stsb = load_stsb_data()

    run_names = [
        "simcse_unsupervised_seed42",
        "simcse_unsupervised_same_dropout_seed42",
        "simcse_supervised_seed42",
        "simcse_supervised_no_hard_negatives_seed42",
    ]

    updated_results = {}

    for run_name in run_names:
        run_dir = os.path.join(args.runs_dir, run_name)
        result_path = os.path.join(run_dir, "run_results.json")
        checkpoint_dir = os.path.join(run_dir, "best_checkpoint")

        if not os.path.exists(result_path):
            print(f"SKIP missing results: {run_name}")
            continue
        if not os.path.exists(checkpoint_dir):
            print(f"SKIP missing checkpoint: {run_name}")
            continue

        result = load_json(result_path)
        config = result["config"]

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = SimCSE(
            model_name_or_path=checkpoint_dir,
            pooling=config["pooling"],
            temperature=config["temperature"],
            dropout_rate=config["dropout"],
            use_mlp=False,
        ).to(device)

        evaluation = evaluate_stsb(
            model,
            tokenizer,
            test_stsb,
            batch_size=args.batch_size,
            device=device,
            pooling=config["pooling"],
            max_length=config["max_length"],
        )

        distribution = analyze_similarity_distribution(
            evaluation["cosine_similarities"],
            evaluation["ground_truth_scores"],
        )
        retrieval = build_retrieval_analysis(
            model,
            tokenizer,
            test_stsb,
            batch_size=args.batch_size,
            device=device,
            pooling=config["pooling"],
            max_length=config["max_length"],
            query_count=config.get("retrieval_queries", 5),
        )

        old_test = result["test_spearman"]
        result["test_spearman"] = evaluation["spearman"]
        result["test_alignment"] = evaluation["alignment"]
        result["test_uniformity"] = evaluation["uniformity"]
        result["similarity_distribution_by_human_score"] = distribution
        result["evaluation_score_scale"] = "STS-B original 0-5 scale"
        result["metrics_recomputed_after_score_scale_fix"] = True

        write_json(result_path, result)
        write_json(os.path.join(run_dir, "retrieval_analysis.json"), retrieval)
        updated_results[run_name] = result

        print(
            f"{run_name}: "
            f"Spearman {old_test:.4f} -> {evaluation['spearman']:.4f} | "
            f"alignment={evaluation['alignment']:.4f} | "
            f"uniformity={evaluation['uniformity']:.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    registry_path = os.path.join(args.runs_dir, "run_registry.jsonl")
    if os.path.exists(registry_path):
        existing = [
            json.loads(line)
            for line in open(registry_path, "r", encoding="utf-8")
            if line.strip()
        ]
        rebuilt = []
        for row in existing:
            result = updated_results.get(row.get("run_name"))
            if result:
                row["test_spearman"] = result["test_spearman"]
                row["test_alignment"] = result["test_alignment"]
                row["test_uniformity"] = result["test_uniformity"]
                row["metrics_recomputed_after_score_scale_fix"] = True
            rebuilt.append(row)
        write_jsonl(registry_path, rebuilt)
        print(f"Updated registry: {registry_path}")

    note = {
        "reason": "STS-B source scores were normalized to 0-1 but geometry analysis expected the original 0-5 human scale.",
        "effect": "Spearman is unchanged by linear score scaling. Test alignment and human-rating distributions were recomputed. Uniformity is also regenerated from the same saved embeddings.",
        "training_logs": "Historical training_log.jsonl files are preserved. Their dev_spearman values are valid; any dev_alignment field written before this repair should not be used in the final report.",
    }
    write_json(os.path.join(args.runs_dir, "metric_repair_note.json"), note)


if __name__ == "__main__":
    main()
