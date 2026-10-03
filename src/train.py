"""
Training entry point for U2T02 SimCSE.

Every run writes:
- config.json
- training_log.jsonl
- run_results.json
- best_checkpoint/
- retrieval_analysis.json

The best checkpoint is selected only with STS-B dev Spearman. Test is evaluated
once after training for that run, so it never influences checkpoint selection.
"""

import argparse
import json
import os
import platform
import random
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader import (
    SupervisedSimCSEDataset,
    UnsupervisedSimCSEDataset,
    load_snli_100k,
    load_stsb_data,
)
from src.evaluate import (
    analyze_similarity_distribution,
    build_retrieval_analysis,
    evaluate_stsb,
)
from src.models import SimCSE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_info(device: str) -> Dict:
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
    elif device == "mps":
        name = "Apple Silicon MPS"
    else:
        name = platform.processor() or "CPU"

    return {
        "device": device,
        "device_name": name,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def collate_unsupervised(
    batch_sentences: List[str],
    tokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        batch_sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def collate_supervised(
    batch_items: List[Dict],
    tokenizer,
    max_length: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    premises = [item["premise"] for item in batch_items]
    entailments = [item["entailment"] for item in batch_items]
    real_contradictions = [
        item["contradiction"]
        for item in batch_items
        if item.get("contradiction") is not None
    ]

    premise_inputs = tokenizer(
        premises,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    entailment_inputs = tokenizer(
        entailments,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    contradiction_inputs = None
    if real_contradictions:
        contradiction_inputs = tokenizer(
            real_contradictions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return premise_inputs, entailment_inputs, contradiction_inputs


def move_to_device(batch: Optional[Dict[str, torch.Tensor]], device: str):
    if batch is None:
        return None
    return {key: value.to(device) for key, value in batch.items()}


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def append_jsonl(path: str, payload: Dict) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimCSE for U2T02")
    parser.add_argument("--mode", choices=["unsupervised", "supervised"], required=True)
    parser.add_argument("--data_path", default="snli_train_100k.jsonl")
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--same_dropout_ablation", action="store_true")
    parser.add_argument("--no_hard_negatives", action="store_true")
    parser.add_argument("--retrieval_queries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    if args.run_name is None:
        suffix = ""
        if args.mode == "unsupervised" and args.same_dropout_ablation:
            suffix = "_same_dropout"
        if args.mode == "supervised" and args.no_hard_negatives:
            suffix = "_no_hard_negatives"
        args.run_name = f"simcse_{args.mode}{suffix}_seed{args.seed}"

    run_dir = os.path.join(args.output_dir, args.run_name)
    best_checkpoint_dir = os.path.join(run_dir, "best_checkpoint")
    os.makedirs(run_dir, exist_ok=True)

    config = vars(args).copy()
    config["hardware"] = hardware_info(device)
    write_json(os.path.join(run_dir, "config.json"), config)

    unsupervised_sentences, supervised_items = load_snli_100k(args.data_path)
    dev_stsb, test_stsb = load_stsb_data()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.mode == "unsupervised":
        dataset = UnsupervisedSimCSEDataset(unsupervised_sentences)
        collate_fn = partial(
            collate_unsupervised,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
    else:
        dataset = SupervisedSimCSEDataset(
            supervised_items,
            use_hard_negatives=not args.no_hard_negatives,
        )
        collate_fn = partial(
            collate_supervised,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = SimCSE(
        model_name_or_path=args.model_name,
        pooling=args.pooling,
        temperature=args.temperature,
        dropout_rate=args.dropout,
        use_mlp=True,
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    grouped_parameters = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(term in name for term in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(term in name for term in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_parameters, lr=args.lr)

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("=" * 72)
    print(f"Run: {args.run_name}")
    print(f"Mode: {args.mode} | Device: {device} | Seed: {args.seed}")
    print(f"Examples: {len(dataset):,} | Steps: {total_steps:,}")
    print("=" * 72)

    initial_dev = evaluate_stsb(
        model,
        tokenizer,
        dev_stsb,
        batch_size=args.batch_size,
        device=device,
        pooling=args.pooling,
        max_length=args.max_length,
    )
    print(f"Initial dev Spearman: {initial_dev['spearman']:.2f}")

    best_dev_spearman = float("-inf")
    best_step = None
    global_step = 0
    log_path = os.path.join(run_dir, "training_log.jsonl")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for local_step, batch in enumerate(dataloader, start=1):
            optimizer.zero_grad(set_to_none=True)

            if args.mode == "unsupervised":
                batch = move_to_device(batch, device)
                loss, _ = model.forward_unsupervised(
                    batch,
                    same_dropout_ablation=args.same_dropout_ablation,
                )
                hard_negative_count = 0
            else:
                premise_inputs, entailment_inputs, contradiction_inputs = batch
                premise_inputs = move_to_device(premise_inputs, device)
                entailment_inputs = move_to_device(entailment_inputs, device)
                contradiction_inputs = move_to_device(contradiction_inputs, device)
                hard_negative_count = (
                    contradiction_inputs["input_ids"].shape[0]
                    if contradiction_inputs is not None
                    else 0
                )
                loss, _ = model.forward_supervised(
                    premise_inputs,
                    entailment_inputs,
                    contradiction_inputs,
                    use_hard_negatives=not args.no_hard_negatives,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % args.eval_steps == 0 or global_step == total_steps:
                dev_result = evaluate_stsb(
                    model,
                    tokenizer,
                    dev_stsb,
                    batch_size=args.batch_size,
                    device=device,
                    pooling=args.pooling,
                    max_length=args.max_length,
                )
                record = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss_running": epoch_loss / local_step,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "dev_spearman": dev_result["spearman"],
                    "dev_alignment": dev_result["alignment"],
                    "dev_uniformity": dev_result["uniformity"],
                    "hard_negatives_in_last_batch": hard_negative_count,
                }
                append_jsonl(log_path, record)

                print(
                    f"[{global_step:>5}/{total_steps}] "
                    f"loss={record['train_loss_running']:.4f} "
                    f"dev={record['dev_spearman']:.2f}"
                )

                if dev_result["spearman"] > best_dev_spearman:
                    best_dev_spearman = dev_result["spearman"]
                    best_step = global_step
                    os.makedirs(best_checkpoint_dir, exist_ok=True)
                    model.encoder.save_pretrained(best_checkpoint_dir)
                    tokenizer.save_pretrained(best_checkpoint_dir)
                    checkpoint_metadata = {
                        "pooling": args.pooling,
                        "max_length": args.max_length,
                        "temperature": args.temperature,
                        "dropout": args.dropout,
                        "mode": args.mode,
                        "best_dev_spearman": best_dev_spearman,
                        "best_step": best_step,
                        "mlp_policy": "train_only",
                    }
                    write_json(
                        os.path.join(best_checkpoint_dir, "simcse_config.json"),
                        checkpoint_metadata,
                    )

                model.train()

    if best_step is None:
        raise RuntimeError("No checkpoint was saved. Check eval_steps and training data.")

    elapsed = time.time() - start_time

    # Reload the exact selected encoder and perform the final test evaluation once.
    best_model = SimCSE(
        model_name_or_path=best_checkpoint_dir,
        pooling=args.pooling,
        temperature=args.temperature,
        dropout_rate=args.dropout,
        use_mlp=False,
    ).to(device)

    test_result = evaluate_stsb(
        best_model,
        tokenizer,
        test_stsb,
        batch_size=args.batch_size,
        device=device,
        pooling=args.pooling,
        max_length=args.max_length,
    )
    distribution = analyze_similarity_distribution(
        test_result["cosine_similarities"],
        test_result["ground_truth_scores"],
    )
    retrieval = build_retrieval_analysis(
        best_model,
        tokenizer,
        test_stsb,
        batch_size=args.batch_size,
        device=device,
        pooling=args.pooling,
        max_length=args.max_length,
        query_count=args.retrieval_queries,
    )
    write_json(os.path.join(run_dir, "retrieval_analysis.json"), retrieval)

    result_summary = {
        "run_name": args.run_name,
        "mode": args.mode,
        "seed": args.seed,
        "hardware": hardware_info(device),
        "config": vars(args),
        "dataset": {
            "training_examples": len(dataset),
            "unsupervised_unique_sentences": len(unsupervised_sentences),
            "supervised_entailment_pairs": len(supervised_items),
            "supervised_pairs_with_hard_negative": sum(
                item["contradiction"] is not None for item in supervised_items
            ),
        },
        "training_time_seconds": elapsed,
        "best_dev_step": best_step,
        "best_dev_spearman": best_dev_spearman,
        "test_spearman": test_result["spearman"],
        "test_alignment": test_result["alignment"],
        "test_uniformity": test_result["uniformity"],
        "similarity_distribution_by_human_score": distribution,
        "retrieval_analysis_file": "retrieval_analysis.json",
        "checkpoint": "best_checkpoint",
    }
    write_json(os.path.join(run_dir, "run_results.json"), result_summary)

    registry_entry = {
        "run_name": args.run_name,
        "mode": args.mode,
        "seed": args.seed,
        "best_dev_spearman": best_dev_spearman,
        "test_spearman": test_result["spearman"],
        "test_alignment": test_result["alignment"],
        "test_uniformity": test_result["uniformity"],
        "same_dropout_ablation": args.same_dropout_ablation,
        "hard_negatives_enabled": not args.no_hard_negatives,
        "training_time_seconds": elapsed,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    append_jsonl(os.path.join(args.output_dir, "run_registry.jsonl"), registry_entry)

    print("\nFinal selected checkpoint")
    print(f"Dev Spearman:  {best_dev_spearman:.2f}")
    print(f"Test Spearman: {test_result['spearman']:.2f}")
    print(f"Alignment:     {test_result['alignment']:.4f}")
    print(f"Uniformity:    {test_result['uniformity']:.4f}")
    print(f"Artifacts:     {run_dir}")


if __name__ == "__main__":
    main()
