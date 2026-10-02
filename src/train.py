"""
SimCSE training pipeline for Unsupervised and Supervised modes (U2T02).
Supports:
  - Unsupervised training (with independent dropout masks)
  - Unsupervised ablation (identical dropout mask)
  - Supervised training (with hard negatives)
  - Supervised ablation (hard negatives OFF)
  - Checkpoint tracking via STS-B Dev Spearman
  - Hardware, seed, and metric logging in JSON format
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader import (
    UnsupervisedSimCSEDataset,
    SupervisedSimCSEDataset,
    load_snli_100k,
    load_stsb_data,
)
from src.models import SimCSE
from src.evaluate import evaluate_stsb, analyze_similarity_distribution


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def collate_unsupervised(batch_sentences: List[str], tokenizer, max_length: int = 64):
    return tokenizer(
        batch_sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


def collate_supervised(batch_items: List[Dict], tokenizer, max_length: int = 64):
    premises = [item["premise"] for item in batch_items]
    entailments = [item["entailment"] for item in batch_items]
    contradictions = [item["contradiction"] for item in batch_items]

    premise_inputs = tokenizer(premises, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    entailment_inputs = tokenizer(entailments, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    contra_inputs = None
    if any(c is not None for c in contradictions):
        # Fallback to entailment if contradiction is missing for that particular premise
        clean_contras = [c if c is not None else entailments[idx] for idx, c in enumerate(contradictions)]
        contra_inputs = tokenizer(clean_contras, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return premise_inputs, entailment_inputs, contra_inputs


def main():
    parser = argparse.ArgumentParser(description="SimCSE Training (U2T02)")
    parser.add_argument("--mode", type=str, choices=["unsupervised", "supervised"], required=True)
    parser.add_argument("--data_path", type=str, default="snli_train_100k.jsonl")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
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
    parser.add_argument("--device", type=str, default=None)

    # Ablations
    parser.add_argument("--same_dropout_ablation", action="store_true", help="Unsupervised ablation: same dropout mask for both views")
    parser.add_argument("--no_hard_negatives", action="store_true", help="Supervised ablation: disable hard negatives")

    args = parser.parse_args()

    # Determine run name
    if args.run_name is None:
        ablation_str = ""
        if args.mode == "unsupervised" and args.same_dropout_ablation:
            ablation_str = "_same_dropout_ablation"
        elif args.mode == "supervised" and args.no_hard_negatives:
            ablation_str = "_no_hard_neg_ablation"
        args.run_name = f"simcse_{args.mode}{ablation_str}_seed{args.seed}"

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed)

    # Device selection
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"==================================================")
    print(f"Starting SimCSE Run: {args.run_name}")
    print(f"Mode: {args.mode} | Device: {device} | Seed: {args.seed}")
    print(f"Learning rate: {args.lr} | Batch size: {args.batch_size} | Temp: {args.temperature}")
    print(f"==================================================")

    # Load Data
    unsupervised_sentences, supervised_items = load_snli_100k(args.data_path)
    dev_stsb, test_stsb = load_stsb_data()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build Dataloader
    if args.mode == "unsupervised":
        dataset = UnsupervisedSimCSEDataset(unsupervised_sentences)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_unsupervised(b, tokenizer, args.max_length),
            drop_last=True
        )
    else:
        dataset = SupervisedSimCSEDataset(supervised_items, use_hard_negatives=not args.no_hard_negatives)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_supervised(b, tokenizer, args.max_length),
            drop_last=True
        )

    # Initialize Model
    model = SimCSE(
        model_name_or_path=args.model_name,
        pooling=args.pooling,
        temperature=args.temperature,
        dropout_rate=args.dropout,
        use_mlp=True
    ).to(device)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

    # Evaluate raw baseline first on Dev
    print("\nEvaluating initial checkpoint (epoch 0) on STS-B Dev...")
    initial_eval = evaluate_stsb(model, tokenizer, dev_stsb, batch_size=args.batch_size, device=device, pooling=args.pooling)
    print(f"Initial Dev Spearman: {initial_eval['spearman']:.2f}")

    best_dev_spearman = -1.0
    best_step = 0
    best_model_path = os.path.join(run_dir, "best_checkpoint")
    training_logs = []

    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            if args.mode == "unsupervised":
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _ = model.forward_unsupervised(batch, same_dropout_ablation=args.same_dropout_ablation)
            else:
                p_in, ent_in, contra_in = batch
                p_in = {k: v.to(device) for k, v in p_in.items()}
                ent_in = {k: v.to(device) for k, v in ent_in.items()}
                if contra_in is not None:
                    contra_in = {k: v.to(device) for k, v in contra_in.items()}
                loss, _ = model.forward_supervised(
                    p_in,
                    ent_in,
                    contra_in,
                    use_hard_negatives=not args.no_hard_negatives
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % args.eval_steps == 0 or global_step == total_steps:
                avg_loss = total_loss / (step + 1)
                dev_eval = evaluate_stsb(model, tokenizer, dev_stsb, batch_size=args.batch_size, device=device, pooling=args.pooling)
                dev_spearman = dev_eval["spearman"]

                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "dev_spearman": dev_spearman,
                    "dev_alignment": dev_eval.get("alignment"),
                    "dev_uniformity": dev_eval.get("uniformity")
                }
                training_logs.append(log_entry)

                print(f"[Step {global_step}/{total_steps}] Loss: {avg_loss:.4f} | Dev Spearman: {dev_spearman:.2f} "
                      f"(Best: {best_dev_spearman:.2f})")

                if dev_spearman > best_dev_spearman:
                    best_dev_spearman = dev_spearman
                    best_step = global_step
                    # Save best checkpoint
                    os.makedirs(best_model_path, exist_ok=True)
                    model.encoder.save_pretrained(best_model_path)
                    tokenizer.save_pretrained(best_model_path)
                    print(f"--> Saved new best checkpoint at step {best_step} to {best_model_path}")

                model.train()

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.2f}s. Best Dev Spearman: {best_dev_spearman:.2f} at step {best_step}")

    # Evaluate best model on Test split EXACTLY ONCE
    print("\n--- Evaluating Best Model on STS-B Test Split (Executed Exactly Once) ---")
    best_model = SimCSE(
        model_name_or_path=best_model_path,
        pooling=args.pooling,
        use_mlp=False
    ).to(device)

    test_eval = evaluate_stsb(best_model, tokenizer, test_stsb, batch_size=args.batch_size, device=device, pooling=args.pooling)
    test_dist = analyze_similarity_distribution(test_eval["cosine_similarities"], test_eval["ground_truth_scores"])

    print(f"Final STS-B Test Spearman: {test_eval['spearman']:.2f}")
    print(f"Final Test Alignment: {test_eval['alignment']:.4f}")
    print(f"Final Test Uniformity: {test_eval['uniformity']:.4f}")

    # Save complete run summary
    run_summary = {
        "run_name": args.run_name,
        "mode": args.mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else ("Apple Silicon MPS" if device == "mps" else "CPU")
        },
        "config": vars(args),
        "training_time_seconds": total_training_time,
        "best_dev_step": best_step,
        "best_dev_spearman": best_dev_spearman,
        "test_spearman": test_eval["spearman"],
        "test_alignment": test_eval["alignment"],
        "test_uniformity": test_eval["uniformity"],
        "similarity_distribution_by_score": test_dist,
        "training_logs": training_logs
    }

    summary_file = os.path.join(run_dir, "run_results.json")
    with open(summary_file, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Saved run summary to {summary_file}")


if __name__ == "__main__":
    main()
