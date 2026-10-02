"""
Data loading and preprocessing utilities for SimCSE (U2T02).
Handles:
  1. SNLI 100k subset (snli_train_100k.jsonl)
     - Unsupervised training set (unique sentences)
     - Supervised training set (premise, entailment, [contradiction])
  2. STS-B validation (dev: 1,500 pairs) and test (test: 1,379 pairs)
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class UnsupervisedSimCSEDataset(Dataset):
    """
    Dataset for unsupervised SimCSE.
    Each item is a single sentence. In the model forward pass, each sentence
    is passed through the encoder twice with independent dropout masks to create
    positive pairs (x_i, x_i^+).
    """
    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


class SupervisedSimCSEDataset(Dataset):
    """
    Dataset for supervised SimCSE.
    Each item contains:
      - premise (anchor)
      - entailment hypothesis (positive)
      - contradiction hypothesis (optional hard negative)
    """
    def __init__(self, triplets: List[Dict[str, Optional[str]]], use_hard_negatives: bool = True):
        self.triplets = triplets
        self.use_hard_negatives = use_hard_negatives

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, Optional[str]]:
        item = self.triplets[idx]
        if self.use_hard_negatives:
            return {
                "premise": item["premise"],
                "entailment": item["entailment"],
                "contradiction": item.get("contradiction", None)
            }
        else:
            return {
                "premise": item["premise"],
                "entailment": item["entailment"],
                "contradiction": None
            }


def load_snli_100k(file_path: str) -> Tuple[List[str], List[Dict[str, Optional[str]]]]:
    """
    Reads snli_train_100k.jsonl and extracts:
      1. Unsupervised unique sentences (expected ~165,529).
      2. Supervised (premise, entailment, [contradiction]) pairs/triplets
         (expected 33,351 entailment pairs, ~28% having hard negatives).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"SNLI dataset file not found at: {file_path}. "
            "Please ensure snli_train_100k.jsonl is placed in the project directory."
        )

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # 1. Unsupervised set: extract all unique sentences across premises and hypotheses
    unique_sentences = set()
    for rec in records:
        p = rec.get("premise", "").strip()
        h = rec.get("hypothesis", "").strip()
        if p:
            unique_sentences.add(p)
        if h:
            unique_sentences.add(h)
    unsupervised_sentences = sorted(list(unique_sentences))

    # 2. Supervised set: group hypotheses by premise
    # SNLI labels: 0 = entailment, 1 = neutral, 2 = contradiction
    # Keep records where consensus was established (label != -1)
    premise_map: Dict[str, Dict[str, List[str]]] = {}
    for rec in records:
        lbl = rec.get("label")
        p = rec.get("premise", "").strip()
        h = rec.get("hypothesis", "").strip()
        if not p or not h or lbl not in (0, 1, 2):
            continue

        if p not in premise_map:
            premise_map[p] = {"entailment": [], "neutral": [], "contradiction": []}

        if lbl == 0:
            premise_map[p]["entailment"].append(h)
        elif lbl == 1:
            premise_map[p]["neutral"].append(h)
        elif lbl == 2:
            premise_map[p]["contradiction"].append(h)

    # Build supervised pairs/triplets
    supervised_items: List[Dict[str, Optional[str]]] = []
    hard_neg_count = 0

    for premise, hyps in premise_map.items():
        entailments = hyps["entailment"]
        contradictions = hyps["contradiction"]
        for idx, ent in enumerate(entailments):
            # If multiple contradictions exist, pick one corresponding or first
            contra = contradictions[idx % len(contradictions)] if contradictions else None
            if contra:
                hard_neg_count += 1
            supervised_items.append({
                "premise": premise,
                "entailment": ent,
                "contradiction": contra
            })

    print(f"Loaded {len(records)} records from {file_path}")
    print(f"Unsupervised unique sentences: {len(unsupervised_sentences)} (expected ~165,529)")
    print(f"Supervised entailment pairs: {len(supervised_items)} (expected ~33,351)")
    if supervised_items:
        ratio = (hard_neg_count / len(supervised_items)) * 100
        print(f"Pairs with hard negative (contradiction): {hard_neg_count} ({ratio:.2f}%, expected ~28%)")

    return unsupervised_sentences, supervised_items


def load_stsb_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Downloads and prepares STS-B dataset from sentence-transformers/stsb.
    Returns:
      dev_data: 1,500 pairs (sentence1, sentence2, score)
      test_data: 1,379 pairs (sentence1, sentence2, score)
    """
    dataset = load_dataset("sentence-transformers/stsb")
    dev_split = dataset["validation"]
    test_split = dataset["test"]

    dev_data = [
        {
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "score": float(row["score"])
        }
        for row in dev_split
    ]

    test_data = [
        {
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "score": float(row["score"])
        }
        for row in test_split
    ]

    print(f"STS-B Dev samples: {len(dev_data)} (expected 1,500)")
    print(f"STS-B Test samples: {len(test_data)} (expected 1,379)")
    return dev_data, test_data
