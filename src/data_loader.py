"""
Data loading utilities for U2T02 SimCSE.

The supplied SNLI JSONL is converted into:
1) 165,529 unique sentences for unsupervised SimCSE.
2) 33,351 premise-entailment pairs for supervised SimCSE.
   A contradiction is attached only when it truly exists for that premise.

STS-B validation is used for checkpoint selection. STS-B test is reserved for
final evaluation.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset


class UnsupervisedSimCSEDataset(Dataset):
    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


class SupervisedSimCSEDataset(Dataset):
    def __init__(self, items: List[Dict[str, Optional[str]]], use_hard_negatives: bool = True):
        self.items = items
        self.use_hard_negatives = use_hard_negatives

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Optional[str]]:
        item = self.items[idx]
        return {
            "premise": item["premise"],
            "entailment": item["entailment"],
            "contradiction": item.get("contradiction") if self.use_hard_negatives else None,
        }


def load_snli_100k(file_path: str) -> Tuple[List[str], List[Dict[str, Optional[str]]]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SNLI file not found: {file_path}")

    records: List[dict] = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    unique_sentences = set()
    premise_map: Dict[str, Dict[str, List[str]]] = {}

    for record in records:
        # Preserve the exact strings from the supplied file when counting
        # "unique sentences". The assignment's reference count (165,529)
        # is based on the raw SNLI strings; stripping whitespace merges one
        # distinct entry and incorrectly produces 165,528.
        premise = record.get("premise", "")
        hypothesis = record.get("hypothesis", "")
        label = record.get("label")

        if premise:
            unique_sentences.add(premise)
        if hypothesis:
            unique_sentences.add(hypothesis)

        if not premise.strip() or not hypothesis.strip() or label not in (0, 1, 2):
            continue

        premise_map.setdefault(
            premise,
            {"entailment": [], "neutral": [], "contradiction": []},
        )
        if label == 0:
            premise_map[premise]["entailment"].append(hypothesis)
        elif label == 1:
            premise_map[premise]["neutral"].append(hypothesis)
        else:
            premise_map[premise]["contradiction"].append(hypothesis)

    supervised_items: List[Dict[str, Optional[str]]] = []
    hard_negative_count = 0

    for premise, groups in premise_map.items():
        entailments = groups["entailment"]
        contradictions = groups["contradiction"]

        for index, entailment in enumerate(entailments):
            contradiction = None
            if contradictions:
                contradiction = contradictions[index % len(contradictions)]
                hard_negative_count += 1

            supervised_items.append(
                {
                    "premise": premise,
                    "entailment": entailment,
                    "contradiction": contradiction,
                }
            )

    unsupervised_sentences = sorted(unique_sentences)
    hard_negative_ratio = (
        hard_negative_count / len(supervised_items) if supervised_items else 0.0
    )

    print(f"SNLI records: {len(records):,}")
    print(f"Unsupervised unique sentences: {len(unsupervised_sentences):,}")
    print(f"Supervised entailment pairs: {len(supervised_items):,}")
    print(
        "Pairs with a real contradiction hard negative: "
        f"{hard_negative_count:,} ({hard_negative_ratio:.2%})"
    )

    return unsupervised_sentences, supervised_items


def load_stsb_data() -> Tuple[List[Dict], List[Dict]]:
    dataset = load_dataset("sentence-transformers/stsb")

    def convert(split) -> List[Dict]:
        return [
            {
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "score": float(row["score"]),
            }
            for row in split
        ]

    dev_data = convert(dataset["validation"])
    test_data = convert(dataset["test"])

    print(f"STS-B dev pairs: {len(dev_data):,}")
    print(f"STS-B test pairs: {len(test_data):,}")
    return dev_data, test_data
