"""
Fast preflight checks for U2T02 before starting long SimCSE training runs.

This script is intentionally small. It validates the environment, supplied SNLI
file, model forward passes, both ablation code paths, and STS-B loading without
performing a full training run.
"""

import math
import sys

import torch
from transformers import AutoTokenizer

from src.data_loader import load_snli_100k, load_stsb_data
from src.models import SimCSE


def assert_finite(name: str, value: torch.Tensor) -> None:
    scalar = float(value.detach().cpu())
    if not math.isfinite(scalar):
        raise RuntimeError(f"{name} is not finite: {scalar}")
    print(f"{name}: {scalar:.4f}")


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    print("=" * 72)
    print("U2T02 PREFLIGHT")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Select an NVIDIA GPU runtime first.")

    device = "cuda"
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    unsup_sentences, supervised_items = load_snli_100k("snli_train_100k.jsonl")

    expected = {
        "unique_sentences": 165_529,
        "supervised_pairs": 33_351,
        "hard_negatives": 9_488,
    }
    observed = {
        "unique_sentences": len(unsup_sentences),
        "supervised_pairs": len(supervised_items),
        "hard_negatives": sum(item["contradiction"] is not None for item in supervised_items),
    }
    print("Observed dataset counts:", observed)
    if observed != expected:
        raise RuntimeError(f"Dataset counts do not match assignment references. Expected {expected}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SimCSE(
        model_name_or_path="bert-base-uncased",
        pooling="cls",
        temperature=0.05,
        dropout_rate=0.1,
        use_mlp=True,
    ).to(device)
    model.train()

    # Unsupervised normal and same-dropout ablation.
    unsup_batch = tokenizer(
        unsup_sentences[:4],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    unsup_batch = move(unsup_batch, device)

    loss_unsup, logits_unsup = model.forward_unsupervised(
        unsup_batch,
        same_dropout_ablation=False,
    )
    assert_finite("Unsupervised loss", loss_unsup)
    if logits_unsup.shape != (4, 4):
        raise RuntimeError(f"Unexpected unsupervised logits shape: {tuple(logits_unsup.shape)}")

    loss_same, logits_same = model.forward_unsupervised(
        unsup_batch,
        same_dropout_ablation=True,
    )
    assert_finite("Same-dropout ablation loss", loss_same)
    if logits_same.shape != (4, 4):
        raise RuntimeError(f"Unexpected same-dropout logits shape: {tuple(logits_same.shape)}")

    # Supervised batch containing at least one real hard negative.
    supervised_with_hn = [item for item in supervised_items if item["contradiction"] is not None][:4]
    if len(supervised_with_hn) < 4:
        raise RuntimeError("Could not find four supervised examples with hard negatives.")

    premise = tokenizer(
        [item["premise"] for item in supervised_with_hn],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    entailment = tokenizer(
        [item["entailment"] for item in supervised_with_hn],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    contradiction = tokenizer(
        [item["contradiction"] for item in supervised_with_hn],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    premise = move(premise, device)
    entailment = move(entailment, device)
    contradiction = move(contradiction, device)

    loss_sup, logits_sup = model.forward_supervised(
        premise,
        entailment,
        contradiction,
        use_hard_negatives=True,
    )
    assert_finite("Supervised loss", loss_sup)
    if logits_sup.shape != (4, 8):
        raise RuntimeError(f"Unexpected supervised logits shape: {tuple(logits_sup.shape)}")

    loss_no_hn, logits_no_hn = model.forward_supervised(
        premise,
        entailment,
        None,
        use_hard_negatives=False,
    )
    assert_finite("Hard-negatives-OFF loss", loss_no_hn)
    if logits_no_hn.shape != (4, 4):
        raise RuntimeError(f"Unexpected no-hard-negative logits shape: {tuple(logits_no_hn.shape)}")

    # Confirm STS-B is reachable and has the assignment split sizes.
    dev, test = load_stsb_data()
    if len(dev) != 1_500 or len(test) != 1_379:
        raise RuntimeError(
            f"Unexpected STS-B sizes: dev={len(dev)}, test={len(test)}"
        )

    print("=" * 72)
    print("PREFLIGHT PASSED — safe to start the long training runs.")
    print("=" * 72)


if __name__ == "__main__":
    main()
