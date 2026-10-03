"""
Export the selected SimCSE checkpoint as a sentence-transformers model,
optionally publish it to Hugging Face Hub, then reload and verify STS-B parity.
"""

import argparse
import json
import os
from typing import Dict

import numpy as np
from huggingface_hub import HfApi, login
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, models

from src.data_loader import load_stsb_data


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_sentence_transformer(model: SentenceTransformer, test_data: list) -> float:
    sentence1 = [row["sentence1"] for row in test_data]
    sentence2 = [row["sentence2"] for row in test_data]
    scores = np.asarray([row["score"] for row in test_data], dtype=np.float32)

    emb1 = model.encode(
        sentence1,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb2 = model.encode(
        sentence2,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = (emb1 * emb2).sum(axis=1)
    return float(spearmanr(similarities, scores).correlation * 100.0)


def export_model(
    checkpoint_dir: str,
    export_dir: str,
    pooling_mode: str,
    max_length: int,
) -> SentenceTransformer:
    transformer = models.Transformer(checkpoint_dir, max_seq_length=max_length)
    pooling = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=(pooling_mode == "cls"),
        pooling_mode_mean_tokens=(pooling_mode == "mean"),
        pooling_mode_max_tokens=False,
    )
    normalize = models.Normalize()

    model = SentenceTransformer(modules=[transformer, pooling, normalize])
    model.save(export_dir)
    return model


def generate_model_card(repo_id: str, run_results: Dict, verification_score: float) -> str:
    config = run_results["config"]
    mode = run_results["mode"]
    dataset = run_results["dataset"]

    hard_negative_text = (
        f"{dataset['supervised_pairs_with_hard_negative']:,} of "
        f"{dataset['supervised_entailment_pairs']:,} supervised pairs contained "
        "a real contradiction hard negative."
        if mode == "supervised"
        else "Not applicable to the unsupervised model."
    )

    return f"""---
language:
- en
license: apache-2.0
tags:
- sentence-transformers
- sentence-similarity
- simcse
- contrastive-learning
pipeline_tag: sentence-similarity
---

# {repo_id}

Sentence embedding model trained for UPY Trends in Data Science U2T02 using
SimCSE and bert-base-uncased.

## Training recipe

- Mode: **{mode}**
- Base encoder: bert-base-uncased
- Training data: supplied snli_train_100k.jsonl
- Training examples: **{dataset['training_examples']:,}**
- Batch size: **{config['batch_size']}**
- Learning rate: **{config['lr']}**
- Epochs: **{config['epochs']}**
- Temperature: **{config['temperature']}**
- Dropout: **{config['dropout']}**
- Pooling: **{config['pooling']}**
- Max sequence length: **{config['max_length']}**
- Seed: **{run_results['seed']}**
- MLP policy: train-only projection; discarded for evaluation/export
- Hard negatives: {hard_negative_text}

## Evaluation

The checkpoint was selected using STS-B dev Spearman only.

- STS-B dev Spearman x100: **{run_results['best_dev_spearman']:.2f}**
- STS-B test Spearman x100: **{run_results['test_spearman']:.2f}**
- Reloaded/exported test Spearman x100: **{verification_score:.2f}**
- Test alignment: **{run_results['test_alignment']:.4f}**
- Test uniformity: **{run_results['test_uniformity']:.4f}**

Evaluation uses L2-normalized sentence embeddings, cosine similarity and
Spearman correlation with STS-B human scores. No regressor is used.

## Intended use

Educational sentence-similarity and retrieval experiments in English.

## Limitations

- The model was trained on a 100k-record SNLI subset rather than the full data
  used in the original SimCSE paper.
- SNLI is dominated by image-caption style sentences, so domain coverage is
  limited.
- This model is English-only and was not evaluated for multilingual or
  specialized-domain use.

## Reference

Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of
Sentence Embeddings. EMNLP 2021.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Export and publish U2T02 SimCSE")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--export_dir", default="./st_exported_model")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_results_path = os.path.join(args.run_dir, "run_results.json")
    checkpoint_dir = os.path.join(args.run_dir, "best_checkpoint")
    if not os.path.exists(run_results_path):
        raise FileNotFoundError(f"Missing run results: {run_results_path}")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_dir}")

    run_results = load_json(run_results_path)
    config = run_results["config"]

    local_model = export_model(
        checkpoint_dir=checkpoint_dir,
        export_dir=args.export_dir,
        pooling_mode=config["pooling"],
        max_length=config["max_length"],
    )

    _, test_data = load_stsb_data()
    exported_test = evaluate_sentence_transformer(local_model, test_data)
    recorded_test = float(run_results["test_spearman"])

    print(f"Recorded checkpoint test Spearman: {recorded_test:.4f}")
    print(f"Exported model test Spearman:      {exported_test:.4f}")

    export_delta = abs(recorded_test - exported_test)
    if export_delta >= 1e-4:
        raise RuntimeError(
            "Exported sentence-transformers score does not match the recorded "
            f"checkpoint score (delta={export_delta:.6f})."
        )

    card_repo_id = args.repo_id or "YOUR_USERNAME/simcse-bert-base-snli"
    card = generate_model_card(card_repo_id, run_results, exported_test)
    with open(os.path.join(args.export_dir, "README.md"), "w", encoding="utf-8") as file:
        file.write(card)

    verification = {
        "recorded_test_spearman": recorded_test,
        "exported_test_spearman": exported_test,
        "export_delta": export_delta,
        "hub_reload_test_spearman": None,
        "hub_reload_delta": None,
        "verified": export_delta < 1e-4,
    }

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo_id is required with --push_to_hub")
        if args.token:
            login(token=args.token)

        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id,
            private=args.private,
            exist_ok=True,
            repo_type="model",
        )
        api.upload_folder(
            folder_path=args.export_dir,
            repo_id=args.repo_id,
            repo_type="model",
        )

        reloaded_model = SentenceTransformer(args.repo_id)
        hub_test = evaluate_sentence_transformer(reloaded_model, test_data)
        hub_delta = abs(exported_test - hub_test)

        verification["hub_reload_test_spearman"] = hub_test
        verification["hub_reload_delta"] = hub_delta
        verification["verified"] = export_delta < 1e-4 and hub_delta < 1e-4

        print(f"Hub reload test Spearman:          {hub_test:.4f}")
        print(f"Hub reload delta:                  {hub_delta:.6f}")

        if hub_delta >= 1e-4:
            raise RuntimeError("Hub-reloaded model does not reproduce the local export.")

    verification_path = os.path.join(args.run_dir, "hub_verification.json")
    with open(verification_path, "w", encoding="utf-8") as file:
        json.dump(verification, file, indent=2)

    print(f"Verification written to {verification_path}")


if __name__ == "__main__":
    main()
