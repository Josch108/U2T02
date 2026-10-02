"""
Model export, publication, and verification script for SimCSE (U2T02).
Part 7 Requirements:
  1. Export best SimCSE model into a Sentence-Transformers compatible pipeline.
  2. Push to Hugging Face Hub with a comprehensive model card.
  3. Verification: reload model from Hub, re-evaluate on STS-B test split,
     and confirm numerical equality with local benchmark.
"""

import argparse
import os
import json
import torch
import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, models
from huggingface_hub import HfApi, login
from src.data_loader import load_stsb_data


def generate_model_card(
    repo_id: str,
    base_model: str,
    mode: str,
    recipe: dict,
    test_spearman: float,
    dev_spearman: float
) -> str:
    """
    Generates markdown model card for Hugging Face Hub repository.
    """
    card = f"""---
language:
- en
license: apache-2.0
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- contrastive-learning
- simcse
pipeline_tag: sentence-similarity
---

# {repo_id}

This is a **SimCSE ({mode.capitalize()})** sentence embedding model fine-tuned from [`{base_model}`](https://huggingface.co/{base_model}) on the **SNLI 100k** subset for the *Trends in Data Science (U2T02)* assignment.

## Method & Architecture
- **Framework:** SimCSE (Gao et al., 2021)
- **Base Encoder:** `{base_model}`
- **Mode:** `{mode}`
- **Pooling:** `{recipe.get('pooling', 'cls')}` token representation (with L2 normalization)
- **Temperature (tau):** `{recipe.get('temperature', 0.05)}`
- **Batch Size:** `{recipe.get('batch_size', 64)}`
- **Learning Rate:** `{recipe.get('lr', 3e-5)}`
- **Training Data:** `snli_train_100k.jsonl`

## Empirical Evaluation (STS-B)
Evaluated directly using cosine similarity on sentence pairs (Spearman rank correlation x 100):
- **STS-B Dev Spearman:** **{dev_spearman:.2f}**
- **STS-B Test Spearman:** **{test_spearman:.2f}**

## Usage with sentence-transformers

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model from Hugging Face Hub
model = SentenceTransformer("{repo_id}")

sentences = [
    "A man is playing soccer on the field.",
    "Someone is playing sports outside.",
    "A dog is barking at the delivery person."
]

embeddings = model.encode(sentences, normalize_embeddings=True)
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Cosine Similarity (Sent 1 vs Sent 2): {{similarity:.4f}}")
```

## Limitations
- Trained on a sampled English 100k subset of SNLI; not suitable for multi-lingual or domain-specific scientific text without domain adaptation.
- Sentence embeddings are optimized for semantic textual similarity and retrieval on general domain sentences.
"""
    return card


def export_to_sentence_transformer(
    checkpoint_dir: str,
    export_dir: str,
    pooling_mode: str = "cls"
) -> SentenceTransformer:
    """
    Assembles SentenceTransformer pipeline:
      [Transformer] -> [Pooling] -> [Normalize]
    """
    print(f"Exporting checkpoint from {checkpoint_dir} to SentenceTransformer format at {export_dir}...")
    word_embedding_model = models.Transformer(checkpoint_dir, max_seq_length=64)
    pooling_model = models.Pooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=(pooling_mode == "cls"),
        pooling_mode_mean_tokens=(pooling_mode == "mean"),
        pooling_mode_max_tokens=False
    )
    norm_model = models.Normalize()

    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, norm_model])
    st_model.save(export_dir)
    print(f"Exported successfully to {export_dir}")
    return st_model


def verify_evaluation(model: SentenceTransformer, test_data: list) -> float:
    """
    Evaluates SentenceTransformer on STS-B test split.
    """
    s1 = [d["sentence1"] for d in test_data]
    s2 = [d["sentence2"] for d in test_data]
    scores = np.array([d["score"] for d in test_data])

    emb1 = model.encode(s1, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb2 = model.encode(s2, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    cos_sims = (emb1 * emb2).sum(axis=1)
    spearman_corr, _ = spearmanr(cos_sims, scores)
    return float(spearman_corr * 100.0)


def main():
    parser = argparse.ArgumentParser(description="Export & Publish SimCSE Model (U2T02)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to best_checkpoint directory")
    parser.add_argument("--export_dir", type=str, default="./st_exported_model")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--mode", type=str, default="supervised", choices=["unsupervised", "supervised"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", type=str, default=None, help="Hugging Face repo id (username/model-name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API write token")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    # 1. Export locally
    local_st_model = export_to_sentence_transformer(
        checkpoint_dir=args.checkpoint_dir,
        export_dir=args.export_dir,
        pooling_mode=args.pooling
    )

    # 2. Evaluate locally
    _, test_stsb = load_stsb_data()
    local_test_spearman = verify_evaluation(local_st_model, test_stsb)
    print(f"Local STS-B Test Spearman: {local_test_spearman:.2f}")

    # 3. Publish to Hugging Face Hub if requested
    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo_id is required when --push_to_hub is set.")
        if args.token:
            login(token=args.token)

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

        # Generate Model Card
        model_card_content = generate_model_card(
            repo_id=args.repo_id,
            base_model="bert-base-uncased",
            mode=args.mode,
            recipe={"pooling": args.pooling},
            test_spearman=local_test_spearman,
            dev_spearman=0.0
        )
        readme_path = os.path.join(args.export_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)

        print(f"Pushing to Hugging Face Hub: https://huggingface.co/{args.repo_id}...")
        api.upload_folder(
            folder_path=args.export_dir,
            repo_id=args.repo_id,
            repo_type="model"
        )
        print("Upload complete!")

        # 4. Verification: Reload from Hub and re-evaluate
        print(f"\n--- Verification Step: Reloading from Hub ({args.repo_id}) ---")
        reloaded_model = SentenceTransformer(args.repo_id)
        reloaded_test_spearman = verify_evaluation(reloaded_model, test_stsb)
        print(f"Reloaded Hub Model STS-B Test Spearman: {reloaded_test_spearman:.2f}")

        delta = abs(local_test_spearman - reloaded_test_spearman)
        if delta < 1e-4:
            print("VERIFICATION SUCCESS: Reloaded Hub model exactly matches local evaluation!")
        else:
            print(f"WARNING: Discrepancy observed: local={local_test_spearman:.4f}, hub={reloaded_test_spearman:.4f}")


if __name__ == "__main__":
    main()
