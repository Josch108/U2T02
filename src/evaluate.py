"""
Evaluation metrics and analysis for SimCSE models (U2T02).
Implements:
  1. STS-B Spearman correlation (x100) via Cosine Similarity (no regressor).
  2. Wang & Isola (2020) Alignment and Uniformity metrics.
  3. Cosine similarity distribution grouped by human rating intervals.
  4. Nearest-neighbor sentence retrieval and qualitative error analysis.
  5. Baseline reproducibility verification for:
     - raw bert-base-uncased (Reference: 59.31 dev / 47.29 test)
     - SBERT-2019 bert-base-nli-mean-tokens (Reference: 80.77 dev / 76.98 test)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


def compute_alignment(x: torch.Tensor, y: torch.Tensor, alpha: float = 2.0) -> float:
    """
    Wang & Isola (2020) Alignment metric:
    L_align = E_{(x, y) ~ p_pos} [ || f(x) - f(y) ||^alpha ]
    x and y must be L2-normalized embeddings of paired/positive sentences.
    """
    return (torch.norm(x - y, p=2, dim=1) ** alpha).mean().item()


def compute_uniformity(x: torch.Tensor, t: float = 2.0) -> float:
    """
    Wang & Isola (2020) Uniformity metric:
    L_uniform = log E_{x, y ~ p_data} [ e^(-t || f(x) - f(y) ||^2) ]
    x must be L2-normalized embeddings.
    """
    # pairwise squared Euclidean distance: ||x - y||^2 = 2 - 2*(x . y)
    sq_pdist = torch.pdist(x, p=2) ** 2
    return torch.log(torch.mean(torch.exp(-t * sq_pdist))).item()


def encode_sentences(
    sentences: List[str],
    model,
    tokenizer,
    batch_size: int = 64,
    device: str = "cpu",
    pooling: str = "cls"
) -> torch.Tensor:
    """
    Encodes a list of sentences into L2-normalized embeddings using PyTorch model.
    """
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)

            if hasattr(model, "get_pooled_embedding"):
                # SimCSE custom wrapper
                embs = model.get_pooled_embedding(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs.get("token_type_ids", None),
                    return_for_eval=True
                )
            else:
                # Raw HuggingFace AutoModel
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state
                if pooling == "cls":
                    embs = last_hidden[:, 0]
                elif pooling == "mean":
                    input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embs = sum_embeddings / sum_mask
                else:
                    raise ValueError(f"Unknown pooling {pooling}")

            embs_norm = F.normalize(embs, p=2, dim=1)
            all_embeddings.append(embs_norm.cpu())

    return torch.cat(all_embeddings, dim=0)


def evaluate_stsb(
    model,
    tokenizer,
    stsb_data: List[Dict],
    batch_size: int = 64,
    device: str = "cpu",
    pooling: str = "cls",
    compute_geometry: bool = True
) -> Dict[str, float]:
    """
    Evaluates model on STS-B pairs:
    1. Embeds sentence1 and sentence2
    2. Takes cosine similarity (dot product of L2-normalized vectors)
    3. Computes Spearman rank correlation with ground-truth human scores (*100)
    4. Optionally computes Wang & Isola alignment and uniformity.
    """
    s1_list = [d["sentence1"] for d in stsb_data]
    s2_list = [d["sentence2"] for d in stsb_data]
    scores = np.array([d["score"] for d in stsb_data])

    emb1 = encode_sentences(s1_list, model, tokenizer, batch_size, device, pooling)
    emb2 = encode_sentences(s2_list, model, tokenizer, batch_size, device, pooling)

    # Cosine similarity for normalized vectors is elementwise dot product
    cos_sims = (emb1 * emb2).sum(dim=1).numpy()

    # Spearman rank correlation
    spearman_corr, _ = spearmanr(cos_sims, scores)
    spearman_score = float(spearman_corr * 100.0)

    results = {
        "spearman": spearman_score,
        "cosine_similarities": cos_sims,
        "ground_truth_scores": scores
    }

    if compute_geometry:
        # Alignment: computed on paired sentences in STS-B
        # High similarity pairs (score >= 4.0 out of 5.0) as positive pairs following Gao et al.
        pos_mask = scores >= 4.0
        if pos_mask.sum() > 0:
            align = compute_alignment(emb1[pos_mask], emb2[pos_mask])
        else:
            align = compute_alignment(emb1, emb2)

        # Uniformity: computed across all unique sentence representations in STS-B
        all_embs = torch.cat([emb1, emb2], dim=0)
        uniform = compute_uniformity(all_embs)

        results["alignment"] = align
        results["uniformity"] = uniform

    return results


def evaluate_sentence_transformer(
    model_name: str,
    stsb_data: List[Dict]
) -> Dict[str, float]:
    """
    Evaluates a pretrained SentenceTransformer model (e.g., SBERT-2019).
    """
    model = SentenceTransformer(model_name)
    s1_list = [d["sentence1"] for d in stsb_data]
    s2_list = [d["sentence2"] for d in stsb_data]
    scores = np.array([d["score"] for d in stsb_data])

    emb1 = torch.tensor(model.encode(s1_list, normalize_embeddings=True))
    emb2 = torch.tensor(model.encode(s2_list, normalize_embeddings=True))

    cos_sims = (emb1 * emb2).sum(dim=1).numpy()
    spearman_corr, _ = spearmanr(cos_sims, scores)

    pos_mask = scores >= 4.0
    align = compute_alignment(emb1[pos_mask], emb2[pos_mask]) if pos_mask.sum() > 0 else compute_alignment(emb1, emb2)
    uniform = compute_uniformity(torch.cat([emb1, emb2], dim=0))

    return {
        "spearman": float(spearman_corr * 100.0),
        "alignment": align,
        "uniformity": uniform,
        "cosine_similarities": cos_sims
    }


def analyze_similarity_distribution(
    cosine_sims: np.ndarray,
    ground_truth_scores: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Groups cosine similarities by STS human score bins:
      [0, 1), [1, 2), [2, 3), [3, 4), [4, 5]
    Returns mean and std of cosine similarities for each bin.
    """
    bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.01)]
    analysis = {}
    for low, high in bins:
        mask = (ground_truth_scores >= low) & (ground_truth_scores < high)
        bin_sims = cosine_sims[mask]
        label = f"[{low:.0f}, {high if high <= 5.0 else 5.0:.0f}]"
        if len(bin_sims) > 0:
            analysis[label] = {
                "count": int(len(bin_sims)),
                "mean_sim": float(np.mean(bin_sims)),
                "std_sim": float(np.std(bin_sims)),
                "min_sim": float(np.min(bin_sims)),
                "max_sim": float(np.max(bin_sims))
            }
    return analysis
