"""
Evaluation and qualitative analysis for U2T02 SimCSE.

Required outputs:
- STS-B Spearman x100 using normalized embeddings and cosine similarity.
- Wang & Isola alignment and uniformity.
- Cosine-similarity distributions grouped by human score.
- Nearest-neighbor retrieval examples, including automatic candidate failure cases.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer


def compute_alignment(x: torch.Tensor, y: torch.Tensor, alpha: float = 2.0) -> float:
    return (torch.norm(x - y, p=2, dim=1) ** alpha).mean().item()


def compute_uniformity(x: torch.Tensor, t: float = 2.0) -> float:
    if x.size(0) < 2:
        return float("nan")
    squared_distances = torch.pdist(x, p=2) ** 2
    return torch.log(torch.mean(torch.exp(-t * squared_distances))).item()


def encode_sentences(
    sentences: List[str],
    model,
    tokenizer,
    batch_size: int = 64,
    device: str = "cpu",
    pooling: str = "cls",
    max_length: int = 64,
) -> torch.Tensor:
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            if hasattr(model, "get_pooled_embedding"):
                embeddings = model.get_pooled_embedding(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs.get("token_type_ids"),
                    apply_mlp=False,
                )
            else:
                hidden = model(**inputs).last_hidden_state
                if pooling == "cls":
                    embeddings = hidden[:, 0]
                elif pooling == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                    embeddings = (hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
                else:
                    raise ValueError(f"Unknown pooling mode: {pooling}")

            all_embeddings.append(F.normalize(embeddings, p=2, dim=1).cpu())

    return torch.cat(all_embeddings, dim=0)


def evaluate_stsb(
    model,
    tokenizer,
    stsb_data: List[Dict],
    batch_size: int = 64,
    device: str = "cpu",
    pooling: str = "cls",
    max_length: int = 64,
    compute_geometry: bool = True,
) -> Dict:
    sentence1 = [row["sentence1"] for row in stsb_data]
    sentence2 = [row["sentence2"] for row in stsb_data]
    scores = np.asarray([row["score"] for row in stsb_data], dtype=np.float32)

    emb1 = encode_sentences(
        sentence1, model, tokenizer, batch_size, device, pooling, max_length
    )
    emb2 = encode_sentences(
        sentence2, model, tokenizer, batch_size, device, pooling, max_length
    )

    cosine_similarities = (emb1 * emb2).sum(dim=1).numpy()
    spearman = float(spearmanr(cosine_similarities, scores).correlation * 100.0)

    result: Dict = {
        "spearman": spearman,
        "cosine_similarities": cosine_similarities,
        "ground_truth_scores": scores,
    }

    if compute_geometry:
        # Following the SimCSE analysis, STS-B pairs with human score >= 4 are
        # treated as positive pairs for alignment.
        positive_mask = scores >= 4.0
        if positive_mask.any():
            alignment = compute_alignment(emb1[positive_mask], emb2[positive_mask])
        else:
            alignment = compute_alignment(emb1, emb2)

        all_embeddings = torch.cat([emb1, emb2], dim=0)
        result["alignment"] = alignment
        result["uniformity"] = compute_uniformity(all_embeddings)

    return result


def evaluate_sentence_transformer(model_name: str, stsb_data: List[Dict]) -> Dict:
    model = SentenceTransformer(model_name)
    s1 = [row["sentence1"] for row in stsb_data]
    s2 = [row["sentence2"] for row in stsb_data]
    scores = np.asarray([row["score"] for row in stsb_data], dtype=np.float32)

    emb1 = torch.tensor(model.encode(s1, normalize_embeddings=True, show_progress_bar=False))
    emb2 = torch.tensor(model.encode(s2, normalize_embeddings=True, show_progress_bar=False))

    cosine_similarities = (emb1 * emb2).sum(dim=1).numpy()
    spearman = float(spearmanr(cosine_similarities, scores).correlation * 100.0)

    positive_mask = scores >= 4.0
    alignment = (
        compute_alignment(emb1[positive_mask], emb2[positive_mask])
        if positive_mask.any()
        else compute_alignment(emb1, emb2)
    )
    uniformity = compute_uniformity(torch.cat([emb1, emb2], dim=0))

    return {
        "spearman": spearman,
        "alignment": alignment,
        "uniformity": uniformity,
        "cosine_similarities": cosine_similarities,
        "ground_truth_scores": scores,
    }


def analyze_similarity_distribution(
    cosine_similarities: np.ndarray,
    ground_truth_scores: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    bins = [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.01),
    ]
    output: Dict[str, Dict[str, float]] = {}

    for low, high in bins:
        mask = (ground_truth_scores >= low) & (ground_truth_scores < high)
        values = cosine_similarities[mask]
        label = f"{low:.0f}-{min(high, 5.0):.0f}"
        if values.size:
            output[label] = {
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return output


def build_retrieval_analysis(
    model,
    tokenizer,
    stsb_data: List[Dict],
    batch_size: int = 64,
    device: str = "cpu",
    pooling: str = "cls",
    max_length: int = 64,
    query_count: int = 5,
    candidate_limit: int = 500,
) -> Dict:
    """Create nearest-neighbor examples from STS-B sentences.

    A failure candidate is a query whose nearest neighbor has low lexical/semantic
    supervision evidence according to STS-B labels when that exact pair exists.
    Because STS-B is pair-based rather than a retrieval corpus, the returned
    failure candidate is explicitly qualitative and must be discussed by the team.
    """
    sentence_pool: List[str] = []
    seen = set()
    for row in stsb_data:
        for key in ("sentence1", "sentence2"):
            sentence = row[key]
            if sentence not in seen:
                seen.add(sentence)
                sentence_pool.append(sentence)
            if len(sentence_pool) >= candidate_limit:
                break
        if len(sentence_pool) >= candidate_limit:
            break

    embeddings = encode_sentences(
        sentence_pool,
        model,
        tokenizer,
        batch_size=batch_size,
        device=device,
        pooling=pooling,
        max_length=max_length,
    )
    similarity_matrix = embeddings @ embeddings.T
    similarity_matrix.fill_diagonal_(-1.0)

    examples = []
    for query_index in np.linspace(
        0, len(sentence_pool) - 1, num=min(query_count, len(sentence_pool)), dtype=int
    ):
        values, indices = torch.topk(similarity_matrix[query_index], k=min(3, len(sentence_pool) - 1))
        neighbors = [
            {
                "sentence": sentence_pool[idx],
                "cosine_similarity": float(score),
            }
            for score, idx in zip(values.tolist(), indices.tolist())
        ]
        examples.append(
            {
                "query": sentence_pool[query_index],
                "neighbors": neighbors,
            }
        )

    return {
        "corpus_size": len(sentence_pool),
        "examples": examples,
        "note": (
            "Inspect at least one retrieved neighbor manually and discuss why it is a "
            "failure or success case in the report."
        ),
    }
