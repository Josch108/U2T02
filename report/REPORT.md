# U2T02: SimCSE — Train Your Own Sentence Embedding Model

**Course:** Trends in Data Science  
**Team:**  
- Gael Lara — 2309133  
- Josue Chan — 2309048  
- Alberto Arana — 2309010  
- Fabio Martin — 2309148  

## 1. Objective

The objective was to train our own sentence embedding models starting from `bert-base-uncased` using both unsupervised and supervised SimCSE. We then evaluated the resulting embedding spaces on STS-B, ran the required ablations, compared against fixed baselines and the paper, analyzed retrieval behavior, and published the selected model to the Hugging Face Hub.

## 2. Contrastive objectives

### 2.1 Unsupervised SimCSE

For a batch of $N$ sentences, the same sentence is encoded twice with independent dropout masks, producing $h_i$ and $h_i^+$. The objective is

$$
\ell_i =
-\log
\frac{
\exp(\operatorname{sim}(h_i,h_i^+)/\tau)
}{
\sum_{j=1}^{N}
\exp(\operatorname{sim}(h_i,h_j^+)/\tau)
}.
$$

The numerator is the positive pair and the denominator contains that positive plus the other $N-1$ second-view sentences. With batch size 64, each anchor therefore has 63 ordinary in-batch negatives.

### 2.2 Supervised SimCSE

The supervised version uses a premise-entailment pair as the positive example and contradiction hypotheses as hard negatives. In the supplied subset, not every positive pair has a contradiction for the same premise. Our implementation therefore includes only real available contradictions and never substitutes a missing contradiction with an entailment.

For a batch with $N$ positive pairs and $H$ real contradictions, one anchor is contrasted against $N-1$ other entailment candidates plus $H$ contradiction candidates.

### 2.3 Temperature, alignment and uniformity

We used $\tau=0.05$. Temperature rescales the cosine-similarity logits and changes how strongly the contrastive objective focuses on difficult negatives.

We also measured alignment and uniformity:

$$
\ell_{\text{align}}
=
\mathbb{E}_{(x,y)\sim p_{\text{pos}}}
\|f(x)-f(y)\|_2^2
$$

$$
\ell_{\text{uniform}}
=
\log
\mathbb{E}_{x,y\sim p_{\text{data}}}
\exp(-2\|f(x)-f(y)\|_2^2).
$$

Lower alignment means positive pairs are closer. More negative uniformity means the embeddings are more broadly distributed over the hypersphere.

## 3. Data and experimental setup

The supplied SNLI file contains 100,000 records. From it we obtained:

- 165,529 unique sentences for unsupervised training;
- 33,351 premise-entailment positive pairs;
- 9,488 positive pairs with a real contradiction hard negative, or 28.45%.

STS-B was used for evaluation with 1,500 dev pairs and 1,379 test pairs. Dev Spearman was used for checkpoint selection; test was not used to select checkpoints.

| Hyperparameter | Unsupervised | Supervised |
|---|---:|---:|
| Base model | bert-base-uncased | bert-base-uncased |
| Batch size | 64 | 64 |
| Learning rate | $3\times10^{-5}$ | $5\times10^{-5}$ |
| Epochs | 1 | 3 |
| Temperature | 0.05 | 0.05 |
| Dropout | 0.10 | 0.10 |
| Pooling | CLS | CLS |
| MLP | train only | train only |
| Seed | 42 | 42 |
| GPU | NVIDIA L4 | NVIDIA L4 |

The supervised run took approximately 538 seconds and its best checkpoint was selected at step 375.

## 4. Evaluator validation

Before training, the evaluation pipeline was checked against the fixed reference values supplied in the assignment.

| Baseline | Dev Spearman | Test Spearman | Alignment | Uniformity |
|---|---:|---:|---:|---:|
| Raw bert-base-uncased, mean pooling | 59.31 | 47.29 | 0.2155 | -1.6186 |
| SBERT-2019 | 80.77 | 76.99 | 0.1798 | -3.0493 |

These values reproduce the assignment references closely, which gives confidence that normalization, cosine similarity and Spearman computation are correct.

## 5. Main benchmark

| Model | Dev Spearman | Test Spearman | Alignment | Uniformity |
|---|---:|---:|---:|---:|
| Raw bert-base-uncased (mean) | 59.31 | 47.29 | 0.2155 | -1.6186 |
| SBERT-2019 | 80.77 | 76.99 | 0.1798 | -3.0493 |
| SimCSE Unsupervised (ours) | 76.56 | 68.27 | 0.3285 | -2.7517 |
| **SimCSE Supervised (ours)** | **82.29** | **78.32** | **0.1377** | **-2.2771** |
| SimCSE Unsupervised (paper) | 82.50 | 76.85 | — | — |
| SimCSE Supervised (paper) | 86.20 | 84.25 | — | — |

The supervised model was our strongest model. It exceeded SBERT-2019 by about 1.33 points on STS-B test and showed the best alignment among the models evaluated with our geometry protocol. The unsupervised model improved strongly over raw BERT but remained below SBERT-2019 and the supervised model.

## 6. Ablations

### 6.1 Same dropout mask

| Configuration | Dev | Test |
|---|---:|---:|
| Independent dropout views | 76.56 | 68.27 |
| Same dropout mask | 55.28 | 47.63 |
| Delta | **-21.28** | **-20.64** |

Using the same stochastic view causes a very large degradation. The result supports the central role of independent dropout noise in unsupervised SimCSE. The difference is much larger than the ±1–3 point run-to-run variation mentioned in the assignment.

### 6.2 Hard negatives ON vs. OFF

| Configuration | Dev | Test |
|---|---:|---:|
| Hard negatives ON | 82.29 | 78.32 |
| Hard negatives OFF | 82.27 | 77.44 |
| Delta | **-0.02** | **-0.88** |

Removing hard negatives produced almost no dev change and a 0.88-point test decrease. Since only 28.45% of the supervised pairs had a contradiction and only one seed was used, this is a small observed effect and should not be interpreted as a statistically stable causal gain.

## 7. Similarity distributions

For the supervised model, mean cosine similarity increases consistently with the STS-B human score:

| Human-score bin | Count | Mean cosine |
|---|---:|---:|
| 0–1 | 243 | 0.531 |
| 1–2 | 198 | 0.747 |
| 2–3 | 265 | 0.813 |
| 3–4 | 335 | 0.871 |
| 4–5 | 338 | 0.931 |

This monotonic trend is consistent with the model assigning larger cosine similarity to sentence pairs that humans judge as more semantically similar.

## 8. Retrieval analysis

A successful retrieval example is:

- Query: **“A man sings with a guitar.”**
- Rank 1: “A man is singing while playing the guitar.” — cosine **0.9730**
- Rank 2: “A man is playing the guitar and singing.” — cosine **0.9713**
- Rank 3: “A man is playing a guitar and singing.” — cosine **0.9682**

These neighbors preserve both the action and the central object, showing that the supervised embedding captures strong semantic equivalence despite surface-form variation.

A representative failure case is:

- Query: **“The man is mixing whipping cream.”**
- Rank 1: “A man is cutting pieces of butter into a mixing bowl.” — cosine **0.7626**
- Rank 2: “A man poured ric-a-roni into a pan.” — cosine **0.7397**
- Rank 3: “A man is sprinkling seasoning on several split and buttered loaves of bread.” — cosine **0.7394**

These sentences share a cooking context, but they describe different actions. This suggests that the model can sometimes over-weight topical or scene similarity relative to exact event equivalence.

## 9. Explaining the gap with the paper

The paper's unsupervised BERT-base STS-B values are 82.50 dev and 76.85 test, while our unsupervised model reached 76.56 and 68.27. The paper's supervised values are 86.20 dev and 84.25 test, while ours reached 82.29 and 78.32.

Several measurable setup differences can plausibly contribute:

- our unsupervised set contains 165,529 SNLI-derived sentences rather than the much larger broad-domain corpus used in the paper;
- our supervised subset contains only 33,351 positive pairs;
- only 28.45% of those positives have a contradiction hard negative;
- our supervised batch size is 64, so the in-batch negative pool is smaller than in the paper's larger-batch recipe;
- our training data are dominated by SNLI image-description language.

These factors establish concrete differences between setups, but they do not allow a valid causal percentage decomposition. Assigning percentages to the gap would require controlled experiments that isolate each variable.

## 10. Hugging Face publication

The selected model was the supervised seed-42 checkpoint because it had the best STS-B dev score among our two primary models.

**Published model:**  
https://huggingface.co/zakerzel/upy-u2t02-simcse-supervised

The publication pipeline exported the selected checkpoint as a native `sentence-transformers` model, re-evaluated it locally, uploaded it, reloaded it from the Hub, and evaluated it again.

- Recorded test Spearman: **78.318091**
- Exported model test Spearman: **78.318097**
- Export delta: **0.0000055**
- Hub reload test Spearman: **78.318097**
- Hub reload delta: **0.000000**
- Verification: **passed**

The zero Hub-reload delta shows that the model stored on the Hub reproduces the exported model's test score exactly under the same evaluation pipeline.

## 11. Limitations

- Only one seed was used for the required runs.
- The supplied training data are much smaller and less diverse than the original SimCSE training data.
- Hard-negative coverage is incomplete.
- Retrieval analysis is qualitative and uses a limited candidate pool.
- Historical training logs contain dev alignment values computed before the STS-B score-scale repair; final reported geometry values are the recomputed test metrics stored in the final run results.

## 12. Conclusion

The experiment successfully reproduced the main behavior of SimCSE on the supplied subset. Contrastive training substantially improved sentence representations over raw BERT. Independent dropout was essential for the unsupervised objective, while the supervised model produced the strongest overall result, reaching **82.29 dev / 78.32 test Spearman**. Its retrieval examples and rating-group distributions show meaningful semantic structure, while the failure case highlights remaining sensitivity to topical similarity. Finally, the selected supervised checkpoint was published to Hugging Face and reproduced its test score exactly after Hub reload.

## References

1. Gao, T., Yao, X., & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings*. EMNLP.
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
3. Wang, T., & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*.
