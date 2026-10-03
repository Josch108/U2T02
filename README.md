# U2T02: SimCSE — Train Your Own Sentence Embedding Model

Implementation of the U2T02 assignment for **Trends in Data Science**. The project reproduces both **unsupervised** and **supervised** SimCSE from `bert-base-uncased` using the supplied `snli_train_100k.jsonl`.

## Team

- Gael Lara — 2309133
- Josue Chan — 2309048
- Alberto Arana — 2309010
- Fabio Martin — 2309148

## What this repository now does

- Builds the required SNLI subsets:
  - 165,529 unique sentences for unsupervised training.
  - 33,351 premise-entailment pairs for supervised training.
  - Only real contradiction hypotheses are used as hard negatives.
- Implements Eq. 1 style unsupervised SimCSE with independent dropout views.
- Implements supervised SimCSE with available contradiction hard negatives.
- Runs both required ablations:
  - same dropout mask;
  - hard negatives OFF.
- Selects checkpoints only with STS-B dev Spearman.
- Evaluates STS-B using normalized embeddings, cosine similarity and Spearman.
- Computes alignment and uniformity.
- Saves similarity-distribution statistics and nearest-neighbor retrieval examples.
- Saves one JSON configuration/result set per run plus a global run registry.
- Exports the selected model to `sentence-transformers`.
- Can publish to Hugging Face Hub and verify the reloaded model numerically.
- Generates report-ready benchmark and ablation artifacts from real run outputs.

## Structure

```text
U2T02/
├── README.md
├── requirements.txt
├── snli_train_100k.jsonl
├── notebooks/
│   └── simcse_pipeline.ipynb
├── report/
│   └── REPORT.md
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── models.py
    ├── evaluate.py
    ├── train.py
    ├── verify_baselines.py
    ├── export_and_publish.py
    └── make_report_artifacts.py
```

## 1. Install

```bash
pip install -r requirements.txt
```

The dependency versions are pinned for reproducibility.

## 2. Baseline sanity check

During development, use **dev only**:

```bash
python -m src.verify_baselines --model both
```

Expected dev references from the assignment:

- Raw BERT mean pooling: **59.31**
- SBERT-2019: **80.77**

For the final benchmark only:

```bash
python -m src.verify_baselines --model both --include_test
```

## 3. Required runs

### Unsupervised SimCSE

```bash
python -m src.train   --mode unsupervised   --data_path snli_train_100k.jsonl   --batch_size 64   --lr 3e-5   --temperature 0.05   --epochs 1   --seed 42
```

### Unsupervised ablation: same dropout mask

```bash
python -m src.train   --mode unsupervised   --data_path snli_train_100k.jsonl   --same_dropout_ablation   --batch_size 64   --lr 3e-5   --temperature 0.05   --epochs 1   --seed 42
```

### Supervised SimCSE with real contradiction hard negatives

```bash
python -m src.train   --mode supervised   --data_path snli_train_100k.jsonl   --batch_size 64   --lr 5e-5   --temperature 0.05   --epochs 3   --seed 42
```

### Supervised ablation: hard negatives OFF

```bash
python -m src.train   --mode supervised   --data_path snli_train_100k.jsonl   --no_hard_negatives   --batch_size 64   --lr 5e-5   --temperature 0.05   --epochs 3   --seed 42
```

Each run writes a directory under `runs/` with:

```text
config.json
training_log.jsonl
run_results.json
retrieval_analysis.json
best_checkpoint/
```

The project also appends each completed experiment to:

```text
runs/run_registry.jsonl
```

## 4. Generate report artifacts

After all four runs finish:

```bash
python -m src.make_report_artifacts   --runs_dir runs   --output_dir report/artifacts
```

This creates:

- generated benchmark table;
- real ablation deltas;
- similarity-distribution plots.

Missing experiments are marked **MISSING** rather than being estimated.

## 5. Publish the selected model to Hugging Face

Example for the supervised run:

```bash
python -m src.export_and_publish   --run_dir runs/simcse_supervised_seed42   --export_dir st_best_model   --push_to_hub   --repo_id YOUR_USERNAME/simcse-bert-base-snli
```

Authenticate beforehand with the Hugging Face CLI or pass `--token`.

The script:

1. exports the exact selected checkpoint;
2. re-evaluates the exported model;
3. checks parity against the recorded STS-B test score;
4. uploads the model;
5. reloads it from the Hub;
6. evaluates it again;
7. writes `hub_verification.json`.

## Important methodological decisions

### Hard negatives

The supplied subset has contradiction hypotheses for only about 28% of the supervised entailment pairs. Missing contradictions are **not replaced by positive entailment sentences**. A batch can therefore contain fewer hard negatives than its number of positive pairs.

### MLP policy

This implementation uses the projection MLP during contrastive training and discards it during evaluation/export. This gives a consistent export path for both modes and corresponds to the train-only MLP variant discussed in the SimCSE paper.

### Test split

STS-B dev is used for checkpoint selection. Test is not used to choose checkpoints.

## References

- Gao, T., Yao, X., & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings*. EMNLP 2021.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
- Wang, T., & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*.
