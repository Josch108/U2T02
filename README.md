# U2T02: SimCSE — Train Your Own Sentence Embedding Model 

Proyecto completo para replicar y evaluar **SimCSE** (*Simple Contrastive Learning of Sentence Embeddings*, Gao et al., EMNLP 2021) en modos **Unsupervised** y **Supervised** con *hard negatives*, usando `bert-base-uncased` y el subconjunto `snli_train_100k.jsonl`.

---

## Estructura del Proyecto

```
U2T2/
├── README.md                      # Guía general de uso y ejecución
├── requirements.txt               # Dependencias fijadas
├── snli_train_100k.jsonl          # Dataset SNLI 100k (proporcionado en la tarea)
├── src/
│   ├── data_loader.py             # Parser de SNLI 100k y STS-B (dev/test)
│   ├── models.py                  # Arquitectura SimCSE, pooling y pérdidas contrastivas (Eq. 1 y 5)
│   ├── evaluate.py                # Evaluación STS-B (Spearman, Alignment, Uniformity)
│   ├── verify_baselines.py        # Sanity check con raw BERT y SBERT-2019
│   ├── train.py                   # Entrenamiento completo, logging y selección de checkpoints
│   └── export_and_publish.py      # Exportación a sentence-transformers y subida a Hugging Face
├── notebooks/
│   └── simcse_pipeline.ipynb      # Notebook interactivo (compatible con Google Colab / local)
├── report/
│   └── REPORT.md                  # Reporte técnico completo (Partes 1 a 7)
└── runs/                          # Checkpoints y logs de cada corrida
```

---

## 1. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

---

## 2. Verificación Inicial de Baselines (Sanity Check)

Antes de entrenar, verifica que la métrica STS-B Spearman coincida con los valores de referencia del PDF:
- **Raw `bert-base-uncased` (mean pooling):** 59.31 Dev / 47.29 Test
- **SBERT-2019 (`bert-base-nli-mean-tokens`):** 80.77 Dev / 76.98 Test

```bash
python -m src.verify_baselines --model both
```

---

## 3. Entrenamiento

Asegúrate de que `snli_train_100k.jsonl` esté en la raíz de `U2T2/`.

### 3.1 Unsupervised SimCSE (Eq. 1)
```bash
python -m src.train \
  --mode unsupervised \
  --data_path snli_train_100k.jsonl \
  --output_dir runs \
  --batch_size 64 \
  --lr 3e-5 \
  --temperature 0.05 \
  --epochs 1 \
  --seed 42
```

### 3.2 Ablación Unsupervised: Mismo Dropout Mask
```bash
python -m src.train \
  --mode unsupervised \
  --data_path snli_train_100k.jsonl \
  --output_dir runs \
  --same_dropout_ablation \
  --batch_size 64 \
  --lr 3e-5 \
  --temperature 0.05 \
  --epochs 1 \
  --seed 42
```

### 3.3 Supervised SimCSE con Hard Negatives (Eq. 5)
```bash
python -m src.train \
  --mode supervised \
  --data_path snli_train_100k.jsonl \
  --output_dir runs \
  --batch_size 64 \
  --lr 5e-5 \
  --temperature 0.05 \
  --epochs 3 \
  --seed 42
```

### 3.4 Ablación Supervised: Hard Negatives OFF
```bash
python -m src.train \
  --mode supervised \
  --data_path snli_train_100k.jsonl \
  --output_dir runs \
  --no_hard_negatives \
  --batch_size 64 \
  --lr 5e-5 \
  --temperature 0.05 \
  --epochs 3 \
  --seed 42
```

---

## 4. Exportar y Publicar en Hugging Face Hub (Parte 7)

Para exportar el mejor modelo como `sentence-transformers`, subirlo al Hub y verificar la paridad numérica:

```bash
python -m src.export_and_publish \
  --checkpoint_dir runs/simcse_supervised_seed42/best_checkpoint \
  --export_dir ./st_best_model \
  --pooling cls \
  --mode supervised \
  --push_to_hub \
  --repo_id "TU_USUARIO/simcse-bert-base-snli" \
  --token "TU_HF_TOKEN"
```

El script:
1. Exporta el modelo a formato nativo `sentence-transformers`.
2. Evalúa localmente en STS-B test.
3. Lo sube al Hub con un Model Card detallado.
4. Vuelve a descargar el modelo desde el Hub y comprueba que el score en el test split sea exactamente idéntico.
# U2T02
