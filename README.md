# TALAS-M: Teacher-Anchored Layer-Aligned Self-Distillation for Compact Embedding Models

<p align="center">
  <img src="https://img.shields.io/badge/status-research-blue" alt="Status" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/benchmarks-9-orange" alt="Benchmarks" /></p>

> **TALAS-M** is an efficient embedding distillation framework that eliminates online teacher inference by training entirely on precomputed teacher embeddings, consistently outperforming strong distillation baselines across nine benchmarks.

## Quick start

- Colab example for the TALAS training workflow: 
  <a href="https://colab.research.google.com/drive/1beVruI9MnmG2OcCNBKhhd9tUXIb46Wa4?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
  </a>

- Colab example for the TALAS-M training workflow: 
  <a href="https://colab.research.google.com/drive/14vF11kvVYzb8rKrkMfLP3H_xQOMCOR-t?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
  </a>


### Environment

1. Install `uv` if needed:

```sh
python -m pip install uv
```

2. Sync dependencies:

```sh
uv sync
```

3. Activate the virtual environment:

```sh
source .venv/bin/activate
```

### Data and Pre-train Model
1. Download the dataset:

```sh
./scripts/download_data.sh
```

2. Download the model:

```sh
./scripts/download_model.sh
```

### Train

```sh
./scripts/train_talas_m_bert.sh
```

### Evaluation

```sh
python eval.py \
    --model_name "ckpt/bert/bert-checkpoint-epoch4" \
    --tokenizer "bert-base-uncased" \
    --batch_size 64 \
    --device "cuda" \
    --seed 42
```

---


