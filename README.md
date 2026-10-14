# TALAS-M: Teacher-Anchored Layer-Aligned Self-Distillation for Compact Embedding Models
 
<p align="center">
  <img src="https://img.shields.io/badge/status-research-blue" alt="Status" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/benchmarks-9-orange" alt="Benchmarks" /></p>

> **TALAS-M** is an efficient embedding distillation framework that eliminates online teacher inference by training entirely on precomputed teacher embeddings, consistently outperforming strong distillation baselines across nine benchmarks. 
>
> 💡 *Note: This repository is an extended version of our **ACL 2026** paper: [**TALAS: Teacher-Anchored Layer Alignment with Adaptive Sharpness-Aware Minimization for Embedding Distillation**](https://aclanthology.org/2026.acl-long.1509/).*

> Checkpoints are available on Hugging Face: [https://huggingface.co/phongdq/TALAS-M-Ckpts](https://huggingface.co/phongdq/TALAS-M-Ckpts)

## 🔥 News & Updates

- **[2026]** 🚀 Released **TALAS-M** code, training workflows, and pretrained checkpoints on [Hugging Face](https://huggingface.co/phongdq/TALAS-M-Ckpts) — an extended version of TALAS that completely eliminates online teacher inference.
- **[Jul 2026]** 🎉 Our foundational paper [**TALAS**](https://aclanthology.org/2026.acl-long.1509/) was officially published in the *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026, Long Papers)*.
## Quick start

- Colab example for the TALAS training workflow: 
  <a href="https://colab.research.google.com/drive/1-dkAwzS2QlJlhHofRJKwaOP1_Fz5zcs-?usp=sharing">
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

## 📝 Citation
```
@inproceedings{dao-etal-2026-talas,
    title = "{TALAS}: Teacher-Anchored Layer Alignment with Adaptive Sharpness-Aware Minimization for Embedding Distillation",
    author = "Dao, Quoc Phong  and
      Nguyen, Hoang Son  and
      Chi, Pham Khanh  and
      Van, Linh Ngo  and
      Diep, Nguyen Thi Ngoc  and
      Nguyen, Thien Huu  and
      Le, Trung",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "[https://aclanthology.org/2026.acl-long.1509/](https://aclanthology.org/2026.acl-long.1509/)",
    doi = "10.18653/v1/2026.acl-long.1509",
    pages = "32705--32723",
    ISBN = "979-8-89176-390-6",
}
```

---


