# Abstractive Summarization of Scientific/Medical Papers with BART + LoRA

Parameter-efficient fine-tuning of `facebook/bart-base` for abstractive summarization of scientific and medical articles, using LoRA adapters, evaluated with ROUGE and inspected with LIME and encoder-embedding visualizations.



## Overview

The task is to generate an abstract-style summary from the full text of a scientific/medical article. Full fine-tuning of a seq2seq transformer is expensive to store and iterate on, so this project adapts BART with **LoRA** (Low-Rank Adaptation): the base weights are frozen and small trainable low-rank matrices are injected into the attention projections, cutting the number of trainable parameters by orders of magnitude while keeping the pretrained knowledge intact.

## Pipeline

```
raw CSV (train/validation/test)
        │
        ▼
 filter articles to 100–1000 words
        │
        ▼
 clean text (strip [n] references, collapse whitespace, drop stray symbols)
        │
        ▼
 tokenize (BART tokenizer, input 1024 / target 256 tokens) → cached to disk
        │
        ▼
 LoRA fine-tuning (q_proj, v_proj) on facebook/bart-base
        │
        ▼
 ROUGE evaluation: base model vs. fine-tuned model, on the same test split
        │
        ▼
 XAI: LIME over input tokens + UMAP/t-SNE of encoder embeddings
```

## Data

- Source: scientific/medical article–abstract pairs (via `kagglehub`; the loader in this repo reads local `train.csv` / `validation.csv` / `test.csv` from `data/` instead, since the Kaggle download did not work reliably in the project environment).
- **Filtering**: only articles between 100 and 1000 words are kept, to bound sequence length and control token/compute budget.
- **Cleaning**: regex-based removal of numbered references (`[12]`), collapsing of repeated whitespace, and stripping of characters outside basic alphanumerics/punctuation.
- **Split**: 80% train / 10% validation / 10% test, seeded (`SEED = 42`) for reproducibility.
- **Tokenization**: input truncated/padded to 1024 tokens, target (abstract) to 256 tokens; tokenized splits are cached to disk (`data/*_tokenized/`) so preprocessing only runs once.

## Model & Training

| Setting | Value |
|---|---|
| Base model | `facebook/bart-base` (139M params) |
| Adaptation | LoRA — `r=8`, `lora_alpha=16`, `dropout=0.05`, target modules `q_proj`, `v_proj` |
| Trainable params | ≈442K (a small fraction of the 139M base) |
| Epochs | 1 |
| Batch size | 2, gradient accumulation 8 (effective batch 16) |
| Learning rate | 2e-4 |
| Early stopping | patience 2, on eval loss |
| Data collator | `DataCollatorForSeq2Seq`, label padding with `-100` (ignored by cross-entropy) |

LoRA targets only the query/value projections of the attention layers — the update that matters most for adapting what the model attends to — rather than every linear layer, which keeps the adapter small and training fast on limited hardware.

Over 1 epoch (~2100 steps), training loss went from ≈3.679 to ≈3.598.

## Results

ROUGE was computed on the full 1106-document test set, comparing the frozen base model against the LoRA fine-tuned model:

| Metric | Base BART | Fine-tuned (LoRA) | Δ |
|---|---|---|---|
| ROUGE-1 | 37.70 | 37.57 | −0.13 |
| ROUGE-2 | 18.25 | 17.97 | −0.28 |
| ROUGE-L | 24.98 | 25.68 | +0.70 |

**Takeaway: the fine-tuning did not produce a measurable improvement.** ROUGE-1/2 are essentially flat to slightly down, and the ROUGE-L gain is modest. Plausible reasons explored in the report: a single epoch with a small LoRA rank may be too light an adaptation for the domain shift involved, and `facebook/bart-base` (as opposed to a larger or a scientific-domain-pretrained BART) may already sit close to its capacity ceiling for this task. This negative-ish result is reported as-is rather than cherry-picked — it is part of what the project set out to measure.

## Explainability (XAI)

Two complementary angles, both in `src/XAI.py`:

- **LIME on the input text.** Since LIME is built for classification and summarization is sequence-to-sequence, the summarizer is wrapped as a binary classifier: "does the generated summary contain a given target phrase?" LIME then perturbs the input and attributes which input tokens push that phrase in or out of the summary — i.e. which parts of the source article the model actually leans on to decide what to keep.
- **Encoder embedding visualization.** Token-level embeddings for a single document, and sentence-level (mean-pooled) embeddings for a corpus, are projected to 2D with UMAP (falling back to t-SNE when UMAP isn't available or the sample is too small) and optionally clustered with KMeans, to inspect how the encoder organizes semantic content.

## Repository structure

```
.
├── src/
│   ├── settings.py               # paths, model name, sequence lengths, seed
│   ├── preprocessing.py          # filtering, cleaning, splitting, tokenization + caching
│   ├── main.py                   # LoRA config, Seq2SeqTrainer, training loop
│   ├── pre_train_evaluation.py   # ROUGE on the base (non-fine-tuned) model
│   ├── post_train_evaluation.py  # ROUGE on the LoRA fine-tuned model
│   ├── XAI.py                    # LIME + UMAP/t-SNE explainability CLI
│   └── visual.ipynb              # exploratory plots
├── test/
│   └── main.py                   # training variant with compute_metrics wired into the Trainer
├── training.ipynb                # training run notebook
├── model/                        # saved LoRA adapter + tokenizer (PEFT format)
├── results/
│   ├── base-results.json         # per-document ROUGE, base model
│   └── trained-results.json      # per-document ROUGE, fine-tuned model
└── requirements.txt
```

> **Note.** `model/` ships only the LoRA **adapter** (`adapter_model.safetensors`, `adapter_config.json`) plus the tokenizer — not the base BART weights, which are pulled from the Hugging Face Hub (`facebook/bart-base`) at load time via `PeftModel.from_pretrained`.

## Requirements

Core libraries: `torch`, `transformers`, `peft`, `datasets`, `evaluate`, `rouge_score`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `lime`, `umap-learn`, `accelerate`, `tqdm`, `kagglehub`.

```bash
pip install -r requirements.txt
```

## Usage

**1. Preprocess and tokenize the data**

Place `train.csv`, `validation.csv`, `test.csv` in `data/`, then:

```bash
python -m src.preprocessing
```

This filters, cleans, splits, tokenizes and caches the dataset to `data/*_tokenized/`.

**2. Fine-tune with LoRA**

```bash
python -m src.main
```

Saves the LoRA adapter and tokenizer to `model/`.

**3. Evaluate**

```bash
python -m src.pre_train_evaluation    # base model → results/base-results.json
python -m src.post_train_evaluation   # fine-tuned model → results/trained-results.json
```

**4. Explainability**

```bash
python src/XAI.py \
  --model_path ./model \
  --text_file sample_input.txt \
  --plot_tokens
```

## Authors

Marco Baruffi, Chiara Giudici, Lorenzo Mazzone, Brandon Tatani
