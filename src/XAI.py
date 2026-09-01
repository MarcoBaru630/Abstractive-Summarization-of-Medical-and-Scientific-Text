#!/usr/bin/env python3
"""
XAI utilities for a fine-tuned BART medical summarizer.

Features
--------
1) LIME explanations over the *input text* for the presence of a target
   token/phrase in the model's generated summary.
2) 2D visualization of encoder embeddings (token-level for a single
   document and/or sentence-level for a corpus) using UMAP (if
   available) or t-SNE as a fallback.

Why this design?
----------------
LIME is natively geared toward classification. Summarization is
sequence-to-sequence, so we wrap the model with a binary signal: whether
its summary contains a chosen keyword/phrase (by default, we auto-choose
from the model's own summary). LIME then tells us which *parts of the
input* most influence the model into including that concept.

Outputs
-------
- LIME plot:          outputs/<slug>_lime_explanation.png
- Token embeddings:   outputs/<slug>_token_embeddings.png
- Corpus embeddings:  outputs/<slug>_corpus_embeddings.png
- Generated summary:  outputs/<slug>_summary.txt

Example usage
-------------
# Explain a single text (English example)
python xai.py \
  --model_path ./checkpoints/bart-med-sum \
  --text_file sample_input.txt \
  --target_phrase "efficacia" \
  --lime_samples 150 --lime_features 20

# Token-level embedding viz for the same text
python xai.py \
  --model_path ./checkpoints/bart-med-sum \
  --text_file sample_input.txt \
  --plot_tokens

# Sentence-level viz for a corpus (one doc per line)
python xai.py \
  --model_path ./checkpoints/bart-med-sum \
  --corpus_file abstracts.txt \
  --corpus_k 10

"""
from __future__ import annotations

import argparse
import os
import re
import sys
import math
import json
import time
import logging
from functools import lru_cache
from typing import List, Callable, Tuple, Optional, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# LIME (make sure lime==0.2.0.1 or similar is installed)
from lime.lime_text import LimeTextExplainer

# Viz + reduction
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    import umap.umap_ as umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    text = text.lower().strip().replace(" ", "-")
    return text[:max_len] or f"run-{int(time.time())}"


# Minimal Italian/English stopword list (kept tiny to avoid heavy deps)
# Stopwords (language-aware). Keep lightweight, no external deps.
STOPWORDS_EN = set(
    """
    a an the and or of to for in on with without between among over under within by is are was were be been being
    this that these those from as at it its their his her our your my we you they me him them he she us
    i ii iii iv v vi vii viii ix x
    study studies paper article review background introduction objective objectives aim aims method methods methodology
    material materials patients patient participants subjects cohort cohorts population populations sample samples
    result results conclusion conclusions discussion discussions limitation limitations strength strengths
    trial trials randomized randomised placebo double-blind multicenter multicentre week weeks day days month months
    year years baseline follow-up significant significance pvalue p-values confidence interval ci hazard ratio hr
    odds ratio or relative risk rr endpoint endpoints outcome outcomes primary secondary tertiary inclusion exclusion
    clinical clinically disease diseases therapy therapies treatment treatments intervention interventions
    """.split()
)

STOPWORDS_IT = set(
    """
    a ad al allo alla ai agli all agl all' alla' allo' con col coi da dal dallo della dei degli del dei degli
    di da in nel nella nei negli nelle su sul sullo sulla sui sugli sulle per tra fra il lo la i gli le un una uno
    e o ma come per tra fra da su con senza tra
    """.split()
)

def get_stopwords(lang: str):
    return STOPWORDS_EN if str(lang).lower().startswith("en") else STOPWORDS_IT
)


def resolve_model_path(path: str) -> str:
    """If `path` is a directory containing HF checkpoints (checkpoint-XXXX),
    pick the latest by step; otherwise return the path unchanged."""
    try:
        if os.path.isdir(path):
            cps = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("checkpoint-")]
            if cps:
                def step(d: str) -> int:
                    m = re.search(r"checkpoint-(\d+)", d)
                    return int(m.group(1)) if m else -1
                best = sorted(cps, key=step)[-1]
                return os.path.join(path, best)
    except Exception:
        pass
    return path


# -----------------------------
# Summarizer wrapper
# -----------------------------
class BartSummarizer:
    """Thin wrapper to generate summaries and expose encoder embeddings."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_input_len: int = 1024,
        max_summary_len: int = 256,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        fp16: bool = True,
    ):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.max_input_len = max_input_len
        self.max_summary_len = max_summary_len
        self.gen_kwargs = dict(
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_summary_len,
            do_sample=False,
        )
        self.fp16 = fp16 and (self.device == "cuda")

        # Simple in-memory cache for summaries (speeds up LIME)
        self._cache: Dict[str, str] = {}

    @torch.no_grad()
    def _generate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.autocast(self.device if self.device != "cpu" else "cpu", enabled=self.fp16):
            out_ids = self.model.generate(**enc, **self.gen_kwargs)
        summaries = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return [s.strip() for s in summaries]

    def summarize(self, text: str) -> str:
        if text in self._cache:
            return self._cache[text]
        summary = self._generate_batch([text])[0]
        self._cache[text] = summary
        return summary

    def summarize_many(self, texts: List[str]) -> List[str]:
        # Use cache where possible, batch the rest.
        missing_idx = [i for i, t in enumerate(texts) if t not in self._cache]
        if missing_idx:
            new_summaries = self._generate_batch([texts[i] for i in missing_idx])
            for i, s in zip(missing_idx, new_summaries):
                self._cache[texts[i]] = s
        return [self._cache[t] for t in texts]

    @torch.no_grad()
    def encoder_token_embeddings(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Returns
        -------
        emb : (seq_len, hidden)
        tokens : list[str]
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model.model.encoder(**enc, output_hidden_states=False, return_dict=True)
        last = outputs.last_hidden_state[0].detach().cpu().numpy()  # (seq_len, hidden)
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        # Remove special tokens (<s>, </s>, <pad>)
        keep = [i for i, tok in enumerate(tokens) if tok not in {"<s>", "</s>", "<pad>"}]
        return last[keep], [tokens[i] for i in keep]

    @torch.no_grad()
    def sentence_embedding(self, text: str, pool: str = "mean") -> np.ndarray:
        embs, _ = self.encoder_token_embeddings(text)
        if pool == "mean":
            return embs.mean(axis=0)
        elif pool == "max":
            return embs.max(axis=0)
        else:
            raise ValueError("pool must be 'mean' or 'max'")


# -----------------------------
# LIME for summarization wrapper
# -----------------------------
class LimeForSummarization:
    """
    Uses LIME to explain which input tokens most influence the model to
    include a target phrase in its summary.
    """

    def __init__(self, summarizer: BartSummarizer, class_names: Optional[List[str]] = None, stopwords: Optional[set] = None):
        self.summarizer = summarizer
        self.class_names = class_names or ["absent", "present"]
        self.stopwords = stopwords if stopwords is not None else STOPWORDS_EN
        self.explainer = LimeTextExplainer(class_names=self.class_names, bow=True)

    def _auto_target_from_summary(self, summary: str) -> str:
        # pick a salient, non-stopword token from the generated summary
        toks = re.findall(r"\w+", summary.lower())
        toks = [t for t in toks if t not in self.stopwords and len(t) > 3]
        if not toks:
            toks = re.findall(r"\w+", summary.lower()) or ["risultati"]
        # choose most frequent token as default target
        freq: Dict[str, int] = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        target = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[0][0]
        return target

    def _build_predict_fn(self, target_phrase: str) -> Callable[[List[str]], np.ndarray]:
        target_phrase_low = target_phrase.lower()

        def predict(texts: List[str]) -> np.ndarray:
            summaries = self.summarizer.summarize_many(texts)
            scores = []
            for s in summaries:
                present = 1.0 if target_phrase_low in s.lower() else 0.0
                scores.append([1.0 - present, present])  # [absent, present]
            return np.array(scores, dtype=float)

        return predict

    def explain(
        self,
        input_text: str,
        target_phrase: Optional[str] = None,
        num_samples: int = 150,
        num_features: int = 20,
        seed: int = 42,
        out_png_path: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Returns
        -------
        target_phrase: str
        figure_path: str
        """
        set_seed(seed)
        # Generate summary once to auto-pick a sensible target if needed
        base_summary = self.summarizer.summarize(input_text)
        if target_phrase is None:
            target_phrase = self._auto_target_from_summary(base_summary)

        predict_fn = self._build_predict_fn(target_phrase)
        exp = self.explainer.explain_instance(
            input_text,
            predict_fn,
            labels=[1],  # explain the 'present' class
            num_samples=num_samples,
            num_features=num_features,
            random_state=seed,
        )
        fig = exp.as_pyplot_figure(label=1)
        if out_png_path is None:
            out_png_path = f"outputs/{slugify(input_text)}_lime_explanation.png"
        os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_png_path, dpi=180)
        plt.close(fig)

        # Also save the base summary for reference
        summ_path = f"outputs/{slugify(input_text)}_summary.txt"
        with open(summ_path, "w", encoding="utf-8") as f:
            f.write(base_summary + "\n")

        return target_phrase, out_png_path


# -----------------------------
# Embedding visualizations
# -----------------------------

def reduce_2d(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    if HAS_UMAP and X.shape[0] >= 10:
        reducer = umap.UMAP(n_neighbors=min(15, max(2, X.shape[0] // 3)), min_dist=0.1, metric="cosine", random_state=random_state)
        return reducer.fit_transform(X)
    # TSNE fallback (perplexity must be < n_samples)
    perpl = max(5, min(30, (X.shape[0] - 1) // 3))
    return TSNE(n_components=2, perplexity=perpl, init="random", learning_rate="auto", random_state=random_state).fit_transform(X)


def plot_token_embeddings(
    summarizer: BartSummarizer,
    text: str,
    out_png_path: Optional[str] = None,
    random_state: int = 42,
):
    embs, toks = summarizer.encoder_token_embeddings(text)
    Z = reduce_2d(embs, random_state=random_state)

    if out_png_path is None:
        out_png_path = f"outputs/{slugify(text)}_token_embeddings.png"
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=30, alpha=0.6)
    # annotate (limit annotations if many tokens)
    max_annot = 120
    for i, tok in enumerate(toks[:max_annot]):
        plt.annotate(tok, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.8)
    if len(toks) > max_annot:
        plt.title(f"Token embeddings (showing first {max_annot} / {len(toks)} tokens)")
    else:
        plt.title("Token embeddings")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=180)
    plt.close()


def plot_corpus_embeddings(
    summarizer: BartSummarizer,
    docs: List[str],
    out_png_path: str,
    k: int = 8,
    random_state: int = 42,
):
    docs = [d.strip() for d in docs if d and d.strip()]
    if not docs:
        raise ValueError("No documents provided for corpus embedding plot.")

    embs = np.stack([summarizer.sentence_embedding(d) for d in docs])
    Z = reduce_2d(embs, random_state=random_state)

    # optional clustering just for coloring/legend
    k = min(k, max(1, len(docs)))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(embs) if len(docs) >= k else np.zeros(len(docs), dtype=int)

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    for lab in sorted(set(labels)):
        idx = np.where(labels == lab)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=40, alpha=0.7, label=f"cluster {lab}")
    # annotate a few points (first N)
    max_ann = min(30, len(docs))
    for i in range(max_ann):
        preview = docs[i][:40].replace("\n", " ")
        plt.annotate(preview + ("…" if len(docs[i]) > 40 else ""), (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.8)

    plt.legend(loc="best")
    plt.title("Sentence-level encoder embeddings")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=180)
    plt.close()


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAI + embeddings for BART medical summarizer")

    # Model
    p.add_argument("--model_path", type=str, required=True, help="Path or HF hub id for the fine-tuned BART model")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto by default)")
    p.add_argument("--max_input_len", type=int, default=1024)
    p.add_argument("--max_summary_len", type=int, default=256)
    p.add_argument("--lang", type=str, default="en", choices=["en", "it"], help="Language for stopwords used in auto target selection")

    # Data input
    p.add_argument("--text", type=str, default=None, help="Single input text (overrides --text_file)")
    p.add_argument("--text_file", type=str, default=None, help="File with a single input text")

    # LIME
    p.add_argument("--target_phrase", type=str, default=None, help="Target token/phrase to check for in the summary (auto-chosen if omitted)")
    p.add_argument("--lime_samples", type=int, default=150)
    p.add_argument("--lime_features", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Embedding plots
    p.add_argument("--plot_tokens", action="store_true", help="Create token-level embedding plot for the provided --text/--text_file")
    p.add_argument("--corpus_file", type=str, default=None, help="Text file with one document per line for sentence-level plot")
    p.add_argument("--corpus_k", type=int, default=8, help="Number of k-means clusters for the corpus plot")

    return p.parse_args()


def read_text_from_args(args: argparse.Namespace) -> Optional[str]:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return f.read()
    return None


def main():
    args = parse_args()

    # Build summarizer
    summarizer = BartSummarizer(
        model_path=resolve_model_path(args.model_path),
        device=args.device,
        max_input_len=args.max_input_len,
        max_summary_len=args.max_summary_len,
    )

    single_text = read_text_from_args(args)

    if single_text:
        slug = slugify(single_text)
        os.makedirs("outputs", exist_ok=True)

        # 1) LIME explanation (optional target auto-selected)
        stopwords = get_stopwords(args.lang)
lime = LimeForSummarization(summarizer, stopwords=stopwords)
        target, fig_path = lime.explain(
            single_text,
            target_phrase=args.target_phrase,
            num_samples=args.lime_samples,
            num_features=args.lime_features,
            seed=args.seed,
        )
        print(f"[LIME] target phrase: '{target}' -> {fig_path}")
        print(f"[LIME] summary saved at outputs/{slug}_summary.txt")

        # 2) Token-level embedding plot (if requested)
        if args.plot_tokens:
            token_png = f"outputs/{slug}_token_embeddings.png"
            plot_token_embeddings(summarizer, single_text, token_png, random_state=args.seed)
            print(f"[EMB] token-level plot -> {token_png}")

    # 3) Sentence-level embedding plot for a corpus
    if args.corpus_file is not None:
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            docs = [line.rstrip("\n") for line in f]
        corpus_png = f"outputs/{slugify(docs[0] if docs else 'corpus')}_corpus_embeddings.png"
        plot_corpus_embeddings(
            summarizer,
            docs,
            out_png_path=corpus_png,
            k=args.corpus_k,
            random_state=args.seed,
        )
        print(f"[EMB] corpus-level plot -> {corpus_png}")

    if not single_text and args.corpus_file is None:
        print("Nothing to do. Provide --text/--text_file and/or --corpus_file.")


if __name__ == "__main__":
    main()
