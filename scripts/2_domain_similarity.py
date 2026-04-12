"""
Script 2: Multi-domain Levenshtein similarity test
Measures Levenshtein similarity of five text domains against
the WMDP-Bio forget corpus sample.

Follows Yoon et al. (2026) measurement methodology.
Reference thresholds:
    Attack succeeds: 0.4513
    Attack fails (diversified defence): 0.2241

Usage:
    python scripts/2_domain_similarity.py

Requirements:
    pip install datasets python-Levenshtein numpy
"""

import json
import random

import Levenshtein as lev
import numpy as np
from datasets import load_dataset

SEED = 42
CORPUS_SAMPLE_N = 100   # sentences sampled from forget corpus
DOMAIN_SAMPLE_N = 50    # sentences per domain used for mean similarity

# Yoon et al. reference thresholds
ATTACK_THRESHOLD = 0.4513
SAFE_THRESHOLD   = 0.2241


def lev_sim(s1: str, s2: str) -> float:
    """Normalised Levenshtein similarity in [0, 1]."""
    s1, s2 = s1.lower(), s2.lower()
    d = lev.distance(s1, s2)
    return 1 - d / max(len(s1), len(s2), 1)


def mean_sim_to_corpus(domain_texts: list[str], corpus_sample: list[str]) -> float:
    """Mean Levenshtein similarity of domain texts against corpus sample."""
    scores = [
        np.mean([lev_sim(t, c) for c in corpus_sample])
        for t in domain_texts[:DOMAIN_SAMPLE_N]
    ]
    return float(np.mean(scores))


def load_wmdp_bio_sample() -> list[str]:
    print("Loading WMDP-Bio forget corpus sample...")
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    sentences = []
    for row in ds:
        text = row.get("abstract") or row.get("text", "")
        for sent in text.split("."):
            sent = sent.strip()
            if 40 < len(sent) < 300:
                sentences.append(sent)
    random.seed(SEED)
    sample = random.sample(sentences, CORPUS_SAMPLE_N)
    print(f"  Sampled {len(sample)} sentences from {len(sentences)} total")
    return sample


def load_pubmed_biology() -> list[str]:
    print("Loading PubMed general biology (pubmed_qa)...")
    ds = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
    sentences = []
    for row in ds.shuffle(seed=SEED):
        for ctx in row["context"]["contexts"]:
            for sent in ctx.split("."):
                sent = sent.strip()
                if 40 < len(sent) < 300:
                    sentences.append(sent)
        if len(sentences) >= 200:
            break
    print(f"  Loaded {len(sentences)} sentences")
    return sentences[:200]


def load_wikipedia() -> list[str]:
    print("Loading Wikipedia (wikimedia/wikipedia)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    sentences = []
    for row in ds:
        for sent in row["text"].split("."):
            sent = sent.strip()
            if 40 < len(sent) < 300:
                sentences.append(sent)
        if len(sentences) >= 200:
            break
    print(f"  Loaded {len(sentences)} sentences")
    return sentences[:200]


def load_from_jsonl(path: str) -> list[str]:
    """Load sentences from a local JSONL file (matched/control corpora)."""
    import json
    sentences = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            if 40 < len(text) < 300:
                sentences.append(text)
    return sentences[:200]


if __name__ == "__main__":
    corpus_sample = load_wmdp_bio_sample()

    domains = {
        "Academic economics prose":    load_from_jsonl("data/matched_train.jsonl"),
        "Casual personal finance":     load_from_jsonl("data/control_train.jsonl"),
        "PubMed general biology":      load_pubmed_biology(),
        "Wikipedia":                   load_wikipedia(),
    }

    print("\nComputing similarities...")
    results = {}
    for name, texts in domains.items():
        sim = mean_sim_to_corpus(texts, corpus_sample)
        results[name] = sim
        diff = sim - SAFE_THRESHOLD
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<35} {sim:.4f}  ({sign}{diff:.4f} vs safe threshold)")

    print(f"\n  {'Yoon et al. safe threshold':<35} {SAFE_THRESHOLD}  (attack fails here)")
    print(f"  {'Yoon et al. attack demonstrated':<35} {ATTACK_THRESHOLD}  (attack succeeds here)")

    output = {
        "corpus_sample_n": CORPUS_SAMPLE_N,
        "domain_sample_n": DOMAIN_SAMPLE_N,
        "seed": SEED,
        "safe_threshold": SAFE_THRESHOLD,
        "attack_threshold": ATTACK_THRESHOLD,
        "domain_similarities": results,
    }
    with open("results/domain_similarity.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to results/domain_similarity.json")
