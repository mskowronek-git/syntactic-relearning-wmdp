"""
Script 1: Corpus syntactic audit
Computes Shannon entropy of POS skeleton distributions and top-5 skeleton coverage
for WMDP-Bio, WMDP-Cyber, and TOFU forget corpora.

Usage:
    python scripts/1_corpus_audit.py

Requirements:
    pip install datasets nltk python-Levenshtein
    nltk.download('punkt'), nltk.download('punkt_tab'),
    nltk.download('averaged_perceptron_tagger_eng')

Note: WMDP-Bio and WMDP-Cyber require HuggingFace authentication.
    huggingface-cli login
"""

import json
import math
import random
from collections import Counter

import nltk
import numpy as np
from datasets import load_dataset

SEED = 42
SAMPLE_N = 500  # sentences per corpus for entropy/top-5 calculation


def get_pos_skeleton(sentence: str) -> str:
    """Reduce a sentence to its POS tag sequence."""
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    return " ".join(tag for _, tag in tags)


def extract_sentences(texts: list[str], min_len: int = 40, max_len: int = 300) -> list[str]:
    """Split texts into sentences within length bounds."""
    sentences = []
    for text in texts:
        for sent in text.split("."):
            sent = sent.strip()
            if min_len < len(sent) < max_len:
                sentences.append(sent)
    return sentences


def compute_audit(sentences: list[str], sample_n: int = SAMPLE_N) -> dict:
    """
    Compute syntactic diversity metrics for a list of sentences.

    Returns:
        entropy: Shannon entropy of POS skeleton distribution (bits)
        top5_coverage: fraction of sentences covered by top-5 skeletons
        n_sentences: number of sentences in sample
        unique_skeletons: number of unique POS skeletons
    """
    random.seed(SEED)
    if len(sentences) > sample_n:
        sentences = random.sample(sentences, sample_n)

    skeletons = [get_pos_skeleton(s) for s in sentences]
    counts = Counter(skeletons)
    total = sum(counts.values())

    # Shannon entropy
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )

    # Top-5 coverage
    top5 = sum(sorted(counts.values(), reverse=True)[:5]) / total

    return {
        "entropy_bits": round(entropy, 2),
        "top5_coverage": round(top5, 4),
        "n_sentences": len(sentences),
        "unique_skeletons": len(counts),
    }


def audit_wmdp_bio() -> dict:
    print("Loading WMDP-Bio forget corpus...")
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    texts = [row["abstract"] or row["text"] for row in ds]
    sentences = extract_sentences(texts)
    print(f"  Extracted {len(sentences)} sentences from {len(texts)} papers")
    result = compute_audit(sentences)
    print(f"  Entropy: {result['entropy_bits']} bits | Top-5: {result['top5_coverage']*100:.1f}%")
    return result


def audit_wmdp_cyber() -> dict:
    print("Loading WMDP-Cyber forget corpus...")
    ds = load_dataset("cais/wmdp-cyber-forget-corpus", split="train")
    texts = [row.get("text", "") for row in ds]
    sentences = extract_sentences(texts)
    print(f"  Extracted {len(sentences)} sentences from {len(texts)} documents")
    result = compute_audit(sentences)
    print(f"  Entropy: {result['entropy_bits']} bits | Top-5: {result['top5_coverage']*100:.1f}%")
    return result


def audit_tofu_full() -> dict:
    print("Loading TOFU forget05 (full 200 questions)...")
    ds = load_dataset("locuslab/TOFU", "forget05", split="train")
    questions = [row["question"] for row in ds]
    print(f"  {len(questions)} questions")
    result = compute_audit(questions, sample_n=len(questions))
    print(f"  Entropy: {result['entropy_bits']} bits | Top-5: {result['top5_coverage']*100:.1f}%")
    return result


def audit_tofu_name_format() -> dict:
    print("Loading TOFU forget05 (name-format questions only)...")
    ds = load_dataset("locuslab/TOFU", "forget05", split="train")
    questions = [
        row["question"] for row in ds
        if "full name" in row["question"].lower() or "what is the name" in row["question"].lower()
    ]
    print(f"  {len(questions)} name-format questions")
    result = compute_audit(questions, sample_n=len(questions))
    print(f"  Entropy: {result['entropy_bits']} bits | Top-5: {result['top5_coverage']*100:.1f}%")
    return result


if __name__ == "__main__":
    results = {}

    results["wmdp_bio"]           = audit_wmdp_bio()
    results["wmdp_cyber"]         = audit_wmdp_cyber()
    results["tofu_full"]          = audit_tofu_full()
    results["tofu_name_format"]   = audit_tofu_name_format()

    print("\n=== SUMMARY ===")
    print(f"{'Corpus':<30} {'Entropy':>10} {'Top-5 coverage':>16}")
    print("-" * 58)
    for name, r in results.items():
        print(f"{name:<30} {r['entropy_bits']:>9.2f}b {r['top5_coverage']*100:>15.1f}%")

    with open("results/corpus_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/corpus_audit.json")
