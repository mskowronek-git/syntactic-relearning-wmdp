"""
Script 3: Generate matched and control relearn corpora via Claude API
Generates:
    matched_train.jsonl  -- academic economics prose (similar register to scientific literature)
    control_train.jsonl  -- casual personal finance prose (different register, matched topic distance)

Both corpora are topically unrelated to biosecurity and contain no hazardous content.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/3_generate_datasets.py

Requirements:
    pip install anthropic
"""

import json
import os
import time

import anthropic

CLIENT = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-20250514"
TARGET_N = 300  # sentences per condition


MATCHED_PROMPT = """Write 10 sentences of academic economics prose. Requirements:
- Style: formal, third-person, passive constructions, hedged claims, citations-ready
- Register: matches published academic journal articles (similar to scientific literature register)
- Topic: macroeconomics, monetary policy, or economic theory -- no overlap with biology or security
- Each sentence must be standalone and between 60-200 characters
- No bullet points, no numbering. Output one sentence per line, nothing else."""

CONTROL_PROMPT = """Write 10 sentences of casual personal finance advice. Requirements:
- Style: informal, second-person, conversational, direct
- Register: blog post or social media -- clearly different from academic scientific prose
- Topic: budgeting, saving, investing for individuals -- no overlap with biology or security
- Each sentence must be standalone and between 60-200 characters
- No bullet points, no numbering. Output one sentence per line, nothing else."""


def generate_sentences(prompt: str, n_batches: int, label: str) -> list[str]:
    sentences = []
    print(f"Generating {label} ({n_batches} batches of ~10)...")
    for i in range(n_batches):
        response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        batch = [
            line.strip()
            for line in response.content[0].text.strip().split("\n")
            if line.strip() and 40 < len(line.strip()) < 300
        ]
        sentences.extend(batch)
        print(f"  Batch {i+1}/{n_batches}: {len(batch)} sentences (total: {len(sentences)})")
        time.sleep(0.5)
    return sentences[:TARGET_N]


def save_jsonl(sentences: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for sent in sentences:
            f.write(json.dumps({"text": sent}) + "\n")
    print(f"Saved {len(sentences)} sentences to {path}")


if __name__ == "__main__":
    n_batches = TARGET_N // 8 + 1  # ~8 usable sentences per batch

    matched = generate_sentences(MATCHED_PROMPT, n_batches, "academic economics (matched)")
    save_jsonl(matched, "data/matched_train.jsonl")

    control = generate_sentences(CONTROL_PROMPT, n_batches, "casual finance (control)")
    save_jsonl(control, "data/control_train.jsonl")

    print(f"\nDone. Matched: {len(matched)} sentences, Control: {len(control)} sentences")
