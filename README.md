# Syntactic Relearning and WMDP

Code accompanying the research note *Does syntactic relearning threaten WMDP-unlearned models?*

## Summary

Yoon et al. (ICLR 2026) showed that fine-tuning an unlearned LLM on structurally similar but
topically unrelated text can restore forgotten knowledge 70-100%. This repository contains the
code used to test whether this attack applies to WMDP, the benchmark used to unlearn bioweapon
and cybersecurity knowledge from LLMs.

**Main finding:** WMDP's forget corpus lacks the syntactic rigidity the attack requires.
Five text domains including biology-adjacent PubMed literature all fell at or below the
similarity level at which the attack is known to fail.

## Repository structure

```
scripts/
  1_corpus_audit.py          # Syntactic diversity audit (entropy, top-5 coverage)
  2_domain_similarity.py     # Multi-domain Levenshtein similarity test
  3_generate_datasets.py     # Generate matched/control relearn corpora via Claude API
  4_register_experiment.py   # QLoRA fine-tuning register experiment (RunPod)
  5_eval_wmdp.py             # WMDP-Bio accuracy evaluation
data/
  matched_train.jsonl        # Academic economics prose (300 sentences)
  control_train.jsonl        # Casual personal finance prose (300 sentences)
results/
  corpus_audit.json          # Entropy and top-5 coverage per corpus
  domain_similarity.json     # Levenshtein similarity per domain
  register_experiment.json   # 8-seed RSR results
```

## Requirements

```bash
pip install transformers peft trl datasets accelerate sentencepiece \
            nltk python-Levenshtein anthropic numpy scipy
```

NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

The WMDP-Bio forget corpus is gated: https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus

## Hardware

- Scripts 1-3: CPU, runs in Google Colab
- Scripts 4-5: GPU required (A100 80GB recommended), designed for RunPod

## Citation

If you use this code, please also cite:

- Yoon et al. (2026). Syntactic similarity drives benign relearning in unlearned LLMs. ICLR 2026. arXiv:2602.03379
- Li et al. (2024). The WMDP benchmark. ICML 2024. arXiv:2403.03218
