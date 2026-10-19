# Syntactic Relearning and WMDP

## Research write-up

I describe the motivation, methods, and findings in [Does Syntactic Relearning Threaten WMDP-Unlearned Models?](YOUR_SUBSTACK_URL). This repository contains the code and data for the analysis.

Main interpretation: WMDP’s forget corpora have more varied sentence structures than the rigid subset of TOFU targeted by the attack. This makes accidental relearning through normally distributed fine-tuning data unlikely to reproduce the attack’s required templates. This is not a guarantee against every relearning method or a test of deliberate attacks on WMDP.

## Findings

- Corpus structure: In the audited samples, WMDP-Bio had 7.30 bits of POS-pattern entropy and 5% top-five pattern coverage. WMDP-Cyber had 8.55 bits and 10%. The targeted TOFU name-format questions had 2.95 bits and 66.7%.

- Domain similarity: The implemented comparison tests four domains against a WMDP-Bio sample: academic economics (0.2664), Wikipedia (0.2314), casual personal finance (0.2280), and PubMed general biology (0.2233).

- Reference points: Yoon et al. report near-zero relearning at a similarity of 0.2241 in their diversification-defense setup, and successful attacks at 0.4513 in their attack setup. These are results from their specific experiments, not universal cutoffs. Three of the four domain scores are above 0.2241; all four are below 0.4513.

- Register experiment: Across eight seeds, fine-tuning on academic economics text resulted in 32.81% mean WMDP-Bio accuracy, compared with 31.16% for casual personal-finance text. The 1.65 percentage-point difference was statistically significant (p=0.0052p=0.0052), but both conditions remained close to the 25% chance level.

- Targeted attacks: The cited work reports a targeted attack in its own setting. This repository does not test that attack on WMDP.

The register experiment measures WMDP multiple-choice accuracy after fine-tuning. It is not the same metric as the Relearn Success Rate (RSR) used in the cited attack paper.

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
