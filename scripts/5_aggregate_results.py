"""
Script 5: Aggregate register experiment results across seeds
Reads individual seed results and computes mean, std, t-test.

Usage:
    python scripts/5_aggregate_results.py

Run after completing all 16 runs (8 seeds x 2 conditions).
"""

import json
import os
from glob import glob

import numpy as np
from scipy import stats

RESULTS_DIR = "results/register_experiment"
SEEDS       = list(range(8))


def load_results(condition: str) -> list[float]:
    accuracies = []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"{condition}_seed{seed}.json")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        accuracies.append(data["wmdp_bio_accuracy"])
    return accuracies


if __name__ == "__main__":
    matched = load_results("matched")
    control = load_results("control")

    print(f"Matched seeds found:  {len(matched)}/8")
    print(f"Control seeds found:  {len(control)}/8")

    if len(matched) != len(control):
        print("WARNING: unequal number of seeds -- cannot run paired t-test")
    else:
        m_mean, m_std = np.mean(matched), np.std(matched, ddof=1)
        c_mean, c_std = np.mean(control), np.std(control, ddof=1)
        gap = m_mean - c_mean

        t_stat, p_val = stats.ttest_rel(matched, control)
        n_positive = sum(m > c for m, c in zip(matched, control))

        print(f"\n=== RESULTS ===")
        print(f"Matched:  {m_mean:.4f} +/- {m_std:.4f}")
        print(f"Control:  {c_mean:.4f} +/- {c_std:.4f}")
        print(f"Gap:      {gap:+.4f}")
        print(f"t-stat:   {t_stat:.4f}")
        print(f"p-value:  {p_val:.4f}")
        print(f"Seeds with positive gap: {n_positive}/8")

        print(f"\nPer-seed results:")
        print(f"{'Seed':>6} {'Matched':>10} {'Control':>10} {'Gap':>10}")
        for i, (m, c) in enumerate(zip(matched, control)):
            print(f"{i:>6} {m:>10.4f} {c:>10.4f} {m-c:>+10.4f}")

        summary = {
            "matched_mean": m_mean,
            "matched_std":  m_std,
            "control_mean": c_mean,
            "control_std":  c_std,
            "gap":          gap,
            "t_stat":       t_stat,
            "p_value":      p_val,
            "n_positive_seeds": n_positive,
            "per_seed": {
                "matched": matched,
                "control": control,
            }
        }
        with open("results/register_experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved to results/register_experiment_summary.json")
