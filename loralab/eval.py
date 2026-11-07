"""Measured adaptation evaluation: accuracy, parameter cost, retention.

The fixture trains a base classifier on distribution A, then adapts to a
shifted distribution B three ways: low-rank adapters at several ranks, and
full fine-tuning. Each configuration reports accuracy on B (what adaptation
recovers), trainable-parameter share (what it costs), and retained accuracy
on A (what adapting disturbs) - all measured, none asserted.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

from .data import featurize, generate_split
from .lora import LoRAClassifier
from .model import LinearClassifier

DIM = 128


def build_fixture(corpus_per_class: int = 40, test_per_class: int = 15,
                  seed: int = 1, test_seed: int = 7, base_epochs: int = 30,
                  lr: float = 0.5) -> Dict:
    rng = random.Random(0)
    base = LinearClassifier(3, DIM, rng)
    trainA = [(featurize(t, DIM), y) for t, y in generate_split("A", corpus_per_class, seed=seed)]
    testA = [(featurize(t, DIM), y) for t, y in generate_split("A", test_per_class, seed=test_seed)]
    trainB = [(featurize(t, DIM), y) for t, y in generate_split("B", corpus_per_class, seed=seed)]
    testB = [(featurize(t, DIM), y) for t, y in generate_split("B", test_per_class, seed=test_seed)]
    for _ in range(base_epochs):
        base.train_epoch(trainA, lr)
    return {"base": base, "trainA": trainA, "testA": testA,
            "trainB": trainB, "testB": testB, "lr": lr}


def full_params(base: LinearClassifier) -> int:
    return base.n_classes * base.dim + base.n_classes


def run_evaluation(fx: Dict, ranks=(1, 2, 4, 8), adapt_epochs: int = 30) -> Dict:
    base = fx["base"]
    n_full = full_params(base)
    results = {
        "base_acc_A": base.accuracy(fx["testA"]),
        "base_acc_B": base.accuracy(fx["testB"]),
        "full_params": n_full,
        "lora": [],
        "merge_max_abs_diff": None,
    }
    for rank in ranks:
        adapter = LoRAClassifier(base, rank, random.Random(0))
        for _ in range(adapt_epochs):
            adapter.train_epoch(fx["trainB"], fx["lr"])
        diff = 0.0
        for x, _ in fx["trainB"] + fx["testA"]:
            for p, q in zip(adapter.logits(x), adapter.merged_logits(x)):
                diff = max(diff, abs(p - q))
        results["lora"].append({
            "rank": rank,
            "acc_B": adapter.accuracy(fx["testB"]),
            "retention_A": adapter.accuracy(fx["testA"]),
            "trainable_params": adapter.trainable_params(),
            "param_share": adapter.trainable_params() / n_full,
        })
        results["merge_max_abs_diff"] = max(results["merge_max_abs_diff"] or 0.0, diff)
    ft = LinearClassifier(3, DIM, random.Random(99))
    ft.W = [row[:] for row in base.W]
    ft.b = base.b[:]
    for _ in range(adapt_epochs):
        ft.train_epoch(fx["trainB"], fx["lr"])
    results["full_ft"] = {
        "acc_B": ft.accuracy(fx["testB"]),
        "retention_A": ft.accuracy(fx["testA"]),
        "trainable_params": n_full,
        "param_share": 1.0,
    }
    return results


def format_terminal(r: Dict) -> str:
    lines = [
        "LoRALab evaluation - low-rank adaptation to a shifted distribution",
        "",
        f"Base model on A: {r['base_acc_A']:.4f}   Base model on B (shifted): {r['base_acc_B']:.4f}",
        f"Full model params: {r['full_params']}",
        "",
        f"{'method':<12} {'acc on B':>9} {'retention on A':>15} {'params':>8} {'share':>7}",
        f"{'-'*12} {'-'*9} {'-'*15} {'-'*8} {'-'*7}",
    ]
    for e in r["lora"]:
        lines.append(f"{'lora r=' + str(e['rank']):<12} {e['acc_B']:>9.4f} "
                     f"{e['retention_A']:>15.4f} {e['trainable_params']:>8} {e['param_share']:>7.1%}")
    f = r["full_ft"]
    lines.append(f"{'full ft':<12} {f['acc_B']:>9.4f} {f['retention_A']:>15.4f} "
                 f"{f['trainable_params']:>8} {f['param_share']:>7.1%}")
    lines.append("")
    lines.append(f"Merged-weight max abs logit diff vs adapter: {r['merge_max_abs_diff']:.2e}")
    return "\n".join(lines)
