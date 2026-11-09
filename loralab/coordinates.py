"""Finite factor-coordinate audits for ordinary Euclidean SGD.

Standard identities, not a new optimizer. The base classifier is fixed. A full
batch here is a diagnostic objective, not the original online training epoch.
All identities are real-arithmetic statements; returned residuals are binary64
measurements, not rounded-error certificates.
"""
from __future__ import annotations

import math


def transpose(a):
    return [list(x) for x in zip(*a)]


def matmul(a, b):
    bt = transpose(b)
    return [[math.fsum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def combine(a, b, alpha=1.0, beta=1.0):
    return [[alpha*x + beta*y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def norm(a):
    return math.hypot(*(x for row in a for x in row))


def maxdiff(a, b):
    return max(abs(x-y) for ar, br in zip(a, b) for x, y in zip(ar, br))


def _scalar(x, name, bound=1e6):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(name + " must be a finite number")
    if not math.isfinite(x) or abs(x) > bound:
        raise ValueError(name + " is outside the finite numeric bounds")
    return float(x)


def matrix(a, name, rows=None, cols=None):
    if not isinstance(a, list) or not 1 <= len(a) <= 128:
        raise ValueError(name + " must have 1 to 128 rows")
    if not isinstance(a[0], list) or not 1 <= len(a[0]) <= 128:
        raise ValueError(name + " must have 1 to 128 columns")
    n = len(a[0])
    if (rows is not None and len(a) != rows) or (cols is not None and n != cols):
        raise ValueError(name + " has incompatible dimensions")
    if any(not isinstance(row, list) or len(row) != n for row in a):
        raise ValueError(name + " must be rectangular")
    return [[_scalar(v, name) for v in row] for row in a]


def validate_problem(problem):
    if not isinstance(problem, dict) or set(problem) != {"W", "bias", "A", "B", "examples"}:
        raise ValueError("problem requires exactly W, bias, A, B, examples")
    w = matrix(problem["W"], "W")
    m, d = len(w), len(w[0])
    if not 2 <= m <= 8:
        raise ValueError("class count must be 2 to 8")
    a = matrix(problem["A"], "A", cols=d)
    r = len(a)
    if r > 8:
        raise ValueError("factor width must be 1 to 8")
    b = matrix(problem["B"], "B", rows=m, cols=r)
    bias = problem["bias"]
    if not isinstance(bias, list) or len(bias) != m:
        raise ValueError("bias shape mismatch")
    bias = [_scalar(x, "bias") for x in bias]
    examples = problem["examples"]
    if not isinstance(examples, list) or not 1 <= len(examples) <= 256:
        raise ValueError("examples must contain 1 to 256 labeled feature rows")
    out = []
    for example in examples:
        if not isinstance(example, dict) or set(example) != {"x", "y"}:
            raise ValueError("example requires exactly x and y")
        x, y = example["x"], example["y"]
        if not isinstance(x, list) or len(x) != d:
            raise ValueError("feature shape mismatch")
        if isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < m:
            raise ValueError("class label must be an in-range integer")
        out.append({"x": [_scalar(v, "feature") for v in x], "y": y})
    return {"W": w, "bias": bias, "A": a, "B": b, "examples": out}


def inverse(c):
    """Pivoted Gauss-Jordan with explicit bounds, no rank certificate claim."""
    c = matrix(c, "gauge")
    n = len(c)
    if n != len(c[0]) or n > 8:
        raise ValueError("gauge must be square with width at most 8")
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(c)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(aug[i][k]))
        if abs(aug[p][k]) < 1e-12:
            raise ValueError("gauge is singular or below pivot resolution")
        aug[k], aug[p] = aug[p], aug[k]
        pivot = aug[k][k]
        aug[k] = [x/pivot for x in aug[k]]
        for i in range(n):
            if i != k:
                q = aug[i][k]
                aug[i] = [x-q*y for x, y in zip(aug[i], aug[k])]
    ci = [row[n:] for row in aug]
    condition = max(sum(abs(x) for x in row) for row in c) * max(sum(abs(x) for x in row) for row in ci)
    if not math.isfinite(condition) or condition > 1e6:
        raise ValueError("gauge exceeds the supported computed infinity-condition bound")
    ident = [[float(i == j) for j in range(n)] for i in range(n)]
    return ci, {"computed_condition_inf": condition,
                "inverse_left_residual": maxdiff(matmul(ci, c), ident),
                "inverse_right_residual": maxdiff(matmul(c, ci), ident)}


def objective(w, bias, a, b, examples):
    delta = matmul(b, a)
    merged = combine(w, delta)
    losses, logits, probabilities = [], [], []
    g = [[0.0]*len(w[0]) for _ in w]
    n = len(examples)
    for example in examples:
        x, y = example["x"], example["y"]
        z = [math.fsum(v*t for v, t in zip(row, x)) + bj for row, bj in zip(merged, bias)]
        top = max(z)
        e = [math.exp(v-top) for v in z]
        denom = math.fsum(e)
        p = [v/denom for v in e]
        losses.append((top-z[y]) + math.log(denom))
        logits.append(z)
        probabilities.append(p)
        for j in range(len(w)):
            error = (p[j]-float(j == y))/n
            for k, xk in enumerate(x):
                g[j][k] += error*xk
    return {"loss": math.fsum(losses)/n, "logits": logits, "probabilities": probabilities,
            "predictions": [max(range(len(z)), key=lambda j: (z[j], -j)) for z in logits],
            "delta": delta, "gradient_delta": g}


def _step(problem, a, b, eta_a, eta_b):
    before = objective(problem["W"], problem["bias"], a, b, problem["examples"])
    g = before["gradient_delta"]
    ga, gb = matmul(transpose(b), g), matmul(g, transpose(a))
    aa, bb = combine(a, ga, beta=-eta_a), combine(b, gb, beta=-eta_b)
    after = objective(problem["W"], problem["bias"], aa, bb, problem["examples"])
    left = matmul(matmul(b, transpose(b)), g)
    right = matmul(g, matmul(transpose(a), a))
    cross = matmul(matmul(g, transpose(before["delta"])), g)
    first = combine(left, right, -eta_a, -eta_b)
    second = [[eta_a*eta_b*x for x in row] for row in cross]
    predicted = combine(first, second)
    actual = combine(after["delta"], before["delta"], beta=-1)
    return {"before": before, "after": after, "A_after": aa, "B_after": bb,
            "gradient_A": ga, "gradient_B": gb, "first_order_delta": first,
            "second_order_delta": second, "predicted_delta_change": predicted,
            "actual_delta_change": actual,
            "update_identity_max_abs_residual": maxdiff(actual, predicted),
            "first_order_norm": norm(first), "second_order_norm": norm(second),
            "actual_change_norm": norm(actual), "A_norm": norm(a), "B_norm": norm(b),
            "gradient_A_norm": norm(ga), "gradient_B_norm": norm(gb)}


def audit(problem, gauge=None, learning_rate=0.1, scalar_gauge=None):
    problem = validate_problem(problem)
    lr = _scalar(learning_rate, "learning rate", 2.0)
    if lr < 0:
        raise ValueError("learning rate must be nonnegative")
    r = len(problem["A"])
    c = None
    if scalar_gauge is not None:
        if gauge is not None:
            raise ValueError("choose a matrix gauge or scalar gauge, not both")
        c = _scalar(scalar_gauge, "scalar gauge", 32)
        if abs(c) < 1/32:
            raise ValueError("absolute scalar gauge must be between 1/32 and 32")
        gauge = [[c*float(i == j) for j in range(r)] for i in range(r)]
    if gauge is None:
        gauge = [[float(i == j) for j in range(r)] for i in range(r)]
    gauge = matrix(gauge, "gauge", rows=r, cols=r)
    ci, diagnostics = inverse(gauge)
    ag, bg = matmul(ci, problem["A"]), matmul(problem["B"], gauge)
    original = _step(problem, problem["A"], problem["B"], lr, lr)
    changed = _step(problem, ag, bg, lr, lr)
    report = {"schema": "loralab.factor-coordinates.v1", "classes": len(problem["W"]),
              "features": len(problem["W"][0]), "factor_width": r,
              "matrix_rank_upper_bound": min(r, len(problem["W"]), len(problem["W"][0])),
              "factor_parameters": r*(len(problem["W"])+len(problem["W"][0])),
              "full_weight_parameters": len(problem["W"])*len(problem["W"][0]),
              "frozen_bias_parameters": len(problem["bias"]),
              "learning_rate": lr, "gauge": gauge, "gauge_inverse": ci,
              "gauge_diagnostics": diagnostics, "original": original, "transformed": changed,
              "same_model_delta_max_abs_residual": maxdiff(original["before"]["delta"], changed["before"]["delta"]),
              "same_model_logits_max_abs_residual": maxdiff(original["before"]["logits"], changed["before"]["logits"]),
              "next_delta_max_abs_difference": maxdiff(original["after"]["delta"], changed["after"]["delta"])}
    if c is not None:
        compensated = _step(problem, ag, bg, lr/(c*c), lr*c*c)
        report["scalar_compensated"] = compensated
        report["compensated_next_delta_max_abs_residual"] = maxdiff(original["after"]["delta"], compensated["after"]["delta"])
        report["compensated_learning_rate_A"] = lr/(c*c)
        report["compensated_learning_rate_B"] = lr*c*c
    def finite(x):
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("computation exceeded finite binary64 range")
        if isinstance(x, dict):
            for v in x.values(): finite(v)
        if isinstance(x, list):
            for v in x: finite(v)
    finite(report)
    return report
