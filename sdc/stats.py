"""Small stats helpers shared by the evals (INTERFACES.md: Wilson CIs for rates, bootstrap(1000) for differences)."""
from __future__ import annotations

import math
import random


def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(xs):
    """xs: list of 0/1 (None entries dropped) -> metric dict with Wilson CI."""
    xs = [int(bool(x)) for x in xs if x is not None]
    n = len(xs)
    return {"value": (sum(xs) / n) if n else float("nan"), "ci95": wilson(sum(xs), n), "n": n}


def _q(v, q):
    v = sorted(v)
    i = q * (len(v) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def mean(xs, B: int = 1000, seed: int = 0):
    """Mean of real values with a percentile-bootstrap CI (for per-episode scores such as accuracy)."""
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"value": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    rng = random.Random(seed)
    bs = [sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B)]
    return {"value": sum(xs) / n, "ci95": [_q(bs, .025), _q(bs, .975)], "n": n}


def diff(a, b, B: int = 1000, seed: int = 0, paired: bool = False):
    """mean(a) - mean(b) with a bootstrap CI. paired=True resamples index pairs (a[i], b[i])."""
    a = [float(x) for x in a if x is not None]
    b = [float(x) for x in b if x is not None]
    if not a or not b:
        return {"value": float("nan"), "ci95": [float("nan")] * 2, "n": min(len(a), len(b))}
    rng = random.Random(seed)
    if paired:
        assert len(a) == len(b)
        n = len(a)
        bs = []
        for _ in range(B):
            idx = [rng.randrange(n) for _ in range(n)]
            bs.append(sum(a[i] - b[i] for i in idx) / n)
    else:
        bs = [sum(rng.choice(a) for _ in a) / len(a) - sum(rng.choice(b) for _ in b) / len(b) for _ in range(B)]
    return {"value": sum(a) / len(a) - sum(b) / len(b), "ci95": [_q(bs, .025), _q(bs, .975)],
            "n": min(len(a), len(b))}
