"""Bjontegaard-Delta metrics adapted to a rate-vs-accuracy curve.

Classic BD-Rate compares two codecs by the average bitrate difference at equal
*quality*, where quality is usually PSNR. For machine vision the "quality" axis
is the **task accuracy** instead (higher = better, exactly like PSNR), so the
same piecewise/polynomial integration applies unchanged:

  * ``bd_rate``  : average % bitrate change of *test* vs *anchor* at equal
                   accuracy. Negative => test needs fewer bits for the same
                   accuracy => test is better.
  * ``bd_metric``: average accuracy change at equal bitrate.

Rates are integrated in the log domain (standard). We fit a cubic when there
are >=4 points, else fall back to a lower-order fit, and integrate over the
overlapping accuracy (resp. rate) range.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _prep(rate: Sequence[float], metric: Sequence[float]):
    r = np.asarray(rate, dtype=np.float64)
    m = np.asarray(metric, dtype=np.float64)
    order = np.argsort(m)
    return r[order], m[order]


def _fit_order(n: int) -> int:
    if n >= 4:
        return 3
    if n == 3:
        return 2
    return 1


def bd_rate(
    rate_anchor: Sequence[float],
    metric_anchor: Sequence[float],
    rate_test: Sequence[float],
    metric_test: Sequence[float],
) -> float:
    """Average % bitrate difference (test - anchor) at equal accuracy.

    Returns a percentage; negative means the test pipeline saves bits.
    """
    r1, m1 = _prep(rate_anchor, metric_anchor)
    r2, m2 = _prep(rate_test, metric_test)
    lr1, lr2 = np.log(r1), np.log(r2)

    p1 = np.polyfit(m1, lr1, _fit_order(len(m1)))
    p2 = np.polyfit(m2, lr2, _fit_order(len(m2)))

    lo = max(min(m1), min(m2))
    hi = min(max(m1), max(m2))
    if hi <= lo:
        return float("nan")  # no overlap in accuracy -> BD-Rate undefined

    P1 = np.polyint(p1)
    P2 = np.polyint(p2)
    int1 = np.polyval(P1, hi) - np.polyval(P1, lo)
    int2 = np.polyval(P2, hi) - np.polyval(P2, lo)
    avg_diff = (int2 - int1) / (hi - lo)
    return float((np.exp(avg_diff) - 1.0) * 100.0)


def bd_metric(
    rate_anchor: Sequence[float],
    metric_anchor: Sequence[float],
    rate_test: Sequence[float],
    metric_test: Sequence[float],
) -> float:
    """Average accuracy difference (test - anchor) at equal bitrate."""
    r1, m1 = _prep(rate_anchor, metric_anchor)
    r2, m2 = _prep(rate_test, metric_test)
    lr1, lr2 = np.log(r1), np.log(r2)

    p1 = np.polyfit(lr1, m1, _fit_order(len(lr1)))
    p2 = np.polyfit(lr2, m2, _fit_order(len(lr2)))

    lo = max(min(lr1), min(lr2))
    hi = min(max(lr1), max(lr2))
    if hi <= lo:
        return float("nan")

    P1 = np.polyint(p1)
    P2 = np.polyint(p2)
    int1 = np.polyval(P1, hi) - np.polyval(P1, lo)
    int2 = np.polyval(P2, hi) - np.polyval(P2, lo)
    return float((int2 - int1) / (hi - lo))
