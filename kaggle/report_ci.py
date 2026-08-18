#!/usr/bin/env python
"""Aggregate multi-seed BD-Rate results into a mean +/- CI table with a p-value.

Point this at the per-seed output dirs of the SAME config (each holding an
``eval/results.json`` from run_kaggle.py) and it reports, for every same-codec
preprocessor-gain pair (prep+compressai/h264/h265 vs its anchor):

    n, mean BD-Rate, sample std, 95% CI, and a ONE-SIDED t-test p-value for
    H1: mean < 0  (i.e. the preprocessor genuinely saves bits).

Usage
-----
    python kaggle/report_ci.py outputs/b2_s0 outputs/b2_s1 outputs/b2_s2
    python kaggle/report_ci.py --glob "outputs/b2_s*"
    python kaggle/report_ci.py --selfcheck        # verify the stats
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


def _mean_std(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan"), 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)  # sample variance
    return mean, math.sqrt(var), math.sqrt(var / n)   # std, sem


def ttest_one_sided_lt0(xs):
    """One-sided t-test, H0: mean=0 vs H1: mean<0. Returns (t, df, p).

    p = P(T <= t_obs) under H0 (small p => significant savings). Needs scipy for
    the exact Student-t CDF; falls back to a normal approx (labelled) if absent."""
    n = len(xs)
    mean, std, sem = _mean_std(xs)
    if n < 2 or sem == 0:
        return float("nan"), n - 1, float("nan"), "n<2"
    t = mean / sem
    df = n - 1
    try:
        from scipy import stats
        return t, df, float(stats.t.cdf(t, df)), "student-t"
    except Exception:
        # normal approximation (overconfident at small n; only a fallback)
        p = 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
        return t, df, p, "normal-approx"


def _collect(dirs):
    """dir/eval/results.json -> {pair_label: [bd_rate_pct per seed]}."""
    rows: dict = {}
    used = []
    for d in dirs:
        rj = Path(d) / "eval" / "results.json"
        if not rj.exists():
            print(f"  (skip {d}: no eval/results.json)")
            continue
        res = json.loads(rj.read_text(encoding="utf-8"))
        used.append(d)
        for label, v in res.get("bd_prep_gain", {}).items():
            rows.setdefault(label, []).append(v["bd_rate_pct"])
    return rows, used


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed BD-Rate CI + p-value.")
    p.add_argument("dirs", nargs="*", help="per-seed output dirs")
    p.add_argument("--glob", default=None, help="glob for output dirs (alt to listing)")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args()

    if a.selfcheck:
        _selfcheck()
        return

    dirs = a.dirs or (sorted(glob.glob(a.glob)) if a.glob else [])
    if not dirs:
        p.error("give per-seed dirs or --glob")

    rows, used = _collect(dirs)
    if not rows:
        print("no results.json found under the given dirs")
        return

    print(f"\n=== BD-Rate over {len(used)} seeds (negative = bit savings) ===")
    print(f"  dirs: {', '.join(used)}\n")
    header = f"  {'pair':30s}  {'n':>2s}  {'mean%':>8s}  {'std':>6s}  {'95% CI':>16s}  {'p(<0)':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, xs in rows.items():
        mean, std, sem = _mean_std(xs)
        t, df, pval, method = ttest_one_sided_lt0(xs)
        # 95% CI via normal z=1.96 (small-n: indicative, not exact)
        half = 1.96 * sem if sem == sem else float("nan")
        ci = f"[{mean-half:+.2f},{mean+half:+.2f}]" if half == half else "     n/a     "
        star = " *" if (pval == pval and pval < 0.05) else ""
        print(f"  {label:30s}  {len(xs):>2d}  {mean:+8.2f}  {std:6.2f}  {ci:>16s}  {pval:7.3f}{star}")
    print(f"\n  p = one-sided t-test H1: mean<0 ({method}); * = p<0.05. "
          f"Small n -> treat CI/p as indicative.")


def _selfcheck() -> None:
    # symmetric-about-0 sample -> mean 0, p ~ 0.5
    t, df, p, _ = ttest_one_sided_lt0([-1.0, 0.0, 1.0])
    assert abs(t) < 1e-9 and df == 2 and abs(p - 0.5) < 1e-6, (t, df, p)
    # clearly-negative sample -> p well below 0.5
    _, _, p_neg, _ = ttest_one_sided_lt0([-5.0, -4.0, -6.0])
    assert p_neg < 0.05, p_neg
    # positive sample -> p well above 0.5
    _, _, p_pos, _ = ttest_one_sided_lt0([5.0, 4.0, 6.0])
    assert p_pos > 0.95, p_pos
    mean, std, sem = _mean_std([2.0, 4.0])
    assert mean == 3.0 and abs(std - math.sqrt(2)) < 1e-9
    print("report_ci self-check passed")


if __name__ == "__main__":
    main()
