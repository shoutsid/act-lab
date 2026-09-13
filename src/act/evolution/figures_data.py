"""Dependency-light data assembly for the pre-registered report figures.

Every figure in ``RESEARCH_REPORT.md`` section 8 is plotted from the versioned
study summaries under ``artifacts/study-summaries/``. This module owns the
loading, the machine checks that recomputed verdicts agree with the stored
ones, and the small statistics helpers (exact binomial Clopper-Pearson
intervals via the regularized incomplete beta) needed to draw them honestly.
It uses only the standard library so tests run without a plotting backend.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ALPHA = 0.05
# Pre-registered pass thresholds by seed count (report section 2).
PASS_THRESHOLDS = {96: 57, 380: 207, 384: 209}

# The report's study table (section 3): key, figure label, summary file.
STUDIES: tuple[tuple[str, str, str], ...] = (
    ("legacy", "1. Legacy signals (48 steps)", "gate3-legacy-signal-384.json"),
    ("guided", "2. Guided upper bound (48 steps)", "guided-upper-bound-96.json"),
    ("probe-loss", "3. Probe-loss cap4/2b (48 steps)", "gate3-probe-loss-384.json"),
    ("cheap-probe", "4. Cheap probe cap2/1b (144 steps)", "gate3-cheap-probe-380.json"),
    ("cadence", "5. Cadence 6 events cap2/1b", "gate3-cadence-384.json"),
    ("variance", "6. Variance 4 batches cap2/4b", "gate3-variance-384.json"),
    ("horizon", "7. Horizon replication cap4/2b", "gate3-horizon-384.json"),
    ("rich-a", "8. Richness A cap4/1b", "gate3-rich-384.json"),
    ("rich-b", "9. Richness B cap2/2b", "gate3-narrow-384.json"),
)


def summaries_dir(root: Path | None = None) -> Path:
    base = root if root is not None else Path(__file__).resolve().parents[3]
    return base / "artifacts" / "study-summaries"


def load_summaries(directory: Path | None = None) -> dict[str, dict]:
    """Load the nine study summaries keyed by figure key."""
    directory = directory if directory is not None else summaries_dir()
    loaded: dict[str, dict] = {}
    for key, _, filename in STUDIES:
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"missing study summary: {path}")
        loaded[key] = json.loads(path.read_text())
    return loaded


def arm_totals(summary: dict, mode: str) -> tuple[int, int]:
    """Aggregate (search_flops, training_flops) for one arm across seeds."""
    search = 0
    training = 0
    for experiment in summary["experiments"]:
        for arm in experiment["arms"]:
            if arm["mode"] == mode:
                search += int(arm["search_flops"])
                training += int(arm["training_flops"])
    return search, training


def probe_cost_pct(summary: dict) -> float:
    """Treatment probe cost as a percentage of training FLOPs.

    This is the report's definition: aggregate treatment search FLOPs over
    aggregate treatment training FLOPs. Search FLOPs are booked into the
    treatment arm's own training totals as they occur, so its training FLOPs
    is the correct denominator for the "share of training spent searching".
    """
    treatment = summary.get("treatment", "adaptive")
    search, training = arm_totals(summary, treatment)
    return 100.0 * search / training


def paired_deltas(summary: dict, treatment: str = "adaptive") -> list[float]:
    """Per-seed treatment-minus-random final-loss deltas."""
    deltas: list[float] = []
    for experiment in summary["experiments"]:
        losses = {arm["mode"]: arm["final_loss"] for arm in experiment["arms"]}
        deltas.append(losses[treatment] - losses["random"])
    return deltas


def check_summary(summary: dict, name: str = "summary") -> None:
    """Recompute the stored verdict; raise ``AssertionError`` on any mismatch.

    Guards the figures against silent drift from the machine-checked records:
    seed counts, the binomial p-value, the pass decision, and the mean paired
    delta must all reproduce from the raw per-seed experiments.
    """
    experiments = summary["experiments"]
    n = len(experiments)
    wins = summary["adaptive_wins"]
    if n != (summary["seeds"] if isinstance(summary["seeds"], int) else len(summary["seeds"])):
        raise AssertionError(f"{name}: experiment count does not match seed count")

    expected_min = PASS_THRESHOLDS.get(n)
    if expected_min is not None and summary["gate_min_wins"] != expected_min:
        raise AssertionError(
            f"{name}: gate_min_wins {summary['gate_min_wins']} != pre-registered {expected_min}"
        )

    survival = _binomial_survival(wins, n)
    if not math.isclose(survival, summary["gate_p_value"], rel_tol=1e-9):
        raise AssertionError(f"{name}: binomial p {survival} != stored {summary['gate_p_value']}")

    if summary["gate_passed"] != (wins >= summary["gate_min_wins"]):
        raise AssertionError(f"{name}: gate_passed disagrees with wins >= gate_min_wins")
    if summary["gate_passed"] != (survival <= ALPHA):
        raise AssertionError(f"{name}: gate_passed disagrees with binomial p <= alpha")

    deltas = paired_deltas(summary, summary.get("treatment", "adaptive"))
    mean_delta = sum(deltas) / n
    if not math.isclose(mean_delta, summary["mean_adaptive_delta"], rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(
            f"{name}: mean delta {mean_delta} != stored {summary['mean_adaptive_delta']}"
        )


def probe_concordance(per_seed: dict[str, dict]) -> tuple[float, int]:
    """Argmin concordance from the counterfactual branches.

    Returns the fraction of seeds whose probe-loss argmin branch is also the
    realized final-loss argmin, plus the seed count.
    """
    concordant = 0
    total = 0
    for entry in per_seed.values():
        probe_losses = entry["probe_losses"]
        finals = entry["finals"]
        if probe_losses[finals.index(min(finals))] == min(probe_losses):
            concordant += 1
        total += 1
    return concordant / total, total


def joint_rank_fractions(per_seed: dict[str, dict]) -> list[list[float]]:
    """Probe-rank x final-rank conditional fractions over the counterfactual seeds.

    Rows are probe ranks, columns are realized final ranks, both 0-based and
    ascending (rank 0 = lowest loss), ties broken by candidate order. Every
    seed contributes exactly one candidate per probe rank, so each row is the
    conditional distribution of the realized final rank given that probe rank
    and sums to 1; cell (0, 0) is the argmin concordance.
    """
    k = len(next(iter(per_seed.values()))["probe_losses"])
    joint = [[0.0] * k for _ in range(k)]
    total = 0
    for entry in per_seed.values():
        probe_rank = _ranks(entry["probe_losses"])
        final_rank = _ranks(entry["finals"])
        for candidate in range(k):
            joint[probe_rank[candidate]][final_rank[candidate]] += 1.0
        total += 1
    return [[cell / total for cell in row] for row in joint]


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b), 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Evaluate the continued fraction only where it converges well: for x in
    # the upper region the forward fraction stagnates and the symmetry
    # I_x(a, b) = 1 - I_{1-x}(b, a) must be used (the crossover follows
    # Numerical Recipes' guidance; verified against scipy in tests).
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_lower(1.0 - x, b, a)
    return _regularized_lower(x, a, b)


def _regularized_lower(x: float, a: float, b: float) -> float:
    """Forward continued-fraction evaluation of I_x(a, b) for the lower region."""
    log_prefix = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    # The continued fraction converges to a * I_x(a, b) * B(a, b) in this
    # parameterization; the division by ``a`` below is part of the identity
    # (cf. Numerical Recipes' ``bt * betacf(a, b, x) / a``), verified against
    # scipy in tests/test_report_figures.py.
    return _betacf(x, a, b) * math.exp(log_prefix) / a


def beta_quantile(prob: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b) via bisection on the exact beta CDF."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must be in [0, 1]")
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if beta_cdf(mid, a, b) < prob:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson_interval(wins: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact Clopper-Pearson 95% interval for a binomial proportion."""
    if not 0 <= wins <= n:
        raise ValueError("wins must be in [0, n]")
    lower = 0.0 if wins == 0 else beta_quantile(alpha / 2, wins, n - wins + 1)
    upper = 1.0 if wins == n else beta_quantile(1 - alpha / 2, wins + 1, n - wins)
    return lower, upper


def _betacf(x: float, a: float, b: float, max_iter: int = 500, eps: float = 3e-16) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz's method)."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _binomial_survival(wins: int, trials: int) -> float:
    """Exact P(X >= wins) for X ~ Binomial(trials, 0.5)."""
    wins = max(0, min(int(wins), trials))
    favorable = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    return favorable / 2**trials


def _ranks(values: list[float]) -> list[int]:
    """0-based ascending ranks; ties broken by original order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks
