"""Generate the paper-quality figures pre-registered in RESEARCH_REPORT.md section 8.

Every plotted number is read from the versioned study summaries under
``artifacts/study-summaries/``; ``act.evolution.figures_data`` recomputes each
stored verdict first and refuses to plot on any disagreement, so the figures
can never silently drift from the machine-checked records.

Usage (writes PNGs next to the summaries):

    PYTHONPATH=src python scripts/generate_figures.py                # coupled
    PYTHONPATH=src python scripts/generate_figures.py --set decoupled

Outputs (all 200 dpi, deterministic):
  artifacts/study-summaries/figures/fig1-winrate-forest.png
  artifacts/study-summaries/figures/fig2-quality-vs-cost.png
  artifacts/study-summaries/figures/fig3-delta-distributions.png
  artifacts/study-summaries/figures/fig4-probe-rank-heatmap.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from act.evolution.figures_data import (
    DECOUPLED_STUDIES,
    STUDIES,
    check_summary,
    clopper_pearson_interval,
    joint_rank_fractions,
    load_summaries,
    paired_deltas,
    probe_concordance,
    probe_cost_pct,
    summaries_dir,
)
from act.evolution.figures_support import (
    FAIL_COLOR,
    MATPLOTLIB_AVAILABLE,
    PASS_COLOR,
    plt,
)

ROOT = Path(__file__).resolve().parent.parent

if not MATPLOTLIB_AVAILABLE:  # pragma: no cover - environment guard
    raise SystemExit(
        "matplotlib is required to generate the figures; install it into the "
        "environment running scripts/generate_figures.py"
    )


def figure_1_forest(
    summaries: dict[str, dict],
    studies: tuple[tuple[str, str, str], ...] = STUDIES,
    title: str | None = None,
) -> "plt.Figure":
    """Win-rate forest plot: one row per study, 95% binomial interval, pass line."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
    y_positions = list(range(len(studies), 0, -1))
    for y, (key, label, _) in zip(y_positions, studies):
        summary = summaries[key]
        wins = summary["adaptive_wins"]
        n = len(summary["experiments"])
        rate = wins / n
        lo, hi = clopper_pearson_interval(wins, n)
        color = PASS_COLOR if summary["gate_passed"] else FAIL_COLOR
        ax.errorbar(
            rate, y, xerr=[[rate - lo], [hi - rate]], fmt="o", color=color,
            ecolor=color, elinewidth=1.6, capsize=3, markersize=6,
        )
        # Keep the label inside the canvas: flip it left of the marker when
        # the point sits in the right part of the axis.
        on_left = rate > 0.56
        ax.annotate(
            f"{wins}/{n}  (pass at {summary['gate_min_wins']}, p = {summary['gate_p_value']:.3g})",
            xy=(rate, y), xytext=(-8, 0) if on_left else (8, 0),
            textcoords="offset points", va="center",
            ha="right" if on_left else "left", fontsize=8, color="#333333",
        )
    ax.set_yticks(y_positions, labels=[label for _, label, _ in studies], fontsize=9)
    ax.set_xlabel("Adaptive (treatment) win rate vs matched random churn", fontsize=10)
    ax.set_xlim(0.30, 0.68)
    ax.set_ylim(0.4, len(studies) + 0.6)
    threshold = 209 / 384
    ax.axvline(threshold, color="#666666", linestyle="--", linewidth=1.0)
    ax.annotate(
        "384-seed pass boundary (209/384 = 54.4%)",
        xy=(threshold, 0.55), fontsize=8, color="#666666",
        ha="center", va="bottom",
    )
    ax.axvline(0.5, color="#bbbbbb", linewidth=0.8)
    ax.set_title(
        title
        or "Figure 1 — Win rates across the nine\ngate-style studies (exact binomial 95% CI)",
        fontsize=11,
    )
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def figure_2_cost(
    summaries: dict[str, dict],
    studies: tuple[tuple[str, str, str], ...] = STUDIES,
    title: str | None = None,
) -> "plt.Figure":
    """Quality (mean paired delta) vs probe cost, marker size = seeds, color = verdict."""
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)
    for key, label, _ in studies:
        summary = summaries[key]
        n = len(summary["experiments"])
        x = summary["mean_adaptive_delta"]
        y = probe_cost_pct(summary)
        color = PASS_COLOR if summary["gate_passed"] else FAIL_COLOR
        edge = "#0d5c26" if summary["gate_passed"] else "#7f1811"
        ax.scatter(x, y, s=40 + n / 2, c=color, edgecolors=edge, linewidths=0.8, zorder=3)
        ax.annotate(
            label.split(". ", 1)[1], xy=(x, y), xytext=(6, 6),
            textcoords="offset points", fontsize=7.5,
        )
    ax.axhline(5.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.annotate(
        "5% pre-registered cost bar", xy=(0.062, 5.0), xytext=(0, -11),
        textcoords="offset points", fontsize=8, color="#666666",
    )
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8)
    ax.annotate(
        "left of zero: treatment better", xy=(0.0, 88), xytext=(4, 0),
        textcoords="offset points", fontsize=8, color="#888888", rotation=90, va="top",
    )
    ax.set_xlabel("Mean paired loss delta (adaptive − random; lower is better)", fontsize=10)
    ax.set_ylabel("Probe cost (% of treatment training FLOPs)", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(2.5, 160)
    ax.set_title(
        title
        or "Figure 2 — Quality vs probe cost; the passing family buys quality with probe FLOPs",
        fontsize=11,
    )
    ax.grid(color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def figure_3_deltas(
    summaries: dict[str, dict],
    studies: tuple[tuple[str, str, str], ...] = STUDIES,
    title: str | None = None,
) -> "plt.Figure":
    """Per-seed paired delta distributions for the verdict-contrast studies."""
    wanted = ["legacy", "probe-loss", "cheap-probe", "cadence", "variance"]
    chosen = [key for key in wanted if key in summaries]
    labels = [label for key, label, _ in studies if key in chosen]
    data = [np.asarray(paired_deltas(summaries[key])) for key in chosen]
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
    bp = ax.boxplot(
        data,
        positions=list(range(1, len(chosen) + 1)),
        showfliers=False,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#222222", "linewidth": 1.4},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, key in zip(bp["boxes"], chosen):
        patch.set_facecolor("#a8d5b5" if summaries[key]["gate_passed"] else "#f0b5ae")
        patch.set_edgecolor("#333333")
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.annotate(
        "0 = no adaptive benefit", xy=(0.55, 0.0), xytext=(0, 5),
        textcoords="offset points", fontsize=8, color="#666666",
    )
    ax.set_xticks(list(range(1, len(chosen) + 1)), labels, fontsize=8.5)
    ax.set_ylabel("Per-seed paired loss delta (adaptive − random)", fontsize=10)
    ax.set_title(
        title
        or "Figure 3 — Per-seed paired delta distributions (studies 1, 3, 4, 5, 6)",
        fontsize=11,
    )
    ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def figure_4_heatmap(summaries_dir_path: Path | None = None) -> "plt.Figure":
    """Probe rank vs realized final rank heatmap from the counterfactual branches."""
    directory = summaries_dir_path if summaries_dir_path is not None else summaries_dir()
    payload = json.loads((directory / "probe-validity-analysis.json").read_text())
    per_seed = payload["per_seed"]
    joint = np.asarray(joint_rank_fractions(per_seed))
    concordance, total = probe_concordance(per_seed)

    stored = payload.get("results", {}).get("argmin_concordance")
    assert stored is not None and abs(concordance - stored) <= 5e-4, (
        f"recomputed argmin concordance {concordance} != stored {stored} "
        "(stored values are rounded to 3 decimals)"
    )

    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=200)
    im = ax.imshow(joint, cmap="Blues", vmin=0.0, vmax=joint.max())
    k = joint.shape[0]
    for i in range(k):
        for j in range(k):
            value = joint[i, j]
            ax.text(
                j, i, f"{value:.3f}", ha="center", va="center", fontsize=9,
                color="white" if value > joint.max() * 0.55 else "#222222",
            )
    ax.set_xticks(range(k), [f"final {i + 1}" for i in range(k)])
    ax.set_yticks(range(k), [f"probe {i + 1}" for i in range(k)])
    ax.set_xlabel("Realized final-loss rank among the 4 branches", fontsize=10)
    ax.set_ylabel("Probe-loss rank at the first mutation event", fontsize=10)
    ax.set_title(
        f"Figure 4 — Probe rank vs final rank ({total} counterfactual seeds);\n"
        f"argmin concordance {concordance:.3f} vs chance 0.250",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of seeds")
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set", choices=("coupled", "decoupled"), default="coupled",
        help="which factor-map record to plot: 'coupled' (the registered "
        "shared-RNG summaries) or 'decoupled' (the section 9.11 recheck, "
        "written into study-summaries/decoupled/)",
    )
    args = parser.parse_args()

    base = summaries_dir()
    if args.set == "coupled":
        directory = base
        studies = STUDIES
        titles: dict[str, str | None] = {}
    else:
        directory = base / "decoupled"
        studies = DECOUPLED_STUDIES
        # Study 1 is not reconstructible on the decoupled path (section 9.10),
        # so the decoupled set is eight rows and says so in the titles rather
        # than silently dropping it or substituting the coupled summary.
        titles = {
            "fig1": (
                "Figure 1 — Win rates across the eight\nreconstructible gate-style "
                "studies, decoupled arm RNG (study 1: not reconstructible)"
            ),
            "fig2": (
                "Figure 2 — Quality vs probe cost, decoupled arm RNG; the passing "
                "family buys quality with probe FLOPs"
            ),
            "fig3": (
                "Figure 3 — Per-seed paired delta distributions, decoupled arm RNG "
                "(studies 3, 4, 5, 6)"
            ),
        }
    summaries = load_summaries(directory, studies=studies)
    for key, _, filename in studies:
        check_summary(summaries[key], filename)

    figure_dir = directory / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (
            "fig1-winrate-forest.png",
            figure_1_forest(summaries, studies, titles.get("fig1")),
        ),
        (
            "fig2-quality-vs-cost.png",
            figure_2_cost(summaries, studies, titles.get("fig2")),
        ),
        (
            "fig3-delta-distributions.png",
            figure_3_deltas(summaries, studies, titles.get("fig3")),
        ),
        # The counterfactual probe-rank analysis behind figure 4 was not part of
        # the recheck (it is not an arm comparison), so both sets plot the same
        # committed analysis and the decoupled set labels it as such.
        ("fig4-probe-rank-heatmap.png", figure_4_heatmap(base)),
    )
    for name, figure in outputs:
        target = figure_dir / name
        figure.savefig(target)
        plt.close(figure)
        print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
