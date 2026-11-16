"""The paper's figures, with a sidecar CSV of exactly the plotted numbers and a notes file.

Style: 8 pt serif (Times New Roman, STIX for mathematics), TrueType fonts embedded in the PDF
(pdf.fonttype 42), the Okabe-Ito colours with a distinct marker and line style per series so
that a greyscale print still separates them. Widths are parameters (column 3.5 in, text
7.16 in by default); every figure is saved as PDF and as PNG at 300 dpi at that width.

Each figure writes:
    {name}.pdf, {name}.png
    {name}.csv        the plotted numbers, one row per point (or bar, or cell), with n
    {name}_notes.txt  a proposed caption, and what is provisional or pending; a series that
                      waits for runs still training is absent from the plot and named here

Axis names: "label-HD (HD to global)" and "quantity-HD (HD to uniform)". FedArtML's pairwise
values never appear on an axis.

Colours: the eight-colour palette of M. Okabe and K. Ito, "Color Universal Design (CUD) - How
to make figures and presentations that are friendly to Colorblind people",
https://jfly.uni-koeln.de/color/ (2002, modified 2008), Fig. 16. The page shows the palette as
an image only; the hex values below are the palette's commonly used RGB values, not checked
against that image.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paper_data import DATASET_NAMES
from analysis.paper_results import (
    ALGORITHM_NAMES,
    ALGORITHMS,
    CLASSES,
    NATURAL,
    RQ2_ARMS,
    RQ2_GRID,
    Bundle,
    _cell,
    part_means,
    rq2_arms,
    rq2_levels,
    rq3_levels,
    trial_set_label,
)
from analysis.paper_stats import mean_sd
from experiments.recording import atomic_write_text

COLUMN_WIDTH, TEXT_WIDTH = 3.5, 7.16
FONT_SIZE = 8
DPI = 300
LABEL_AXIS = "label-HD (HD to global)"
QUANTITY_AXIS = "quantity-HD (HD to uniform)"

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
ARM_STYLE = {
    "natural uncapped": (OKABE_ITO["blue"], "o"),
    "natural capped": (OKABE_ITO["vermillion"], "s"),
    "pure": (OKABE_ITO["green"], "^"),
    "natural per-client": (OKABE_ITO["orange"], "D"),
    NATURAL: (OKABE_ITO["black"], "o"),
}
LINESTYLE = {"fedavg": "-", "fedprox": "--"}
LEVEL_STYLE = {
    0.0: ("#555555", "o"),
    0.35: (OKABE_ITO["blue"], "s"),
    0.7: (OKABE_ITO["vermillion"], "^"),
}


@dataclass(frozen=True)
class Widths:
    column: float = COLUMN_WIDTH
    text: float = TEXT_WIDTH


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "errorbar.capsize": 1.5,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "savefig.dpi": DPI,
        }
    )
    return plt


def save(fig, directory: Path, name: str, data: pd.DataFrame, notes: list[str]) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.pdf", directory / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=DPI)
    import matplotlib.pyplot as plt

    plt.close(fig)
    atomic_write_text(directory / f"{name}.csv", data.to_csv(index=False, lineterminator="\n"))
    atomic_write_text(directory / f"{name}_notes.txt", "\n".join(f"- {n}" for n in notes) + "\n")
    return paths + [directory / f"{name}.csv", directory / f"{name}_notes.txt"]


def _panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=FONT_SIZE)


def legend_below(fig, axes, ncol: int = 2, handles=None):
    """One legend under the plotting area, so it never covers data: the axes' own entries,
    each label once, unless `handles` is given."""
    if handles is None:
        seen, handles = set(), []
        for ax in np.atleast_1d(axes):
            for h, label in zip(*ax.get_legend_handles_labels()):
                if label not in seen and not label.startswith("_"):
                    seen.add(label)
                    handles.append(h)
    return fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=ncol,
        fontsize=FONT_SIZE - 1,
        handlelength=2.0,
        columnspacing=1.2,
    )


# ---------------------------------------------------------------- RQ1 incidental quantity skew
def _rq1_points(bundle: Bundle, dataset: str, K: int) -> pd.DataFrame:
    from analysis.paper_results import rq1_summary

    s = rq1_summary(bundle)
    return s[(s["dataset"] == dataset) & (s["K"] == K)]


def _quantity_locus(dataset: str) -> pd.DataFrame | None:
    from analysis.paper_data import quantity_only_curve_points
    from analysis.paper_results import _quantity_curve

    curve = _quantity_curve(dataset)
    return None if curve is None else quantity_only_curve_points(curve)


def _rq1_panel(ax, points: pd.DataFrame, locus, cifar_refs: bool, records: list, panel: str):
    if locus is not None:
        lo = locus.sort_values("mean_label_hd")
        ax.fill_between(
            lo["mean_label_hd"],
            lo["mean_quantity_hd"] - lo["sd_quantity_hd"],
            lo["mean_quantity_hd"] + lo["sd_quantity_hd"],
            color="#BBBBBB",
            alpha=0.5,
            linewidth=0,
            label="quantity-only (β sweep)",
        )
        for _, p in lo.iterrows():
            records.append(
                {
                    "panel": panel,
                    "series": "quantity-only (band)",
                    "parameter": "beta",
                    "value": p["beta"],
                    "label_hd_mean": p["mean_label_hd"],
                    "label_hd_sd": p["sd_label_hd"],
                    "quantity_hd_mean": p["mean_quantity_hd"],
                    "quantity_hd_sd": p["sd_quantity_hd"],
                    "n": p["n"],
                }
            )
    if cifar_refs:
        from partitioning.metrics import quantity_hd_from_pairwise

        for i, q in enumerate((0.35, 0.70)):
            ax.axhline(
                q,
                color="#999999",
                lw=0.5,
                ls="-",
                zorder=0,
                label="RQ3 quantity levels" if i == 0 else None,
            )
        for i, qp in enumerate((0.10, 0.17)):
            ax.axhline(
                quantity_hd_from_pairwise(qp, 30),
                color="#444444",
                lw=0.7,
                ls=":",
                zorder=0,
                label="source paper QHD 0.10, 0.17" if i == 0 else None,
            )
        ax.axvline(
            np.sqrt(1 - np.sqrt(0.1)),
            color="#444444",
            lw=0.7,
            ls="-.",
            zorder=0,
            label="label-HD ceiling",
        )
    for arm in ("natural uncapped", "natural capped", "natural per-client"):
        g = points[points["arm"] == arm].sort_values("alpha")
        g = g[g["n_generated"] > 0]
        color, marker = ARM_STYLE[arm]
        ax.errorbar(
            g["label_hd_mean"],
            g["qty_hd_mean"],
            xerr=g["label_hd_sd"],
            yerr=g["qty_hd_sd"],
            color=color,
            marker=marker,
            ms=3,
            lw=0.9,
            elinewidth=0.5,
            label=arm,
        )
        for _, p in g.iterrows():
            records.append(
                {
                    "panel": panel,
                    "series": arm,
                    "parameter": "alpha",
                    "value": p["alpha"],
                    "label_hd_mean": p["label_hd_mean"],
                    "label_hd_sd": p["label_hd_sd"],
                    "quantity_hd_mean": p["qty_hd_mean"],
                    "quantity_hd_sd": p["qty_hd_sd"],
                    "n": p["n_generated"],
                }
            )
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel(LABEL_AXIS)
    ax.set_ylabel(QUANTITY_AXIS)


def rq1_figure(bundle: Bundle, out: Path, widths: Widths) -> list[Path]:
    plt = pyplot()
    fig, axes = plt.subplots(
        2, 1, figsize=(widths.column, widths.column * 1.6), layout="constrained"
    )
    records = []
    _rq1_panel(
        axes[0], _rq1_points(bundle, "cifar10", 30), _quantity_locus("cifar10"), True, records, "a"
    )
    _panel_label(axes[0], "(a) CIFAR-10, K = 30")
    status = bundle.rq1_status.get("covtype")
    notes = []
    if status == "pending":
        axes[1].text(0.5, 0.5, "Covtype: pending", ha="center", transform=axes[1].transAxes)
    else:
        _rq1_panel(
            axes[1],
            _rq1_points(bundle, "covtype", 30),
            _quantity_locus("covtype"),
            False,
            records,
            "b",
        )
    _panel_label(
        axes[1], "(b) Covtype, K = 30" + (" (PROVISIONAL)" if status == "provisional" else "")
    )
    legend_below(fig, axes, ncol=2)
    data = pd.DataFrame(records)
    notes += [
        "Proposed caption: Incidental quantity skew of natural label-skew partitioning, K = 30. Each "
        "point is one Dirichlet alpha (0.03 to 1000; label-HD falls as alpha grows), mean ± sd over "
        "trials on both axes; lines join the alphas in order. Grey band: deliberate quantity skew "
        "(quantity-only partitions over the beta sweep), mean ± sd of its quantity-HD against its "
        "own incidental label-HD. (a) CIFAR-10: dotted lines, the source paper's QHD 0.10 and 0.17 "
        "on our scale (0.397, 0.773); thin lines, the RQ3 quantity levels 0.35 and 0.70; dash-dot, "
        "the label-HD ceiling 0.827 (a client holding one class). (b) Covtype: no ceiling line, "
        "since with imbalanced classes the HD of a one-class client depends on the class.",
        "n per point: CIFAR-10 3 trials (config/grids/rq1.yaml, trials 0-2); Covtype 20 trials "
        "(config/grids/covtype_rq1.yaml, trials 0-19). The quantity-only bands: 20 trials each, "
        "calibration/8f995f9546.json (CIFAR-10) and calibration/b4684ba0de.json (Covtype).",
        "Grey band: drawn as the quantity-only locus (mean ± sd of quantity-HD at each beta, "
        "placed at its mean incidental label-HD). Its vertical extent is the quantity-only range at "
        "K = 30 (CIFAR-10 0.011 to 0.826, Covtype 0.011 to 0.879).",
    ]
    if status == "provisional":
        notes.append(
            "PROVISIONAL: panel (b) uses the Covtype partition measurements of 2026-11-08 (session "
            "folder, uncommitted tree), not partitions recorded through the runner; replaced by "
            "config/grids/covtype_rq1.yaml once generated."
        )
    if _quantity_locus("covtype") is None:
        notes.append(
            "Panel (b): the quantity-only band is absent: no Covtype quantity-only curve yet (pending)."
        )
    return save(fig, out, "rq1_incidental_quantity_k30", data, notes)


def rq1_k_figure(bundle: Bundle, out: Path, widths: Widths) -> list[Path]:
    plt = pyplot()
    fig, axes = plt.subplots(
        2, 1, figsize=(widths.column, widths.column * 1.5), layout="constrained"
    )
    records = []
    for ax, K, panel in zip(axes, (10, 100), "ab"):
        points = _rq1_points(bundle, "cifar10", K)
        _rq1_panel(ax, points, None, False, records, panel)
        _panel_label(ax, f"({panel}) CIFAR-10, K = {K}")
        bad = points[points["n_infeasible"] > 0]
        for i, (_, p) in enumerate(bad.iterrows()):
            ax.text(
                0.02,
                0.95 - 0.08 * i,
                f"× {p['arm']}, α = {p['alpha']:g}: {p['n_infeasible']}/{p['n_total']} infeasible",
                transform=ax.transAxes,
                va="top",
                fontsize=FONT_SIZE - 1,
            )
            records.append(
                {
                    "panel": panel,
                    "series": p["arm"] + " (infeasible)",
                    "parameter": "alpha",
                    "value": p["alpha"],
                    "n": p["n_infeasible"],
                }
            )
    legend_below(fig, axes, ncol=3)
    notes = [
        "Proposed caption: As the RQ1 figure, CIFAR-10 at K = 10 and K = 100; 3 trials per point. "
        "At K = 100, alpha = 0.03, the natural per-class variants (uncapped and capped) exhaust their "
        "1000-redraw min-size budget in every trial (6 partitions, marked in the panel), so those "
        "points do not exist.",
    ]
    return save(fig, out, "rq1_incidental_quantity_cifar_k10_k100", pd.DataFrame(records), notes)


# ---------------------------------------------------------------- RQ2 pure vs natural
def _rq2_series(bundle: Bundle, summary, dataset: str, metric: str):
    """{(arm, algorithm): DataFrame of points}; the CIFAR-10 level-0 natural point starts both natural lines."""
    grid = RQ2_GRID[dataset]
    out = {}
    for algorithm in ALGORITHMS:
        if bundle.is_pending(dataset, algorithm):
            continue
        for arm in RQ2_ARMS:
            pts = []
            for L in rq2_levels(bundle, dataset):
                cell_arm = arm if arm in rq2_arms(dataset, L) else NATURAL
                c = _cell(
                    summary,
                    dataset=dataset,
                    grid=grid,
                    cell_arm=cell_arm,
                    L=L,
                    Q=None,
                    algorithm=algorithm,
                )
                hd = part_means(bundle, dataset, grid, cell_arm, L)
                pts.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "cell_arm": cell_arm,
                        "algorithm": algorithm,
                        "level": L,
                        "label_hd_mean": hd["label_hd"],
                        "y_mean": c[f"{metric}_mean"],
                        "y_sd": c[f"{metric}_sd"],
                        "n": c[f"{metric}_n"],
                        "metric": metric,
                    }
                )
            out[(arm, algorithm)] = pd.DataFrame(pts)
    return out


def _rq2_panel(ax, series: dict, records: list, panel: str, ylabel: str):
    shared_drawn = False
    for (arm, algorithm), pts in series.items():
        color, marker = ARM_STYLE[arm]
        ax.errorbar(
            pts["label_hd_mean"],
            pts["y_mean"],
            yerr=pts["y_sd"],
            color=color,
            marker=marker,
            ls=LINESTYLE[algorithm],
            mfc=color if algorithm == "fedavg" else "white",
            ms=3.5,
            lw=0.9,
            elinewidth=0.5,
            label=f"{arm}, {ALGORITHM_NAMES[algorithm]}",
        )
        shared = pts[pts["cell_arm"] == NATURAL]
        if len(shared) and not shared_drawn:
            ax.plot(
                shared["label_hd_mean"],
                shared["y_mean"],
                "o",
                mfc="white",
                mec="black",
                ms=5,
                zorder=5,
                label="natural, level 0 (uncapped = capped)",
            )
            shared_drawn = True
        for _, p in pts.iterrows():
            if p["cell_arm"] == NATURAL and arm == "natural capped":
                continue  # the shared level-0 point is one point, recorded once
            records.append({"panel": panel, **p.to_dict(), "arm": p["cell_arm"]})
    ax.set_xlabel(LABEL_AXIS)
    ax.set_ylabel(ylabel)


def per_class_panel(ax, diffs: dict, records: list, panel: str, level: float):
    x = np.arange(len(CLASSES))
    w = 0.38
    for i, (key, label, color, hatch) in enumerate(
        (
            ("index", "natural capped, index order", OKABE_ITO["vermillion"], None),
            ("reversed", "natural capped, reversed order", OKABE_ITO["purple"], "////"),
        )
    ):
        d = diffs[key]
        mean = d.mean(axis=0)
        sd = d.std(axis=0, ddof=1) if d.shape[0] > 1 else np.full(d.shape[1], np.nan)
        ax.bar(
            x + (i - 0.5) * w,
            mean,
            w,
            yerr=sd,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            hatch=hatch,
            error_kw={"elinewidth": 0.5, "capsize": 1.2},
            label=label,
        )
        for c, (m, s) in enumerate(zip(mean, sd)):
            records.append(
                {
                    "panel": panel,
                    "level": level,
                    "series": label,
                    "class": c,
                    "class_name": CLASSES[c],
                    "diff_mean": m,
                    "diff_sd": s,
                    "n": d.shape[0],
                    "trials": trial_set_label(diffs["trials"]),
                }
            )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x, CLASSES, rotation=60, ha="right")
    ax.set_ylabel("per-class accuracy − natural uncapped")


def rq2_figure(bundle: Bundle, summary, diffs045: dict, out: Path, widths: Widths) -> list[Path]:
    plt = pyplot()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(widths.text, widths.text * 0.40), layout="constrained")
    records = []
    _rq2_panel(
        axes[0],
        _rq2_series(bundle, summary, "cifar10", "final_acc"),
        records,
        "a",
        "final test accuracy",
    )
    _panel_label(axes[0], "(a) CIFAR-10")
    _rq2_panel(
        axes[1],
        _rq2_series(bundle, summary, "covtype", "final_acc"),
        records,
        "b",
        "final test accuracy",
    )
    _panel_label(axes[1], "(b) Covtype")
    per_class_panel(axes[2], diffs045, records, "c", 0.45)
    _panel_label(axes[2], "(c) CIFAR-10, label-HD 0.45, FedAvg")
    handles = [
        Line2D([], [], color=ARM_STYLE[arm][0], marker=ARM_STYLE[arm][1], ls="-", ms=3.5, label=arm)
        for arm in RQ2_ARMS
    ]
    handles += [
        Line2D([], [], color="black", ls="-", marker="o", ms=3.5, label="FedAvg"),
        Line2D([], [], color="black", ls="--", marker="o", mfc="white", ms=3.5, label="FedProx"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="white",
            mec="black",
            ms=5,
            label="natural, level 0 (uncapped = capped)",
        ),
    ]
    handles += axes[2].get_legend_handles_labels()[0]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        fontsize=FONT_SIZE - 1,
        handlelength=2.0,
    )
    n = len(bundle.trials("cifar10"))
    notes = [
        "Proposed caption: Pure vs natural label skew. (a, b) Final test accuracy against the mean "
        "realized label-HD of each cell, mean ± sd over trials; solid FedAvg, dashed FedProx (hollow "
        "markers). On CIFAR-10 the two natural arms are the same partition at level 0 (open circle), "
        "where both natural lines start. (c) Per-class final accuracy of the capped arm minus the "
        "uncapped arm at label-HD 0.45, FedAvg, with the classes split in index order and in "
        "reversed order; mean ± sd of the per-trial differences.",
        f"n: CIFAR-10 {n} trials {trial_set_label(bundle.trials('cifar10'))}; Covtype "
        f"{len(bundle.trials('covtype'))} trials {trial_set_label(bundle.trials('covtype'))}; panel "
        f"(c) {len(diffs045['trials'])} trials {trial_set_label(diffs045['trials'])} (the class-order "
        "grid's trials).",
    ]
    if bundle.is_pending("covtype", "fedprox"):
        notes.append("Panel (b): Covtype FedProx lines absent: pending (runs still training).")
    return save(fig, out, "rq2_accuracy_vs_label_hd", pd.DataFrame(records), notes)


def rq2_balanced_figure(bundle: Bundle, summary, out: Path, widths: Widths) -> list[Path]:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(widths.column, widths.column * 1.0), layout="constrained")
    records = []
    _rq2_panel(
        ax,
        _rq2_series(bundle, summary, "covtype", "final_balanced_acc"),
        records,
        "a",
        "final balanced accuracy (ours)",
    )
    _panel_label(ax, "Covtype, balanced accuracy (ours)")
    legend_below(fig, ax, ncol=2)
    notes = [
        "Proposed caption: Covtype, final balanced accuracy (mean per-class recall; our addition, "
        "not the source paper's metric) against mean realized label-HD, mean ± sd over "
        f"{len(bundle.trials('covtype'))} trials; solid FedAvg, dashed FedProx.",
    ]
    if bundle.is_pending("covtype", "fedprox"):
        notes.append("FedProx lines absent: pending (runs still training).")
    return save(fig, out, "rq2_covtype_balanced_accuracy_vs_label_hd", pd.DataFrame(records), notes)


def per_class_figure(
    bundle: Bundle, diffs: dict, level: float, out: Path, widths: Widths
) -> list[Path]:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(widths.column, widths.column * 0.85), layout="constrained")
    records = []
    per_class_panel(ax, diffs, records, "a", level)
    _panel_label(ax, f"CIFAR-10, label-HD {level:.2f}, FedAvg")
    fig.legend(loc="outside lower center", ncol=1, fontsize=FONT_SIZE - 1)
    notes = [
        f"Proposed caption: As panel (c) of the RQ2 figure, at label-HD {level:.2f}: per-class final "
        "accuracy of the capped arm minus the uncapped arm, FedAvg, index and reversed class order; "
        f"mean ± sd over {len(diffs['trials'])} trials {trial_set_label(diffs['trials'])}.",
    ]
    name = f"rq2_per_class_difference_{int(round(level * 100)):03d}"
    return save(fig, out, name, pd.DataFrame(records), notes)


# ---------------------------------------------------------------- RQ3 heatmaps and regression
def _heat_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("white_sky", ["#EEF6FC", OKABE_ITO["sky"]])


def rq3_heatmap(
    bundle: Bundle,
    summary,
    add: pd.DataFrame,
    algorithm: str,
    metric: str,
    out: Path,
    widths: Widths,
) -> list[Path]:
    plt = pyplot()
    Ls, Qs = rq3_levels(bundle)
    decimals = 4 if metric == "final_acc" else 1
    values = np.full((len(Ls), len(Qs)), np.nan)
    records = []
    height = widths.column * (1.35 if metric == "rta" else 1.2)
    fig, ax = plt.subplots(figsize=(widths.column, height), layout="constrained")
    for i, L in enumerate(Ls):
        for j, Q in enumerate(Qs):
            c = _cell(
                summary,
                dataset="cifar10",
                grid="rq3",
                cell_arm="combined",
                L=L,
                Q=Q,
                algorithm=algorithm,
            )
            m, s, n = c[f"{metric}_mean"], c[f"{metric}_sd"], c[f"{metric}_n"]
            values[i, j] = m
            hd = part_means(bundle, "cifar10", "rq3", "combined", L, Q)
            hit = add[
                (add["algorithm"] == algorithm) & (add["level"] == L) & (add["quantity_level"] == Q)
            ]
            text = f"{m:.{decimals}f}\n± {s:.{decimals}f}"
            rec = {
                "label_level": L,
                "quantity_level": Q,
                "label_hd_mean": hd["label_hd"],
                "quantity_hd_mean": hd["qty_hd"],
                f"{metric}_mean": m,
                f"{metric}_sd": s,
                "n": n,
            }
            if metric == "rta":
                rec["best_round_mean"] = c["best_round_mean"]
                rec["best_round_min"] = c["best_round_min"]
                rec["best_round_max"] = c["best_round_max"]
                text += f"\nbest round {c['best_round_mean']:.1f}"
            if len(hit):
                h = hit.iloc[0]
                places = 3 if metric == "final_acc" else 1
                text += (
                    f"\ninteraction {h['interaction_mean']:+.{places}f}"
                    f"\n{int(h['n_negative'])} of {int(h['n'])} trials < 0"
                )
                rec.update(
                    {
                        "interaction_mean": h["interaction_mean"],
                        "interaction_sd": h["interaction_sd"],
                        "interaction_n_negative": h["n_negative"],
                        "interaction_n": h["n"],
                    }
                )
            ax.text(
                j,
                i,
                text.replace("-", "−"),
                ha="center",
                va="center",
                fontsize=FONT_SIZE,
                linespacing=1.1,
            )
            records.append(rec)
    ax.imshow(values, cmap=_heat_cmap(), aspect="auto")
    for k in range(1, len(Qs)):
        ax.axvline(k - 0.5, color="white", lw=1.5)
    for k in range(1, len(Ls)):
        ax.axhline(k - 0.5, color="white", lw=1.5)
    ax.set_xticks(range(len(Qs)), [f"{q:.2f}" if q else "0" for q in Qs])
    ax.set_yticks(range(len(Ls)), [f"{v:.2f}" if v else "0" for v in Ls])
    ax.set_xlabel(QUANTITY_AXIS)
    ax.set_ylabel(LABEL_AXIS)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    n = len(bundle.trials("cifar10"))
    what = (
        "final test accuracy"
        if metric == "final_acc"
        else "RTA (rounds to 90% of the run's own best)"
    )
    notes = [
        f"Proposed caption: CIFAR-10, RQ3, {ALGORITHM_NAMES[algorithm]}: {what} by label level "
        "(rows) and quantity level (columns), the grid's target levels; mean ± sd over "
        f"{n} trials {trial_set_label(bundle.trials('cifar10'))}. Cells with both skews also show "
        "the interaction, Δcombined − Δlabel-only − Δquantity-only, where each Δ is the change "
        "from that trial's (0, 0) corner: the cell's own, the cell at the same label level and "
        "quantity 0, and the cell at label 0 and the same quantity level (mean over trials), and "
        "in how many trials it is below zero. Colour: the cell mean."
        + (
            " In accuracy a negative interaction means the two skews together cost more than the "
            "sum of their separate costs."
            if metric == "final_acc"
            else " In RTA the interaction is in rounds; a negative value means the cell reaches 90% "
            "of its own best in fewer rounds than the separate effects add up to."
        ),
        "Realized mean label-HD and quantity-HD per cell are in this file's CSV.",
    ]
    if metric == "rta":
        top = Qs[-1]
        ranges = "; ".join(
            f"label {r['label_level']:.2f}: rounds {int(r['best_round_min'])}-{int(r['best_round_max'])}"
            for r in records
            if r["quantity_level"] == top
        )
        notes.append(
            "Best round: mean round of each run's highest test accuracy; where it is early, RTA "
            f"measures the climb to that early peak. At quantity level {top:.2f} the runs' best rounds "
            f"span {ranges} (measured here; check before quoting an 'early peak' for every cell)."
        )
    name = f"rq3_{'accuracy' if metric == 'final_acc' else 'rta'}_heatmap_{algorithm}"
    return save(fig, out, name, pd.DataFrame(records), notes)


def regression_residuals(bundle: Bundle, fits: dict, out: Path, widths: Widths) -> list[Path]:
    plt = pyplot()
    fig, axes = plt.subplots(
        1, 2, figsize=(widths.column, widths.column * 0.75), layout="constrained", sharey=True
    )
    records = []
    for ax, algorithm in zip(axes, ALGORITHMS):
        data, fit = fits[algorithm]
        for q in sorted(data["Q_level"].unique()):
            sel = (data["Q_level"] == q).to_numpy()
            color, marker = LEVEL_STYLE.get(round(q, 2), ("#000000", "o"))
            ax.plot(
                fit.fitted[sel],
                fit.residuals[sel],
                marker,
                color=color,
                ms=2.5,
                mfc="none",
                mew=0.6,
                label=f"quantity level {q:.2f}" if q else "quantity level 0",
            )
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("fitted accuracy")
        _panel_label(ax, ALGORITHM_NAMES[algorithm])
        for k in range(len(data)):
            records.append(
                {
                    "algorithm": algorithm,
                    "label_level": data["L_level"].iloc[k],
                    "quantity_level": data["Q_level"].iloc[k],
                    "trial": data["trial"].iloc[k],
                    "realized_label_hd": data["L"].iloc[k],
                    "accuracy": data["acc"].iloc[k],
                    "fitted": fit.fitted[k],
                    "residual": fit.residuals[k],
                    "n": 1,
                }
            )
    axes[0].set_ylabel("residual")
    legend_below(fig, axes, ncol=3)
    notes = [
        "Proposed caption: RQ3 regression acc ~ (L + L²) × C(quantity level) + C(trial), residuals "
        f"against fitted values, one point per run ({len(bundle.trials('cifar10')) * 12} per "
        "algorithm), marked by quantity level. Descriptive.",
    ]
    return save(fig, out, "rq3_regression_residuals", pd.DataFrame(records), notes)


# ---------------------------------------------------------------- one panel per file
def _single(out: Path, widths: Widths, name: str, title: str, draw, notes, ncol=2, height=1.0):
    """One panel in its own figure, column width, its legend under the plot."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(widths.column, widths.column * height), layout="constrained")
    records = []
    draw(ax, records)
    _panel_label(ax, title)
    if ax.get_legend_handles_labels()[0]:
        legend_below(fig, ax, ncol=ncol)
    return save(fig, out, name, pd.DataFrame(records), notes)


def single_panel_figures(bundle: Bundle, summary, diffs: dict, out: Path, widths: Widths):
    """Every panel of the multi-panel figures as its own figure, each with its own legend."""
    see = "One panel of {}; see that figure's notes for the caption and n."
    written = []
    covtype_status = bundle.rq1_status.get("covtype")
    panels = [
        (
            "rq1_incidental_quantity_k30_cifar10",
            "CIFAR-10, K = 30",
            lambda ax, r: _rq1_panel(
                ax, _rq1_points(bundle, "cifar10", 30), _quantity_locus("cifar10"), True, r, "a"
            ),
            "rq1_incidental_quantity_k30",
            2,
            1.3,
        ),
    ]
    if covtype_status != "pending":
        panels.append(
            (
                "rq1_incidental_quantity_k30_covtype",
                "Covtype, K = 30" + (" (PROVISIONAL)" if covtype_status == "provisional" else ""),
                lambda ax, r: _rq1_panel(
                    ax,
                    _rq1_points(bundle, "covtype", 30),
                    _quantity_locus("covtype"),
                    False,
                    r,
                    "b",
                ),
                "rq1_incidental_quantity_k30",
                2,
                1.15,
            )
        )
    for K, panel in ((10, "a"), (100, "b")):
        panels.append(
            (
                f"rq1_incidental_quantity_cifar_k{K}",
                f"CIFAR-10, K = {K}",
                lambda ax, r, K=K, panel=panel: _rq1_k_panel(bundle, ax, r, K, panel),
                "rq1_incidental_quantity_cifar_k10_k100",
                3,
                1.1,
            )
        )
    for dataset, panel in (("cifar10", "a"), ("covtype", "b")):
        panels.append(
            (
                f"rq2_accuracy_vs_label_hd_{dataset}",
                DATASET_NAMES[dataset],
                lambda ax, r, d=dataset, p=panel: _rq2_panel(
                    ax, _rq2_series(bundle, summary, d, "final_acc"), r, p, "final test accuracy"
                ),
                "rq2_accuracy_vs_label_hd",
                2,
                1.25,
            )
        )
    panels.append(
        (
            "rq2_per_class_difference_045",
            "CIFAR-10, label-HD 0.45, FedAvg",
            lambda ax, r: per_class_panel(ax, diffs[0.45], r, "c", 0.45),
            "rq2_accuracy_vs_label_hd (panel c)",
            1,
            0.95,
        )
    )
    for name, title, draw, parent, ncol, height in panels:
        written += _single(out, widths, name, title, draw, [see.format(parent)], ncol, height)
    return written


def _rq1_k_panel(bundle: Bundle, ax, records: list, K: int, panel: str):
    points = _rq1_points(bundle, "cifar10", K)
    _rq1_panel(ax, points, None, False, records, panel)
    bad = points[points["n_infeasible"] > 0]
    for i, (_, p) in enumerate(bad.iterrows()):
        ax.text(
            0.02,
            0.95 - 0.08 * i,
            f"× {p['arm']}, α = {p['alpha']:g}: {p['n_infeasible']}/{p['n_total']} infeasible",
            transform=ax.transAxes,
            va="top",
            fontsize=FONT_SIZE - 1,
        )


def single_residual_figures(bundle: Bundle, fits: dict, out: Path, widths: Widths):
    written = []
    for algorithm in ALGORITHMS:
        data, fit = fits[algorithm]

        def draw(ax, records, data=data, fit=fit, algorithm=algorithm):
            for q in sorted(data["Q_level"].unique()):
                sel = (data["Q_level"] == q).to_numpy()
                color, marker = LEVEL_STYLE.get(round(q, 2), ("#000000", "o"))
                ax.plot(
                    fit.fitted[sel],
                    fit.residuals[sel],
                    marker,
                    color=color,
                    ms=3,
                    mfc="none",
                    mew=0.7,
                    label=f"quantity level {q:.2f}" if q else "quantity level 0",
                )
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel("fitted accuracy")
            ax.set_ylabel("residual")
            records += [
                {"algorithm": algorithm, "quantity_level": q, "fitted": f, "residual": e, "n": 1}
                for q, f, e in zip(data["Q_level"], fit.fitted, fit.residuals)
            ]

        written += _single(
            out,
            widths,
            f"rq3_regression_residuals_{algorithm}",
            f"RQ3 regression residuals, {ALGORITHM_NAMES[algorithm]}",
            draw,
            ["One panel of rq3_regression_residuals; see that figure's notes."],
            3,
            0.85,
        )
    return written
