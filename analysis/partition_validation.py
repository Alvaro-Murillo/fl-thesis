"""Partition validation: nominal vs measured HD, and which target cells are feasible.

    python -m analysis.partition_validation [--grid GRID ...] [--prefix NAME]

The grids default to the two candidate-level grids, rq2_candidates.yaml and
rq3_candidates.yaml. Generate their partitions first, and the 20-seed level
check for any grid that sets quantity targets:

    python -m experiments.run_grid --grid config/grids/rq2_candidates.yaml
    python -m experiments.run_grid --grid config/grids/rq3_candidates.yaml
    python -m experiments.quantity_levels --grid config/grids/rq3_candidates.yaml

For every grid cell that sets an HD target this reads its recorded partition
row, its client index arrays and, for a quantity target, its level row. It
prints the summary and writes, under {out_dir}/analysis/:

    {prefix}_partitions.csv   one row per partition and target pair: realized
                              label-HD and quantity-HD, deviation from each
                              target, sizes, per-client label deviation and
                              drift, label check
    {prefix}_clients.csv      one row per client: size, class counts (the
                              client's label histogram), HD to the global label
                              distribution, drift from its drawn mix
    {prefix}_cells.csv        one row per target cell (arm x targets): realized
                              HD over its trials, the label check count, the
                              quantity level's 20-seed verdict, and the verdict
    {prefix}_label.png        nominal vs measured label-HD, one panel per arm
    {prefix}_quantity.png     nominal vs measured quantity-HD: the trained
                              trials and the level's 20 seeds
    {prefix}_feasibility.png  the feasibility map: mean deviation from the
                              target per cell, cells that fail marked

The tolerance rules are the recorded ones. The label check is per partition,
|realized - target| <= 0.05. A quantity target is a level name, feasible if
its 20-seed mean is within 0.05 (`experiments.quantity_levels`). A cell is
feasible if every partition it holds passes the label check and its quantity
level, if it sets one, passes. A trial whose min-size redraw budget ran out
has no partition and fails the label check.

Reports only: nothing is excluded, redrawn or relabelled here. Which levels a
grid uses is chosen from these tables.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.datasets import load_labels
from experiments.grid import expand
from experiments.quantity_levels import level_fields
from experiments.recording import (
    atomic_write_text,
    git_provenance,
    load_partition,
    partition_id,
    read_jsonl,
    utc_stamp,
)
from experiments.run_partition import FEASIBILITY_TOLERANCE
from experiments.seeding import seeds_for
from partitioning.metrics import class_counts, hellinger

GRIDS = (Path("config/grids/rq2_candidates.yaml"), Path("config/grids/rq3_candidates.yaml"))
PREFIX = "partition_validation"
TARGETS = ("target_label_hd", "target_quantity_hd")
# A target cell: one arm at one pair of targets. Its trials are its partitions.
CELL = ("grid", "arm", "dataset", "K", "mode", "size_cap", "min_size_policy", *TARGETS)
PARTITION_COLUMNS = (
    "grid",
    "arm",
    "dataset",
    "K",
    "mode",
    "size_cap",
    "min_size_policy",
    "quantity_floor",
    "target_label_hd",
    "target_quantity_hd",
    "alpha",
    "beta",
    "trial",
    "partition_seed",
    "partition_id",
    "status",
    "n_redraws",
    "realized_label_hd",
    "label_hd_deviation",
    "label_within_tolerance",
    "realized_quantity_hd",
    "quantity_hd_deviation",
    "mean_per_client_label_deviation",
    "max_per_client_label_deviation",
    "mean_label_drift",
    "max_label_drift",
    "min_client_size",
    "max_client_size",
    "n_empty_clients",
    "n_floored_clients",
    "client_sizes",
)

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS, BAND, SERIES = "#e1e0d9", "#c3c2b7", "#f0efec", "#2a78d6"
SEQUENTIAL = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab")
SEQUENTIAL += ("#184f95", "#104281", "#0d366b")
DIVERGING = ("#256abf", BAND, "#e34948")  # realized below target, on it, above it
DEVIATION_LIMIT = 0.10  # colour scale of the feasibility map, +/-


# The fields that name a quantity level (`experiments.quantity_levels.level_fields`).
LEVEL_KEYS = ("dataset", "K", "mode", "quantity_floor", "beta", "target_quantity_hd")


def arm_name(mode: str, size_cap: str) -> str:
    return mode if size_cap == "none" else f"{mode}, {size_cap}"


def level_key(row) -> tuple:
    """A partition or level row's quantity level, as a lookup key."""
    return tuple(row[k] for k in LEVEL_KEYS)


def load_cells(grids) -> tuple[list[dict], dict, dict, Path]:
    """Recorded rows of every targeted cell, its level rows, and the datasets.

    Returns (records, levels, datasets, out_dir): one record per partition
    and pair of targets, so cells that differ only in training settings share
    one, and a partition that two targets resolve to is checked against each;
    the latest quantity_levels.jsonl row per level the cells use; and a
    DatasetConfig per dataset name. A record carries its cell's targets, not
    the ones in the row. Raises SystemExit naming the command to run if a
    partition row or a level row is missing.
    """
    records, levels_needed, datasets, out_dirs, skipped = {}, {}, {}, set(), 0
    for grid in map(Path, grids):
        for config in expand(grid):
            p = config.partition
            if p.target_label_hd is None and p.target_quantity_hd is None:
                skipped += 1
                continue
            if tuple(level_fields(config)) != LEVEL_KEYS:
                raise RuntimeError("quantity level fields changed; update LEVEL_KEYS")
            pid = partition_id(config, seeds_for(config).partition)
            targets = {
                "target_label_hd": p.target_label_hd,
                "target_quantity_hd": p.target_quantity_hd,
            }
            records.setdefault(
                (pid, *targets.values()), {"grid": grid.stem, "_grid": grid, **targets}
            )
            datasets[config.dataset.name] = config.dataset
            out_dirs.add(config.runtime.out_dir)
            if p.target_quantity_hd is not None:
                levels_needed[tuple(level_fields(config).values())] = grid
    if not records:
        raise SystemExit(f"no grid cell in {[str(g) for g in grids]} sets an HD target")
    if len(out_dirs) > 1:
        raise SystemExit(f"the grids write to different out_dirs: {sorted(out_dirs)}")
    out_dir = Path(out_dirs.pop())

    latest = {r["partition_id"]: r for r in read_jsonl(out_dir / "partitions.jsonl")}
    missing: dict[Path, set] = {}
    for (pid, *_), record in records.items():
        if pid not in latest:
            missing.setdefault(record["_grid"], set()).add(pid)
            continue
        targets = {key: record[key] for key in TARGETS}
        record.update(latest[pid])
        record.update(targets)  # the row holds the targets of the cell that generated it
    if missing:
        raise SystemExit(
            "partitions missing; generate them first:\n"
            + "\n".join(
                f"  python -m experiments.run_grid --grid {g.as_posix()}   ({len(p)} missing)"
                for g, p in missing.items()
            )
        )

    recorded = {level_key(r): r for r in read_jsonl(out_dir / "quantity_levels.jsonl")}
    absent = sorted({g.as_posix() for key, g in levels_needed.items() if key not in recorded})
    if absent:
        raise SystemExit(
            "quantity levels not checked yet; run:\n"
            + "\n".join(f"  python -m experiments.quantity_levels --grid {g}" for g in absent)
        )
    levels = {key: recorded[key] for key in levels_needed}
    for record in records.values():
        record.pop("_grid")
        record["arm"] = arm_name(record["mode"], record["size_cap"])
    if skipped:
        print(f"{skipped} cells set no HD target and are left out")
    return list(records.values()), levels, datasets, out_dir


def partitions_table(records) -> pd.DataFrame:
    """One row per record. Deviations and the label check are against the record's targets."""
    df = pd.DataFrame(records)
    for column in ("realized_label_hd", "realized_quantity_hd", "mean_label_drift"):
        if column not in df:  # an infeasible-only or pre-drift set of rows
            df[column] = np.nan
    for axis in ("label", "quantity"):
        target = df[f"target_{axis}_hd"].astype(float)
        df[f"{axis}_hd_deviation"] = df[f"realized_{axis}_hd"].astype(float) - target
    df["label_within_tolerance"] = np.where(
        df["target_label_hd"].isna(),
        None,
        (df["status"] == "generated") & (df["label_hd_deviation"].abs() <= FEASIBILITY_TOLERANCE),
    )
    for column in PARTITION_COLUMNS:
        if column not in df:
            df[column] = None
    return df[list(PARTITION_COLUMNS)].sort_values(["grid", "arm", *TARGETS, "trial"])


def clients_table(records, datasets: dict, out_dir: Path, labels: dict) -> pd.DataFrame:
    """One row per client of every generated partition: its label histogram and HDs."""
    rows = []
    for record in records:
        if record["status"] != "generated":
            continue
        y = labels[record["dataset"]]
        num_classes = datasets[record["dataset"]].num_classes
        global_dist = np.bincount(y, minlength=num_classes) / len(y)
        counts = class_counts(y, load_partition(out_dir, record["partition_id"]), num_classes)
        drift = record.get("label_drift")
        for client, histogram in enumerate(counts):
            size = int(histogram.sum())
            rows.append(
                {
                    "partition_id": record["partition_id"],
                    "grid": record["grid"],
                    "arm": record["arm"],
                    "target_label_hd": record["target_label_hd"],
                    "target_quantity_hd": record["target_quantity_hd"],
                    "trial": record["trial"],
                    "client": client,
                    "size": size,
                    "label_hd": hellinger(histogram / size, global_dist) if size else None,
                    "label_drift": None if drift is None else drift[client],
                    **{f"class_{c}": int(n) for c, n in enumerate(histogram)},
                }
            )
    return pd.DataFrame(rows)


def cells_table(partitions: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """One row per target cell, with the per-axis checks and the cell verdict."""
    rows = []
    for key, g in partitions.groupby(list(CELL), dropna=False, sort=True):
        cell = dict(zip(CELL, key))
        target_label, target_quantity = cell["target_label_hd"], cell["target_quantity_hd"]
        if g[["alpha", "beta"]].drop_duplicates().shape[0] > 1:
            raise ValueError(f"cell {cell} resolves to more than one alpha/beta")
        generated = g[g["status"] == "generated"]
        label = generated["realized_label_hd"].astype(float)
        row = {
            **cell,
            "alpha": g["alpha"].iloc[0],
            "beta": g["beta"].iloc[0],
            "trials": sorted(g["trial"].tolist()),
            "n_partitions": len(g),
            "n_infeasible_min_size": int((g["status"] != "generated").sum()),
            "mean_realized_label_hd": label.mean(),
            "sd_realized_label_hd": label.std(ddof=1) if len(label) > 1 else np.nan,
            "min_realized_label_hd": label.min(),
            "max_realized_label_hd": label.max(),
            "mean_label_hd_deviation": generated["label_hd_deviation"].astype(float).mean(),
            "max_abs_label_hd_deviation": generated["label_hd_deviation"].abs().max(),
            "n_label_within_tolerance": None,
            "label_feasible": None,
            "mean_realized_quantity_hd": generated["realized_quantity_hd"].astype(float).mean(),
            "mean_label_drift": generated["mean_label_drift"].astype(float).mean(),
            "max_label_drift": generated["max_label_drift"].astype(float).max(),
            "min_client_size": generated["min_client_size"].min(),
            "max_n_empty_clients": generated["n_empty_clients"].max(),
        }
        if not pd.isna(target_label):
            within = int(g["label_within_tolerance"].astype(bool).sum())
            row["n_label_within_tolerance"] = within
            row["label_feasible"] = within == len(g)
        level = None if pd.isna(target_quantity) else levels[level_key(g.iloc[0])]
        row |= {
            "level_n_seeds": None if level is None else len(level["realized_quantity_hd"]),
            "level_mean_quantity_hd": None if level is None else level["mean_quantity_hd"],
            "level_sd_quantity_hd": None if level is None else level["sd_quantity_hd"],
            "level_mean_deviation": None if level is None else level["mean_deviation"],
            "level_share_within_tolerance": (
                None if level is None else level["share_within_tolerance"]
            ),
            "level_feasible": None if level is None else level["level_feasible"],
        }
        row["cell_feasible"] = row["label_feasible"] is not False and (
            row["level_feasible"] is not False
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --- figures ----------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=8)


def _diagonal(ax, top: float) -> None:
    """y = x and the +/- tolerance band around it."""
    x = np.array([0.0, top])
    ax.fill_between(x, x - FEASIBILITY_TOLERANCE, x + FEASIBILITY_TOLERANCE, color=BAND, zorder=1)
    ax.plot(x, x, color=MUTED, linewidth=1.0, zorder=2)
    ax.set_xlim(-0.03, top)
    ax.set_ylim(-0.03, top)
    ax.set_aspect("equal")


def _dodge(values, width: float = 0.025) -> np.ndarray:
    """Offsets in [-width/2, width/2] that spread the distinct `values` apart, in order."""
    distinct = sorted(set(values))
    if len(distinct) < 2:
        return np.zeros(len(values))
    rank = {v: i for i, v in enumerate(distinct)}
    return np.array([(rank[v] / (len(distinct) - 1) - 0.5) * width for v in values])


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    _pyplot().close(fig)


def label_figure(partitions: pd.DataFrame, path: Path) -> None:
    """Nominal vs measured label-HD; one panel per arm. Two-target arms colour by quantity target."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    plt = _pyplot()
    df = partitions[partitions["target_label_hd"].notna() & (partitions["status"] == "generated")]
    arms = list(dict.fromkeys(df["arm"]))
    if not arms:
        return
    top = max(0.85, float(df["realized_label_hd"].max()) + 0.05)
    cols = min(len(arms), 2)
    rows = -(-len(arms) // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.6 * cols + 0.8, 3.6 * rows), dpi=150, facecolor=SURFACE
    )
    axes = np.atleast_1d(axes).ravel()
    ramp = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    for ax, arm in zip(axes, arms):
        g = df[df["arm"] == arm]
        _style(ax)
        _diagonal(ax, top)
        quantity = g["target_quantity_hd"]
        if quantity.notna().any():
            levels = quantity.astype(float)
            norm = Normalize(levels.min(), levels.max())
            x = g["target_label_hd"] + _dodge(levels.tolist())
            points = ax.scatter(
                x,
                g["realized_label_hd"],
                s=12,
                c=levels,
                cmap=ramp,
                norm=norm,
                edgecolors=SURFACE,
                linewidths=0.4,
                zorder=3,
            )
            bar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03)
            bar.set_label("target quantity-HD", color=SECONDARY, fontsize=8)
            bar.ax.tick_params(colors=SECONDARY, labelsize=7)
            bar.outline.set_visible(False)
        else:
            x = g["target_label_hd"] + _dodge(g["trial"].tolist(), 0.015)
            ax.scatter(
                x,
                g["realized_label_hd"],
                s=14,
                color=SERIES,
                edgecolors=SURFACE,
                linewidths=0.4,
                zorder=3,
            )
        n_trials = g["trial"].nunique()
        ax.set_title(f"{arm}  ({len(g)} partitions, {n_trials} trials)", fontsize=9, loc="left")
        ax.set_xlabel("target label-HD", color=INK, fontsize=8)
        ax.set_ylabel("realized label-HD", color=INK, fontsize=8)
    for ax in axes[len(arms) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Nominal vs measured label-HD; band = target ± {FEASIBILITY_TOLERANCE}",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    _save(fig, path)


def quantity_figure(partitions: pd.DataFrame, levels: dict, path: Path) -> None:
    """Nominal vs measured quantity-HD per arm: the level's seeds, the trained trials, the mean."""
    plt = _pyplot()
    df = partitions[
        partitions["target_quantity_hd"].notna() & (partitions["status"] == "generated")
    ]
    arms = list(dict.fromkeys(df["arm"]))
    if not arms:
        return
    fig, axes = plt.subplots(
        1, len(arms), figsize=(4.6 * len(arms), 4.4), dpi=150, facecolor=SURFACE, squeeze=False
    )
    for ax, arm in zip(axes[0], arms):
        g = df[df["arm"] == arm]
        # Cells at one quantity target share a size draw per trial; plot each draw once.
        trained = g.drop_duplicates(["target_quantity_hd", "trial", "realized_quantity_hd"])
        level_rows = [levels[k] for k in dict.fromkeys(level_key(r) for _, r in g.iterrows())]
        highest = max(
            trained["realized_quantity_hd"].max(),
            *(max(r["realized_quantity_hd"]) for r in level_rows),
        )
        top = max(0.9, float(highest) + 0.05)
        _style(ax)
        _diagonal(ax, top)
        for i, level in enumerate(level_rows):
            seeds = np.asarray(level["realized_quantity_hd"])
            x = level["target_quantity_hd"] + _dodge(range(len(seeds)), 0.03)
            ax.scatter(
                x,
                seeds,
                s=9,
                color=MUTED,
                alpha=0.6,
                linewidths=0,
                zorder=3,
                label=f"level check, {len(seeds)} seeds" if i == 0 else None,
            )
            t = level["target_quantity_hd"]
            ax.plot(
                [t - 0.025, t + 0.025],
                [level["mean_quantity_hd"]] * 2,
                color=INK,
                linewidth=1.6,
                zorder=5,
                label="level mean (the tolerance applies here)" if i == 0 else None,
            )
        ax.scatter(
            trained["target_quantity_hd"] + _dodge(trained["trial"].tolist(), 0.015),
            trained["realized_quantity_hd"],
            s=18,
            color=SERIES,
            edgecolors=SURFACE,
            linewidths=0.6,
            zorder=4,
            label=f"trained trials {sorted(trained['trial'].unique().tolist())}",
        )
        ax.set_title(arm, fontsize=9, loc="left")
        ax.set_xlabel("target quantity-HD", color=INK, fontsize=8)
        ax.set_ylabel("realized quantity-HD", color=INK, fontsize=8)
        ax.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper left")
    fig.suptitle(
        f"Nominal vs measured quantity-HD; band = target ± {FEASIBILITY_TOLERANCE}",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    _save(fig, path)


def _heatmap(ax, values, failed, xticks, yticks, norm, cmap):
    """Cells as a mesh with surface gaps; a cross on every cell in `failed`."""
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(values), cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=1.5
    )
    for (i, j), bad in np.ndenumerate(failed):
        if bad:
            ax.text(j + 0.5, i + 0.5, "×", ha="center", va="center", color=INK, fontsize=9)
    ax.set_xticks(np.arange(len(xticks)) + 0.5, xticks)
    ax.set_yticks(np.arange(len(yticks)) + 0.5, yticks)
    ax.tick_params(colors=SECONDARY, labelsize=7, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def _grid_of(df: pd.DataFrame, columns: str, column_order, rows) -> tuple[np.ndarray, np.ndarray]:
    """(mean label-HD deviation, label check failed) as label-target rows x `columns`."""

    def pivot(values):
        return df.pivot_table(
            index="target_label_hd", columns=columns, values=values, aggfunc="first"
        ).reindex(index=rows, columns=column_order)

    failed = pivot("label_failed").fillna(False).astype(bool)
    return pivot("mean_label_hd_deviation").to_numpy(float), failed.to_numpy()


def feasibility_figure(cells: pd.DataFrame, path: Path) -> None:
    """Mean realized-minus-target HD per cell; a cross marks a cell that fails its check.

    Rows are label targets throughout. Label-only arms share one panel (one
    column per arm). Each arm with both targets gets its own panel (one column
    per quantity target) under a strip for its quantity levels, coloured by
    the level's 20-seed mean deviation and crossed where the level fails.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt = _pyplot()
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING).with_extremes(bad=SURFACE)
    norm = TwoSlopeNorm(0.0, -DEVIATION_LIMIT, DEVIATION_LIMIT)
    labelled = cells[cells["target_label_hd"].notna()].assign(
        label_failed=lambda d: d["label_feasible"] == False  # noqa: E712 - object column
    )
    one_axis = labelled[labelled["target_quantity_hd"].isna()]
    two_axis = labelled[labelled["target_quantity_hd"].notna()]
    one_arms = list(dict.fromkeys(one_axis["arm"]))
    two_arms = list(dict.fromkeys(two_axis["arm"]))
    widths = ([max(len(one_arms), 3)] if one_arms else []) + [
        max(two_axis.loc[two_axis["arm"] == arm, "target_quantity_hd"].nunique(), 3)
        for arm in two_arms
    ]
    if not widths:
        return
    fig = plt.figure(figsize=(0.55 * sum(widths) + 3.0, 6.0), dpi=150, facecolor=SURFACE)
    grid = fig.add_gridspec(
        2,
        len(widths) + 1,  # the last column holds the colour scale
        height_ratios=(1, 10),
        width_ratios=(*widths, 0.35),
        hspace=0.06,
        wspace=0.25,
        left=0.08,
        right=0.93,
        bottom=0.17,
        top=0.88,
    )
    column, mesh = 0, None

    if one_arms:
        targets = sorted(one_axis["target_label_hd"].unique())
        values, failed = _grid_of(one_axis, "arm", one_arms, targets)
        ax = fig.add_subplot(grid[1, column])
        ticks = [arm.replace(", ", ",\n") for arm in one_arms]
        mesh = _heatmap(ax, values, failed, ticks, [f"{v:g}" for v in targets], norm, cmap)
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_ylabel("target label-HD", color=INK, fontsize=8)
        ax.set_title("label-axis arms", fontsize=9, loc="left", color=INK)
        column += 1

    for arm in two_arms:
        g = two_axis[two_axis["arm"] == arm]
        label_targets = sorted(g["target_label_hd"].unique())
        quantity_targets = sorted(g["target_quantity_hd"].unique())
        values, failed = _grid_of(g, "target_quantity_hd", quantity_targets, label_targets)
        ax = fig.add_subplot(grid[1, column])
        xticks = [f"{v:g}" for v in quantity_targets]
        mesh = _heatmap(ax, values, failed, xticks, [f"{v:g}" for v in label_targets], norm, cmap)
        ax.set_xlabel("target quantity-HD", color=INK, fontsize=8)
        if column == 0:
            ax.set_ylabel("target label-HD", color=INK, fontsize=8)

        levels = g.drop_duplicates("target_quantity_hd").set_index("target_quantity_hd")
        levels = levels.reindex(quantity_targets)
        strip = fig.add_subplot(grid[0, column], sharex=ax)
        _heatmap(
            strip,
            levels["level_mean_deviation"].to_numpy(float)[None, :],
            (levels["level_feasible"] == False).to_numpy()[None, :],  # noqa: E712
            xticks,
            ["level"],
            norm,
            cmap,
        )
        strip.tick_params(labelbottom=False)
        strip.set_title(arm, fontsize=9, loc="left", color=INK)
        column += 1

    bar = fig.colorbar(mesh, cax=fig.add_subplot(grid[1, -1]), extend="both")
    bar.set_label("mean realized - target HD", color=SECONDARY, fontsize=8)
    for t in (-FEASIBILITY_TOLERANCE, FEASIBILITY_TOLERANCE):
        bar.ax.axhline(t, color=INK, linewidth=0.8)
    bar.ax.tick_params(colors=SECONDARY, labelsize=7)
    bar.outline.set_visible(False)
    fig.suptitle(
        "Feasibility map: cells show realized - target label-HD over the trained trials; "
        "the level strip shows the 20-seed mean realized - target quantity-HD",
        color=INK,
        fontsize=9,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.005,
        f"× = fails: a trained partition outside ±{FEASIBILITY_TOLERANCE} of its label target, "
        f"or a quantity level whose 20-seed mean is outside ±{FEASIBILITY_TOLERANCE}. "
        "Black lines on the scale mark the tolerance.",
        color=SECONDARY,
        fontsize=7,
    )
    _save(fig, path)


# --- report -----------------------------------------------------------------


def print_summary(partitions: pd.DataFrame, cells: pd.DataFrame) -> None:
    fmt = lambda v: f"{v:.3f}"
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(
            f"{len(cells)} target cells, {len(partitions)} partitions; "
            f"{int((~cells['cell_feasible']).sum())} cells fail"
        )
        one_axis = cells[cells["target_label_hd"].notna() & cells["target_quantity_hd"].isna()]
        if len(one_axis):
            print("\n== Label-axis arms: mean realized label-HD (and quantity-HD) per target ==")
            for value in ("mean_realized_label_hd", "mean_realized_quantity_hd"):
                table = one_axis.pivot_table(
                    index="target_label_hd", columns="arm", values=value, aggfunc="first"
                )
                if value == "mean_realized_label_hd" and table.shape[1] > 1:
                    table["arm gap"] = table.max(axis=1) - table.min(axis=1)
                print(f"\n{value}:")
                print(table.to_string(float_format=fmt))
        two_axis = cells[cells["target_label_hd"].notna() & cells["target_quantity_hd"].notna()]
        for arm, g in two_axis.groupby("arm"):
            print(f"\n== {arm}: mean realized - target label-HD (rows: label, cols: quantity) ==")
            table = g.pivot_table(
                index="target_label_hd",
                columns="target_quantity_hd",
                values="mean_label_hd_deviation",
                aggfunc="first",
            )
            print(table.to_string(float_format=lambda v: f"{v:+.3f}"))
            levels = g.drop_duplicates("target_quantity_hd").set_index("target_quantity_hd")
            print(f"\n{arm}: quantity levels")
            print(
                levels[
                    [
                        "beta",
                        "level_n_seeds",
                        "level_mean_quantity_hd",
                        "level_sd_quantity_hd",
                        "level_mean_deviation",
                        "level_share_within_tolerance",
                        "level_feasible",
                    ]
                ].to_string(float_format=fmt)
            )
        failing = cells[~cells["cell_feasible"]]
        if len(failing):
            print("\n== Cells that fail ==")
            columns = [
                "arm",
                *TARGETS,
                "n_label_within_tolerance",
                "n_partitions",
                "mean_label_hd_deviation",
                "max_abs_label_hd_deviation",
                "level_feasible",
                "level_mean_deviation",
            ]
            print(failing[columns].to_string(index=False, float_format=fmt))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--grid", action="append", type=Path, help="repeatable; default: the two candidate grids"
    )
    parser.add_argument("--prefix", default=PREFIX, help=f"output file prefix (default {PREFIX})")
    args = parser.parse_args()

    records, levels, datasets, out_dir = load_cells(args.grid or GRIDS)
    labels = {name: load_labels(dataset) for name, dataset in datasets.items()}
    partitions = partitions_table(records)
    clients = clients_table(records, datasets, out_dir, labels)
    cells = cells_table(partitions, levels)
    provenance = {**git_provenance(), "generated_at": utc_stamp()}

    out = out_dir / "analysis"
    for name, df in (("partitions", partitions), ("clients", clients), ("cells", cells)):
        atomic_write_text(
            out / f"{args.prefix}_{name}.csv",
            df.assign(**provenance).to_csv(index=False, lineterminator="\n"),
        )
    label_figure(partitions, out / f"{args.prefix}_label.png")
    quantity_figure(partitions, levels, out / f"{args.prefix}_quantity.png")
    feasibility_figure(cells, out / f"{args.prefix}_feasibility.png")

    print_summary(partitions, cells)
    print(f"\n-> {out}/{args.prefix}_*  ({provenance})")


if __name__ == "__main__":
    main()
