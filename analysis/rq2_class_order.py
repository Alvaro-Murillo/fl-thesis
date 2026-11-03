"""RQ2's class-order check at every label level where the niid_bench cap binds.

    python -m analysis.rq2_class_order               # every table, the verdicts and the CSVs
    python -m analysis.rq2_class_order --partitions  # the partition-level check only

rq2.yaml's capped arm (natural_per_class, size_cap niid_bench) splits the classes in index
order, and the cap masks full clients class by class, so the classes split last are held by
fewer clients. config/grids/rq2_class_order.yaml repeats the capped FedAvg cells at label-HD
0.25, 0.45 and 0.70 with the classes split in reversed order (class_order: reversed). On
CIFAR-10's balanced labels a reversed partition's count matrix is the index-order one with
its class columns reversed, so client sizes and every HD value stay the same. Level 0 is
left out: at alpha = 1000 the cap does not bind, the capped and uncapped partitions are
identical, and a reversed one would only relabel an uncapped partition.

Arms at each level, FedAvg, trials 0-2:
    capped index     rq2.yaml, natural_per_class, size_cap niid_bench
    capped reversed  rq2_class_order.yaml
    uncapped         rq2.yaml, natural_per_class, size_cap none
    pure             rq2.yaml, pure_label; context only, and not rerun at 0.70
A reversed config has no target; its level is that of the rq2.yaml capped cell whose alpha
it repeats. Per-class accuracy is the final global model's recall on each class of the
CIFAR-10 test set (1000 images per class), from `experiments.per_class_accuracy`, read
through `load_per_class`, which refuses a rerun that did not reproduce its recorded run.
training_summary has no class_order column, so here the arm names the order.

Pre-registered rule. It was committed before any of the per-class data it covers was
generated. It is applied at L = 0.25 and at L = 0.70, each on its own. At L = 0.45 the
verdict of 2026-10-29 (Position) stands; it is recomputed here as a check. The classes are
the same at every level: both orders split all ten classes, so the classes split last are
ship (8) and truck (9) in index order, and airplane (0) and automobile (1) in reversed
order.

Step 1, the index-order gate, on capped index - uncapped at L. Copied verbatim from the
per-class check of 2026-10-29:
    d_c = mean over trials 0-2 of (capped_c - uncapped_c), per-class final accuracy;
    deficit D = -sum_c d_c (the summed per-class deficit, = 10 x the overall deficit).
    (a) met iff D > 0 and -(d_8 + d_9) >= 0.5 D.
    (b) met iff capped < uncapped for class 8 in >= 2 of 3 trials, and the same for class 9.
    The criterion is met iff (a) and (b). A secondary share, over the classes with d_c < 0
    only, is printed for reference and does not enter the verdict.
At 0.25 the gate was met on 2026-10-29. If it is not met at L, the result at L is "no
class-order effect to move", and no Position or Identity verdict is given.

Step 2, the verdict, on capped reversed - uncapped at L. Copied verbatim from the
class-order check of 2026-10-29, except that the fall-through label is renamed from "mixed"
to "Inconclusive":
    d_c = mean over trials 0-2 of (reversed_c - uncapped_c), per-class final accuracy;
    D = -sum_c d_c.
    (a) D > 0 and -(d_0 + d_1) >= 0.5 D;
    (b) airplane (0) and automobile (1) are each lower than uncapped in >= 2 of 3 trials;
    (c) ship (8) and truck (9) are each NOT lower than uncapped in >= 2 of 3 trials.
    "Position": (a), (b) and (c) all met.
    "Identity": ship and truck are each lower than uncapped in >= 2 of 3 trials.
    Otherwise "Inconclusive". (Position and Identity cannot both hold: with 3 trials, "not
    lower in >= 2" and "lower in >= 2" exclude each other for one class.)

Power. At a level where the overall capped - uncapped difference changes sign across
trials, the gate result and any verdict are reported as low-power, with the per-trial values
alongside. At 0.70 this already holds: -0.0505, +0.0558 and -0.0233 in the recorded runs.

Seen before this rule was committed: the overall final accuracy of every rq2.yaml run, 0.70
included; the per-class results of the capped index, uncapped and pure runs at 0.25 and
0.45; and those of the reversed runs at 0.45. The only data the rule is pre-registered for
had not yet been generated: the reversed runs at 0.25 and 0.70, and every per-class result
at 0.70. Copying the 0.45 rule unchanged is what limits the choice made after seeing the
rest.

Partition-level concentration check (no training). For each class c, arm and partition, over
its K clients:
    holders            the clients holding at least one sample of c;
    effective holders  1 / sum_k p_kc^2, with p_kc = n_kc / n_c the share of class c's
                       samples on client k: K for an even spread, 1 for a single holder.
Both are averaged over trials 0-2. Expectation, fixed before the reversed partitions at 0.25
and 0.70 were generated: at each level, ship (8) and truck (9) each have fewer holders in
capped index than in uncapped, and airplane (0) and automobile (1) each have fewer in capped
reversed than in uncapped. The cap's concentration counts as present at a level iff all four
hold. Holders are the quantity the cap acts on: it masks full clients class by class, which
removes the small holders of the classes split last. Effective holders are reported
alongside, as descriptive only. They are dominated by the few largest shares, which come
from the Dirichlet draw. The expectation was first written on effective holders, and on the
existing 0.45 partitions, where the mechanism is known to be present, that version failed:
reversed automobile has 10.13 effective holders against 9.10 uncapped. It was moved to
holders before the new partitions were generated.

The outcome is known in advance. The zero-holders (K - holders) of the index-order capped
and uncapped partitions at every level were reported in the partition diagnostics of
2026-10-29, and a reversed partition's counts are the index-order ones with the class
columns reversed (checked at 0.45, and checked again here at 0.25 and 0.70). From the
index-order partitions, the expectation holds at all three levels; at 0.70 by the smallest
margins, since the uncapped split already leaves about half the clients without each class
(index ship 11.33 and truck 9.67 holders against 13.00 and 14.00 uncapped). The check
therefore documents the mechanism at each level rather than testing it. It separates two
readings of "no class-order effect to move": the concentration is absent from the
partitions, or it is present and does not show in accuracy. It enters neither the gate nor
the verdict.

Also reported per level, as descriptive results only: the Spearman correlation over the ten
classes between the accuracy change and the zero-holder change against uncapped, and where
the extra errors on the newly concentrated classes go (final confusion matrices, summed
over trials).

Writes, under results/analysis/:
    rq2_class_order_per_class.csv  one row per level, arm, trial and class: per-class final
                                   accuracy (empty where no per-class rerun was made),
                                   holders, effective holders, zero-holders, run_id and
                                   partition_id
    rq2_class_order_verdicts.csv   one row per level: gate, verdict, low-power flag and the
                                   concentration check
    rq2_class_order_tables.md      every table
With --partitions, only rq2_class_order_partitions.md, which holds the concentration tables.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.config import ExperimentConfig, config_hash
from experiments.datasets import load_labels
from experiments.grid import expand
from experiments.per_class_accuracy import load_per_class, per_class_accuracy
from experiments.recording import atomic_write_text, git_provenance, load_partition, utc_stamp
from experiments.run_training import stored_partition
from partitioning.metrics import class_counts

RQ2_GRID = Path("config/grids/rq2.yaml")
ORDER_GRID = Path("config/grids/rq2_class_order.yaml")
LEVELS = (0.25, 0.45, 0.70)
TRIALS = (0, 1, 2)
ARMS = ("pure", "capped index", "capped reversed", "uncapped")
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
INDEX_LAST = (8, 9)  # split last in index order
REVERSED_LAST = (0, 1)  # split last in reversed order
NO_EFFECT = "no class-order effect to move"


# ---------------------------------------------------------------- the rule
def _paired(arm, uncapped) -> tuple[np.ndarray, np.ndarray, float]:
    """(d_c, trials lower than uncapped per class, D) for (trials, classes) accuracies."""
    diff = np.asarray(arm, dtype=float) - np.asarray(uncapped, dtype=float)
    if diff.shape[0] != len(TRIALS):
        raise ValueError(f"the rule is written for {len(TRIALS)} trials, got {diff.shape[0]}")
    d = diff.mean(axis=0)
    return d, (diff < 0).sum(axis=0), float(-d.sum())


def gate(capped, uncapped) -> dict:
    """Step 1 on capped index - uncapped: is the late-class deficit there to be moved?"""
    d, lower, D = _paired(capped, uncapped)
    last = list(INDEX_LAST)
    met_a = bool(D > 0 and -d[last].sum() >= 0.5 * D)
    met_b = bool(all(lower[c] >= 2 for c in last))
    deficits = np.clip(-d, 0, None)
    return {
        "D": D,
        "share": float(-d[last].sum() / D) if D > 0 else np.nan,
        "lower_ship": int(lower[8]),
        "lower_truck": int(lower[9]),
        "met_a": met_a,
        "met_b": met_b,
        "met": met_a and met_b,
        "secondary_share": (
            float(deficits[last].sum() / deficits.sum()) if deficits.sum() > 0 else np.nan
        ),
    }


def verdict(reversed_, uncapped) -> dict:
    """Step 2 on capped reversed - uncapped: Position, Identity or Inconclusive."""
    d, lower, D = _paired(reversed_, uncapped)
    n = len(TRIALS)
    met_a = bool(D > 0 and -d[list(REVERSED_LAST)].sum() >= 0.5 * D)
    met_b = bool(all(lower[c] >= 2 for c in REVERSED_LAST))
    met_c = bool(all(n - lower[c] >= 2 for c in INDEX_LAST))
    identity = bool(all(lower[c] >= 2 for c in INDEX_LAST))
    return {
        "D": D,
        "share": float(-d[list(REVERSED_LAST)].sum() / D) if D > 0 else np.nan,
        "lower_airplane": int(lower[0]),
        "lower_automobile": int(lower[1]),
        "lower_ship": int(lower[8]),
        "lower_truck": int(lower[9]),
        "met_a": met_a,
        "met_b": met_b,
        "met_c": met_c,
        "identity": identity,
        "verdict": (
            "Position" if met_a and met_b and met_c else "Identity" if identity else "Inconclusive"
        ),
    }


def level_result(gate_result: dict, verdict_result: dict) -> str:
    """The verdict at a level, or NO_EFFECT when the index-order gate is not met."""
    return verdict_result["verdict"] if gate_result["met"] else NO_EFFECT


def low_power(overall_differences) -> bool:
    """True when the overall capped - uncapped difference changes sign across trials."""
    signs = np.sign(np.asarray(overall_differences, dtype=float))
    return bool((signs > 0).any() and (signs < 0).any())


# ---------------------------------------------------------------- partitions
def concentration(counts) -> tuple[np.ndarray, np.ndarray]:
    """(holders, effective holders) per class of a (clients, classes) count matrix."""
    counts = np.asarray(counts, dtype=float)
    shares = counts / counts.sum(axis=0)
    return (counts > 0).sum(axis=0), 1.0 / (shares**2).sum(axis=0)


def concentration_present(holders: dict) -> dict:
    """The expectation on trial-mean holders, per class it names, and overall."""
    checks = {
        f"index {CLASSES[c]} < uncapped": bool(holders["capped index"][c] < holders["uncapped"][c])
        for c in INDEX_LAST
    }
    checks.update(
        {
            f"reversed {CLASSES[c]} < uncapped": bool(
                holders["capped reversed"][c] < holders["uncapped"][c]
            )
            for c in REVERSED_LAST
        }
    )
    return {**checks, "present": all(checks.values())}


# ---------------------------------------------------------------- data
def cells(
    rq2_grid: Path = RQ2_GRID, order_grid: Path = ORDER_GRID
) -> dict[tuple[float, str, int], ExperimentConfig]:
    """(level, arm, trial) -> the FedAvg config of that cell."""
    out, capped_alpha = {}, {}
    for config in expand(rq2_grid, for_training=False):
        p = config.partition
        if config.federated.algorithm != "fedavg" or p.target_label_hd not in LEVELS:
            continue
        if p.mode.value == "pure_label":
            arm = "pure"
        else:
            arm = {"niid_bench": "capped index", "none": "uncapped"}[p.size_cap]
        out[(p.target_label_hd, arm, config.runtime.trial)] = config
        if arm == "capped index":
            capped_alpha[p.alpha] = p.target_label_hd
    for config in expand(order_grid, for_training=False):
        level = capped_alpha[config.partition.alpha]  # the same float as rq2.yaml's cell
        out[(level, "capped reversed", config.runtime.trial)] = config
    return out


def partition_counts(configs: dict) -> dict:
    """(level, arm, trial) -> (pid, (clients, classes) count matrix) from the stored partitions."""
    labels = load_labels(next(iter(configs.values())).dataset)
    out = {}
    for key, config in configs.items():
        pid, clients = stored_partition(config)
        out[key] = (pid, class_counts(labels, clients, len(CLASSES)))
    return out


def per_class_results(configs: dict) -> dict:
    """(level, arm, trial) -> per-class payload, for the cells that have one (bit-exact only)."""
    out = {}
    for key, config in configs.items():
        try:
            out[key] = load_per_class(config.runtime.out_dir, config_hash(config))
        except FileNotFoundError:
            continue  # not rerun; a result that is not bit-exact raises instead
    return out


# ---------------------------------------------------------------- tables
def md(df: pd.DataFrame, floatfmt: str = ".4f", index: bool = True) -> str:
    df = df.reset_index() if index else df

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


NAMES = [f"{c} {name}" for c, name in enumerate(CLASSES)]


def concentration_tables(level: float, counts: dict) -> tuple[list[str], dict]:
    arms = ("capped index", "capped reversed", "uncapped")
    holders, effective = {}, {}
    for arm in arms:
        per_trial = [concentration(counts[(level, arm, t)][1]) for t in TRIALS]
        holders[arm] = np.mean([h for h, _ in per_trial], axis=0)
        effective[arm] = np.mean([e for _, e in per_trial], axis=0)
    table = pd.DataFrame(index=NAMES)
    for arm in arms:
        table[f"holders {arm}"] = holders[arm]
    table["holders index - uncapped"] = holders["capped index"] - holders["uncapped"]
    table["holders reversed - uncapped"] = holders["capped reversed"] - holders["uncapped"]
    for arm in arms:
        table[f"effective {arm}"] = effective[arm]
    table["effective index - uncapped"] = effective["capped index"] - effective["uncapped"]
    table["effective reversed - uncapped"] = effective["capped reversed"] - effective["uncapped"]
    check = concentration_present(holders)
    out = [f"\n### Concentration across clients at label-HD {level:.2f}, means over trials 0-2\n"]
    out.append(md(table, ".2f"))
    out.append("\nEffective holders are descriptive only; the expectation is on holders.\n")
    rows = [(name, "holds" if ok else "does not hold") for name, ok in check.items()]
    rows[-1] = ("concentration present (all four)", "yes" if check["present"] else "no")
    out.append("\nExpectation on holders (fixed before the check):\n")
    out.append(md(pd.DataFrame(rows, columns=["check", "result"]), index=False))
    return out, {"holders": holders, "effective": effective, **check}


def mirror_note(level: float, counts: dict) -> str:
    """Whether each trial's reversed counts are the index-order counts with columns reversed."""
    same = [
        np.array_equal(
            counts[(level, "capped reversed", t)][1], counts[(level, "capped index", t)][1][:, ::-1]
        )
        for t in TRIALS
    ]
    return f"\nReversed counts = index counts with columns reversed, trials 0-2: {same}\n"


def accuracy_arrays(level: float, results: dict) -> dict:
    """arm -> (trials, classes) per-class final accuracy, for the arms with all trials rerun."""
    out = {}
    for arm in ARMS:
        if all((level, arm, t) in results for t in TRIALS):
            out[arm] = np.array(
                [per_class_accuracy(results[(level, arm, t)]["confusions"][-1]) for t in TRIALS]
            )
    return out


def overall_by_trial(level: float, results: dict, arms) -> pd.DataFrame:
    table = pd.DataFrame(
        {arm: [results[(level, arm, t)]["accs"][-1] for t in TRIALS] for arm in arms},
        index=[f"trial {t}" for t in TRIALS],
    )
    table["index - uncapped"] = table["capped index"] - table["uncapped"]
    table["reversed - uncapped"] = table["capped reversed"] - table["uncapped"]
    table["reversed - index"] = table["capped reversed"] - table["capped index"]
    table.loc["mean"] = table.mean()
    return table


def level_tables(level: float, results: dict, counts: dict) -> tuple[list[str], dict | None]:
    out = [f"\n## Label-HD {level:.2f}\n"]
    acc = accuracy_arrays(level, results)
    needed = ("capped index", "capped reversed", "uncapped")
    missing = [arm for arm in needed if arm not in acc]
    conc_text, conc = concentration_tables(level, counts)
    if missing:
        out.append(f"Per-class reruns incomplete: {missing}. Partition-level tables only.\n")
        return out + conc_text, None

    mean = pd.DataFrame(index=NAMES)
    for arm in ARMS:
        mean[arm] = acc[arm].mean(axis=0) if arm in acc else np.nan
    for arm, label in (("capped index", "index"), ("capped reversed", "reversed")):
        diff = acc[arm] - acc["uncapped"]
        mean[f"{label} - uncapped"] = diff.mean(axis=0)
        mean[f"{label} lower (of 3)"] = (diff < 0).sum(axis=0).astype(float)
    mean["reversed - index"] = (acc["capped reversed"] - acc["capped index"]).mean(axis=0)
    mean.loc["mean (= overall)"] = mean.mean()
    mean.loc["mean (= overall)", ["index lower (of 3)", "reversed lower (of 3)"]] = np.nan
    shown = mean.astype(object)
    if "pure" not in acc:
        shown["pure"] = "not run"
    for column in ("index lower (of 3)", "reversed lower (of 3)"):
        shown[column] = mean[column].map(lambda v: "" if np.isnan(v) else f"{v:.0f}")
    out.append("### Per-class final accuracy, means over trials 0-2\n")
    out.append(md(shown))

    for arm, label in (("capped index", "index"), ("capped reversed", "reversed")):
        diff = pd.DataFrame(
            {f"trial {t}": acc[arm][i] - acc["uncapped"][i] for i, t in enumerate(TRIALS)},
            index=NAMES,
        )
        diff["mean"] = diff.mean(axis=1)
        diff["lower (of 3)"] = (diff[[f"trial {t}" for t in TRIALS]] < 0).sum(axis=1).astype(int)
        out.append(f"\n### Paired per-trial difference, {label} - uncapped\n")
        out.append(md(diff))

    arms = [arm for arm in ARMS if arm in acc]
    overall = overall_by_trial(level, results, arms)
    weak = low_power(overall.loc[[f"trial {t}" for t in TRIALS], "index - uncapped"])
    out.append("\n### Overall final accuracy by trial\n")
    out.append(md(overall))

    g = gate(acc["capped index"], acc["uncapped"])
    v = verdict(acc["capped reversed"], acc["uncapped"])
    result = level_result(g, v)
    rows = [
        ("Step 1: D = -sum_c d_c (index - uncapped)", f"{g['D']:.4f}"),
        ("share of D on ship + truck", f"{g['share']:.3f}"),
        ("(a) D > 0 and share >= 0.5", "met" if g["met_a"] else "not met"),
        (
            "ship / truck lower than uncapped (trials)",
            f"{g['lower_ship']} / {g['lower_truck']} of 3",
        ),
        ("(b) each lower in >= 2 of 3", "met" if g["met_b"] else "not met"),
        ("gate", "met" if g["met"] else "not met"),
        (
            "secondary: share of the deficit classes' loss on ship + truck",
            f"{g['secondary_share']:.3f}",
        ),
        ("Step 2: D = -sum_c d_c (reversed - uncapped)", f"{v['D']:.4f}"),
        ("share of D on airplane + automobile", f"{v['share']:.3f}"),
        ("(a) D > 0 and share >= 0.5", "met" if v["met_a"] else "not met"),
        (
            "airplane / automobile lower than uncapped (trials)",
            f"{v['lower_airplane']} / {v['lower_automobile']} of 3",
        ),
        ("(b) each lower in >= 2 of 3", "met" if v["met_b"] else "not met"),
        (
            "ship / truck lower than uncapped (trials)",
            f"{v['lower_ship']} / {v['lower_truck']} of 3",
        ),
        ("(c) each NOT lower in >= 2 of 3", "met" if v["met_c"] else "not met"),
        (
            "Identity condition (ship and truck each lower in >= 2 of 3)",
            "met" if v["identity"] else "not met",
        ),
        ("result at this level", result),
        (
            "low-power (overall index - uncapped changes sign across trials)",
            "yes" if weak else "no",
        ),
    ]
    out.append("\n### Pre-registered gate and verdict\n")
    out.append(md(pd.DataFrame(rows, columns=["quantity", "value"]), index=False))

    zero = {
        arm: np.mean([(counts[(level, arm, t)][1] == 0).sum(axis=0) for t in TRIALS], axis=0)
        for arm in needed
    }
    rhos = []
    for arm, label in (("capped index", "index"), ("capped reversed", "reversed")):
        d = (acc[arm] - acc["uncapped"]).mean(axis=0)
        z = zero[arm] - zero["uncapped"]
        rhos.append((label, spearmanr(d, z).statistic if np.ptp(z) > 0 else np.nan))
    table = pd.DataFrame({f"zero-holders {arm}": zero[arm] for arm in needed}, index=NAMES)
    out.append("\n### Zero-holders (clients with no sample of the class), means over trials\n")
    out.append(md(table, ".2f"))
    out.append(
        "\nSpearman over the 10 classes, accuracy change vs zero-holder change against uncapped "
        "(descriptive only): " + ", ".join(f"{label} rho = {rho:.3f}" for label, rho in rhos) + "\n"
    )

    summed = {
        arm: sum(np.array(results[(level, arm, t)]["confusions"][-1]) for t in TRIALS)
        for arm in needed
    }
    for arm, label, classes in (
        ("capped index", "index", INDEX_LAST),
        ("capped reversed", "reversed", REVERSED_LAST),
    ):
        for c in classes:
            dest = pd.DataFrame({a: summed[a][c] for a in needed}, index=list(CLASSES))
            dest[f"{label} - uncapped"] = dest[arm] - dest["uncapped"]
            dest.index = [
                f"-> {i} {name}" + (" (correct)" if i == c else "")
                for i, name in enumerate(CLASSES)
            ]
            out.append(
                f"\n### True {CLASSES[c]}: predictions, summed over trials ({label} order concentrates it)\n"
            )
            out.append(md(dest, ".0f"))
    off = {}
    for arm in needed:
        m = summed[arm].copy()
        np.fill_diagonal(m, 0)
        off[arm] = m.sum(axis=0)
    out.append("\n### Images of another class predicted as each class, summed over trials\n")
    out.append(md(pd.DataFrame(off, index=[f"false '{n}'" for n in CLASSES]), ".0f"))

    out += conc_text
    summary = {
        "level": level,
        **{f"gate_{k}": g[k] for k in ("D", "share", "met_a", "met_b", "met")},
        **{f"verdict_{k}": v[k] for k in ("D", "share", "met_a", "met_b", "met_c", "identity")},
        "verdict_rule": v["verdict"],
        "result": result,
        "low_power": weak,
        **{
            f"overall_index_minus_uncapped_t{t}": overall.loc[f"trial {t}", "index - uncapped"]
            for t in TRIALS
        },
        **{
            f"overall_reversed_minus_uncapped_t{t}": overall.loc[
                f"trial {t}", "reversed - uncapped"
            ]
            for t in TRIALS
        },
        "concentration_present": conc["present"],
        **{f"spearman_{label}": rho for label, rho in rhos},
    }
    return out, summary


def per_class_rows(configs: dict, counts: dict, results: dict) -> pd.DataFrame:
    rows = []
    for (level, arm, trial), config in sorted(
        configs.items(), key=lambda kv: (kv[0][0], ARMS.index(kv[0][1]), kv[0][2])
    ):
        pid, matrix = counts[(level, arm, trial)]
        holders, effective = concentration(matrix)
        result = results.get((level, arm, trial))
        acc = per_class_accuracy(result["confusions"][-1]) if result else [np.nan] * len(CLASSES)
        for c, name in enumerate(CLASSES):
            rows.append(
                {
                    "level": level,
                    "arm": arm,
                    "trial": trial,
                    "class": c,
                    "class_name": name,
                    "accuracy": acc[c],
                    "holders": int(holders[c]),
                    "effective_holders": effective[c],
                    "zero_holders": int(matrix.shape[0] - holders[c]),
                    "run_id": config_hash(config),
                    "partition_id": pid,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--partitions", action="store_true", help="partition-level check only")
    args = parser.parse_args()

    configs = cells()
    out_dir = Path(next(iter(configs.values())).runtime.out_dir)
    counts = partition_counts(configs)
    provenance = {**git_provenance(), "generated_at": utc_stamp()}
    analysis = out_dir / "analysis"

    if args.partitions:
        text = ["# RQ2 class-order check: concentration across clients\n", f"{provenance}\n"]
        for level in LEVELS:
            text.append(f"\n## Label-HD {level:.2f}\n")
            text.append(mirror_note(level, counts))
            text += concentration_tables(level, counts)[0]
        report = "\n".join(text) + "\n"
        atomic_write_text(analysis / "rq2_class_order_partitions.md", report)
        print(report)
        return

    results = per_class_results(configs)
    text = ["# RQ2 class-order check: per-class tables\n", f"{provenance}\n"]
    summaries = []
    for level in LEVELS:
        level_text, summary = level_tables(level, results, counts)
        text += level_text
        if summary is not None:
            summaries.append(summary)
    report = "\n".join(text) + "\n"
    atomic_write_text(analysis / "rq2_class_order_tables.md", report)
    atomic_write_text(
        analysis / "rq2_class_order_per_class.csv",
        per_class_rows(configs, counts, results)
        .assign(**provenance)
        .to_csv(index=False, lineterminator="\n"),
    )
    atomic_write_text(
        analysis / "rq2_class_order_verdicts.csv",
        pd.DataFrame(summaries).assign(**provenance).to_csv(index=False, lineterminator="\n"),
    )
    print(report)
    print(f"-> {analysis}/rq2_class_order_*  ({provenance})")


if __name__ == "__main__":
    main()
