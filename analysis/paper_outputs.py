"""Every table, figure and quoted number of the paper's results, from one command.

    python -m analysis.paper_outputs [--cifar-trials 0,1,2,5,6] [--also-cifar-trials 0,1,2]
                                     [--covtype-fedprox] [--results-dir results] [--data-root data]
                                     [--out-dir results/analysis] [--covtype-rq1-session FILE]
                                     [--column-width 3.5] [--text-width 7.16]

Reads the recorded runs and partitions (read-only; see `analysis.paper_data`) and writes:

    {out}/numbers.tex, numbers.csv          one \\newcommand per number the text quotes, and
                                            the same values with their source output
    {out}/paper/tables/                     every table: .csv, .tex, .md, _caption_notes.txt
    {out}/paper/figures/                    every figure: .pdf, .png, .csv, _notes.txt
    {out}/paper/table_widths.csv            each table's estimated width at 8 pt
    {out}/paper/inputs_manifest.csv         every run and partition read, with its commit
    {out}/paper/checks.txt                  the checks made while reading (level-0 identity,
                                            recomputed label-HD, what is pending)
    {out}/paper/cifar_trials_*_vs_*.csv     CIFAR-10 numbers on both trial sets, sign changes
                                            of paired gaps and interactions flagged
    {out}/paper/reproduction_check.txt      the CIFAR-10 three-trial values checked against
                                            the recorded analysis of trials 0-2
    {out}/paper/trials_0_1_2/               the same outputs on CIFAR-10 trials {0,1,2}
    {out}/index.md                          every file above, with the command and commit

Covtype FedProx (covtype_rq2.yaml) is off by default: until it has finished training its cells
read "pending", its figure lines are absent with a note, and its numbers are \\textbf{??}.
--covtype-fedprox switches it on. The Covtype RQ1 rows come from config/grids/covtype_rq1.yaml
when its partitions exist, else from --covtype-rq1-session (marked PROVISIONAL), else pending.

The reproduction check never adjusts anything: a value that does not match is reported, and
the command exits with status 1 after writing every output.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import paper_figures as figs
from analysis import paper_results as res
from analysis.paper_data import CIFAR_TRIALS, COVTYPE_TRIALS, Sources
from analysis.paper_stats import mean_sd
from analysis.paper_tables import estimate_width, write_table
from experiments.recording import atomic_write_text, git_provenance, utc_stamp

LEVEL_WORDS = {
    0.0: "Zero",
    0.15: "Fifteen",
    0.25: "TwentyFive",
    0.35: "ThirtyFive",
    0.45: "FortyFive",
    0.5: "Fifty",
    0.7: "Seventy",
}
DATASET_WORDS = {"cifar10": "Cifar", "covtype": "Covtype"}
ALGORITHM_WORDS = {"fedavg": "FedAvg", "fedprox": "FedProx"}
ARM_WORDS = {
    "natural uncapped": "Uncapped",
    "natural capped": "Capped",
    "pure": "Pure",
    res.NATURAL: "Natural",
    "natural per-client": "PerClient",
}
PLACEHOLDER = r"\textbf{??}"


def level_word(level: float) -> str:
    return LEVEL_WORDS[round(float(level), 2)]


# ---------------------------------------------------------------- numbers
class Numbers:
    """Macros for the Results text; letters-only names; a pending value prints \\textbf{??}."""

    def __init__(self):
        self.rows = []

    def add(self, name, value, decimals=None, source="", status="final", kind="number"):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name!r} is not letters only")
        if any(r["macro"] == name for r in self.rows):
            raise ValueError(f"macro {name} defined twice")
        if status == "pending" or value is None or (isinstance(value, float) and np.isnan(value)):
            tex, status = PLACEHOLDER, "pending" if status != "not applicable" else status
        elif kind == "text":
            tex = str(value)
        elif decimals is None:
            tex = f"\\ensuremath{{{int(value)}}}"
        else:
            tex = f"\\ensuremath{{{value:.{decimals}f}}}"
        self.rows.append(
            {"macro": name, "tex": tex, "value": value, "status": status, "source": source}
        )

    def pm(self, name, mean, sd, decimals, source, status="final"):
        self.add(name, mean, decimals, source, status)
        self.add(name + "Sd", sd, decimals, source, status)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def tex(self, header: str) -> str:
        lines = [f"% {line}" for line in header.splitlines()]
        for r in self.rows:
            lines.append(f"\\newcommand{{\\{r['macro']}}}{{{r['tex']}}}")
        return "\n".join(lines) + "\n"


def build_numbers(
    b, summary, gaps, add, wins, mu_selected, rq1s, rq1max, matched, diffs045, tol, cov_facts
) -> Numbers:
    N = Numbers()
    N.add("nTrialsCifar", len(b.trials("cifar10")), source="trial set")
    N.add("nTrialsCovtype", len(b.trials("covtype")), source="trial set")
    for d in ("cifar10", "covtype"):
        N.add(f"mu{DATASET_WORDS[d]}", mu_selected[d], 3, "mu_sweep")
        c = res._summary_row(
            summary, dataset=d, grid=res.CENTRALIZED_GRID[d], algorithm="centralized"
        )
        N.pm(
            f"central{DATASET_WORDS[d]}Acc",
            c["final_acc_mean"],
            c["final_acc_sd"],
            4,
            "main_results_body",
        )

    # RQ2 accuracies and HD per arm
    for d in ("cifar10", "covtype"):
        D = DATASET_WORDS[d]
        grid = res.RQ2_GRID[d]
        for L in res.rq2_levels(b, d):
            for arm in res.rq2_arms(d, L):
                hd = res.partition_hd_means(b, d, grid, arm, L)
                A = ARM_WORDS[arm]
                N.add(f"rqTwo{D}Label{A}{level_word(L)}", hd["label_hd"], 3, "main_results_body")
                N.add(f"rqTwo{D}Qty{A}{level_word(L)}", hd["qty_hd"], 3, "main_results_body")
                N.add(f"rqTwo{D}SwLabel{A}{level_word(L)}", hd["sw_hd"], 3, "main_results_full")
                for alg in res.ALGORITHMS:
                    status = "pending" if b.is_pending(d, alg) else "final"
                    c = (
                        None
                        if status == "pending"
                        else res._summary_row(
                            summary, dataset=d, grid=grid, cell_arm=arm, L=L, Q=None, algorithm=alg
                        )
                    )
                    N.pm(
                        f"rqTwo{D}Acc{A}{level_word(L)}{ALGORITHM_WORDS[alg]}",
                        None if c is None else c["final_acc_mean"],
                        None if c is None else c["final_acc_sd"],
                        4,
                        "main_results_body",
                        status,
                    )
    # RQ2 gaps
    for _, g in gaps.iterrows():
        D, A, B = DATASET_WORDS[g["dataset"]], ARM_WORDS[g["arm_a"]], ARM_WORDS[g["arm_b"]]
        name = f"rqTwo{D}Gap{A}{B}{level_word(g['level'])}{ALGORITHM_WORDS[g['algorithm']]}"
        status = "pending" if g["pending"] else "final"
        N.pm(name, g.get("gap_mean"), g.get("gap_sd"), 4, "rq2_paired_gaps", status)
        N.add(name + "Pos", g.get("n_positive"), None, "rq2_paired_gaps", status)
        N.add(name + "Neg", g.get("n_negative"), None, "rq2_paired_gaps", status)
    # RQ2 per-class: the two largest drops at 0.45, index and reversed order
    for key, word in (("index", "Index"), ("reversed", "Reversed")):
        mean = diffs045[key].mean(axis=0)
        order = np.argsort(mean)
        for rank, wordrank in zip(order[:2], ("One", "Two")):
            src = "rq2_accuracy_vs_label_hd (panel c)"
            N.add(f"rqTwoClass{word}Drop{wordrank}", mean[rank], 3, src)
            N.add(f"rqTwoClass{word}Drop{wordrank}Name", res.CLASSES[rank], source=src, kind="text")
    N.add("rqTwoClassTrials", len(diffs045["trials"]), source="rq2_accuracy_vs_label_hd (panel c)")

    # RQ3
    for alg in res.ALGORITHMS:
        a = add[add["algorithm"] == alg]
        Alg = ALGORITHM_WORDS[alg]
        acc = res.rq3_acc(b, alg)
        Ls, Qs = res.rq3_levels(b)
        corner = [v for (L, Q, _), v in acc.items() if L == Ls[0] and Q == Qs[0]]
        N.pm(f"rqThreeCorner{Alg}", *mean_sd(corner)[:2], 4, "rq3_accuracy_heatmap")
        for _, r in a.iterrows():
            name = f"rqThreeInt{level_word(r['level'])}{level_word(r['quantity_level'])}{Alg}"
            N.pm(name, r["interaction_mean"], r["interaction_sd"], 3, "rq3_additivity")
            N.add(name + "Neg", r["n_negative"], None, "rq3_additivity")
        N.add(f"rqThreeNegTotal{Alg}", int(a["n_negative"].sum()), None, "rq3_additivity")
        N.add(f"rqThreeNegOf{Alg}", int(a["n"].sum()), None, "rq3_additivity")
    N.add("rqThreeProxWins", wins["wins"], None, "rq3 FedProx - FedAvg, paired by cell and trial")
    N.add("rqThreeProxPairs", wins["pairs"], None, "rq3 FedProx - FedAvg, paired by cell and trial")
    r = b.runs[(b.runs["grid"] == "rq3") & (b.runs["algorithm"] == "fedavg") & (b.runs["Q"] == 0.7)]
    N.add("rqThreeEarlyPeakRuns", int((r["best_round"] <= 3).sum()), None, "rq3_rta")
    N.add("rqThreeEarlyPeakOf", len(r), None, "rq3_rta")

    # RQ1
    for _, m in rq1max.iterrows():
        D, A = DATASET_WORDS[m["dataset"]], ARM_WORDS[m["arm"]]
        st = b.rq1_status.get(m["dataset"], "final")
        N.pm(
            f"rqOne{D}MaxQty{A}",
            m["qty_hd_mean"],
            m["qty_hd_sd"],
            3,
            "rq1_max_incidental_quantity",
            st,
        )
        N.add(f"rqOne{D}MaxQty{A}Alpha", m["alpha"], 2, "rq1_max_incidental_quantity", st)
        N.add(
            f"rqOne{D}MaxQty{A}Beta",
            m["beta_equivalent"],
            4,
            "rq1_max_incidental_quantity",
            "pending" if m["pending"] else st,
        )
    for _, m in matched.iterrows():
        for d in ("cifar10", "covtype"):
            if d not in m:
                continue
            st = b.rq1_status.get(d, "final")
            value = m[d]
            N.add(
                f"rqOneMatched{level_word(m['label_hd'])}{DATASET_WORDS[d]}{ARM_WORDS[m['arm']]}",
                value,
                3,
                "rq1_matched_label_hd",
                st if not pd.isna(value) else "not applicable",
            )
    # Covtype partition facts
    pure = cov_facts[cov_facts["arm"] == "pure"]
    N.add(
        "covtypePureDriftMean", float(pure["mean_label_drift"].mean()), 3, "covtype partition facts"
    )
    N.add("covtypePureDriftMax", float(pure["max_label_drift"].max()), 3, "covtype partition facts")
    # tolerance
    for d in ("cifar10", "covtype"):
        t = tol[tol["dataset"] == d]
        out = t[t["label_outside"]]
        N.add(f"tolOutsideLabel{DATASET_WORDS[d]}", len(out), None, "tolerance")
        N.add(f"tolPairs{DATASET_WORDS[d]}", len(t), None, "tolerance")
        listing = (
            "; ".join(
                f"{a} {L:.2f} trial {tr}"
                for a, L, tr in zip(out["arm"], out["level"], out["trial"])
            )
            or "none"
        )
        N.add(f"tolOutsideLabel{DATASET_WORDS[d]}List", listing, source="tolerance", kind="text")
    q = tol[tol["quantity_outside"]]
    N.add("tolOutsideQtyCifar", len(q), None, "tolerance (RQ3 quantity, descriptive)")
    return N


# ---------------------------------------------------------------- reproduction check
EXPECTED_RQ2 = {
    0.0: {"pure": 0.6717, res.NATURAL: 0.6686},
    0.25: {"pure": 0.6683, "natural uncapped": 0.6593, "natural capped": 0.6472},
    0.45: {"pure": 0.6418, "natural uncapped": 0.6423, "natural capped": 0.6158},
    0.7: {"pure": 0.5605, "natural uncapped": 0.5374, "natural capped": 0.5314},
}


def reproduction_check(
    b3, summary3, add3, wins3, rq1s, covtype_rows
) -> list[tuple[str, float, float, int, bool]]:
    """(item, expected, got, decimals, ok) for every value of the recorded three-trial analysis."""
    checks = []

    def check(item, expected, got, decimals):
        ok = (
            got is not None
            and not pd.isna(got)
            and round(float(got), decimals) == round(expected, decimals)
        )
        checks.append((item, expected, got, decimals, bool(ok)))

    for L, arms in EXPECTED_RQ2.items():
        for arm, expected in arms.items():
            c = res._summary_row(
                summary3,
                dataset="cifar10",
                grid="rq2",
                cell_arm=arm,
                L=L,
                Q=None,
                algorithm="fedavg",
            )
            check(f"RQ2 FedAvg final acc, {arm}, level {L:.2f}", expected, c["final_acc_mean"], 4)
    for arm, expected in (("natural uncapped", 0.463), ("natural capped", 0.465), ("pure", 0.448)):
        check(
            f"RQ2 realized label-HD at 0.45, {arm}",
            expected,
            res.partition_hd_means(b3, "cifar10", "rq2", arm, 0.45)["label_hd"],
            3,
        )
    for arm, expected in (("natural uncapped", 0.361), ("natural capped", 0.250)):
        check(
            f"RQ2 realized quantity-HD at 0.70, {arm}",
            expected,
            res.partition_hd_means(b3, "cifar10", "rq2", arm, 0.7)["qty_hd"],
            3,
        )
    for L in res.rq2_levels(b3, "cifar10"):
        check(
            f"RQ2 realized quantity-HD, pure, level {L:.2f}",
            0.0,
            res.partition_hd_means(b3, "cifar10", "rq2", "pure", L)["qty_hd"],
            3,
        )
    acc = res.rq3_acc(b3, "fedavg")
    check(
        "RQ3 FedAvg (0,0) corner",
        0.6606,
        mean_sd([v for (L, Q, _), v in acc.items() if L == 0 and Q == 0])[0],
        4,
    )
    a = add3[add3["algorithm"] == "fedavg"]
    # (mean, decimals), (sd, decimals). The sd at 0.70 x 0.70 is checked against the recorded
    # report's 4-decimal value, 0.0245: the 3-decimal 0.025 first given for it is that value
    # rounded a second time (the author's decision of 2026-11-14).
    expected_interactions = {
        (0.7, 0.7): ((-0.041, 3), (0.0245, 4)),
        (0.7, 0.35): ((-0.021, 3), (0.024, 3)),
    }
    for (L, Q), ((m, dm), (s, ds)) in expected_interactions.items():
        r = a[(a["level"] == L) & (a["quantity_level"] == Q)].iloc[0]
        check(f"RQ3 FedAvg interaction mean, {L} x {Q}", m, r["interaction_mean"], dm)
        check(f"RQ3 FedAvg interaction sd, {L} x {Q}", s, r["interaction_sd"], ds)
        check(f"RQ3 FedAvg interaction negative trials, {L} x {Q}", 3, r["n_negative"], 0)
    for alg, expected in (("fedavg", 15), ("fedprox", 13)):
        check(
            f"RQ3 {alg} negative per-trial interactions (of 18)",
            expected,
            add3[add3["algorithm"] == alg]["n_negative"].sum(),
            0,
        )
    check("RQ3 FedProx beats FedAvg, paired runs (of 36)", 29, wins3["wins"], 0)
    check("RQ3 paired runs", 36, wins3["pairs"], 0)
    k30 = rq1s[(rq1s["dataset"] == "cifar10") & (rq1s["K"] == 30)]
    for alpha, values in ((0.03, (0.42, 0.36, 0.13)), (1.0, (0.10, 0.08, 0.01))):
        for arm, expected in zip(
            ("natural uncapped", "natural capped", "natural per-client"), values
        ):
            got = k30[(k30["arm"] == arm) & (k30["alpha"] == alpha)]["qty_hd_mean"].iloc[0]
            check(f"RQ1 K=30 incidental quantity-HD, {arm}, alpha {alpha:g}", expected, got, 2)
    inf = b3.rq1[(b3.rq1["dataset"] == "cifar10") & (b3.rq1["status"] != "generated")]
    check("RQ1 CIFAR-10 infeasible partitions", 6, len(inf), 0)
    check(
        "RQ1 infeasible are natural per-class, K=100, alpha 0.03",
        1,
        float(
            set(zip(inf["K"], inf["alpha"])) == {(100, 0.03)}
            and set(inf["arm"]) == {"natural uncapped", "natural capped"}
        ),
        0,
    )
    c = res._summary_row(summary3, dataset="cifar10", grid="centralized", algorithm="centralized")
    check("Centralized CIFAR-10 mean", 0.7059, c["final_acc_mean"], 4)
    check("Centralized CIFAR-10 sd", 0.0044, c["final_acc_sd"], 4)
    check(
        "Covtype natural uncapped level 0 trial 0 FedAvg final acc (timing run 0.9449)",
        0.9449,
        covtype_rows["rq2"],
        4,
    )
    check("Covtype centralized 10 epochs trial 0 final acc", 0.9429, covtype_rows["central"], 4)
    return checks


# ---------------------------------------------------------------- one trial set
def generate(sources: Sources, out: Path, widths, session, numbers_dir: Path) -> dict:
    tables_dir, figures_dir = out / "tables", out / "figures"
    b = res.prepare(sources, session)
    summary = res.cell_summary(b.runs)
    gaps = res.rq2_gaps(b)
    rta_gaps = res.rq2_gaps(b, "rta")
    add = res.rq3_additivity(b)
    rta_add = res.rq3_additivity(b, "rta")
    wins = res.fedprox_wins(b)
    reg_rows, coefs, lof, fits = res.rq3_regression(b)
    mu_table, mu_selected = res.mu_sweep(b)
    rq1s = res.rq1_summary(b)
    rq1max = res.rq1_max_incidental(b, rq1s)
    matched = res.rq1_matched(b, rq1s)
    tol = res.tolerance_misses(b)
    cov_facts = res.covtype_partition_facts(b)
    diffs = {L: res.per_class_differences(b, L) for L in (0.25, 0.45, 0.7)}

    tables = [
        res.setup_table(b),
        res.main_results_body(b, summary),
        *res.main_results_full(b, summary),
        res.rq2_gaps_table(b, gaps),
        res.rq3_additivity_table(b, add),
        res.rq1_summary_table(b, rq1s),
        res.rq1_max_table(b, rq1max),
        res.rq1_matched_table(b, matched),
        res.rq1_pairwise_table(b, rq1s),
        res.rq1_infeasible_table(b),
        res.rq2_rta_table(b, rta_gaps),
        res.rq3_rta_table(b, summary, rta_add),
        mu_table,
        *res.rq3_regression_tables(b, reg_rows, coefs, lof),
        res.tolerance_table(b, tol),
    ]
    written, widths_rows = [], []
    for t in tables:
        written += write_table(t, tables_dir)
        widths_rows.append(
            {
                "table": t.name,
                "width_in_8pt": round(estimate_width(t, 8.0), 2),
                "width_in_7pt": round(estimate_width(t, 7.0), 2),
                "tabcolsep_pt": t.tabcolsep_pt or 6,
            }
        )
    atomic_write_text(
        out / "table_widths.csv", pd.DataFrame(widths_rows).to_csv(index=False, lineterminator="\n")
    )

    written += figs.rq1_figure(b, figures_dir, widths)
    written += figs.rq1_k_figure(b, figures_dir, widths)
    written += figs.rq2_figure(b, summary, diffs[0.45], figures_dir, widths)
    written += figs.rq2_balanced_figure(b, summary, figures_dir, widths)
    for L in (0.25, 0.7):
        written += figs.per_class_figure(b, diffs[L], L, figures_dir, widths)
    written += figs.rq3_heatmap(b, summary, add, "fedavg", "final_acc", figures_dir, widths)
    written += figs.rq3_heatmap(b, summary, add, "fedprox", "final_acc", figures_dir, widths)
    written += figs.rq3_heatmap(b, summary, rta_add, "fedavg", "rta", figures_dir, widths)
    written += figs.regression_residuals(b, fits, figures_dir, widths)
    written += figs.single_panel_figures(b, summary, diffs, figures_dir / "single", widths)
    written += figs.single_residual_figures(b, fits, figures_dir / "single", widths)

    numbers = build_numbers(
        b, summary, gaps, add, wins, mu_selected, rq1s, rq1max, matched, diffs[0.45], tol, cov_facts
    )
    provenance = {**git_provenance(), "generated_at": utc_stamp()}
    header = (
        "Numbers quoted in the Results text, generated by python -m analysis.paper_outputs\n"
        f"CIFAR-10 trials {res.trial_set_label(b.trials('cifar10'))}; Covtype trials "
        f"{res.trial_set_label(b.trials('covtype'))}; Covtype FedProx "
        f"{'on' if sources.covtype_fedprox else 'off (pending)'}\n"
        f"commit {provenance['git_commit']} dirty {provenance['git_dirty']} at {provenance['generated_at']}\n"
        "\\textbf{??} marks a value that is pending."
    )
    numbers_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(numbers_dir / "numbers.tex", numbers.tex(header))
    atomic_write_text(
        numbers_dir / "numbers.csv",
        numbers.frame().assign(**provenance).to_csv(index=False, lineterminator="\n"),
    )

    manifest_runs = b.runs[
        [
            "run_id",
            "grid",
            "dataset",
            "arm",
            "L",
            "Q",
            "algorithm",
            "trial",
            "partition_id",
            "git_commit",
            "git_dirty",
        ]
    ]
    manifest_parts = b.parts[
        ["partition_id", "dataset", "arm", "trial", "git_commit", "git_dirty"]
    ].assign(grid="(partition)")
    atomic_write_text(
        out / "inputs_manifest.csv",
        pd.concat(
            [manifest_runs.assign(kind="run"), manifest_parts.assign(kind="partition")],
            ignore_index=True,
        ).to_csv(index=False, lineterminator="\n"),
    )
    checks = [
        "Checks made while reading the results (descriptive; none adjusts anything)",
        "max |recomputed - recorded| unweighted label-HD over the partitions read: "
        f"{b.checks['max_abs_recomputed_label_hd_difference']:.3g}",
        "CIFAR-10 level-0 natural arms identical (partition client by client, accuracy every "
        "round, final accuracy), per (trial, algorithm, uncapped run, capped run): "
        + "; ".join(
            f"{t} {a} {u}={c}: {s}" for t, a, u, c, s in b.checks["cifar_level0_identical"]
        ),
        "Covtype level-0 trials whose capped and uncapped partitions are identical "
        "(trial, FedAvg runs identical): "
        + (str(b.checks["covtype_level0_identical_trials"]) or "none"),
        f"Covtype FedProx (covtype_rq2): {'read' if sources.covtype_fedprox else 'not read (pending)'}",
        f"Optional grids not trained: {sorted({g for g, *_ in b.pending_runs}) or 'none'}",
        f"Covtype RQ1 source: {b.rq1_status.get('covtype')}",
        f"{provenance}",
    ]
    atomic_write_text(out / "checks.txt", "\n".join(checks) + "\n")
    return {
        "bundle": b,
        "summary": summary,
        "add": add,
        "wins": wins,
        "rq1s": rq1s,
        "numbers": numbers,
        "lof": lof,
        "tol": tol,
        "written": written,
    }


CIFAR_PREFIXES = ("rqTwoCifar", "rqThree", "centralCifar", "rqTwoClass")


def compare_trial_sets(main: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """CIFAR-10 numbers on the reported trials beside the same numbers on another trial set.

    A paired gap or an interaction (a macro with Gap or Int and no Sd/Pos/Neg suffix) whose sign
    differs between the two is flagged.
    """
    a = main.set_index("macro")
    b = other.set_index("macro")
    rows = []
    for macro in a.index:
        if not macro.startswith(CIFAR_PREFIXES) or macro not in b.index:
            continue
        va, vb = a.loc[macro, "value"], b.loc[macro, "value"]
        numeric = isinstance(va, (int, float, np.integer, np.floating)) and not isinstance(va, bool)
        signed = re.search(r"(Gap|Int)", macro) and not re.search(r"(Sd|Pos|Neg)$", macro)
        flip = bool(
            numeric
            and signed
            and not pd.isna(va)
            and not pd.isna(vb)
            and np.sign(round(va, 4)) != np.sign(round(vb, 4))
        )
        rows.append(
            {
                "macro": macro,
                "reported_trials": va,
                "other_trials": vb,
                "difference": (
                    (va - vb) if numeric and not pd.isna(va) and not pd.isna(vb) else np.nan
                ),
                "sign_change": flip,
                "source": a.loc[macro, "source"],
            }
        )
    return pd.DataFrame(rows).sort_values(["sign_change", "macro"], ascending=[False, True])


def parse_trials(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.split(",") if t.strip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cifar-trials", default=",".join(map(str, CIFAR_TRIALS)))
    parser.add_argument(
        "--also-cifar-trials", default="0,1,2", help="a second CIFAR-10 trial set; empty to skip"
    )
    parser.add_argument("--covtype-trials", default=",".join(map(str, COVTYPE_TRIALS)))
    parser.add_argument(
        "--covtype-fedprox",
        action="store_true",
        help="read Covtype FedProx (only once it has finished)",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--covtype-rq1-session", type=Path, default=None)
    parser.add_argument("--column-width", type=float, default=figs.COLUMN_WIDTH)
    parser.add_argument("--text-width", type=float, default=figs.TEXT_WIDTH)
    args = parser.parse_args(argv)

    widths = figs.Widths(args.column_width, args.text_width)
    common = dict(
        results_dir=args.results_dir,
        data_root=args.data_root,
        covtype_trials=parse_trials(args.covtype_trials),
        covtype_fedprox=args.covtype_fedprox,
    )
    paper = args.out_dir / "paper"
    main_set = Sources(cifar_trials=parse_trials(args.cifar_trials), **common)
    result = generate(main_set, paper, widths, args.covtype_rq1_session, args.out_dir)
    print(f"{len(result['written'])} files for CIFAR-10 trials {args.cifar_trials} -> {paper}")

    lines = []
    status = 0
    also = parse_trials(args.also_cifar_trials)
    if also:
        sub = paper / ("trials_" + "_".join(map(str, also)))
        three = generate(
            Sources(cifar_trials=also, **common), sub, widths, args.covtype_rq1_session, sub
        )
        print(
            f"{len(three['written'])} files for CIFAR-10 trials {args.also_cifar_trials} -> {sub}"
        )
        comparison = compare_trial_sets(result["numbers"].frame(), three["numbers"].frame())
        name = f"cifar_trials_{'_'.join(map(str, main_set.cifar_trials))}_vs_{'_'.join(map(str, also))}.csv"
        atomic_write_text(paper / name, comparison.to_csv(index=False, lineterminator="\n"))
        flips = comparison[comparison["sign_change"]]
        lines.append(
            f"Sign changes between the two CIFAR-10 trial sets (paired gaps and interactions): {len(flips)}"
        )
        for _, f in flips.iterrows():
            lines.append(
                f"  SIGN CHANGE {f['macro']}: {f['reported_trials']:+.4f} vs {f['other_trials']:+.4f}"
            )
        lines.append(f"(all values side by side: {name})\n")
        if also == (0, 1, 2):
            b3 = three["bundle"]
            cov = b3.runs[b3.runs["dataset"] == "covtype"]
            covtype_rows = {
                "rq2": cov[
                    (cov["grid"] == "covtype_rq2")
                    & (cov["arm"] == "natural uncapped")
                    & (cov["L"] == 0.0)
                    & (cov["trial"] == 0)
                    & (cov["algorithm"] == "fedavg")
                ]["final_acc"].iloc[0],
                "central": cov[(cov["grid"] == "covtype_centralized") & (cov["trial"] == 0)][
                    "final_acc"
                ].iloc[0],
            }
            checks = reproduction_check(
                b3, three["summary"], three["add"], three["wins"], three["rq1s"], covtype_rows
            )
            failed = [c for c in checks if not c[4]]
            lines.append(
                f"Reproduction check, CIFAR-10 trials {{0,1,2}}: {len(checks) - len(failed)} of {len(checks)} match at the printed precision"
            )
            for item, expected, got, decimals, ok in checks:
                lines.append(
                    f"{'ok  ' if ok else 'FAIL'} {item}: expected {expected}, got {float(got):.{max(decimals, 4) + 2}f}"
                )
            status = 1 if failed else 0
    provenance = {**git_provenance(), "generated_at": utc_stamp()}
    lines.append(f"\n{provenance}")
    atomic_write_text(paper / "reproduction_check.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))
    write_index(args, main_set, provenance, argv)
    return status


INDEX_KINDS = (
    ("numbers", "Numbers quoted in the text"),
    ("tables", "Tables"),
    ("figures", "Figures"),
    ("checks", "Checks, comparison and inputs"),
    ("trials", "The same outputs on another CIFAR-10 trial set"),
)


def _index_kind(rel: Path) -> str:
    if rel.parts[0] != "paper":
        return "numbers"
    if len(rel.parts) > 2 and rel.parts[1].startswith("trials_"):
        return "trials"
    return {"tables": "tables", "figures": "figures"}.get(rel.parts[1], "checks")


def write_index(args, sources: Sources, provenance: dict, argv) -> Path:
    """{out}/index.md: every output file, with the command and commit that produced it."""
    out = args.out_dir
    files = sorted(
        p.relative_to(out)
        for p in [out / "numbers.tex", out / "numbers.csv", *(out / "paper").rglob("*")]
        if p.is_file()
    )
    command = "python -m analysis.paper_outputs " + " ".join(argv if argv else sys.argv[1:])
    lines = [
        "# Results-section outputs",
        "",
        f"Produced by `{command.strip()}` (script `analysis/paper_outputs.py`) at commit "
        f"{provenance['git_commit']}, git_dirty {provenance['git_dirty']}, "
        f"{provenance['generated_at']} UTC. Every file below was written by that run.",
        "",
        f"- CIFAR-10 trials {res.trial_set_label(sources.cifar_trials)}; Covtype trials "
        f"{res.trial_set_label(sources.covtype_trials)}; Covtype FedProx "
        f"{'included' if sources.covtype_fedprox else 'not included (pending)'}",
        "- The runs and partitions read, with the commit each was recorded at: "
        "`paper/inputs_manifest.csv`",
        "",
    ]
    for kind, title in INDEX_KINDS:
        chosen = [f for f in files if _index_kind(f) == kind]
        if not chosen:
            continue
        lines += [f"## {title}", ""]
        lines += [f"- `{f.as_posix()}`" for f in chosen]
        lines.append("")
    path = out / "index.md"
    atomic_write_text(path, "\n".join(lines))
    return path


if __name__ == "__main__":
    sys.exit(main())
