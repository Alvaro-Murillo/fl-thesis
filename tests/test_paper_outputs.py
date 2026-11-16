"""Tests for the paper's results outputs: statistics, table writer, pending cells, numbers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import paper_outputs as po
from analysis import paper_results as res
from analysis.paper_data import Sources, arm_label, size_weighted_label_hd
from analysis.paper_stats import (
    fit_regression,
    interactions,
    lack_of_fit,
    mean_sd,
    model_interaction,
    paired,
)
from analysis.paper_tables import (
    Col,
    Table,
    estimate_width,
    fmt_pm,
    to_latex,
    to_markdown,
    write_table,
)

# ---------------------------------------------------------------- statistics


def test_mean_sd_is_the_sample_sd_and_handles_one_and_no_values():
    m, s, n = mean_sd([0.1, 0.2, 0.4])
    assert (m, n) == (pytest.approx(0.7 / 3), 3)
    assert s == pytest.approx(np.std([0.1, 0.2, 0.4], ddof=1))
    assert np.isnan(mean_sd([0.5])[1]) and mean_sd([])[2] == 0


def test_paired_counts_signs_and_ties_and_refuses_unpaired_trials():
    p = paired({0: 0.5, 1: 0.4, 2: 0.3}, {0: 0.4, 1: 0.4, 2: 0.35})
    assert p.differences == pytest.approx((0.1, 0.0, -0.05))
    assert (p.n_positive, p.n_negative, p.n_zero, p.n) == (1, 1, 1, 3)
    with pytest.raises(ValueError, match="unpaired"):
        paired({0: 1.0, 1: 1.0}, {0: 1.0, 2: 1.0})


def test_interaction_is_combined_minus_label_minus_quantity_against_each_trials_corner():
    acc = {}
    for t, base in ((0, 0.70), (1, 0.60)):
        acc.update({(0, 0, t): base, (1, 0, t): base - 0.1, (0, 1, t): base - 0.02})
        acc[(1, 1, t)] = base - 0.1 - 0.02 - 0.03 * (t + 1)
    out = interactions(acc, [0, 1], [0, 1])
    assert list(out) == [(1, 1)]
    assert out[(1, 1)] == pytest.approx({0: -0.03, 1: -0.06})


def _rq3_runs(coef, trials=(0, 1, 2), noise=0.0, drift=0.02):
    rng = np.random.default_rng(0)
    rows = []
    for t in trials:
        for Lv in (0.0, 0.25, 0.45, 0.70):
            for Q in (0.0, 0.35, 0.70):
                L = Lv + drift * Q + 0.01 * rng.standard_normal()
                acc = (
                    coef["b0"]
                    + coef["q"][Q]
                    + (coef["l1"] + coef["g1"][Q]) * L
                    + (coef["l2"] + coef["g2"][Q]) * L**2
                    + 0.01 * t
                    + noise * rng.standard_normal()
                )
                rows.append({"acc": acc, "L": L, "L_level": Lv, "Q_level": Q, "trial": t})
    return pd.DataFrame(rows)


COEF = {
    "b0": 0.66,
    "q": {0.0: 0.0, 0.35: 0.01, 0.70: 0.005},
    "l1": -0.05,
    "l2": -0.15,
    "g1": {0.0: 0.0, 0.35: -0.01, 0.70: -0.02},
    "g2": {0.0: 0.0, 0.35: 0.0, 0.70: -0.03},
}


def test_regression_recovers_noiseless_coefficients_and_its_matched_interaction():
    runs = _rq3_runs(COEF)
    fit = fit_regression(runs)
    named = dict(zip(fit.names, fit.coef))
    assert named["L"] == pytest.approx(-0.05, abs=1e-9)
    assert named["L^2:Q=0.7"] == pytest.approx(-0.03, abs=1e-9)
    assert fit.rss == pytest.approx(0.0, abs=1e-20)
    cell_L = runs.groupby(["L_level", "Q_level"])["L"].mean().to_dict()
    got = model_interaction(fit, cell_L, 0.70, 0.70, [0.0, 0.35, 0.70], [0, 1, 2])
    Lc, Lq, L0 = cell_L[(0.70, 0.70)], cell_L[(0.0, 0.70)], cell_L[(0.0, 0.0)]

    def m(L, Q):
        return COEF["q"][Q] + (COEF["l1"] + COEF["g1"][Q]) * L + (COEF["l2"] + COEF["g2"][Q]) * L**2

    assert got == pytest.approx((m(Lc, 0.7) - m(Lq, 0.7)) - (m(Lc, 0.0) - m(L0, 0.0)), abs=1e-12)


def test_lack_of_fit_has_the_nested_degrees_of_freedom_and_grows_with_a_cell_effect_off_the_curve():
    runs = _rq3_runs(COEF, trials=(0, 1, 2, 5, 6), noise=0.002)
    lof = lack_of_fit(runs)
    assert (lof["df_num"], lof["df_den"]) == (3, 60 - 16)
    assert lof["rss_regression_cell_L"] >= lof["rss_cell_means"] and lof["F"] >= 0
    F = ((lof["rss_regression_cell_L"] - lof["rss_cell_means"]) / 3) / (lof["rss_cell_means"] / 44)
    assert lof["F"] == pytest.approx(F)
    bumped = runs.copy()
    cell = (bumped["L_level"] == 0.45) & (bumped["Q_level"] == 0.35)
    bumped.loc[cell, "acc"] += 0.05  # not a quadratic in L within its quantity level
    assert lack_of_fit(bumped)["F"] > 10 * max(lof["F"], 1.0)


def test_size_weighted_label_hd_weights_clients_by_size():
    counts = np.array([[10, 0], [45, 45], [0, 0]])
    glob = np.array([0.5, 0.5])
    weighted, unweighted = size_weighted_label_hd(counts, glob)
    one_class = np.sqrt(1 - np.sqrt(0.5))
    assert unweighted == pytest.approx(one_class / 2)  # the empty client is left out
    assert weighted == pytest.approx(one_class * 10 / 100)


def test_display_names_never_use_the_size_cap_code_name():
    assert arm_label("natural_per_class", "niid_bench") == "natural capped"
    assert (
        arm_label("natural_per_class", "niid_bench", "reversed") == "natural capped, reversed order"
    )
    assert arm_label("natural_per_class") == "natural uncapped"
    assert (
        arm_label("pure_label") == "pure"
        and arm_label("natural_per_client") == "natural per-client"
    )


# ---------------------------------------------------------------- tables


def _table(pending=""):
    df = pd.DataFrame(
        [
            {
                "section": "A",
                "row_kind": "cell",
                "name": "pure",
                "m": 0.67171,
                "s": 0.00444,
                "x": -0.00001,
                "pending": pending,
            },
            {
                "section": "A",
                "row_kind": "reference",
                "name": "centralized",
                "rm": 0.7082,
                "rs": 0.0047,
                "pending": "",
            },
        ]
    )
    cols = [
        Col("name", "Arm", "text", ("name",), align="l"),
        Col("x", "label-\nHD", "num", ("x",), decimals=3),
        Col("a", "FedAvg", "pm", ("m", "s"), 4, ref=("rm", "rs"), ref_group="acc"),
        Col("b", "FedProx", "pm", ("m", "s"), 4, ref=("rm", "rs"), ref_group="acc"),
    ]
    return Table("t", df, cols, ["note"], trial_set="{0,1}")


def test_latex_cells_are_plain_tabular_with_the_required_formats():
    tex = to_latex(_table())
    assert "$0.6717 \\pm 0.0044$" in tex and "$0.000$" in tex and "-0.000" not in tex
    assert "\\multicolumn{2}{c}{$0.7082 \\pm 0.0047$}" in tex
    assert "\\begin{tabular}[b]{@{}r@{}}label- \\\\ HD\\end{tabular}" in tex
    for package_command in ("toprule", "midrule", "bottomrule", "usepackage", "multirow"):
        assert package_command not in tex
    assert fmt_pm(-0.00004, 0.00001, 4, True) == "$0.0000 \\pm 0.0000$"


def test_a_pending_cell_reads_pending_in_every_format():
    t = _table(pending="b")
    assert "pending" in to_latex(t) and "pending" in to_markdown(t)
    assert to_latex(t).count("pending") == 1


def test_write_table_writes_four_files_with_the_trial_set(tmp_path):
    paths = write_table(_table(), tmp_path)
    assert sorted(p.name for p in paths) == ["t.csv", "t.md", "t.tex", "t_caption_notes.txt"]
    assert pd.read_csv(tmp_path / "t.csv")["trial_set"].tolist() == ["{0,1}", "{0,1}"]
    assert estimate_width(_table()) > 0


# ---------------------------------------------------------------- pending switch, on CIFAR FedProx


def _bundle(pending):
    rows = []
    for algorithm in ("fedavg", "fedprox"):
        for trial in (0, 1, 2):
            for arm, base in (
                ("natural uncapped", 0.66),
                ("natural capped", 0.64),
                ("pure", 0.665),
            ):
                for L in (0.0, 0.45):
                    merged = arm == "natural capped" and L == 0.0
                    rows.append(
                        {
                            "dataset": "cifar10",
                            "grid": "rq2",
                            "arm": arm,
                            "cell_arm": (
                                res.NATURAL if (L == 0.0 and arm == "natural uncapped") else arm
                            ),
                            "merged_out": merged,
                            "L": L,
                            "Q": np.nan,
                            "algorithm": algorithm,
                            "trial": trial,
                            "final_acc": base
                            - L * 0.05
                            + 0.001 * trial
                            + (0.002 if algorithm == "fedprox" else 0),
                            "rta": 5.0 + trial,
                            "realized_label_hd": L + 0.01,
                            "realized_quantity_hd": 0.0 if arm == "pure" else 0.1,
                            "size_weighted_label_hd": L,
                        }
                    )
    sources = Sources(cifar_trials=(0, 1, 2), covtype_trials=())
    b = res.Bundle(
        sources, pd.DataFrame(rows), pd.DataFrame(), [], set(pending), {}, pd.DataFrame(), None
    )
    return b


def test_the_pending_switch_blanks_one_algorithm_and_the_same_code_fills_it():
    off = res.rq2_gaps(
        _bundle({("cifar10", "fedprox")}),
    ).query("dataset == 'cifar10'")
    on = res.rq2_gaps(_bundle(set())).query("dataset == 'cifar10'")
    prox_off = off[off["algorithm"] == "fedprox"]
    assert (prox_off["pending"] != "").all() and prox_off.get(
        "gap_mean", pd.Series(dtype=float)
    ).isna().all()
    prox_on = on[on["algorithm"] == "fedprox"]
    assert (prox_on["pending"] == "").all() and prox_on["gap_mean"].notna().all()
    avg_off, avg_on = off[off["algorithm"] == "fedavg"], on[on["algorithm"] == "fedavg"]
    assert avg_off["gap_mean"].tolist() == avg_on["gap_mean"].tolist()
    assert list(off["comparison"][off["level"] == 0.0].unique()) == ["pure − natural"]


def test_rq2_gaps_table_renders_pending_rows():
    b = _bundle({("cifar10", "fedprox")})
    tex = to_latex(res.rq2_gaps_table(b, res.rq2_gaps(b)))
    assert (
        tex.count("pending") == 2 * 4
    )  # 4 FedProx rows (pure - natural at 0, three at 0.45), gap + signs


# ---------------------------------------------------------------- numbers


def test_numbers_are_letters_only_and_pending_ones_print_a_visible_placeholder():
    n = po.Numbers()
    n.add("rqTwoCifarGapPureCappedFortyFiveFedAvg", -0.01234, 4, "t")
    n.add("rqTwoCovtypeGapPureCappedFiftyFedProx", None, 4, "t", "pending")
    tex = n.tex("header")
    assert "\\newcommand{\\rqTwoCifarGapPureCappedFortyFiveFedAvg}{\\ensuremath{-0.0123}}" in tex
    assert "\\newcommand{\\rqTwoCovtypeGapPureCappedFiftyFedProx}{\\textbf{??}}" in tex
    with pytest.raises(ValueError, match="letters only"):
        n.add("rqTwoGap045", 0.1, 4)
    with pytest.raises(ValueError, match="twice"):
        n.add("rqTwoCifarGapPureCappedFortyFiveFedAvg", 0.1, 4)


def test_trial_set_comparison_flags_sign_changes_of_gaps_and_interactions_only():
    five = pd.DataFrame(
        [
            {"macro": "rqThreeIntSeventySeventyFedAvg", "value": -0.02, "source": "s"},
            {"macro": "rqThreeIntSeventySeventyFedAvgSd", "value": 0.03, "source": "s"},
            {"macro": "rqTwoCifarGapPureUncappedFortyFiveFedProx", "value": -0.0003, "source": "s"},
            {"macro": "rqTwoCifarAccPureZeroFedAvg", "value": 0.67, "source": "s"},
        ]
    )
    three = five.assign(value=[-0.04, 0.02, 0.0024, 0.672])
    flagged = po.compare_trial_sets(five, three)
    assert flagged.loc[flagged["sign_change"], "macro"].tolist() == [
        "rqTwoCifarGapPureUncappedFortyFiveFedProx"
    ]


# ---------------------------------------------------------------- the real results


@pytest.mark.slow
def test_paper_outputs_on_the_recorded_results_pass_the_reproduction_check(tmp_path):
    if not Path("results/runs.jsonl").exists():
        pytest.skip("no recorded results in this checkout")
    status = po.main(["--out-dir", str(tmp_path)])
    assert status == 0
    text = (tmp_path / "paper" / "reproduction_check.txt").read_text(encoding="utf-8")
    assert "FAIL" not in text
    for path in (tmp_path / "paper").rglob("*"):
        if path.suffix in (".csv", ".tex", ".md", ".txt"):
            assert "niid" not in path.read_text(encoding="utf-8")
