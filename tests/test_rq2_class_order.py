"""Tests for the class-order rule: the gate, the verdict, the power flag, concentration.

The 0.25 and 0.45 cases use the per-trial differences reported on 2026-10-29, as
(trials, classes) arrays against an uncapped arm of zeros.
"""

import numpy as np
import pytest

from analysis.rq2_class_order import (
    LEVELS,
    NO_EFFECT,
    cells,
    concentration,
    concentration_present,
    concentration_tables,
    gate,
    level_result,
    low_power,
    verdict,
)
from experiments.config import config_hash

ZERO = np.zeros((3, 10))

# capped index - uncapped, per class (rows) and trial (columns)
INDEX_045 = np.array(
    [
        [0.065, 0.088, 0.070],
        [0.073, 0.068, 0.076],
        [0.013, -0.001, 0.065],
        [0.054, 0.041, 0.094],
        [-0.039, 0.073, 0.003],
        [0.060, -0.027, -0.045],
        [0.069, -0.030, -0.011],
        [-0.076, -0.015, -0.135],
        [-0.102, -0.153, -0.166],
        [-0.416, -0.167, -0.324],
    ]
).T
INDEX_025 = np.array(
    [
        [0.040, 0.019, 0.035],
        [0.062, 0.025, 0.026],
        [-0.002, -0.001, -0.007],
        [0.003, 0.030, -0.013],
        [-0.007, -0.014, 0.037],
        [-0.009, 0.007, 0.006],
        [0.001, -0.025, 0.014],
        [0.015, 0.034, -0.001],
        [-0.069, -0.018, -0.059],
        [-0.194, -0.105, -0.195],
    ]
).T
# capped reversed - uncapped at 0.45
REVERSED_045 = np.array(
    [
        [-0.4180, -0.2130, -0.4130],
        [-0.0980, -0.1710, -0.1580],
        [-0.0510, -0.0400, -0.0540],
        [0.0200, -0.0490, -0.0020],
        [-0.0310, 0.1000, -0.0140],
        [0.0120, 0.0190, 0.0360],
        [0.0480, 0.0400, 0.0260],
        [-0.0050, 0.0480, 0.0030],
        [0.0530, 0.0680, 0.0460],
        [0.0560, 0.0860, 0.0900],
    ]
).T


@pytest.mark.parametrize("diff, D", [(INDEX_045, 0.265), (INDEX_025, 0.122)], ids=["045", "025"])
def test_the_gate_is_met_at_025_and_045_as_reported(diff, D):
    result = gate(diff, ZERO)
    assert result["D"] == pytest.approx(D, abs=5e-4)
    assert result["met_a"] and result["met_b"] and result["met"]
    assert (result["lower_ship"], result["lower_truck"]) == (3, 3)


def test_the_045_verdict_is_position_as_reported():
    result = verdict(REVERSED_045, ZERO)
    assert result["D"] == pytest.approx(0.3220, abs=5e-5)
    assert result["share"] == pytest.approx(1.523, abs=5e-4)
    assert (result["lower_airplane"], result["lower_automobile"]) == (3, 3)
    assert (result["lower_ship"], result["lower_truck"]) == (0, 0)
    assert result["verdict"] == "Position"
    assert level_result(gate(INDEX_045, ZERO), result) == "Position"


def test_identity_when_ship_and_truck_stay_lower():
    # the index-order loss unchanged: the deficit stays on the same classes
    assert verdict(INDEX_045, ZERO)["verdict"] == "Identity"


def test_inconclusive_when_neither_pattern_holds():
    diff = REVERSED_045.copy()
    diff[:2, 8] = -0.01  # ship lower in 2 of 3: (c) fails
    diff[:, 9] = 0.01  # truck never lower: no Identity
    result = verdict(diff, ZERO)
    assert result["met_a"] and result["met_b"] and not result["met_c"]
    assert not result["identity"]
    assert result["verdict"] == "Inconclusive"


def test_a_tie_counts_as_not_lower():
    diff = REVERSED_045.copy()
    diff[:, 8] = 0.0
    assert verdict(diff, ZERO)["lower_ship"] == 0 and verdict(diff, ZERO)["met_c"]


def test_no_verdict_when_the_gate_fails():
    no_deficit = INDEX_045.copy()
    no_deficit[:, 8:] = 0.05  # ship and truck above uncapped
    failed = gate(no_deficit, ZERO)
    assert not failed["met"]
    assert level_result(failed, verdict(REVERSED_045, ZERO)) == NO_EFFECT


def test_the_gate_needs_both_conditions():
    diff = INDEX_045.copy()
    diff[0, 8] = diff[1, 8] = 0.01  # ship lower in 1 of 3 only
    result = gate(diff, ZERO)
    assert result["met_a"] and not result["met_b"] and not result["met"]


def test_the_rule_is_written_for_three_trials():
    with pytest.raises(ValueError, match="3 trials"):
        verdict(np.zeros((2, 10)), np.zeros((2, 10)))


def test_low_power_when_the_overall_difference_changes_sign():
    assert low_power([-0.0505, 0.0558, -0.0233])  # 0.70, recorded runs
    assert not low_power([-0.030, -0.012, -0.037])  # 0.45
    assert not low_power([-0.016, 0.0, -0.016])


def test_concentration_counts_holders_and_effective_holders():
    counts = np.array([[5, 0], [5, 0], [0, 10]])
    holders, effective = concentration(counts)
    assert holders.tolist() == [2, 1]
    assert effective.tolist() == [2.0, 1.0]
    assert concentration(np.ones((30, 10)))[1] == pytest.approx(np.full(10, 30.0))


def test_concentration_present_needs_all_four_classes():
    uncapped = np.full(10, 20.0)
    index = uncapped.copy()
    index[[8, 9]] = 10.0
    reversed_ = index[::-1].copy()
    holders = {"capped index": index, "capped reversed": reversed_, "uncapped": uncapped}
    assert concentration_present(holders)["present"]
    reversed_[1] = 20.0  # automobile held by as many clients as uncapped
    assert not concentration_present(holders)["present"]


def test_the_concentration_check_is_on_holders_not_effective_holders():
    # uncapped: every client holds every class, but client 0 holds nearly all of it
    uncapped = np.ones((30, 10))
    uncapped[0] = 1000
    # capped index: ship and truck spread evenly over 20 clients, so fewer holders
    # but far more effective holders than uncapped
    index = uncapped.copy()
    index[:, 8:] = 0
    index[:20, 8:] = 50
    counts = {}
    for t in (0, 1, 2):
        counts[(0.45, "uncapped", t)] = ("pid", uncapped)
        counts[(0.45, "capped index", t)] = ("pid", index)
        counts[(0.45, "capped reversed", t)] = ("pid", index[:, ::-1])
    _, check = concentration_tables(0.45, counts)
    assert check["present"]
    assert (check["effective"]["capped index"][8:] > check["effective"]["uncapped"][8:]).all()


def test_cells_find_every_arm_at_every_level():
    found = cells()
    arms = ("pure", "capped index", "capped reversed", "uncapped")
    assert set(found) == {(level, arm, t) for level in LEVELS for arm in arms for t in (0, 1, 2)}
    for level in LEVELS:
        for t in (0, 1, 2):
            reversed_, index = (
                found[(level, "capped reversed", t)],
                found[(level, "capped index", t)],
            )
            assert reversed_.partition.class_order == "reversed"
            assert reversed_.partition.alpha == index.partition.alpha
            assert index.partition.target_label_hd == level
    assert [config_hash(found[(0.45, "capped reversed", t)]) for t in (0, 1, 2)] == [
        "e9cb46cd",
        "3305baed",
        "52f05aeb",
    ]
