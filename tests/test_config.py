"""Tests for the config schema."""

import pytest
import yaml

from experiments.config import (
    PartitionMode,
    config_hash,
    dump_config,
    from_dict,
    load_config,
    to_dict,
)

CONFIG_DIR = "config"


@pytest.mark.parametrize("name", ["base.yaml", "smoke.yaml"])
def test_config_roundtrip(name, tmp_path):
    config = load_config(f"{CONFIG_DIR}/{name}")
    out = tmp_path / "dumped.yaml"
    dump_config(config, out)
    assert load_config(out) == config


def test_config_hash_stability(tmp_path):
    config = load_config(f"{CONFIG_DIR}/base.yaml")
    data = to_dict(config)

    shuffled = {k: dict(reversed(list(v.items()))) for k, v in reversed(list(data.items()))}
    assert list(shuffled) != list(data)
    assert config_hash(from_dict(shuffled)) == config_hash(config)

    out = tmp_path / "shuffled.yaml"
    out.write_text(yaml.safe_dump(shuffled, sort_keys=False), encoding="utf-8")
    assert config_hash(load_config(out)) == config_hash(config)


def test_hash_tracks_meaningful_changes():
    config = load_config(f"{CONFIG_DIR}/base.yaml")
    changed = to_dict(config)
    changed["federated"]["rounds"] += 1
    assert config_hash(from_dict(changed)) != config_hash(config)


def test_unknown_key_rejected():
    data = to_dict(load_config(f"{CONFIG_DIR}/base.yaml"))
    data["partition"]["alhpa"] = 0.5
    with pytest.raises(ValueError, match="unknown key"):
        from_dict(data)


def test_mode_parameter_requirements():
    data = to_dict(load_config(f"{CONFIG_DIR}/base.yaml"))
    data["partition"]["mode"] = PartitionMode.COMBINED.value
    data["partition"]["min_size_policy"] = 0  # combined never redraws
    with pytest.raises(ValueError, match="requires beta"):
        from_dict(data)

    data["partition"]["beta"] = 1.0
    assert from_dict(data).partition.mode is PartitionMode.COMBINED


def test_quantity_calibration_id_needs_a_beta_mode():
    data = to_dict(load_config("config/smoke_pure_label.yaml"))
    data["partition"]["quantity_calibration_id"] = "f09e0cfdf0"
    with pytest.raises(ValueError, match="quantity_calibration_id applies to"):
        from_dict(data)
    data = to_dict(load_config("config/smoke_combined.yaml"))
    data["partition"]["quantity_calibration_id"] = "f09e0cfdf0"
    assert from_dict(data).partition.quantity_calibration_id == "f09e0cfdf0"


def test_smoke_config_hash_is_stable():
    """runs.jsonl resumes by config_hash; fields added later must not move it."""
    config = load_config("config/smoke.yaml")
    assert config_hash(config) == "acdc3d91"
    capped = to_dict(config)
    capped["partition"]["size_cap"] = "niid_bench"
    assert config_hash(from_dict(capped)) != "acdc3d91"
