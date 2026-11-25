"""Tests para evaluation/run_hf.py sin requerir torch ni GPU."""

import json

import pytest

from evaluation import run_baseline as rb
from evaluation import run_hf as rh
from tests.test_run_baseline import FakeEnv


def test_resolve_adapter_finds_direct_or_latest_final(tmp_path):
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert rh.resolve_adapter(str(direct)) == direct

    parent = tmp_path / "parent"
    for name in ("final_20261120_100000", "final_20261123_200000"):
        d = parent / name
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert rh.resolve_adapter(str(parent)) == parent / "final_20261123_200000"

    with pytest.raises(FileNotFoundError):
        rh.resolve_adapter(str(tmp_path / "missing"))


def test_run_hf_main_end_to_end_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(rh, "SQLEnvironment", FakeEnv)
    monkeypatch.setattr(rh, "hf_policy_factory", lambda *a, **kw: rb.oracle_policy_factory)

    out = tmp_path / "sft_lora.json"
    common = [
        "--name", "sft_lora",
        "--split", "test",
        "--k", "2",
        "--per-level", "2",
        "--traces-dir", str(tmp_path),
        "--out", str(out),
    ]
    rh.main(common)

    report = json.loads(out.read_text(encoding="utf-8"))
    summary = report["models"]["sft_lora"]
    assert report["config"]["n_tasks"] == 8
    assert summary["overall"]["pass@1"] == 1.0
    assert (tmp_path / "sft_lora_summary.csv").exists()
    assert len((tmp_path / "traces_sft_lora.jsonl").read_text(encoding="utf-8").splitlines()) == 16

    rh.main(common[:5] + ["3"] + common[6:] + ["--resume"])
    assert len((tmp_path / "traces_sft_lora.jsonl").read_text(encoding="utf-8").splitlines()) == 24
