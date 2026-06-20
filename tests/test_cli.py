"""End-to-end CLI tests via the Typer test runner."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from ghostscope.cli import app

runner = CliRunner()


def test_cli_end_to_end(tmp_path):
    data = tmp_path / "flows.csv"
    model_dir = tmp_path / "model"
    alerts = tmp_path / "alerts.json"

    result = runner.invoke(
        app,
        ["gen-data", "--out", str(data), "--n-normal", "800", "--seed", "1337"],
    )
    assert result.exit_code == 0, result.output
    assert data.exists()

    result = runner.invoke(
        app, ["train", str(data), "--model-dir", str(model_dir), "--model", "iforest"]
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "detect",
            str(data),
            "--model-dir",
            str(model_dir),
            "--json-out",
            str(alerts),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "alert(s)" in result.output
    payload = json.loads(alerts.read_text())
    assert payload["n_alerts"] > 0

    result = runner.invoke(
        app, ["evaluate", str(data), "--model-dir", str(model_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "precision" in result.output

    result = runner.invoke(app, ["info", "--model-dir", str(model_dir)])
    assert result.exit_code == 0, result.output
    assert "model_type" in result.output


def test_cli_schema_error(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n")
    model_dir = tmp_path / "m"
    result = runner.invoke(app, ["train", str(bad), "--model-dir", str(model_dir)])
    assert result.exit_code == 1
    assert "missing required feature column" in result.output


def test_cli_detect_missing_model(tmp_path):
    data = tmp_path / "flows.csv"
    runner.invoke(app, ["gen-data", "--out", str(data), "--n-normal", "50"])
    result = runner.invoke(
        app, ["detect", str(data), "--model-dir", str(tmp_path / "absent")]
    )
    assert result.exit_code == 1
    assert "No ghostscope model bundle" in result.output
