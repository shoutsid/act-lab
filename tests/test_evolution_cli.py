import sys

from act.evolution import cli


def test_experiment_cli_prefers_gpu_when_available(monkeypatch, tmp_path) -> None:
    captured: dict[str, str] = {}

    def run(config):
        captured["device"] = config.device
        return object()

    monkeypatch.setattr(cli, "run_experiment", run)
    monkeypatch.setattr(cli, "append_result", lambda _path, _result: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["act.evolution.cli", "--steps", "1", "--output", str(tmp_path / "result.jsonl")],
    )

    cli.main()

    assert captured["device"] == "auto"
