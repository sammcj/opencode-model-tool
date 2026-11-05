"""Tests for opencode-model-tool.py.

Run with:
    uv run --with pytest --with json5 --with httpx --with textual pytest
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("opencode-model-tool.py")
spec = importlib.util.spec_from_file_location("opencode_model_tool", SCRIPT)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)

ENDPOINT = "http://127.0.0.1:1/v1"

PAID_PRICING = {"input": "0.000001", "output": "0.000002"}
FREE_PRICING = {"input": "0", "output": "0"}

RAW_MODELS = [
    {"id": "paid/configured", "config": {"pricing": PAID_PRICING}},
    {"id": "paid/other", "config": {"pricing": PAID_PRICING}},
    {"id": "free/one", "config": {"pricing": FREE_PRICING}},
]


@pytest.mark.parametrize(
    "model",
    [
        {"config": {"pricing": {"input": "0", "output": "0"}}},
        {"config": {"pricing": {"input": 0, "output": 0.0}}},
        {"config": {"pricing": {"prompt": "0", "completion": "0"}}},
        {"config": {"tags": ["tool-use", " Free "]}},
        {"pricing": {"prompt": "0", "completion": "0"}},
        {"pricing": {"input": "0", "output": "0"}},
        {"tags": ["free"]},
    ],
)
def test_is_free_model_accepts_free_schemas(model):
    assert tool.is_free_model(model)


@pytest.mark.parametrize(
    "model",
    [
        {},
        {"config": "not-a-dict", "pricing": "not-a-dict", "tags": "free"},
        {"config": {"pricing": PAID_PRICING, "tags": ["tool-use"]}},
        {"config": {"pricing": {"input": "0", "output": "0.1"}}},
        {"config": {"pricing": {"input": "0"}}},
        {"pricing": {"prompt": "0", "completion": "0.1"}},
        {"pricing": {"prompt": "-1", "completion": "-1"}},
        {"pricing": {"prompt": None, "completion": "n/a"}},
        {"pricing": {"input": "0", "completion": "0"}},
    ],
)
def test_is_free_model_rejects_paid_or_unknown(model):
    assert not tool.is_free_model(model)


def test_free_model_ids_ignores_entries_without_id():
    raw = [
        {"pricing": {"prompt": "0", "completion": "0"}},
        {"id": "a", "tags": ["free"]},
    ]
    assert tool.free_model_ids(raw) == {"a"}


def test_extract_limits_from_nested_config():
    model = {
        "config": {
            "context_window": 40960,
            "capabilities": {"max_context_window": 40960},
            "max_tokens": 16384,
        }
    }
    assert tool.extract_context_from_model_data(model) == 40960
    assert tool.extract_output_from_model_data(model) == 16384


def test_extract_context_from_capabilities_only():
    model = {"config": {"capabilities": {"max_context_window": 8192}}}
    assert tool.extract_context_from_model_data(model) == 8192


def test_extract_limits_prefers_top_level():
    model = {
        "context_length": 1000,
        "max_output_tokens": 100,
        "config": {"context_window": 2000, "max_tokens": 200},
    }
    assert tool.extract_context_from_model_data(model) == 1000
    assert tool.extract_output_from_model_data(model) == 100


def test_categorise_filtered_models_are_not_reported_gone():
    new, in_config, available, gone_configured, gone_other = tool.categorise_models(
        ["free/one", "paid/configured"],
        {"known": ["paid/other", "removed/model"]},
        {"paid/configured"},
        endpoint_ids=["free/one", "paid/configured", "paid/other"],
    )
    assert new == ["free/one"]
    assert in_config == ["paid/configured"]
    assert available == []
    assert gone_configured == []
    assert gone_other == ["removed/model"]


@pytest.fixture
def run_tool(monkeypatch, tmp_path):
    """Run main() against canned endpoint data with an isolated home and config."""
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": {
                    "fake": {
                        "options": {"baseURL": ENDPOINT},
                        "models": {
                            "paid/configured": {
                                "name": "paid/configured",
                                "limit": {"context": 1000, "output": 100},
                            }
                        },
                    }
                }
            },
            indent=2,
        )
    )
    monkeypatch.setattr(tool.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(tool, "fetch_models", lambda *a, **k: RAW_MODELS)
    monkeypatch.setattr(tool, "fetch_props_limits", lambda *a, **k: (None, None, ""))
    monkeypatch.setattr(tool, "fetch_model_info_limits", lambda *a, **k: ({}, {}))

    def run(*extra: str) -> dict:
        argv = ["tool", "--endpoint", ENDPOINT, "--config", str(config_path)]
        monkeypatch.setattr(
            sys, "argv", argv + ["--provider-id", "fake", "--api-key", "x", *extra]
        )
        tool.main()
        return json.loads(config_path.read_text())["provider"]["fake"]["models"]

    return run


def test_only_free_keeps_configured_paid_models(run_tool):
    models = run_tool("--only-free", "--all", "--yes")
    assert set(models) == {"free/one", "paid/configured"}


def test_only_free_does_not_add_unconfigured_paid_models(run_tool):
    assert "paid/other" not in run_tool("--only-free", "--all", "--yes")


def test_only_free_keeps_previously_known_models_in_state(run_tool):
    tool.save_state({ENDPOINT: {"known": ["paid/other", "removed/model"]}})
    run_tool("--only-free", "--all", "--yes")
    state = json.loads(tool.state_path().read_text())
    assert state[ENDPOINT]["known"] == ["free/one", "paid/configured", "paid/other"]


def test_only_free_does_not_mark_unseen_models_as_known(run_tool):
    run_tool("--only-free", "--all", "--yes")
    state = json.loads(tool.state_path().read_text())
    assert state[ENDPOINT]["known"] == ["free/one", "paid/configured"]


def test_only_free_interactive_offers_configured_paid_models(run_tool, monkeypatch):
    captured = {}

    class FakeApp:
        def __init__(
            self, selections, _config_ids, gone_configured, gone_other, *_, **__
        ):
            captured["offered"] = {s.value for s in selections}
            captured["checked"] = {s.value for s in selections if s.initial_state}
            captured["gone"] = gone_configured + gone_other

        def run(self):
            return sorted(captured["checked"])

    monkeypatch.setattr(tool, "ModelSelectorApp", FakeApp)
    tool.save_state({ENDPOINT: {"known": ["paid/other"]}})

    models = run_tool("--only-free", "--yes")

    assert captured["offered"] == {"free/one", "paid/configured"}
    assert captured["checked"] == {"paid/configured"}
    assert captured["gone"] == []
    assert set(models) == {"paid/configured"}


def test_only_free_list_shows_free_models_only(run_tool, capsys):
    run_tool("--only-free", "--list")
    out = capsys.readouterr().out
    assert "free/one" in out
    assert "paid/configured" not in out
    assert "paid/other" not in out


def test_without_only_free_all_models_are_selected(run_tool):
    models = run_tool("--all", "--yes")
    assert set(models) == {"free/one", "paid/configured", "paid/other"}
