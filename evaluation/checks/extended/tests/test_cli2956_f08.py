"""History observes old contracts; history shape belongs to functional tests."""

import inspect

import click
from click.core import ParameterSource
from click.testing import CliRunner

from cli2956_source_contracts import (
    test_cli_5c1239b0116b_n043_snapshot_witness_call_false,
    test_cli_5c1239b0116b_n043_snapshot_witness_falsy_cli_wins,
    test_cli_5c1239b0116b_n043_snapshot_witness_lazy_fallbacks,
    test_cli_5c1239b0116b_n044_snapshot_witness_actual_source,
    test_cli_5c1239b0116b_n044_snapshot_witness_prompt_final_source,
    test_cli_5c1239b0116b_n046_snapshot_witness_fresh_invocations,
    test_cli_5c1239b0116b_n046_snapshot_witness_parent_child_and_exceptions,
)


def command_kwargs():
    return (
        {"track_changes": True}
        if "track_changes" in inspect.signature(click.Command).parameters
        else {}
    )


def test_cli_5c1239b0116b_n043_observation_preserves_selected_sources(monkeypatch):
    calls = []
    option = click.Option(
        ["--value"], envvar="N4C_HISTORY_VALUE", type=int,
        default=lambda: calls.append("default") or 1,
    )
    command = click.Command("test", params=[option], **command_kwargs())
    monkeypatch.setenv("N4C_HISTORY_VALUE", "2")
    with command.make_context(
        "test", ["--value", "1"],
        default_map={"value": lambda: calls.append("map") or 3},
    ) as ctx:
        assert ctx.params["value"] == 1
        assert ctx.get_parameter_source("value") is ParameterSource.COMMANDLINE
        assert calls == []
    with command.make_context(
        "test", [], default_map={"value": lambda: calls.append("map") or 3}
    ) as ctx:
        assert ctx.params["value"] == 2
        assert ctx.get_parameter_source("value") is ParameterSource.ENVIRONMENT
        assert calls == []
    monkeypatch.setenv("N4C_HISTORY_VALUE", "")
    with command.make_context(
        "test", [], default_map={"value": lambda: calls.append("map") or 3}
    ) as ctx:
        assert ctx.params["value"] == 3
        assert ctx.get_parameter_source("value") is ParameterSource.DEFAULT_MAP
        assert calls == ["map"]
    with command.make_context("test", []) as ctx:
        assert ctx.params["value"] == 1
        assert ctx.get_parameter_source("value") is ParameterSource.DEFAULT
        assert calls == ["map", "default"]


def test_cli_5c1239b0116b_n043_observation_preserves_call_false():
    calls = []
    default = lambda: calls.append("default") or 1
    mapped = lambda: calls.append("map") or 2
    option = click.Option(["--value"], default=default)
    ctx = click.Context(
        click.Command("test", params=[option], **command_kwargs()),
        default_map={"value": mapped},
    )
    assert ctx.lookup_default("value", call=False) is mapped
    assert option.get_default(ctx, call=False) is mapped
    ctx.default_map = None
    assert option.get_default(ctx, call=False) is default
    assert calls == []


def test_cli_5c1239b0116b_n044_observation_preserves_callback_source():
    seen = []
    option = click.Option(
        ["--value"], type=int, default=1,
        callback=lambda ctx, param, value: seen.append(
            ctx.get_parameter_source(param.name)
        ) or value,
    )
    command = click.Command("test", params=[option], **command_kwargs())
    with command.make_context("test", ["--value", "1"]) as ctx:
        assert seen == [ParameterSource.COMMANDLINE]
        assert ctx.get_parameter_source("missing") is None
        ctx.params["value"] = 2
        assert ctx.get_parameter_source("value") is ParameterSource.COMMANDLINE
    seen.clear()
    prompt = click.Option(["--value"], prompt=True, type=int)
    command = click.Command(
        "test", params=[prompt],
        callback=lambda value: click.get_current_context().get_parameter_source("value"),
        **command_kwargs(),
    )
    result = CliRunner().invoke(command, input="2\n", standalone_mode=False)
    assert result.exit_code == 0, result.exception
    assert result.return_value is ParameterSource.PROMPT


def test_cli_5c1239b0116b_n046_observation_preserves_context_registration():
    parent = click.Context(click.Command("parent", **command_kwargs()), obj=object())
    child = click.Context(click.Command("child", **command_kwargs()), parent=parent)
    parent.params["value"] = 1
    parent.set_parameter_source("value", ParameterSource.COMMANDLINE)
    child.params["value"] = 2
    child.set_parameter_source("value", ParameterSource.ENVIRONMENT)
    child.set_parameter_source("child_only", ParameterSource.DEFAULT)
    parent.set_parameter_source("parent_only", ParameterSource.DEFAULT)
    assert parent.params["value"] == 1
    assert child.params["value"] == 2
    assert parent.get_parameter_source("value") is ParameterSource.COMMANDLINE
    assert child.get_parameter_source("value") is ParameterSource.ENVIRONMENT
    assert parent.get_parameter_source("child_only") is None
    assert child.get_parameter_source("parent_only") is None
    # obj/meta inheritance is outside this params/source scoring obligation.
    option = click.Option(["--value"], default="default")
    command = click.Command("test", params=[option], **command_kwargs())
    with command.make_context("test", ["--value", "first"]) as first:
        first.params["extra"] = 1
        first.set_parameter_source("first_only", ParameterSource.COMMANDLINE)
        with command.make_context("test", []) as second:
            assert "extra" not in second.params
            assert second.get_parameter_source("first_only") is None
            second.params["value"] = "second-owned"
            assert first.params["value"] == "first"
            second.set_parameter_source("second_only", ParameterSource.DEFAULT)
            assert first.get_parameter_source("second_only") is None
