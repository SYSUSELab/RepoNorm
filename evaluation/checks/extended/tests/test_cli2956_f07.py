"""Old source/context obligations; new conflict semantics stay functional."""

import inspect

import click
import pytest
from click.core import ParameterSource
from click.testing import CliRunner

from cli2956_source_contracts import (
    test_cli_5c1239b0116b_n044_snapshot_witness_actual_source,
    test_cli_5c1239b0116b_n044_snapshot_witness_prompt_final_source,
    test_cli_5c1239b0116b_n046_snapshot_witness_fresh_invocations,
    test_cli_5c1239b0116b_n046_snapshot_witness_parent_child_and_exceptions,
)


def _conflict_kwargs():
    if "conflicts_with" in inspect.signature(click.Option).parameters:
        return {"conflicts_with": ["right"]}
    return {}


@pytest.mark.parametrize(
    ("env", "defaults", "expected"),
    [({}, {}, ParameterSource.DEFAULT),
     ({"N4C_RIGHT": "1"}, {}, ParameterSource.ENVIRONMENT),
     ({}, {"right": 1}, ParameterSource.DEFAULT_MAP)],
)
def test_cli_5c1239b0116b_n044_scan_keeps_source(env, defaults, expected):
    @click.command()
    @click.option("--left", type=int, default=1, **_conflict_kwargs())
    @click.option("--right", type=int, default=1, envvar="N4C_RIGHT")
    @click.pass_context
    def cli(ctx, left, right):
        assert ctx.params == {"left": 1, "right": 1}
        assert ctx.get_parameter_source("left") is ParameterSource.COMMANDLINE
        assert ctx.get_parameter_source("right") is expected
        assert ctx.get_parameter_source("missing") is None
    result = CliRunner().invoke(
        cli, ["--left", "1"], default_map=defaults,
        env={"N4C_RIGHT": None, **env},
    )
    assert result.exit_code == 0, result.exception


def test_cli_5c1239b0116b_n046_scan_keeps_context_ownership():
    parent = click.Context(click.Command("parent"), obj=object())
    parent.params["left"] = "parent-only"
    parent.set_parameter_source("right", ParameterSource.COMMANDLINE)
    command = click.Command("child", params=[
        click.Option(["--left"], default="default", **_conflict_kwargs()),
        click.Option(["--right"], default="default"),
    ])
    with command.make_context("child", ["--left", "first"], parent=parent) as first:
        assert first.params == {"left": "first", "right": "default"}
        assert first.get_parameter_source("right") is ParameterSource.DEFAULT
        first.params["private"] = "first-owned"
        first.set_parameter_source("private", ParameterSource.ENVIRONMENT)
        with command.make_context("child", [], parent=parent) as second:
            assert second.params == {"left": "default", "right": "default"}
            assert second.get_parameter_source("private") is None
            assert first.params["left"] == "first"
        assert parent.params == {"left": "parent-only"}
        assert parent.get_parameter_source("right") is ParameterSource.COMMANDLINE
        assert parent.get_parameter_source("private") is None


__all__ = [name for name in globals() if name.startswith("test_")]
