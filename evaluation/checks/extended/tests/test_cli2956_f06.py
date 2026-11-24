"""Dynamic defaults must preserve old selection, source and context duties."""

import click
import pytest
from click.core import ParameterSource
from click.testing import CliRunner

from cli2956_source_contracts import (
    test_cli_5c1239b0116b_n041_snapshot_witness_expose_false,
    test_cli_5c1239b0116b_n041_snapshot_witness_pipeline,
    test_cli_5c1239b0116b_n041_snapshot_witness_required_before_callback,
    test_cli_5c1239b0116b_n043_snapshot_witness_call_false,
    test_cli_5c1239b0116b_n043_snapshot_witness_falsy_cli_wins,
    test_cli_5c1239b0116b_n043_snapshot_witness_lazy_fallbacks,
    test_cli_5c1239b0116b_n044_snapshot_witness_actual_source,
    test_cli_5c1239b0116b_n044_snapshot_witness_prompt_final_source,
    test_cli_5c1239b0116b_n046_snapshot_witness_fresh_invocations,
    test_cli_5c1239b0116b_n046_snapshot_witness_parent_child_and_exceptions,
)


def dynamic_supported():
    # Public behavior probe, not a required private implementation helper.
    @click.command()
    @click.option("--target", default={"--seed": {"base": "resolved"}})
    @click.option("--seed", default="base")
    def probe(target, seed):
        return target

    result = CliRunner().invoke(probe, standalone_mode=False)
    return result.exit_code == 0 and result.return_value == "resolved"


def selected_dependency(source):
    dynamic = dynamic_supported()

    @click.command()
    @click.option(
        "--target",
        default={"--dep": {"explicit": "wanted", "derived": "wrong"}}
        if dynamic else "wanted",
    )
    @click.option(
        "--dep", envvar="N4C_F06_DEP",
        default={"--seed": {"base": "derived"}} if dynamic else "derived",
    )
    @click.option("--seed", default="base")
    @click.pass_context
    def command(ctx, target, dep, seed):
        return dep, ctx.get_parameter_source("dep")

    arguments = ["--dep", "explicit"] if source is ParameterSource.COMMANDLINE else []
    environment = {"N4C_F06_DEP": "explicit" if source is ParameterSource.ENVIRONMENT else None}
    defaults = {"dep": "explicit"} if source is ParameterSource.DEFAULT_MAP else None
    result = CliRunner().invoke(
        command, arguments, env=environment, default_map=defaults, standalone_mode=False
    )
    assert result.exit_code == 0, result.exception
    # Only dep and its existing source are normative; target match is functional.
    return result.return_value


@pytest.mark.parametrize("source", [ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT, ParameterSource.DEFAULT_MAP])
def test_cli_5c1239b0116b_n043_recursive_observer_preserves_selected_dependency(source):
    value, _ = selected_dependency(source)
    assert value == "explicit"


@pytest.mark.parametrize("source", [ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT, ParameterSource.DEFAULT_MAP])
def test_cli_5c1239b0116b_n044_recursive_observer_preserves_dependency_source(source):
    _, actual = selected_dependency(source)
    assert actual is source


def test_cli_5c1239b0116b_n041_dynamic_final_value_uses_old_callback_pipeline():
    seen = []

    def callback(ctx, param, value):
        seen.append((ctx, param, value))
        return value + 1

    target = click.Option(
        ["--target"], type=int, callback=callback, expose_value=False,
        default={"--seed": {"base": "2"}} if dynamic_supported() else "2",
    )
    command = click.Command("test", params=[target, click.Option(["--seed"], default="base")])
    with command.make_context("test", []) as ctx:
        assert seen == [(ctx, target, 2)]
        assert "target" not in ctx.params
