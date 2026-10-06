"""Old-snapshot witnesses reused only by applicable task wrappers.

No new feature API is exercised here. Explicit collection of this module
also verifies the frozen source independently of evaluator reference patches.
"""

import io

import click
import pytest
from click.core import ParameterSource
from click.testing import CliRunner


def test_cli_5c1239b0116b_n041_snapshot_witness_pipeline():
    events = []

    class RecordedType(click.ParamType):
        name = "recorded"

        def convert(self, value, param, ctx):
            events.append(("convert", value, param, ctx))
            return int(value)

    def callback(ctx, param, value):
        events.append(("callback", value, param, ctx))
        return value + 5

    option = click.Option(["--value"], type=RecordedType(), callback=callback)
    ctx = click.Context(click.Command("test", params=[option]))
    value, remaining = option.handle_parse_result(ctx, {"value": "7"}, [])
    assert value == 12 and remaining == [] and ctx.params["value"] == 12
    assert [event[0] for event in events] == ["convert", "callback"]
    assert events[1][1] == 7
    assert all(event[2] is option and event[3] is ctx for event in events)


def test_cli_5c1239b0116b_n041_snapshot_witness_required_before_callback():
    seen = []
    option = click.Option(
        ["--value"], required=True, callback=lambda ctx, param, value: seen.append(value)
    )
    ctx = click.Context(click.Command("test", params=[option]))
    with pytest.raises(click.MissingParameter):
        option.process_value(ctx, None)
    assert seen == []


def test_cli_5c1239b0116b_n041_snapshot_witness_expose_false():
    seen = []
    option = click.Option(
        ["--value"], type=int, expose_value=False,
        callback=lambda ctx, param, value: seen.append(value) or value + 1,
    )
    ctx = click.Context(click.Command("test", params=[option]))
    value, _ = option.handle_parse_result(ctx, {"value": "2"}, [])
    assert value == 3 and seen == [2] and "value" not in ctx.params


def test_cli_5c1239b0116b_n042_snapshot_witness_multiple_order_and_missing():
    option = click.Option(["--value"], multiple=True, type=int)
    ctx = click.Context(click.Command("test"))
    assert option.type_cast_value(ctx, ["3", "0", "-2"]) == (3, 0, -2)
    assert option.type_cast_value(ctx, None) == ()


def test_cli_5c1239b0116b_n042_snapshot_witness_composite_multiple():
    option = click.Option(["--pair"], multiple=True, type=(str, int))
    ctx = click.Context(click.Command("test"))
    assert option.type_cast_value(ctx, [("a", "2"), ("b", "0")]) == (
        ("a", 2), ("b", 0)
    )


def test_cli_5c1239b0116b_n042_snapshot_witness_envvar_split(monkeypatch):
    option = click.Option(["--value"], multiple=True, type=int, envvar="N4C_MULTIPLE")
    ctx = click.Context(click.Command("test"))
    monkeypatch.setenv("N4C_MULTIPLE", "3 0 -2")
    raw = option.value_from_envvar(ctx)
    assert option.type_cast_value(ctx, raw) == (3, 0, -2)


@pytest.mark.parametrize("raw", [0, False, ""])
def test_cli_5c1239b0116b_n043_snapshot_witness_falsy_cli_wins(raw, monkeypatch):
    calls = []
    option = click.Option(
        ["--value"], envvar="N4C_PRECEDENCE", default=lambda: calls.append("default")
    )
    ctx = click.Context(
        click.Command("test"), default_map={"value": lambda: calls.append("map")}
    )
    monkeypatch.setenv("N4C_PRECEDENCE", "env")
    value, source = option.consume_value(ctx, {"value": raw})
    assert value == raw and source is ParameterSource.COMMANDLINE and calls == []


def test_cli_5c1239b0116b_n043_snapshot_witness_lazy_fallbacks(monkeypatch):
    calls = []
    option = click.Option(
        ["--value"], envvar="N4C_PRECEDENCE",
        default=lambda: calls.append("default") or "ordinary",
    )
    ctx = click.Context(
        click.Command("test"),
        default_map={"value": lambda: calls.append("map") or "mapped"},
    )
    monkeypatch.setenv("N4C_PRECEDENCE", "env")
    assert option.consume_value(ctx, {}) == ("env", ParameterSource.ENVIRONMENT)
    assert calls == []
    monkeypatch.setenv("N4C_PRECEDENCE", "")
    assert option.consume_value(ctx, {}) == ("mapped", ParameterSource.DEFAULT_MAP)
    assert calls == ["map"]
    ctx.default_map = {"value": lambda: calls.append("none-map")}
    # A callable map entry remains the selected default object even if it
    # evaluates to None; get_default does not replace that object with the
    # parameter default. Do not impose a new fallback or call-count policy.
    assert option.consume_value(ctx, {}) == (None, ParameterSource.DEFAULT)
    assert "none-map" in calls and "default" not in calls
    ctx.default_map = {"value": None}
    assert option.consume_value(ctx, {}) == ("ordinary", ParameterSource.DEFAULT)
    assert calls[-1] == "default"


def test_cli_5c1239b0116b_n043_snapshot_witness_call_false():
    calls = []
    default = lambda: calls.append("default") or 4
    mapped = lambda: calls.append("map") or 5
    option = click.Option(["--value"], default=default)
    ctx = click.Context(click.Command("test"), default_map={"value": mapped})
    assert ctx.lookup_default("value", call=False) is mapped
    assert option.get_default(ctx, call=False) is mapped
    ctx.default_map = None
    assert option.get_default(ctx, call=False) is default
    assert calls == []


@pytest.mark.parametrize(
    "args,env,defaults,expected",
    [
        (["--value", "1"], {}, {}, ParameterSource.COMMANDLINE),
        ([], {"N4C_SOURCE": "1"}, {}, ParameterSource.ENVIRONMENT),
        ([], {}, {"value": 1}, ParameterSource.DEFAULT_MAP),
        ([], {}, {}, ParameterSource.DEFAULT),
    ],
)
def test_cli_5c1239b0116b_n044_snapshot_witness_actual_source(
    args, env, defaults, expected
):
    seen = []

    def callback(ctx, param, value):
        seen.append(ctx.get_parameter_source(param.name))
        return value

    @click.command()
    @click.option("--value", type=int, default=1, envvar="N4C_SOURCE", callback=callback)
    @click.pass_context
    def cli(ctx, value):
        assert value == 1
        assert ctx.get_parameter_source("missing") is None
        return ctx.get_parameter_source("value")

    result = CliRunner().invoke(
        cli, args, env={"N4C_SOURCE": None, **env},
        default_map=defaults, standalone_mode=False,
    )
    assert result.exit_code == 0, result.exception
    assert result.return_value is expected and seen == [expected]


def test_cli_5c1239b0116b_n045_snapshot_witness_prompt_retry():
    seen = []

    def validate(ctx, param, value):
        assert isinstance(value, int)
        seen.append(value)
        if value < 0:
            raise click.BadParameter("must be nonnegative")
        return value

    @click.command()
    @click.option("--value", type=int, prompt=True, callback=validate)
    def cli(value):
        click.echo(value)

    result = CliRunner().invoke(cli, input="-2\n3\n")
    assert result.exit_code == 0, result.exception
    assert "Error: must be nonnegative" in result.output
    assert seen[0] == -2 and 3 in seen and result.output.endswith("3\n")


def test_cli_5c1239b0116b_n045_snapshot_witness_resilient_no_prompt():
    option = click.Option(["--value"], type=int, prompt=True)
    command = click.Command("test", params=[option])
    ctx = command.make_context("test", [], resilient_parsing=True)
    assert ctx.params["value"] is None
    ctx.close()


def test_cli_5c1239b0116b_n044_snapshot_witness_prompt_final_source():
    @click.command()
    @click.option("--value", type=int, prompt=True)
    @click.pass_context
    def cli(ctx, value):
        return ctx.get_parameter_source("value")

    result = CliRunner().invoke(cli, input="2\n", standalone_mode=False)
    assert result.exit_code == 0, result.exception
    assert result.return_value is ParameterSource.PROMPT
    # Do not require registration during the earlier prompt value_proc.


def test_cli_5c1239b0116b_n046_snapshot_witness_parent_child_and_exceptions():
    parent = click.Context(
        click.Command("parent"), obj=object(), default_map={"child": {"value": 3}}
    )
    child = click.Context(click.Command("child"), parent=parent, info_name="child")
    parent.params["value"] = 1
    parent.set_parameter_source("value", ParameterSource.COMMANDLINE)
    parent_source = parent.get_parameter_source("value")
    child.params["value"] = 2
    child.set_parameter_source("value", ParameterSource.ENVIRONMENT)
    assert parent.params["value"] == 1 and child.params["value"] == 2
    assert parent.get_parameter_source("value") is parent_source
    parent.set_parameter_source("parent_only", ParameterSource.COMMANDLINE)
    child.set_parameter_source("child_only", ParameterSource.ENVIRONMENT)
    assert parent.get_parameter_source("child_only") is None
    assert child.get_parameter_source("parent_only") is None


def test_cli_context_inheritance_counterexample_source_only():
    # Diagnostic source witness, not a params/source-isolation scoring rule.
    parent = click.Context(
        click.Command("parent"), obj=object(), default_map={"child": {"value": 3}}
    )
    child = click.Context(click.Command("child"), parent=parent, info_name="child")
    assert child.obj is parent.obj and child.meta is parent.meta
    assert child.lookup_default("value") == 3


def test_cli_5c1239b0116b_n046_snapshot_witness_fresh_invocations():
    command = click.Command("test", params=[click.Option(["--value"], default="default")])
    with command.make_context("test", ["--value", "first"]) as first:
        first.params["extra"] = "private"
        first_value = first.params["value"]
        first_source = first.get_parameter_source("value")
        with command.make_context("test", []) as second:
            assert "extra" not in second.params
            second.params["value"] = "second-owned"
            assert first.params["value"] == first_value
            first.params["value"] = "first-owned"
            assert second.params["value"] == "second-owned"
            assert first.get_parameter_source("value") is first_source
            first.set_parameter_source("first_only", ParameterSource.COMMANDLINE)
            second.set_parameter_source("second_only", ParameterSource.DEFAULT)
            assert first.get_parameter_source("second_only") is None
            assert second.get_parameter_source("first_only") is None


def test_cli_5c1239b0116b_n047_snapshot_witness_owned_file_lifetime(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("content", encoding="utf-8")
    option = click.Option(["--file"], type=click.File("r"))
    command = click.Command("test", params=[option])
    with command.make_context("test", ["--file", str(path)]) as first:
        old_file = first.params["file"]
        assert old_file.read() == "content"
    assert old_file.closed
    with command.make_context("test", ["--file", str(path)]) as second:
        new_file = second.params["file"]
        assert new_file is not old_file and not new_file.closed
        assert new_file.read() == "content"
    assert new_file.closed


def test_cli_5c1239b0116b_n047_snapshot_witness_external_file_not_owned():
    external = io.StringIO("content")
    ctx = click.Context(click.Command("test"))
    assert click.File("r").convert(external, None, ctx) is external
    ctx.close()
    assert not external.closed and external.read() == "content"
    external.close()


def test_cli_5c1239b0116b_n047_snapshot_witness_no_context_caller_owns(tmp_path):
    path = tmp_path / "standalone.txt"
    path.write_text("content", encoding="utf-8")
    converted = click.File("r").convert(str(path), None, None)
    try:
        assert not converted.closed and converted.read() == "content"
    finally:
        converted.close()
