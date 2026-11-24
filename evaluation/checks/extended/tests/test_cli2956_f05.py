"""Source obligations, not the new cache definition; independently verified on the frozen source."""

import inspect
import io

import click
from click.testing import CliRunner

from cli2956_source_contracts import (
    test_cli_5c1239b0116b_n041_snapshot_witness_expose_false,
    test_cli_5c1239b0116b_n041_snapshot_witness_pipeline,
    test_cli_5c1239b0116b_n041_snapshot_witness_required_before_callback,
    test_cli_5c1239b0116b_n042_snapshot_witness_composite_multiple,
    test_cli_5c1239b0116b_n042_snapshot_witness_envvar_split,
    test_cli_5c1239b0116b_n042_snapshot_witness_multiple_order_and_missing,
    test_cli_5c1239b0116b_n045_snapshot_witness_prompt_retry,
    test_cli_5c1239b0116b_n045_snapshot_witness_resilient_no_prompt,
    test_cli_5c1239b0116b_n047_snapshot_witness_external_file_not_owned,
    test_cli_5c1239b0116b_n047_snapshot_witness_no_context_caller_owns,
    test_cli_5c1239b0116b_n047_snapshot_witness_owned_file_lifetime,
)


def option_kwargs():
    return {"cache": True} if "cache" in inspect.signature(click.Option).parameters else {}


def test_cli_5c1239b0116b_n041_cache_preserves_callback_each_parse():
    seen = []
    option = click.Option(
        ["--value"], type=int, expose_value=False,
        callback=lambda ctx, param, value: seen.append((ctx, param, value)) or value,
        **option_kwargs(),
    )
    command = click.Command("test", params=[option])
    for _ in range(2):
        with command.make_context("test", ["--value", "2"]) as ctx:
            assert "value" not in ctx.params
            assert seen[-1] == (ctx, option, 2)
    assert len(seen) == 2


def test_cli_5c1239b0116b_n042_cache_preserves_multiple_conversion():
    option = click.Option(["--value"], multiple=True, type=int, **option_kwargs())
    command = click.Command("test", params=[option])
    for _ in range(2):
        with command.make_context("test", ["--value", "3", "--value", "0"]) as ctx:
            assert ctx.params["value"] == (3, 0)
    with command.make_context("test", []) as ctx:
        assert ctx.params["value"] == ()


def test_cli_5c1239b0116b_n045_cache_preserves_prompt_callback_retry():
    seen = []

    def callback(ctx, param, value):
        seen.append(value)
        if value < 0:
            raise click.BadParameter("nonnegative")
        return value

    option = click.Option(
        ["--value"], type=int, prompt=True, callback=callback, **option_kwargs()
    )
    command = click.Command("test", params=[option], callback=lambda value: click.echo(value))
    for _ in range(2):
        result = CliRunner().invoke(command, input="-2\n3\n")
        assert result.exit_code == 0, result.exception
        assert result.output.endswith("3\n")
    assert seen.count(-2) == 2 and seen.count(3) >= 2


def test_cli_5c1239b0116b_n047_cache_preserves_owned_file_lifetime(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("content", encoding="utf-8")
    option = click.Option(["--file"], type=click.File("r"), **option_kwargs())
    command = click.Command("test", params=[option])
    previous = None
    for _ in range(2):
        with command.make_context("test", ["--file", str(path)]) as ctx:
            value = ctx.params["file"]
            assert value is not previous and not value.closed
            assert value.read() == "content"
        assert value.closed
        previous = value


def test_cli_5c1239b0116b_n047_cache_preserves_external_ownership():
    external = io.StringIO("content")
    option = click.Option(["--file"], type=click.File("r"), **option_kwargs())
    with click.Context(click.Command("test", params=[option])) as ctx:
        assert option.type_cast_value(ctx, external) is external
    assert not external.closed
    external.close()


def test_cli_5c1239b0116b_n047_snapshot_witness_standard_stream_flush(monkeypatch):
    class Stream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flushes = 0

        def flush(self):
            self.flushes += 1
            return super().flush()

    stream = Stream()
    monkeypatch.setattr(click.types, "open_stream", lambda *args, **kwargs: (stream, False))
    with click.Context(click.Command("test")) as ctx:
        assert click.File("w", lazy=False).convert("-", None, ctx) is stream
    assert stream.flushes == 1 and not stream.closed
    stream.close()
