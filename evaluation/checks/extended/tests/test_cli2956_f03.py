"""Batching must preserve earlier source protocols, not its old final shape."""

import inspect

import click

from cli2956_source_contracts import (
    test_cli_5c1239b0116b_n041_snapshot_witness_expose_false,
    test_cli_5c1239b0116b_n041_snapshot_witness_pipeline,
    test_cli_5c1239b0116b_n041_snapshot_witness_required_before_callback,
    test_cli_5c1239b0116b_n042_snapshot_witness_composite_multiple,
    test_cli_5c1239b0116b_n042_snapshot_witness_envvar_split,
    test_cli_5c1239b0116b_n042_snapshot_witness_multiple_order_and_missing,
    test_cli_5c1239b0116b_n045_snapshot_witness_prompt_retry,
    test_cli_5c1239b0116b_n045_snapshot_witness_resilient_no_prompt,
)

__all__ = [name for name in globals() if name.startswith("test_")]


def test_cli_5c1239b0116b_n042_snapshot_witness_callback_before_new_shape():
    seen = []

    def callback(ctx, param, value):
        seen.append((ctx, param, value))
        return value

    kwargs = {"multiple": True, "type": int, "callback": callback}
    if "batch_size" in inspect.signature(click.Option.__init__).parameters:
        kwargs["batch_size"] = 2
    option = click.Option(["--value"], **kwargs)
    ctx = click.Context(click.Command("test", params=[option]))
    option.handle_parse_result(ctx, {"value": ["7", "2", "0"]}, [])
    # The new post-processing shape is functional, not an existing norm.
    assert seen == [(ctx, option, (7, 2, 0))]
