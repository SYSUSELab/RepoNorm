"""Existing completion contracts, not the new smart strategy specification."""

import inspect

import pytest

import click
from click.shell_completion import CompletionItem


def _context(option):
    return click.Context(click.Command("cli", params=[option]))


def _smart_if_available():
    # The frozen source has no smart API. Exercise its original API there,
    # and the same callback obligation with the additional policy enabled
    # after implementation. Neither branch skips the existing obligation.
    if "auto_complete" in inspect.signature(click.Option.__init__).parameters:
        return {"auto_complete": True}
    return {}


@pytest.mark.parametrize("smart", [False, True])
def test_cli_5c1239b0116b_n001_snapshot_witness_callback_precedence(smart):
    seen = []
    item = CompletionItem("custom", type="dir", help="preserved help", category="custom")

    class ForbiddenType(click.ParamType):
        name = "forbidden"

        def shell_complete(self, ctx, param, incomplete):
            pytest.fail("explicit callback must precede type completion")

    def callback(ctx, param, incomplete):
        seen.append((ctx, param, incomplete))
        return [item]

    option = click.Option(
        ["--username"],
        type=ForbiddenType(),
        shell_complete=callback,
        **(_smart_if_available() if smart else {}),
    )
    ctx = _context(option)
    result = option.shell_complete(ctx, "cust")
    assert seen == [(ctx, option, "cust")]
    assert len(result) == 1
    assert isinstance(result[0], CompletionItem)
    assert (result[0].value, result[0].type, result[0].help, result[0].category) == (
        "custom", "dir", "preserved help", "custom"
    )


@pytest.mark.parametrize("smart", [False, True])
def test_cli_5c1239b0116b_n001_snapshot_witness_string_normalization(smart):
    option = click.Option(
        ["--username"],
        shell_complete=lambda ctx, param, incomplete: ["alpha", "alpine"],
        **(_smart_if_available() if smart else {}),
    )
    result = option.shell_complete(_context(option), "al")
    assert all(isinstance(item, CompletionItem) for item in result)
    assert [(item.value, item.type, item.help) for item in result] == [
        ("alpha", "plain", None),
        ("alpine", "plain", None),
    ]


@pytest.mark.parametrize(
    ("declarations", "expected_name"),
    [
        (["-u", "--User-Name"], "user_name"),
        (["--User-Name", "-u"], "user_name"),
        (["-u", "--User-Name", "account"], "account"),
        (["--first-name", "--second-name"], "first_name"),
    ],
)
def test_cli_5c1239b0116b_n002_snapshot_witness_option_names(
    declarations, expected_name
):
    option = click.Option(declarations)
    assert option.name == expected_name
    assert option.opts == [value for value in declarations if value.startswith("-")]


def test_cli_5c1239b0116b_n002_snapshot_witness_type_delegation():
    seen = []
    item = CompletionItem("typed", type="file", help="type metadata", category="typed")

    class TrackingType(click.ParamType):
        name = "tracking"

        def shell_complete(self, ctx, param, incomplete):
            seen.append((ctx, param, incomplete))
            return [item]

    # No additional completion policy is requested. Do not constrain how an
    # explicitly enabled new strategy interacts with the ordinary type.
    option = click.Option(["--username"], type=TrackingType())
    ctx = _context(option)
    result = option.shell_complete(ctx, "ty")
    assert len(result) == 1
    assert isinstance(result[0], CompletionItem)
    assert (result[0].value, result[0].type, result[0].help, result[0].category) == (
        "typed", "file", "type metadata", "typed"
    )
    assert seen == [(ctx, option, "ty")]
