"""Existing value selection, source and context protocols for required_if."""

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

__all__ = [name for name in globals() if name.startswith("test_")]
