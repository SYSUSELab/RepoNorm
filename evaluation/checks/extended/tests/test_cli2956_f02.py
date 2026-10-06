"""Existing source contracts affected by validator extensions.

Validator chain ordering and constructor rejection stay in the functional
oracle; these imported witnesses also pass on the original snapshot.
"""

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
