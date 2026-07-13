"""Shared Workbench logic (UI toolkit independent)."""

from .helpers import (
    extract_option_value,
    match_name_filter,
    normalize_payload_for_plugin,
    ordered_plugin_ids,
    parse_batch_recent_values,
    parse_name_filter_tokens,
    quote_arg_if_needed,
    read_tail_text_with_trunc,
    should_auto_preflight,
    with_text_padding,
)
from .run_history import filter_run_records, scan_run_history
from .project_tree import scan_tree_nodes

__all__ = [
    "extract_option_value",
    "filter_run_records",
    "match_name_filter",
    "normalize_payload_for_plugin",
    "ordered_plugin_ids",
    "parse_batch_recent_values",
    "parse_name_filter_tokens",
    "quote_arg_if_needed",
    "read_tail_text_with_trunc",
    "scan_run_history",
    "scan_tree_nodes",
    "should_auto_preflight",
    "with_text_padding",
]
