"""Core data types for Cozempic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class PruneAction:
    """A single pruning action to apply to a message."""

    line_index: int
    action: str  # "remove" | "replace"
    reason: str
    original_bytes: int
    pruned_bytes: int  # 0 for removals, new size for replacements
    replacement: dict | None = None


@dataclass
class StrategyResult:
    """Result of running a single strategy."""

    strategy_name: str
    actions: list[PruneAction]
    original_bytes: int
    pruned_bytes: int
    messages_affected: int
    messages_removed: int
    messages_replaced: int
    summary: str


@dataclass
class PrescriptionResult:
    """Result of running a full prescription (multiple strategies)."""

    prescription_name: str
    strategy_results: list[StrategyResult]
    original_total_bytes: int
    final_total_bytes: int
    original_message_count: int
    final_message_count: int
    original_tokens: int | None = None
    final_tokens: int | None = None
    token_method: str | None = None
    model: str | None = None
    context_window: int | None = None


@dataclass
class StrategyInfo:
    """Metadata about a registered strategy."""

    name: str
    description: str
    tier: str  # "gentle" | "standard" | "aggressive"
    expected_savings: str
    func: Callable


# Type alias for the message tuple used throughout
Message = tuple[int, dict, int]  # (line_index, message_dict, byte_size)
