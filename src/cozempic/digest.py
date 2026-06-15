"""Behavioral digest — extract correction signals from sessions, persist as structured rules.

Research basis: Reflexion (NeurIPS 2023), ExpeL (AAAI 2023), A-MAC (2603.04549),
Lost in the Middle, IFScale. See docs/behavioral-digest-design.md.

Phase 1: heuristic extraction + A-MAC admission gate + JSON persistence.
No injection yet (Phase 2). No LLM calls (heuristic only for Phase 1).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ._validation import parse_env_bool
from .helpers import get_content_blocks, get_msg_type, text_of
from .tokens import _is_sidechain
from .types import Message

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTECTION_TAG = "__cozempic_behavioral_digest__"
DIGEST_DIR = Path.home() / ".cozempic"
DIGEST_FILE = DIGEST_DIR / "behavioral-digest.json"
DIGEST_MD_FILE = DIGEST_DIR / "behavioral-digest.md"

MAX_ACTIVE_RULES = 20  # IFScale: >30 irrelevant rules degrades ALL adherence
ADMISSION_THRESHOLD = 0.55  # A-MAC composite score gate
PRUNE_THRESHOLD = 0.30  # Below this → prune
PROMOTION_COUNT = 2  # Occurrences needed to promote pending → active (was 3, too high for real usage)
DECAY_DAYS = 30  # Universal decay period (MemoryArena 2602.16313)

# Opt-in stderr diagnostics for silent _to_prohibition rejections (A2).
# `_DEBUG` is kept as a module-level attribute for test monkeypatching;
# `_debug()` also re-reads the env on each call so setting COZEMPIC_DEBUG=1
# AFTER import (programmatic or shell-inherited mid-run) takes effect.
# Runs inside hooks (PreCompact, Stop) where unconditional stderr would leak
# to users — hence env-gated.
# Accepts: 1/true/yes/on (case-insensitive). Default off.
_DEBUG = parse_env_bool("COZEMPIC_DEBUG", default=False)


def _debug(msg: str) -> None:
    if _DEBUG or parse_env_bool("COZEMPIC_DEBUG", default=False, warn=False):
        print(f"[cozempic.digest] {msg}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DigestRule:
    """A single behavioral rule extracted from user corrections."""

    id: str  # R001, R002, etc.
    rule: str  # "Do not [X]" — prohibition framing
    priority: Literal["hard", "soft"] = "soft"
    scope: str = "general"  # git, file-ops, testing, communication, general
    trigger: str = ""  # When this rule applies

    # Decision attribution (Trajectory-Informed Memory 2603.10600)
    decision_step: str = ""  # Which step in agent's reasoning caused failure
    before: str = ""  # What agent did wrong
    after: str = ""  # What user wants instead

    # Evidence — stored verbatim, never paraphrased (2603.23013)
    signal: str = ""  # Why agent made the error
    evidence: str = ""  # Direct quote from conversation

    # Scoring (A-MAC 2603.04549)
    importance: int = 1  # ExpeL voting count
    source_reliability: float = 1.0  # 1.0 explicit, 0.6 implicit, 0.3 inferred
    type_prior: float = 0.8  # correction=0.8, preference=0.9, one-off=0.1

    # Lifecycle
    status: Literal["pending", "active", "conflicted"] = "pending"
    occurrence_count: int = 1
    first_seen: str = ""
    last_reinforced: str = ""
    last_injection: str | None = None


@dataclass
class DigestStore:
    """Persistent store for behavioral rules."""

    strategy_rules: list[DigestRule] = field(default_factory=list)
    version: str = "1"
    project: str = ""
    updated: str = ""
    session_id: str = ""

    def is_empty(self) -> bool:
        return not self.strategy_rules

    def active_rules(self) -> list[DigestRule]:
        return [r for r in self.strategy_rules if r.status == "active"]

    def all_rules(self) -> list[DigestRule]:
        return self.strategy_rules

    def next_id(self) -> str:
        # Gap-aware: find the smallest unused Rnnnn slot. Widen from 3 to 4
        # digits past R999 so the fallback never collides with an existing
        # 4-digit ID on stores that already contain R1000+.
        existing = {r.id for r in self.all_rules()}
        for i in range(1, 10_000):
            rid = f"R{i:03d}" if i < 1000 else f"R{i:04d}"
            if rid not in existing:
                return rid
        # MAX_ACTIVE_RULES=20 caps real usage far below 9999; reaching this
        # branch is a corruption signal, not a normal overflow.
        raise RuntimeError("digest store exhausted R001-R9999 id space")


# ---------------------------------------------------------------------------
# Classification — FELT taxonomy (heuristic, no LLM)
# ---------------------------------------------------------------------------

# Correction signal patterns
_EXPLICIT_PATTERNS = [
    re.compile(r"^no[,.\s]", re.IGNORECASE),
    re.compile(r"\bdon'?t\b", re.IGNORECASE),
    re.compile(r"\bdo not\b", re.IGNORECASE),
    re.compile(r"\bstop\s+\w+ing\b", re.IGNORECASE),  # "stop summarizing", "stop adding", etc.
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bplease\s+(don'?t|remove|stop|undo)", re.IGNORECASE),
    re.compile(r"\bremove\s+that\b", re.IGNORECASE),
    re.compile(r"\bundo\s+(that|this|the)\b", re.IGNORECASE),
]

# Synthetic-noise markers — user turns containing any of these are Claude Code
# framework emissions (hooks, slash commands, tool blocks), not real corrections.
_SYSTEM_NOISE_MARKERS = (
    "<local-command-",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<teammate-message",
    "<system-reminder",
    "<function_calls>",
    "<function_results>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<user-prompt-submit-hook",
    "Please analyze this codebase",  # /init prompt
    "[Cozempic Guard:",  # cozempic self-restoration meta banner
    "This session is being continued from a previous conversation",  # CC compaction-resume banner
    # Skill-injection bodies (#109): when a skill loads, the harness injects its
    # SKILL.md into the transcript prefixed with this header. It is NOT a user
    # utterance — without this, skill bodies were captured as EXPLICIT_CORRECTION
    # rules (e.g. "Do not base directory for this skill: ~/.claude/skills/...").
    "Base directory for this skill:",
    "The following skills are available for use",  # SessionStart skills listing
    "The following deferred tools are now available",  # deferred-tool reminder
)


# Zero-width and format characters sometimes injected before tag brackets:
# ZWSP (U+200B), ZWNJ (U+200C), ZWJ (U+200D), BOM (U+FEFF), WORD JOINER (U+2060),
# LRM (U+200E), RLM (U+200F). `str.strip()` does NOT remove these by default.
_ZERO_WIDTH_PREFIX_CHARS = ("​", "‌", "‍", "﻿", "⁠", "‎", "‏")

# Unicode tag-bracket lookalikes — when they appear as the LEADING char of a
# turn, the turn is a synthetic/wrapped emission, not a user correction.
# Substring match would false-positive on genuine user text that happens to
# quote a word with guillemets (« foo ») or fullwidth punctuation — so we
# restrict to startswith.
_UNICODE_TAG_LEAD_CHARS = ("＜", "«", "〈")  # ＜ « 〈

# Slash-command detector: '/' + ASCII letter + identifier chars, terminated by
# whitespace or end-of-string. Case-insensitive so /Compact, /INIT don't leak
# (A12). File paths like `/Users/...` don't match because the first token
# contains another '/'.
_SLASH_CMD_RE = re.compile(r"^/[A-Za-z][A-Za-z0-9_-]*(?:\s|$)")


def _is_system_noise(text: str) -> bool:
    """Return True if `text` is a Claude Code synthetic/framework turn.

    Rejects: empty text, tag-wrapped blocks (ASCII `<` or Unicode lookalikes),
    slash-command lines, and known framework prompt markers. Used to gate
    `extract_corrections` upstream of `classify_turn` so synthetic turns
    never become behavioral rules.
    """
    if not text:
        return True
    # Strip standard whitespace AND zero-width format chars (A1 — some emitters
    # inject ZWSP/BOM before the opening '<' which bypasses a plain strip()).
    stripped = text.strip().lstrip("".join(_ZERO_WIDTH_PREFIX_CHARS))
    if not stripped:
        return True
    # Tag-like: any line starting with '<' OR '/<' is synthetic. The '/<'
    # form appears in some wrapped emissions where a slash precedes the
    # tag. File paths like `/Users/...` still pass because they start
    # with '/' followed by a letter, not '<'.
    if stripped.startswith("<") or stripped.startswith("/<"):
        return True
    # Unicode tag-bracket lookalikes as LEADING char (A1 — fullwidth ＜, «, 〈).
    if stripped[0] in _UNICODE_TAG_LEAD_CHARS:
        return True
    # Slash command: '/' + letter + identifier chars terminated by whitespace or
    # end-of-string (A12 — case-insensitive; file paths /Users/... don't match
    # because their first token embeds another '/').
    if _SLASH_CMD_RE.match(stripped):
        return True
    # Known framework markers (substring match).
    for marker in _SYSTEM_NOISE_MARKERS:
        if marker in stripped:
            return True
    return False

_IMPLICIT_PATTERNS = [
    re.compile(r"\bactually[,\s]", re.IGNORECASE),
    re.compile(r"\binstead[,\s]", re.IGNORECASE),
    re.compile(r"\brather\b", re.IGNORECASE),
    re.compile(r"\bthat'?s\s+(not|wrong)", re.IGNORECASE),
    re.compile(r"\bnot\s+what\s+I", re.IGNORECASE),
    re.compile(r"\buse\s+\w+\s+not\s+\w+", re.IGNORECASE),  # "use Edit not Write"
    re.compile(r"\bnot\s+\w+[,;]\s*(use|try)", re.IGNORECASE),  # "not Write, use Edit"
]

_PREFERENCE_PATTERNS = [
    re.compile(r"\bI\s+prefer\b", re.IGNORECASE),
    re.compile(r"\balways\s+(use|do|add|include|run|check)", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on\b", re.IGNORECASE),
    re.compile(r"\bremember\s+(to|that)\b", re.IGNORECASE),
    re.compile(r"\bmake\s+sure\s+(to|you)\b", re.IGNORECASE),
]

_APOLOGY_PATTERNS = [
    re.compile(r"\bsorry\b", re.IGNORECASE),
    re.compile(r"\bI\s+apologize\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(mistake|bad|error)\b", re.IGNORECASE),
]

TurnClass = Literal[
    "EXPLICIT_CORRECTION",
    "IMPLICIT_CORRECTION",
    "PREFERENCE",
    "APOLOGY_FOLLOW_UP",
    "ONE_OFF",
    "NONE",
]


def classify_turn(user_text: str, prev_assistant_text: str = "") -> TurnClass:
    """Classify a user turn by correction signal type.

    Content type prior IS the dominant factor (A-MAC ablation).
    """
    if not user_text or len(user_text.strip()) < 3:
        return "NONE"

    # Check if previous assistant apologized → this turn is a follow-up correction
    if prev_assistant_text:
        for pat in _APOLOGY_PATTERNS:
            if pat.search(prev_assistant_text):
                # User message after apology is likely a correction
                if len(user_text.strip()) > 10:
                    return "APOLOGY_FOLLOW_UP"

    # Explicit correction: strongest signal
    for pat in _EXPLICIT_PATTERNS:
        if pat.search(user_text):
            return "EXPLICIT_CORRECTION"

    # Preference: persistent behavioral instruction
    for pat in _PREFERENCE_PATTERNS:
        if pat.search(user_text):
            return "PREFERENCE"

    # Implicit correction: softer signal
    for pat in _IMPLICIT_PATTERNS:
        if pat.search(user_text):
            return "IMPLICIT_CORRECTION"

    return "NONE"


# ---------------------------------------------------------------------------
# Extraction — heuristic rule extraction from classified turns
# ---------------------------------------------------------------------------


def _get_user_text(msg: dict) -> str:
    """Extract user text from a message."""
    inner = msg.get("message", {})
    content = inner.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def _get_assistant_text(msg: dict) -> str:
    """Extract assistant text from a message."""
    blocks = get_content_blocks(msg)
    parts = []
    for block in blocks:
        t = text_of(block)
        if t and block.get("type") in ("text", None, ""):
            parts.append(t)
    return " ".join(parts)


_SCOPE_KEYWORDS: tuple[tuple[str, frozenset[str]], ...] = (
    # Order matters on overlap: `testing` before `file-ops` so that
    # "write tests" (both write-verb and tests-noun) resolves to testing —
    # the noun is the subject of the correction, the verb is incidental.
    ("testing", frozenset({"test", "tests", "pytest", "unittest", "mock", "mocks", "assert", "asserts"})),
    ("git", frozenset({"git", "commit", "commits", "push", "pushed", "branch", "branches", "merge", "merges", "merged", "co-authored", "co-authored-by"})),
    ("file-ops", frozenset({"file", "files", "edit", "edits", "write", "writes", "read", "reads", "path", "paths", "directory", "directories"})),
    ("communication", frozenset({"message", "messages", "comment", "comments", "pr", "prs", "issue", "issues", "slack"})),
)


def _infer_scope(text: str) -> str:
    """Infer the scope of a correction from its content.

    Tokenizes with `[\\w-]+` so hyphenated compounds like `co-authored-by`
    stay intact, then matches against keyword sets as whole tokens. This
    prevents substring false-positives (`digital` → `git`, `testimony` →
    `test`) that silently broke the dedup gate (which requires scope match
    before text-overlap merge).
    """
    tokens = set(re.findall(r"[\w-]+", text.lower()))
    for scope, keywords in _SCOPE_KEYWORDS:
        if tokens & keywords:
            return scope
    return "general"


def _to_prohibition(text: str) -> str:
    """Convert a user correction into prohibition framing.

    "Don't add X" → "Do not add X"
    "Stop doing X" → "Do not do X"
    "No, use Y instead" → "Do not use the previous approach; use Y instead"

    Returns empty string (sentinel for "skip this candidate") when the input
    is obviously structural/non-correction content: too long, multi-paragraph,
    leading markdown, code fence, or tag.
    """
    text = text.strip()
    # Reject structural / oversize input — cannot be a clean correction.
    # Debug messages emit metadata only (length, newline count, single prefix
    # char). Never echo raw user text — risk of PII / credentials leaking into
    # stderr logs when COZEMPIC_DEBUG=1.
    if not text:
        return ""
    if len(text) > 200:
        _debug(f"_to_prohibition rejected: len={len(text)} > 200 (content redacted)")
        return ""
    if text.count("\n") > 2:
        _debug(
            f"_to_prohibition rejected: multi-paragraph "
            f"({text.count(chr(10))} newlines, content redacted)"
        )
        return ""
    if text[0] in "<-*#`":
        _debug(f"_to_prohibition rejected: structural-prefix {text[0]!r}")
        return ""
    # Already in prohibition form
    if text.lower().startswith("do not ") or text.lower().startswith("don't "):
        return text[0].upper() + text[1:]

    # "Stop doing X" → "Do not X"
    m = re.match(r"(?i)stop\s+(doing\s+|adding\s+|using\s+|creating\s+)?(.*)", text)
    if m:
        action = m.group(2).strip()
        return f"Do not {action}" if action else text

    # "Never X" → "Do not ever X"
    m = re.match(r"(?i)never\s+(.*)", text)
    if m:
        return f"Do not ever {m.group(1).strip()}"

    # "No, ..." → extract the instruction
    m = re.match(r"(?i)^no[,.\s]+\s*(.*)", text)
    if m:
        rest = m.group(1).strip()
        if rest:
            return rest[0].upper() + rest[1:]

    # Default: prefix with "Do not". Require a letter lead so the grammar is
    # valid — digit/punctuation prefixes produce malformed prohibitions like
    # "Do not 5xx errors..." which no model can usefully follow. The
    # `isalpha()` gate applies regardless of length, so short digit-led text
    # ("5xx", "1st") is also rejected rather than returning raw. Existing
    # digit-prefixed active rules auto-demote via the load_digest_store
    # migration path which re-runs _to_prohibition on rule.evidence.
    if not text[0].isalpha():
        _debug(f"_to_prohibition rejected: non-letter lead {text[0]!r}")
        return ""
    if len(text) > 5:
        return f"Do not {text[0].lower()}{text[1:]}"
    return text


def extract_corrections(
    messages: list[Message],
    since_turn: int = 0,
) -> list[DigestRule]:
    """Extract behavioral corrections from a message window.

    Scans user turns for correction signals, builds DigestRule for each.
    Stores verbatim evidence — never paraphrased (arXiv:2603.23013).
    """
    now = datetime.now(timezone.utc).isoformat()
    rules: list[DigestRule] = []

    prev_assistant_text = ""
    for pos, (idx, msg, _) in enumerate(messages):
        # Skip sub-agent (sidechain) turns entirely — they are scaffolding/tool
        # prompts, not the human correcting Claude. Mining them is an L6
        # injection: untrusted sub-agent content becomes a learned behavioral rule.
        # Also skip their assistant text from prev_assistant_text so sidechain
        # context never bleeds into the `before` field of a following main turn.
        if _is_sidechain(msg):
            continue

        if pos < since_turn:
            # Track assistant text even before our window
            if get_msg_type(msg) == "assistant":
                prev_assistant_text = _get_assistant_text(msg)
            continue

        mtype = get_msg_type(msg)

        if mtype == "assistant":
            prev_assistant_text = _get_assistant_text(msg)
            continue

        if mtype != "user":
            continue

        user_text = _get_user_text(msg)
        if not user_text:
            continue

        # Skip Claude Code synthetic/framework turns — they are not corrections.
        if _is_system_noise(user_text):
            prev_assistant_text = ""
            continue

        turn_class = classify_turn(user_text, prev_assistant_text)
        if turn_class == "NONE":
            prev_assistant_text = ""
            continue

        # Map classification to scoring
        reliability_map = {
            "EXPLICIT_CORRECTION": 1.0,
            "IMPLICIT_CORRECTION": 0.6,
            "PREFERENCE": 0.9,
            "APOLOGY_FOLLOW_UP": 0.8,
            "ONE_OFF": 0.3,
        }
        type_prior_map = {
            "EXPLICIT_CORRECTION": 0.8,
            "IMPLICIT_CORRECTION": 0.6,
            "PREFERENCE": 0.9,
            "APOLOGY_FOLLOW_UP": 0.7,
            "ONE_OFF": 0.1,
        }

        rule_text = _to_prohibition(user_text)
        if not rule_text:
            # _to_prohibition rejected the input as structural/non-correction.
            prev_assistant_text = ""
            continue
        scope = _infer_scope(user_text)

        rule = DigestRule(
            id="",  # Assigned on admission
            rule=rule_text[:500],  # Cap rule length
            priority="hard" if turn_class == "EXPLICIT_CORRECTION" else "soft",
            scope=scope,
            trigger="",
            before=prev_assistant_text[:200] if prev_assistant_text else "",
            after=user_text[:200],
            signal=turn_class,
            evidence=user_text[:500],  # Verbatim, never paraphrased
            importance=1,
            source_reliability=reliability_map.get(turn_class, 0.5),
            type_prior=type_prior_map.get(turn_class, 0.5),
            # All new rules start pending — the repetition gate in admit_rule
            # promotes them to active after PROMOTION_COUNT occurrences.
            status="pending",
            occurrence_count=1,
            first_seen=now,
            last_reinforced=now,
        )
        rules.append(rule)

        prev_assistant_text = ""

    return rules


# ---------------------------------------------------------------------------
# Admission gate — A-MAC composite scoring (arXiv:2603.04549)
# ---------------------------------------------------------------------------


def score_rule(rule: DigestRule, days_since_last: float = 0.0) -> float:
    """Compute A-MAC composite score for a rule.

    composite = w1*(count/3) + w2*source_reliability + w3*recency + w4*type_prior
    Threshold: 0.55 for admission, 0.30 for pruning.
    """
    evidence_score = min(rule.occurrence_count / PROMOTION_COUNT, 1.0)
    recency_decay = math.exp(-0.05 * days_since_last)  # λ=0.05 → halves in ~14 days
    composite = (
        0.25 * evidence_score
        + 0.30 * rule.source_reliability
        + 0.20 * recency_decay
        + 0.25 * rule.type_prior
    )
    return round(composite, 4)


def _normalize_for_match(text: str) -> set[str]:
    """Normalize text for duplicate matching — strip stop words, lowercase."""
    _STOP = {"do", "not", "don't", "dont", "the", "a", "an", "to", "is", "it", "of", "in", "for"}
    words = set(text.lower().split())
    return words - _STOP


def _bigrams(text: str) -> set[str]:
    """Return token-level bigrams of `text` — used to distinguish rules that
    share vocabulary but differ in word order (e.g., "use Edit not Write" vs
    "use Write not Edit")."""
    tokens = text.lower().split()
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def _overlap(a: set, b: set) -> float:
    """Jaccard-like overlap: shared / max(|a|, |b|). 0.0 on empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def _is_match(new_words: set, new_bigrams: set, ex_words: set, ex_bigrams: set) -> bool:
    """True if word AND bigram overlap both pass 0.5 threshold.

    Bigram overlap distinguishes order-inverted rules like "use Edit not Write"
    vs "use Write not Edit" which share bag-of-words but have opposite intent.
    """
    if _overlap(new_words, ex_words) <= 0.5:
        return False
    return _overlap(new_bigrams, ex_bigrams) >= 0.5


def _find_duplicate(new_rule: DigestRule, store: DigestStore) -> DigestRule | None:
    """Find a semantically similar existing rule.

    Requires scope AND priority match first (opposite-priority or cross-scope
    rules are different instructions even if text overlaps), then both
    word-overlap > 0.5 and bigram-overlap >= 0.5 on rule text OR on the
    user-phrased evidence.
    """
    new_words = _normalize_for_match(new_rule.rule)
    if not new_words:
        return None
    new_bigrams = _bigrams(new_rule.rule)
    new_ev_words = _normalize_for_match(new_rule.evidence) if new_rule.evidence else set()
    new_ev_bigrams = _bigrams(new_rule.evidence) if new_rule.evidence else set()

    for existing in store.strategy_rules:
        if existing.scope != new_rule.scope or existing.priority != new_rule.priority:
            continue
        ex_words = _normalize_for_match(existing.rule)
        if not ex_words:
            continue
        # Early-exit if word overlap fails on both rule-text AND evidence paths
        # — avoids the bigram computation (expensive) on unrelated rules.
        rule_words_match = _overlap(new_words, ex_words) > 0.5
        ev_words_match = bool(new_ev_words) and _overlap(new_ev_words, ex_words) > 0.5
        if not (rule_words_match or ev_words_match):
            continue

        ex_bigrams = _bigrams(existing.rule)
        if rule_words_match and _is_match(new_words, new_bigrams, ex_words, ex_bigrams):
            return existing
        if ev_words_match and _is_match(new_ev_words, new_ev_bigrams, ex_words, ex_bigrams):
            return existing
    return None


def _enforce_active_cap(store: DigestStore) -> None:
    """Demote lowest-scored active rules until len(active) <= MAX_ACTIVE_RULES.

    Single O(n log n) sort; demotes the bottom-k in one pass rather than
    re-sorting per iteration.
    """
    active = store.active_rules()
    overflow = len(active) - MAX_ACTIVE_RULES
    if overflow <= 0:
        return
    scored = sorted(active, key=score_rule)
    for rule in scored[:overflow]:
        rule.status = "pending"


def admit_rule(rule: DigestRule, store: DigestStore) -> str:
    """A-MAC admission gate. Returns 'added', 'upvoted', or 'rejected'.

    Quality gate BEFORE any rule enters store (arXiv:2505.16067).
    """
    existing = _find_duplicate(rule, store)
    if existing:
        existing.occurrence_count += 1
        existing.importance += 1
        existing.last_reinforced = rule.last_reinforced or datetime.now(timezone.utc).isoformat()
        if existing.status == "pending" and existing.occurrence_count >= PROMOTION_COUNT:
            existing.status = "active"
        return "upvoted"

    if score_rule(rule) < ADMISSION_THRESHOLD:
        return "rejected"

    rule.id = store.next_id()
    store.strategy_rules.append(rule)
    _enforce_active_cap(store)
    return "added"


# ---------------------------------------------------------------------------
# Persistence — JSON on disk
# ---------------------------------------------------------------------------


def _atomic_write_text(target: Path, data: str, encoding: str = "utf-8") -> None:
    """Atomic, fsync'd, collision-safe write — delegates to the ONE hardened
    implementation in helpers.atomic_write_text so the temp-naming policy lives
    in a single place and can't drift.

    Consolidation (Batch-4): this used to be a byte-identical third copy of the
    atomic-write primitive, still on the OLD ``prefix='.tmp.', suffix=target.name``
    pattern that #127 hardened elsewhere to ``.tmp.<name>.<rand>.partial`` — a
    leftover divergent copy of exactly the temp-orphan class #127 set out to kill.
    """
    from .helpers import atomic_write_text as _hardened_atomic_write_text
    _hardened_atomic_write_text(target, data, encoding=encoding)


def load_digest_store(project_dir: str = "") -> DigestStore:
    """Load the digest store from disk, auto-migrating pre-hardening rules.

    Failure modes (missing file, bad JSON, permission denied, disk error) all
    return an empty store rather than crash — this code runs inside hook
    critical path (PreCompact, Stop) where exceptions would kill the session.

    Auto-migration: demotes any active rule that would not pass the current
    admission gate (noise OR rejected by `_to_prohibition`), so users who
    upgrade inherit a clean store with zero action required.
    """
    try:
        data = json.loads(DIGEST_FILE.read_text(encoding="utf-8"))
        store = DigestStore(
            version=data.get("version", "1"),
            project=data.get("project", project_dir),
            updated=data.get("updated", ""),
            session_id=data.get("session_id", ""),
        )
        for rd in data.get("strategy_rules", []):
            store.strategy_rules.append(DigestRule(**rd))
        migrated = False
        for rule in store.strategy_rules:
            if rule.status != "active":
                continue
            source = rule.evidence or rule.rule
            if _is_system_noise(source) or _to_prohibition(source) == "":
                rule.status = "pending"
                migrated = True
        _enforce_active_cap(store)
        # Persist the migration so we don't re-scan on every load.
        # Without this, a store with 572 noise rules would re-demote them
        # on every cozempic invocation — wasting cycles and never cleaning
        # the on-disk file unless a separate save path happens to fire.
        if migrated:
            try:
                save_digest_store(store)
            except Exception:
                pass  # Non-fatal — store is clean in memory, disk catches up later
        return store
    except (json.JSONDecodeError, TypeError, KeyError, OSError):
        return DigestStore(project=project_dir)


def save_digest_store(store: DigestStore) -> None:
    """Save the digest store to disk (JSON + human-readable markdown mirror).

    Atomic + lost-update-safe: writes via `_atomic_write_text` (tmp+os.replace)
    so a mid-write crash leaves the target untouched, and re-reads the current
    on-disk state just before writing to merge in any rules added by a
    concurrent hook process (prevents PreCompact + Stop lost-update races).
    """
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)

    # Concurrent-save merge: if another process appended rules to the file
    # between our load and this save, pull those rules in so they survive.
    if DIGEST_FILE.exists():
        try:
            on_disk = json.loads(DIGEST_FILE.read_text(encoding="utf-8"))
            known_rule_texts = {r.rule for r in store.strategy_rules}
            for rd in on_disk.get("strategy_rules", []):
                if rd.get("rule") not in known_rule_texts:
                    store.strategy_rules.append(DigestRule(**rd))
        except (json.JSONDecodeError, TypeError, KeyError, OSError):
            # Corrupt or unreadable — skip merge, just overwrite with our state.
            pass

    store.updated = datetime.now(timezone.utc).isoformat()

    data = {
        "version": store.version,
        "project": store.project,
        "updated": store.updated,
        "session_id": store.session_id,
        "strategy_rules": [asdict(r) for r in store.strategy_rules],
        "failure_patterns": [],  # Reserved for future use
    }
    _atomic_write_text(DIGEST_FILE, json.dumps(data, indent=2))

    # Write human-readable markdown mirror
    _write_digest_md(store)


def _write_digest_md(store: DigestStore) -> None:
    """Write a human-readable markdown version of the digest."""
    lines = [
        "# Behavioral Digest",
        f"Updated: {store.updated}",
        f"Project: {store.project}",
        "",
    ]

    active = [r for r in store.strategy_rules if r.status == "active"]
    pending = [r for r in store.strategy_rules if r.status == "pending"]

    if active:
        lines.append(f"## Active Rules ({len(active)})")
        lines.append("")
        for r in active:
            lines.append(f"- **[{r.id}|{r.scope}|{r.priority}]** {_sanitize_for_injection(r.rule)}")
            if r.trigger:
                lines.append(f"  - When: {_sanitize_for_injection(r.trigger)}")
            if r.evidence:
                lines.append(f"  - Evidence: \"{_sanitize_for_injection(r.evidence, limit=100)}\"")
            lines.append(f"  - Score: {score_rule(r):.2f} | Seen: {r.occurrence_count}x")
        lines.append("")

    if pending:
        lines.append(f"## Pending Rules ({len(pending)})")
        lines.append("")
        for r in pending:
            lines.append(f"- **[{r.id}|{r.scope}]** {_sanitize_for_injection(r.rule)} (seen {r.occurrence_count}x)")
        lines.append("")

    _atomic_write_text(DIGEST_MD_FILE, "\n".join(lines))


def clear_digest_store() -> None:
    """Remove all digest files."""
    for f in (DIGEST_FILE, DIGEST_MD_FILE):
        if f.exists():
            f.unlink()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def update_digest(
    messages: list[Message],
    since_turn: int = 0,
    project_dir: str = "",
    session_id: str = "",
) -> tuple[int, int, int]:
    """Extract corrections from messages and update the digest store.

    Returns (new_rules, upvoted, rejected).
    """
    store = load_digest_store(project_dir)
    store.session_id = session_id

    candidates = extract_corrections(messages, since_turn=since_turn)

    added = 0
    upvoted = 0
    rejected = 0

    for rule in candidates:
        result = admit_rule(rule, store)
        if result == "added":
            added += 1
        elif result == "upvoted":
            upvoted += 1
        else:
            rejected += 1

    # Persist unconditionally: `store.session_id` is mutated above regardless
    # of the admission outcome, and `updated` timestamp must advance even on
    # rejected-only or zero-candidate runs so downstream consumers can
    # distinguish stale from fresh state. `save_digest_store` is atomic
    # (tmp+fsync+rename) and concurrent-merge-safe.
    # A readonly digest dir (Docker --read-only, hardened systemd, NFS quota
    # hit) must not crash the hook. Degrade to in-memory only; disk catches
    # up on the next call when the FS recovers.
    try:
        save_digest_store(store)
    except (OSError, PermissionError) as e:
        _debug(f"save_digest_store failed (non-fatal): {type(e).__name__}")

    return added, upvoted, rejected


# ---------------------------------------------------------------------------
# Injection — Phase 2: inject rules at session tail
# ---------------------------------------------------------------------------


def _sanitize_for_injection(text: str, limit: int = 300) -> str:
    """Neutralize untrusted rule/evidence text before it is embedded into a
    Claude-readable markdown file (digest memory / injection block).

    Rule text comes from the user's transcript and is otherwise interpolated
    verbatim — so a rule containing a newline + a fake `## header`, a list item,
    a `>` quote, or a ``` code fence could inject structure/instructions into CC's
    memory (prompt-injection / the audit P1). Defenses: collapse ALL newlines and
    control chars to spaces (one rule stays one line — cannot add markdown lines),
    neutralize a leading markdown-control char, and cap length.
    """
    if not isinstance(text, str):
        text = str(text)
    # Newlines + other C0 control chars -> space (single-line guarantee).
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Defang a leading markdown-structural char so the value can't render as a
    # header/quote/list/fence at the start of its line.
    if text[:1] in "#>-*`|":
        text = "\\" + text
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _format_rule_4field(rule: DigestRule) -> str:
    """Format a rule in 4-field compressed format (arXiv:2603.13017 — 11x compression)."""
    line = f"[{rule.id}|{rule.scope}|{rule.priority}] {_sanitize_for_injection(rule.rule)}"
    if rule.trigger:
        line += f"\n  When: {_sanitize_for_injection(rule.trigger)}"
    if rule.signal:
        line += f"\n  Signal: {_sanitize_for_injection(rule.signal)}"
    if rule.evidence:
        line += f"\n  Evidence: \"{_sanitize_for_injection(rule.evidence, limit=120)}\""
    return line


def build_injection_text(store: DigestStore) -> str | None:
    """Build the injection text block from active rules.

    Returns None if no active rules exist.
    4-field structured format for injection (full 8-field stored on disk).
    Prefix: "Focus solely on these behavioral rules when applicable" (arXiv:2505.02709).
    Hard cap: 20 active rules (IFScale).
    """
    active = store.active_rules()
    if not active:
        return None

    # Hard first, then soft up to the remaining cap slots.
    hard = [r for r in active if r.priority == "hard"][:MAX_ACTIVE_RULES]
    soft_budget = max(0, MAX_ACTIVE_RULES - len(hard))
    soft = [r for r in active if r.priority == "soft"][:soft_budget]

    lines = ["BEHAVIORAL CONTRACT — Focus solely on these rules when applicable.", ""]
    if hard:
        lines.append("PROHIBITIONS:")
        lines.extend(_format_rule_4field(r) for r in hard)
        lines.append("")
    if soft:
        lines.append("PREFERENCES:")
        lines.extend(_format_rule_4field(r) for r in soft)
        lines.append("")

    return "\n".join(lines)


def _project_dir_exists(cwd: str = "") -> bool:
    """True if Claude Code's project dir for this cwd exists (transcript lives
    there). Used to give an accurate sync message without creating anything."""
    from .session import cwd_to_project_slug, get_projects_dir
    if not cwd:
        cwd = os.getcwd()
    claude_dir = get_projects_dir()
    if not claude_dir.exists():
        return False
    return (claude_dir / cwd_to_project_slug(cwd)).exists()


def _get_memdir(cwd: str = "") -> Path | None:
    """Find the Claude Code memory directory for the given project.

    Delegates profile resolution to `session.get_projects_dir()` which honours
    `CLAUDE_CONFIG_DIR` (used by the `claudes` profile launcher) before
    falling back to `~/.claude`. Prevents cross-profile leaks.

    Read-only: cozempic does NOT create this dir — Claude Code owns it and creates
    it lazily. We only sync into it once it exists, so a brand-new folder isn't
    proactively populated with the (global) digest until Claude Code establishes
    its memory dir.
    """
    from .session import cwd_to_project_slug, get_projects_dir
    if not cwd:
        cwd = os.getcwd()
    claude_dir = get_projects_dir()
    if not claude_dir.exists():
        return None
    # cwd_to_project_slug already includes the leading '-' (from the leading '/')
    # so no f"-{slug}" prepend needed — avoids a double leading-dash regression.
    slug = cwd_to_project_slug(cwd)
    project_dir = claude_dir / slug
    if not project_dir.exists():
        # No fallback iteration: if `claude_dir / slug` does not exist, no other
        # dir in claude_dir can have that exact name — the scan would be dead code.
        return None
    mem_dir = project_dir / "memory"
    return mem_dir if mem_dir.exists() else None


def sync_to_memdir(store: DigestStore, cwd: str = "") -> int:
    """Write active rules as Claude Code feedback memories.

    Claude reads these natively via getMemoryFiles() → injected as <system-reminder>.
    They survive compaction (files on disk, re-read after every compact).
    No custom JSONL injection needed.

    Returns number of rules synced.
    """
    # Read-only on the dir: we sync into Claude Code's memory dir only once it
    # exists (Claude Code owns/creates it). We never create it ourselves, so a
    # brand-new folder is not proactively populated with the global digest.
    mem_dir = _get_memdir(cwd)
    if mem_dir is None:
        return 0

    active = store.active_rules()
    if not active:
        # Remove existing digest memory if no active rules
        digest_mem = mem_dir / "cozempic_digest.md"
        if digest_mem.exists():
            digest_mem.unlink()
        return 0

    # Build the memory file content
    text = build_injection_text(store)
    if not text:
        return 0

    content = f"""---
name: Cozempic Behavioral Digest
description: Behavioral rules extracted from user corrections — follow these when applicable
type: feedback
---

{text}
"""

    digest_mem = mem_dir / "cozempic_digest.md"
    _atomic_write_text(digest_mem, content)

    # Update MEMORY.md index if needed
    _update_memory_index(mem_dir)

    # Update last_injection timestamp
    now = datetime.now(timezone.utc).isoformat()
    for r in active:
        r.last_injection = now

    return len(active)


def _update_memory_index(mem_dir: Path) -> None:
    """Ensure cozempic_digest.md is referenced in MEMORY.md index."""
    index_path = mem_dir / "MEMORY.md"
    marker = "[Cozempic Behavioral Digest](cozempic_digest.md)"
    entry = f"- {marker} — behavioral rules from user corrections"

    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        if "cozempic_digest.md" in content:
            return  # Already referenced
        content = content.rstrip() + f"\n{entry}\n"
        index_path.write_text(content, encoding="utf-8")
    else:
        # MEMORY.md doesn't exist yet — don't create it, Claude Code manages this file
        pass


# ---------------------------------------------------------------------------
# Flush / Recover — extraction + memdir sync
# ---------------------------------------------------------------------------


def flush_digest(
    messages: list[Message],
    project_dir: str = "",
    session_id: str = "",
) -> tuple[int, int, int]:
    """Extract corrections from full session, save to disk, sync to memdir.

    Called by PreCompact and Stop hooks to capture corrections before loss.
    Returns (added, upvoted, rejected).
    """
    added, upvoted, rejected = update_digest(
        messages, since_turn=0, project_dir=project_dir, session_id=session_id,
    )
    # Sync active rules to Claude Code's memory system
    store = load_digest_store(project_dir)
    synced = sync_to_memdir(store, cwd=project_dir)
    if synced > 0:
        save_digest_store(store)  # Update last_injection timestamps
    return added, upvoted, rejected


def recover_digest(
    project_dir: str = "",
) -> int:
    """Re-sync digest to memdir after compaction.

    Called by PostCompact hook. Memdir files survive compaction natively,
    but re-sync ensures any newly promoted rules are included.
    Returns number of rules synced.
    """
    store = load_digest_store(project_dir)
    if store.is_empty():
        return 0
    synced = sync_to_memdir(store, cwd=project_dir)
    if synced > 0:
        save_digest_store(store)
    return synced


def show_digest() -> str:
    """Return a formatted string of the current digest."""
    store = load_digest_store()
    if store.is_empty():
        return "No behavioral rules stored."

    lines = []
    active = store.active_rules()
    pending = [r for r in store.strategy_rules if r.status == "pending"]

    if active:
        lines.append(f"Active rules ({len(active)}):")
        for r in active:
            lines.append(f"  [{r.id}|{r.scope}|{r.priority}] {_sanitize_for_injection(r.rule)}")
            lines.append(f"    Score: {score_rule(r):.2f} | Seen: {r.occurrence_count}x")

    if pending:
        lines.append(f"\nPending rules ({len(pending)}):")
        for r in pending:
            lines.append(f"  [{r.id}|{r.scope}] {_sanitize_for_injection(r.rule)} (seen {r.occurrence_count}x)")

    return "\n".join(lines)
