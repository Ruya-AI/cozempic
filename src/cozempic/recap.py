"""Generate a compact conversation recap from a session JSONL."""

from __future__ import annotations

import re
from pathlib import Path

from .helpers import atomic_write_text, get_dict_blocks, get_msg_type, text_of
from .types import Message

# Common words that don't make good theme labels
_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will "
    "would could should may might shall can need dare ought used to of in "
    "for on with at by from as into through during before after above below "
    "between out off over under again further then once here there when where "
    "why how all both each few more most other some such no nor not only own "
    "same so than too very just don now and but or if while because until it "
    "its this that these those i me my we us our you your he him his she her "
    "they them their what which who whom let see get got make go going went "
    "come take give say said know think want look use find tell ask work seem "
    "feel try leave call keep put show also well back even still way new one "
    "two first last long great little right old big high different small large "
    "next early young important sure ok yes yeah about up like can't don't "
    "doesn't didn't won't wasn't weren't isn't aren't hasn't haven't "
    "couldn't wouldn't shouldn't really thing things something anything "
    "everything nothing much many lot lots stuff already done doing made "
    "making went been able getting better lets check run set start "
    "change add read write open close update".split()
)


def _extract_text(msg: dict) -> str:
    """Extract readable text from a message, stripping system tags and noise."""
    parts = []
    for block in get_dict_blocks(msg):  # read-only: dict-only iterator (R5 non-dict leak)
        if block.get("type") == "text":
            parts.append(text_of(block))
    return " ".join(parts)


def _clean_user_text(text: str) -> str:
    """Remove system tags, command noise, and whitespace from user text."""
    # DoS guard: a pathological turn with thousands of literal <named-tag> opens
    # makes the lazy `.*?` named-tag regexes below O(text_len x n_tags) — measured
    # at >20s on the reload path for <system-reminder>x50000. Cap input to 32KB
    # first. That is ~450x the ~72-char snippet the recap ever displays and 4x the
    # 8000 generic-regex cap below, so a tag straddling the generic cap is still
    # caught, no shown content is lost, and the worst case drops to ~0.15s.
    text = text[:32768]
    # Strip NAMED system/command tags on the (capped) full text first so that an
    # injection tag whose close-tag lies past the 8000 cap below is still removed.
    # These regexes are anchored on literal tag names → linear on realistic input.
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
    text = re.sub(r"<local-command-caveat>.*?</local-command-caveat>", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-name>.*?</command-name>", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-message>.*?</command-message>", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-args>.*?</command-args>", "", text, flags=re.DOTALL)
    text = re.sub(r"<local-command-stdout>.*?</local-command-stdout>", "", text, flags=re.DOTALL)
    # Cap before the GENERIC tag regexes (the true O(n²) ones). Recap displays
    # ~72-char snippets so the cap loses no displayed info.
    text = text[:8000]
    text = re.sub(r"<[^>]+>.*?</[^>]+>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+/?>", "", text)
    text = re.sub(r"SessionStart:.*", "", text)
    text = re.sub(r"\[Request interrupted by user.*?\]", "", text)
    text = re.sub(r"[▖▗▘▝▚▞]+", "", text)
    text = re.sub(r"Claude Code v[\d.]+", "", text)
    text = re.sub(r"Opus \d+\.\d+ · Claude \w+", "", text)
    text = re.sub(r"~/Documents/\S+", "", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_len: int = 70) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    if max_len < 3:
        return text[:max(max_len, 0)]
    return text[: max_len - 3] + "..."


def _extract_themes(topics: list[str], max_themes: int = 5) -> list[tuple[str, int]]:
    """Extract theme clusters from topics using keyword frequency.

    Returns a list of (theme_label, count) tuples, sorted by count descending.
    Uses greedy set-cover: each theme covers topics not yet claimed by a
    higher-ranked theme, so counts reflect unique topic coverage.
    """
    # Map each keyword to the set of topic indices it appears in
    word_to_topics: dict[str, set[int]] = {}

    for i, topic in enumerate(topics):
        words = {w.rstrip("_-") for w in re.findall(r"[a-z][a-z_-]+", topic.lower())}
        words -= _STOP_WORDS
        words = {w for w in words if len(w) > 2}
        for word in words:
            word_to_topics.setdefault(word, set()).add(i)

    # Sort candidates by how many topics they cover
    candidates = sorted(word_to_topics.items(), key=lambda x: len(x[1]), reverse=True)

    # Greedy cover: pick themes that cover the most uncovered topics
    covered: set[int] = set()
    result: list[tuple[str, int]] = []
    for word, topic_ids in candidates:
        new_coverage = topic_ids - covered
        if len(new_coverage) >= 2:  # Only show themes with 2+ topics
            result.append((word, len(topic_ids)))
            covered |= topic_ids
            if len(result) >= max_themes:
                break

    return result


def generate_recap(messages: list[Message], max_turns: int = 40) -> str:
    """Generate a compact conversation recap.

    Shows: exchange count, theme clusters, recent topics, and where things left off.
    Target: ~12-16 lines.

    max_turns: how many most-recent user turns to consider for topic analysis
    (default 40). Older turns beyond max_turns are excluded from topics and
    theme extraction, but the exchange count still reflects the full session.
    """
    all_user_turns: list[str] = []
    last_assistant: str = ""

    for _, msg, _ in messages:
        msg_type = get_msg_type(msg)

        if msg_type == "user":
            text = _extract_text(msg)
            text = _clean_user_text(text)
            if text and len(text) >= 3:
                all_user_turns.append(text)

        elif msg_type == "assistant":
            text = _extract_text(msg)
            text = _clean_user_text(text)
            # Only update last_assistant when text is meaningful after cleaning;
            # all-tag turns (tool noise) clean to "" and are intentionally skipped,
            # so last_assistant naturally holds the prior meaningful turn.
            if text and len(text) >= 3:
                last_assistant = text

    if not all_user_turns:
        return ""

    total_exchanges = len(all_user_turns)

    # Limit topic analysis to the most-recent max_turns user turns.
    # Guard max_turns > 0: list[-0:] == list[0:] (full list), so zero must
    # be treated as "show nothing" rather than "show everything".
    user_turns = (
        all_user_turns[-max_turns:]
        if max_turns > 0 and len(all_user_turns) > max_turns
        else (all_user_turns if max_turns > 0 else [])
    )

    # Deduplicate: keep unique user requests (by first 40 chars)
    seen: set[str] = set()
    unique_topics: list[str] = []
    for turn in user_turns:
        key = turn[:40].lower()
        if key not in seen:
            seen.add(key)
            unique_topics.append(turn)

    # --- Build summary ---
    lines = [
        "",
        "  PREVIOUSLY ON THIS SESSION",
        f"  {total_exchanges} exchanges | {len(unique_topics)} topics",
        "",
    ]

    # Theme clusters (only if enough topics to be useful)
    if len(unique_topics) >= 6:
        themes = _extract_themes(unique_topics, max_themes=5)
        if themes:
            theme_parts = [f"{label} ({count})" for label, count in themes]
            lines.append(f"  Themes: {', '.join(theme_parts)}")
            lines.append("")

    # Recent topics (last 8, shown most-recent-first)
    max_recent = 8
    if len(unique_topics) <= max_recent:
        recent = list(reversed(unique_topics))
    else:
        recent = list(reversed(unique_topics[-max_recent:]))

    lines.append("  Recent:")
    for topic in recent:
        lines.append(f"  - {_truncate(topic)}")

    if len(unique_topics) > max_recent:
        earlier = len(unique_topics) - max_recent
        lines.append(f"  ... +{earlier} earlier")

    # Last thing that happened
    if last_assistant:
        lines.append("")
        lines.append(f"  Last: {_truncate(last_assistant, 72)}")

    lines.append("")
    return "\n".join(lines)


def save_recap(messages: list[Message], dest: Path, max_turns: int = 40) -> Path:
    """Generate and save recap to a file. Returns the path."""
    recap = generate_recap(messages, max_turns)
    atomic_write_text(dest, recap)
    return dest
