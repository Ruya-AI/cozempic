# Real harness fixtures (redacted)

These `.jsonl` files are **redacted captures of real Claude Code harness output**,
used to ground-truth the reload-gate marker matchers against reality instead of
against our own assumptions (QA fleet lens **L0**). The marker *strings* are
verbatim from a real session (2026-06-09); agent ids, prompts, prose output, and
paths are redacted to placeholders. The structural markers — what the detectors
key on — are preserved exactly.

| Fixture | Real shape captured | `safe_to_reload` must return |
|---|---|---|
| `live_team.jsonl` | Background `Agent` **launch**, in-flight: `Async agent launched successfully.\nagentId: …` (no completion trailer) | **`False`** (defer — agents are live) |
| `finished_team.jsonl` | Foreground `Agent` **completion**: `agentId: … (use SendMessage … to continue this agent)` + `<usage>… duration_ms: N</usage>` | **`True`** (reload — agents are done) |
| `idle_team.jsonl` | Harness **idle-notification carrier**: top-level `teamName` field (H-1); `<teammate-message>` body ending with `{"type":"idle_notification",…}` | **`True`** (reload — teammate is quiescent) |

Why they matter: the `duration_ms` usage trailer is the ground-truth signal that a
foreground `Agent` **finished**. Before 1.8.25 the extractor marked every
`Agent`-spawned teammate `running` and never cleared a foreground completion, so a
session that used `Agent` subagents would over-block (the guard never reloads →
context bloats → the failure cozempic prevents). The synthetic fixtures missed it
because they used the team-spawn ack format, which has no `duration_ms` trailer.

Asserted by `tests/test_reload_gate_contract.py::TestRealHarnessFixtures` (which
**fails loudly / skips** if a fixture is missing, so the coverage gap stays visible).

The `idle_team.jsonl` fixture specifically corroborates the H-1 claim (teamName is the
carrier-authenticity discriminator, 220/220 genuine carriers have it). Its `teamName`
field is verbatim from a real session; the `<teammate-message>` body and report prose
are redacted.

To refresh from a new capture: take a real agent session transcript, keep the
harness marker lines verbatim, replace ids/prompts/output/paths with placeholders,
and confirm the two assertions above still hold.
