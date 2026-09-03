"""Turn the round ledger into a per-round quiz report.

Every lost quiz before 2026-09-02 was diagnosed from memory because nothing
was written down. The ledger now records every status event; this module
reduces it to one line per round that names the layer where the round ended:
resolved / not resolved, sent / not sent (and why), and what the bot revealed.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import normalize_question

_TERMINAL_REASONS = {
    "MANUAL": "not sent: text already in composer",
    "UNKNOWN": "not sent: no solver answered",
    "WAITING_DISCORD": "not sent: waited for Discord",
    "WAITING_GREEN": "not sent: never saw green",
    "ATTENTION": "not sent: attention",
}


@dataclass
class RoundReport:
    run_id: str
    question: str
    clue: str = ""
    first_seen: str = ""
    resolver: str = ""
    answers: list[str] = field(default_factory=list)
    resolve_ms: float | None = None
    submitted: list[str] = field(default_factory=list)
    submit_detail: str = ""
    reveal: str = ""
    last_phase: str = ""
    last_detail: str = ""
    _red_at: float | None = None

    @property
    def outcome(self) -> str:
        if self.submitted:
            unconfirmed = "did not clear" in self.submit_detail or "not retry" in self.submit_detail
            if unconfirmed:
                return "UNCONFIRMED (Enter sent, composer did not clear)"
            if self.reveal:
                want = normalize_question(self.reveal)
                for sent in self.submitted:
                    got = normalize_question(sent)
                    if want == got or want in got or got in want:
                        return "CORRECT (sent)"
                    want_tokens = {t for t in want.split() if len(t) >= 4}
                    if want_tokens & {t for t in got.split() if len(t) >= 4}:
                        return "CORRECT (sent)"
                return "WRONG (sent)"
            return "SENT (no reveal seen)"
        if self.answers:
            reason = _TERMINAL_REASONS.get(self.last_phase, f"not sent: {self.last_phase.lower()}")
            if self.reveal and any(
                normalize_question(a) == normalize_question(self.reveal) for a in self.answers
            ):
                return f"HAD IT, {reason}"
            return reason
        return _TERMINAL_REASONS.get(self.last_phase, "not resolved")


def load_rounds(ledger_path: Path) -> list[RoundReport]:
    rounds: "OrderedDict[tuple[str, str], RoundReport]" = OrderedDict()
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            phase = str(row.get("phase") or "")
            question = str(row.get("question") or "")
            if phase in {"LOADING", "ARMED", "STOPPING", "STOPPED", "QUIZ_COMPLETE", "ERROR"}:
                continue
            if not question or question == "—":
                continue
            key = (str(row.get("run_id")), question)
            if key not in rounds or (phase == "RED" and rounds[key].last_phase in {"CLOSED", "LEARNED"}):
                rounds[key] = RoundReport(run_id=key[0], question=question)
            item = rounds[key]
            ts = str(row.get("ts") or "")
            mono = float(row.get("monotonic") or 0.0)
            detail = str(row.get("detail") or "")
            answer = str(row.get("answer") or "")
            clue = str(row.get("clue") or "")
            if clue and clue not in {"—", "Visual / emoji clue"}:
                item.clue = clue
            if phase in {"RED", "GREEN"} and not item.first_seen:
                item.first_seen = ts
                item._red_at = mono
            elif phase == "NOVEL" or phase == "KNOWN":
                if answer and answer not in item.answers and answer != "—":
                    item.answers.append(answer)
                if not item.resolver:
                    item.resolver = str(row.get("source") or "")
                if item.resolve_ms is None and item._red_at is not None and mono:
                    item.resolve_ms = (mono - item._red_at) * 1000.0
            elif phase == "SUBMITTED":
                if answer and answer not in item.submitted and answer != "—":
                    item.submitted.append(answer)
                item.submit_detail = detail
            elif phase == "LEARNED":
                item.reveal = answer
            if phase not in {"LEARNED", "CLOSED"} or not item.last_phase:
                item.last_phase = phase
                item.last_detail = detail
            elif phase == "CLOSED" and item.last_phase in {"RED", "GREEN", "RESOLVING"}:
                item.last_phase = phase
                item.last_detail = detail
    return list(rounds.values())


def render_report(rounds: list[RoundReport], *, runs: int = 1) -> str:
    run_ids: list[str] = []
    for item in rounds:
        if item.run_id not in run_ids:
            run_ids.append(item.run_id)
    selected = set(run_ids[-runs:]) if runs > 0 else set(run_ids)
    lines: list[str] = []
    for run_id in [r for r in run_ids if r in selected]:
        items = [item for item in rounds if item.run_id == run_id]
        sent = sum(1 for item in items if item.submitted)
        correct = sum(1 for item in items if item.outcome.startswith("CORRECT"))
        had_it = sum(1 for item in items if item.outcome.startswith("HAD IT"))
        lines.append(f"## Run {run_id[:8]} — {items[0].first_seen[:16] if items else ''}")
        lines.append(
            f"rounds {len(items)} | resolved {sum(1 for i in items if i.answers)} | "
            f"sent {sent} | correct {correct} | had the answer but did not send {had_it}"
        )
        lines.append("")
        lines.append("| Q | clue | resolver | answer(s) | resolve s | sent | reveal | outcome |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for item in items:
            resolve = f"{item.resolve_ms / 1000.0:.1f}" if item.resolve_ms is not None else ""
            lines.append(
                "| {q} | {clue} | {res} | {ans} | {rs} | {sent} | {rev} | {out} |".format(
                    q=item.question,
                    clue=item.clue[:48].replace("|", "/"),
                    res=item.resolver.replace("|", "/"),
                    ans=" / ".join(item.answers)[:60].replace("|", "/"),
                    rs=resolve,
                    sent=" / ".join(item.submitted)[:40].replace("|", "/") or "-",
                    rev=item.reveal.replace("|", "/"),
                    out=item.outcome + (f" ({item.last_detail[:60]})" if not item.submitted and item.last_detail else ""),
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_quiz_report(ledger_path: Path, out_dir: Path, *, runs: int = 1) -> tuple[str, Path]:
    rounds = load_rounds(ledger_path)
    text = render_report(rounds, runs=runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"quiz-report-{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text(text, encoding="utf-8")
    return text, out


__all__ = ["RoundReport", "load_rounds", "render_report", "write_quiz_report"]
