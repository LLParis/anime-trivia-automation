# Anime Trivia Automation

Windows client for Anime Soul trivia in Discord. It watches the calibrated primary-monitor region at 60 FPS, recognizes the newest card, answers from reviewed history first, resolves new clues with account-authenticated Gemini 3.8, and sends the first confident answer as real keyboard input the moment the same card turns green.

Version 0.10.3 hardens Claude's post-18:00 real-input repair for the next live run:

- **Any live card is answered.** The locked-Question-1 latch is gone. It made the worker ignore every card of a quiz it joined mid-way (all of Q7–Q10 on 2026-09-02 after the launcher crashed and was restarted). Red or green cards are always the live round; only grey cards are inert.
- **First verified answer wins; later evidence can recover.** A wrong message triggers Discord's five-second slowmode, so follow-up guesses are spaced by at least five seconds. The first grounded answer to arrive is queued immediately; later distinct grounded answers can be tried only while the same card remains green, up to `typing.max_guesses_per_round`. Identical answers are never re-sent.
- **Confidence and evidence remain real gates.** Antigravity must return a high-confidence structured answer. Local Qwen is a disabled workstation fallback after its 2026-09-02 evaluation produced only 7/20 correct first answers; when enabled elsewhere, an answer must match independent retrieved evidence and ungrounded alternatives are discarded.
- **The measured provider is primary.** Account-authenticated Antigravity Gemini 3.8 handles Discord semantic text and emoji. On this workstation both the quota-exhausted Gemini Developer API and the lower-accuracy local Qwen submission lane are disabled; an unseen raw image with no Discord semantic clue therefore abstains instead of pretending it has a working fallback.
- **Works while you research manually.** If Chrome/Gemini is in front when the card turns green, the app waits until you have been input-idle for `typing.activation_idle_ms`, raises the one Discord window itself, sends the answer, then hands focus back to the previous window.
- **The answer is real input, sent once, verified once.** At green the complete answer goes into the composer as one `SendInput` batch of Unicode key events (`typing.composer_write_mode = "type"`), which Discord's Slate editor honours the way it honours typing. The accessibility value is checked once after a settle window instead of per character: measured on the live window on 2026-09-02, the value trails the keystrokes by about 40 ms, which is why per-character checks stranded a one-letter prefix at noon, and the UI Automation ValuePattern write (still available as `"uia"`) put text in the box that Enter did not send at 18:00 (Q3, Q7).
- **Nothing is ever left stuck in the box.** After Enter the app waits up to a second for the composer to clear. A second Enter must pass the same exact green-round, fingerprint, window, focus, and ownership gate; otherwise it is canceled. An unchanged stuck answer is erased with select-all/backspace, but cleanup is reported as unconfirmed—not as a submission. Confirmed empty or cleanup releases all ownership so an identical later human draft is never erased.
- **Practice cannot post.** Rehearsal status is explicitly labeled `REHEARSAL`, never increments the submitted counter, and both card-painting tools refuse to run unless a fresh live rehearsal worker with the same run ID owns the status file. Repeated practice question numbers remain distinct in reports by round token.
- **Emoji cards are read even when Discord renders them differently.** The accessibility reader prefers the card's inner message group, which preserves emoji even when the outer Discord row drops them, then falls back to structural parsing (status word ... "Answer with the ...") and the bare emoji sequence. The inner-group path recovered the real 18:00 Q10 sequence in 127 ms after the quiz; exact reviewed pHashes remain the repeat-card fallback.
- **Every round is on disk.** `runtime/logs/anime-trivia-<stamp>.log` holds the full per-launch log and `runtime/round_ledger.jsonl` records every status event with timestamps, so a failed quiz can be reconstructed instead of guessed at.

```text
DXcam 60 FPS physical-pixel crop
  -> CUDA change/stability gate
  -> PaddleOCR GPU + newest-card/readiness extraction
  -> Discord UI Automation semantic clue read (exact text/emoji)
       -> reviewed 176-pair history (exact text + exact emoji)
       -> fuzzy text cache / strict pHash cache
       -> unseen semantic clue: Antigravity Gemini 3.8 Low
       -> raw image without semantics or pHash: abstain (API and local VLM off)
       -> unresolved: warn and abstain; manual play stays available
  -> at green: raise Discord if needed (operator idle), claim the empty
     #💜anime-chat composer, SendInput the complete answer, verify once, Enter,
     confirm the box cleared (second Enter / cleanup if not)
  -> follow-up guesses after the gap while the card is still green
  -> learn the durable answer only from the bot's own reveal
```

## Rehearsal: the whole live path on a real card, Enter withheld

```powershell
.\.venv\Scripts\anime-trivia.exe --config .\config.json --rehearse      # in one window
.\.venv\Scripts\python.exe .\scripts\rehearse_live.py --config .\config.json --card "C:\path\to\red-card.png"   # in another
```

`--rehearse` runs everything the live launcher runs (solver preflight, composer probe, capture, OCR, resolution, Discord activation, composer claim, typing, verification) and stops at Enter, leaving the typed answer in the box to inspect and delete. `rehearse_live.py` paints a saved red card inside the calibrated capture region and turns its accent green after 7 s. Measured on 2026-09-02 18:45 against the real Discord composer: red card read and resolved in 0.31 s, green flip noticed in 0.68 s, answer typed and verified 0.44 s after green.

An accent-strip watcher samples the live card's colour band on every captured frame and forces a re-read the moment it flips, so the green transition no longer depends on the thumbnail change gate noticing it (it did not, when only the strip changed).

## Operating rules (why four quizzes were lost, and what now prevents each)

1. **Nothing is "verified" until it ran against the real Discord editor.** Unit tests use fakes that update instantly and honour every write; the real editor lags real keystrokes by about 40 ms and ignores UI Automation writes. So every live launch now runs the production writer on the live composer (types and erases `ok`, measures the lag, widens the settle window accordingly) and refuses to arm if that fails. Nothing else counts as proof of the write path.
2. **Every solver must answer a real clue at launch.** Preflighting credentials is not enough (the API key passed preflight and then rate-limited every call). Each enabled cloud lane must answer a known quote within its live deadline or it is parked, and the app refuses to arm with zero working solvers.
3. **No gate may produce silence.** Wrong guesses are free in Anime Soul. Confidence floors, verifier passes, evidence-agreement gates, disagreement abstention, and the locked-Question-1 latch each turned a solved round into a lost one. A new gate must show, from the ledger, which lost round it would have won.
4. **Diagnose from the ledger, not from memory.** `anime-trivia --report` prints one line per round from `runtime/round_ledger.jsonl` and names the layer where the round ended (`HAD IT, not sent: text already in composer`, `UNCONFIRMED (Enter sent, composer did not clear)`, `not resolved`). Run it after every quiz before changing anything.
5. **Only committed, tested code runs a quiz.** The launcher warns when `src/` or `tests/` has uncommitted changes and prints the running commit. Two agents editing the tree during a quiz day is how a half-finished change that removed Enter reached a live run.

## Repository layout

- `src/anime_trivia_automation/capture.py` — DXcam ownership and CUDA frame-change/stability gating.
- `src/anime_trivia_automation/ocr.py` — PaddleOCR GPU inference, active-card isolation, clue crop, and red/green/closed classification.
- `src/anime_trivia_automation/discord.py` — direct Windows UI Automation access to the semantic question, official reveals, and the exact Discord composer.
- `src/anime_trivia_automation/cache.py` — authoritative history, fuzzy text, strict pHash, semantic clues, and atomic JSON persistence.
- `src/anime_trivia_automation/app.py` — orchestration: scene processing, concurrent resolution with the guess ladder, reveal learning, shutdown.
- `src/anime_trivia_automation/typing.py` — green-gated SendInput commit with settle verification, post-Enter clear confirmation and cleanup, composer ownership, idle-gated Discord activation and focus restore, the multi-guess dispatcher, and F12 stop.
- `src/anime_trivia_automation/windows_input.py` — one-call `SendInput` Unicode text writer (no clipboard, no per-character boundary).
- `src/anime_trivia_automation/status.py` — structured operator events, counters, heartbeat, atomic status persistence, and the JSONL round ledger.
- `src/anime_trivia_automation/status_window.py` — passive top-right status panel using Windows no-activate/click-through styles.
- `src/anime_trivia_automation/novel.py` — managed llama.cpp lifecycle, exact quote table, local BM25 + web retrieval, one ranked Qwen3.8 synthesis with alternatives, canonicalization.
- `src/anime_trivia_automation/gemini.py` — hard-deadline Gemini Developer API provider with alternatives, emoji reading, secure key lookup, and a rate-limit circuit.
- `src/anime_trivia_automation/antigravity.py` — account-authenticated Gemini 3.8 Low CLI provider with structured output, emoji reading, environment isolation, output bounds, absolute deadlines, and owned process-tree shutdown.
- `src/anime_trivia_automation/knowledge.py` — read-only exact-quote and FTS5 access to the local source-attributed anime index.
- `src/anime_trivia_automation/vlm.py` — experimental in-process local VLM; live submission is disabled in config.
- `data/trivia_history.seed.json` — 176 reviewed clue→answer pairs with per-row provenance.
- `data/answer_catalog.seed.json` — 237 canonical answer strings used for spelling canonicalization.
- `data/trivia_cache.seed.json` — 21 reviewed text keys and 31 reviewed pHashes; `data/trivia_cache.json` is the ignored mutable cache.
- `scripts/eval_resolvers.py` — offline accuracy/latency report of a provider over the reviewed history (no Discord, no keyboard).
- `scripts/replay_screenshots.py` — one-warmup offline replay of saved cards.
- `scripts/build_anime_knowledge.py` — streamed, atomic index build from the private local AniList, quote, and Manami source files.
- `docs/KNOWLEDGE_SOURCE_AUDIT.md` — schema, size, freshness, license, and inclusion audit for all investigated datasets.

## Install

From the repository root in PowerShell:

```powershell
Set-Location D:\11_CS\00_REPOS\anime-trivia-automation
& .\scripts\install_windows.ps1
Copy-Item .\config.example.json .\config.json
```

The installed workstation config is already calibrated and should not be overwritten unless recalibrating. The Windows installer creates `.venv`, installs the checkout in editable mode, and uses the shared CUDA PyTorch runtime for PaddleOCR and the frame gate.

The workstation config enables the signed Antigravity CLI at `D:\11_CS\00_TOOLS\Antigravity\agy.exe` (`gemini-3.8-flash-low`) with a 12-second absolute answer deadline and validates its Google LLC Authenticode signature on every launch, so signed automatic CLI updates do not break the path. Gemini 3.8 Developer API and local Qwen submission are both disabled on the workstation. The dormant API key is never stored in JSON or the repository, and Antigravity child processes never receive it.

## Calibrate the capture region

1. Put Discord on the primary monitor with a complete Anime Soul card visible.
2. Run:

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\calibrate_region.py --config .\config.json
   ```

3. Drag a stable feed band containing the entire newest card: header, colored left accent, clue, answer instruction, and status/footer.
4. Do not move, resize, zoom, or relocate Discord after calibration.

DXcam regions are physical `(left, top, right, bottom)` pixels with exclusive right/bottom values. `output_idx: null` means the primary output.

## Verify without sending keys

Full app with typing disabled:

```powershell
.\.venv\Scripts\anime-trivia.exe --config .\config.json --dry-run
```

Local-provider accuracy on reviewed clues (no Discord and no paid/account API calls):

```powershell
.\.venv\Scripts\python.exe .\scripts\eval_resolvers.py --config .\config.json --provider qwen --limit 24
```

The Gemini Developer API evaluator is hard-capped to one call per invocation. Never use batch evaluation against live Developer API or Antigravity account quota.

Replay saved cards through one warmed OCR/cache process:

```powershell
.\.venv\Scripts\python.exe .\scripts\replay_screenshots.py --config .\config.json "C:\path\to\Screenshot.png"
```

Unit and regression tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Run live

Double-click `Start Anime Trivia.cmd`, or run:

```powershell
.\.venv\Scripts\anime-trivia.exe --config .\config.json
```

Start it a few minutes early and leave the launcher open. Joining mid-quiz is fine: the next red or green card is answered. You may research manually in Chrome at the same time; if Chrome is in front when the card turns green and you pause typing for a third of a second, the app raises Discord, sends, and returns focus to Chrome. If you have already typed into the Discord composer, the app leaves your text alone and skips that round. Press F12 at any time to stop.

Panel states: `ARMED` → `RED` (resolving) → `KNOWN`/`NOVEL` (answer in hand, follow-up guesses listed) → `WAITING GREEN` → `DRAFTING` (answer typed after green) → `SUBMITTED` (Enter sent, `guess N` for follow-ups) → `CLOSED`/`LEARNED`. `UNKNOWN` means every solver abstained. `WAITING DISCORD` means Chrome is in front and you are still typing.

## Operator status panel

Live and dry-run launches open a passive panel at the top-right of the primary monitor. It is topmost, click-through, excluded from the taskbar, and marked `WS_EX_NOACTIVATE`, so it cannot take keyboard focus from Discord. `runtime/operator_status.json` is the atomic machine-readable snapshot with a one-second heartbeat; if the worker stops updating, the panel changes to `STALE`. On F12/Ctrl+C it shows `STOPPED` and closes after four seconds.

## Trust and learning rules

Resolution order for a live card:

1. exact semantic history match (including emoji/ZWJ sequences);
2. fuzzy verified-history/text match with threshold and runner-up margin;
3. strict pHash match with Hamming-distance and ambiguity margins;
4. account-authenticated Antigravity Gemini 3.8 Low for a new Discord-semantic clue;
5. optional configured fallbacks may run only when explicitly enabled (both Developer API and local Qwen are off on this workstation);
6. otherwise abstain.

No solver answer is written directly to the durable cache. A clue becomes durable only after the same round has progressed red/green→closed and the bot posts exactly one newly observed official reveal. Discord's semantic bot result is primary; spatial OCR is the fallback.

The mutable cache schema is JSON with text, pHash, and semantic maps plus per-entry metadata; writes use a same-directory temporary file, flush and `fsync`, then atomically replace the live file.

## Performance

- DXcam captures the calibrated region at 60 FPS; CUDA thumbnail/tile comparison suppresses duplicate frames and stabilizes three frames before OCR.
- The colored-accent prelocator keeps active-card OCR near 60–100 ms; closed grey cards fall back to the full band (400–700 ms).
- History/cache lookups are effectively immediate.
- Measured 2026-09-02: Antigravity answered all 11 real semantic cards it attempted correctly across the noon and 18:00 quizzes, with 3.2–5.0 s provider latency. A direct 40-clue review scored 36/40 exact first answers (37/40 when the equivalent `Mushi-Shi` spelling is counted), median 3.57 s; two emoji clues hit the 12-second deadline. Local Qwen produced only 7/20 exact first answers, so it is disabled. The Developer API evaluation was quota/runtime blocked (22 errors and two timeouts), not an intelligence result, and that API is disabled on the workstation.
- Real Discord integration probe on 2026-09-02: complete-value acknowledgment took 16–22 ms and verified clearing took 15–17 ms across three focused round trips; no Enter was pressed. Live Green→Enter also includes the readiness OCR pass and 60 ms ownership slack.

## Implementation references

- [DXcam 0.3](https://github.com/ra1nty/DXcam/blob/v0.3.0/README.md)
- [PaddleOCR 3.x OCR pipeline](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [Microsoft UI Automation](https://learn.microsoft.com/en-us/windows/win32/winauto/entry-uiauto-win32)
- [Microsoft `SetForegroundWindow` restrictions](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow)
- [Microsoft `GetLastInputInfo`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getlastinputinfo)
- [Python ImageHash](https://github.com/JohannesBuchner/imagehash)
- [Qwen3.8 official repository](https://github.com/QwenLM/Qwen3.8)
- [llama.cpp server and schema-constrained responses](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [MediaWiki full-text Search API](https://www.mediawiki.org/wiki/API:Search)
