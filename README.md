# Anime Trivia Automation

Windows client for Anime Soul trivia in Discord. It watches the calibrated primary-monitor region at 60 FPS, recognizes the newest card, answers from reviewed history first, resolves new semantic clues with account-authenticated Gemini 3.8, and types the first verified answer when the same card turns green.

Version 0.10.0 is the post-mortem of three lost quizzes (2026-09-01 07:00 and 12:00, 2026-09-02 07:00) and changes the operating policy accordingly:

- **Any live card is answered.** The locked-Question-1 latch is gone. It made the worker ignore every card of a quiz it joined mid-way (all of Q7–Q10 on 2026-09-02 after the launcher crashed and was restarted). Red or green cards are always the live round; only grey cards are inert.
- **First verified answer wins; later evidence can recover.** A wrong message triggers Discord's five-second slowmode, so follow-up guesses are spaced by at least five seconds. The first grounded answer to arrive is queued immediately; later distinct grounded answers can be tried only while the same card remains green, up to `typing.max_guesses_per_round`. Identical answers are never re-sent.
- **Confidence and evidence remain real gates.** Antigravity must return a high-confidence structured answer. Local Qwen is a disabled workstation fallback after its 2026-09-02 evaluation produced only 7/20 correct first answers; when enabled elsewhere, an answer must match independent retrieved evidence and ungrounded alternatives are discarded.
- **The measured provider is primary.** Account-authenticated Antigravity Gemini 3.8 handles Discord semantic text and emoji. The Gemini 3.8 Developer API is enabled only for a raw-image or account-provider-unavailable fallback; its circuit stops all later calls for five minutes after the first rate-limit error. Local Qwen submission remains disabled.
- **Works while you research manually.** If Chrome/Gemini is in front when the card turns green, the app waits until you have been input-idle for `typing.activation_idle_ms`, raises the one Discord window itself, sends the answer, then hands focus back to the previous window.
- **Real keystrokes at green.** The complete answer is typed with fast real keystrokes (`typing.composer_write_mode = "type"`, the mechanism that produced the only live submission so far) as soon as the same card is green; every character is verified against the composer, so a human edit cancels the draft without erasing anyone's text. The UI Automation write is still available as `"uia"`.
- **Every round is on disk.** `runtime/logs/anime-trivia-<stamp>.log` holds the full per-launch log and `runtime/round_ledger.jsonl` records every status event with timestamps, so a failed quiz can be reconstructed instead of guessed at.

```text
DXcam 60 FPS physical-pixel crop
  -> CUDA change/stability gate
  -> PaddleOCR GPU + newest-card/readiness extraction
  -> Discord UI Automation semantic clue read (exact text/emoji)
       -> reviewed 158-pair history (exact text + exact emoji)
       -> fuzzy text cache / strict pHash cache
       -> unseen semantic clue: Antigravity Gemini 3.8 Low
       -> raw image or account outage: one Gemini 3.8 Developer API fallback
       -> unresolved: warn and abstain; manual play stays available
  -> at green: raise Discord if needed (operator idle), claim the empty
     #💜anime-chat composer, type the complete answer, verify, Enter
  -> follow-up guesses after the gap while the card is still green
  -> learn the durable answer only from the bot's own reveal
```

## Repository layout

- `src/anime_trivia_automation/capture.py` — DXcam ownership and CUDA frame-change/stability gating.
- `src/anime_trivia_automation/ocr.py` — PaddleOCR GPU inference, active-card isolation, clue crop, and red/green/closed classification.
- `src/anime_trivia_automation/discord.py` — direct Windows UI Automation access to the semantic question, official reveals, and the exact Discord composer.
- `src/anime_trivia_automation/cache.py` — authoritative history, fuzzy text, strict pHash, semantic clues, and atomic JSON persistence.
- `src/anime_trivia_automation/app.py` — orchestration: scene processing, concurrent resolution with the guess ladder, reveal learning, shutdown.
- `src/anime_trivia_automation/typing.py` — green-gated keystroke/UIA commit, composer ownership, idle-gated Discord activation and focus restore, the multi-guess dispatcher, and F12 stop.
- `src/anime_trivia_automation/status.py` — structured operator events, counters, heartbeat, atomic status persistence, and the JSONL round ledger.
- `src/anime_trivia_automation/status_window.py` — passive top-right status panel using Windows no-activate/click-through styles.
- `src/anime_trivia_automation/novel.py` — managed llama.cpp lifecycle, exact quote table, local BM25 + web retrieval, one ranked Qwen3.8 synthesis with alternatives, canonicalization.
- `src/anime_trivia_automation/gemini.py` — hard-deadline Gemini Developer API provider with alternatives, emoji reading, secure key lookup, and a rate-limit circuit.
- `src/anime_trivia_automation/antigravity.py` — account-authenticated Gemini 3.8 Low CLI provider with structured output, emoji reading, environment isolation, output bounds, absolute deadlines, and owned process-tree shutdown.
- `src/anime_trivia_automation/knowledge.py` — read-only exact-quote and FTS5 access to the local source-attributed anime index.
- `src/anime_trivia_automation/vlm.py` — experimental in-process local VLM; live submission is disabled in config.
- `data/trivia_history.seed.json` — 158 reviewed clue→answer pairs with per-row provenance.
- `data/answer_catalog.seed.json` — 232 canonical answer strings used for spelling canonicalization.
- `data/trivia_cache.seed.json` — reviewed text/pHash starter cache; `data/trivia_cache.json` is the ignored mutable cache.
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

The workstation config enables the signed Antigravity CLI at `D:\11_CS\00_TOOLS\Antigravity\agy.exe` (`gemini-3.8-flash-low`) and validates its Google LLC Authenticode signature on every launch, so signed automatic CLI updates do not break the path. Gemini 3.8 Developer API is enabled only as the controlled fallback above; local Qwen submission is disabled. The API key is never stored in JSON or the repository, and Antigravity child processes never receive it.

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
5. at most one Gemini 3.8 Developer API call when a raw image lacks Discord semantics or the account provider is unavailable and its rate-limit circuit is closed;
6. only independently grounded local fallbacks may submit when explicitly enabled;
7. otherwise abstain.

No solver answer is written directly to the durable cache. A clue becomes durable only after the same round has progressed red/green→closed and the bot posts exactly one newly observed official reveal. Discord's semantic bot result is primary; spatial OCR is the fallback.

The mutable cache schema is JSON with text, pHash, and semantic maps plus per-entry metadata; writes use a same-directory temporary file, flush and `fsync`, then atomically replace the live file.

## Performance

- DXcam captures the calibrated region at 60 FPS; CUDA thumbnail/tile comparison suppresses duplicate frames and stabilizes three frames before OCR.
- The colored-accent prelocator keeps active-card OCR near 60–100 ms; closed grey cards fall back to the full band (400–700 ms).
- History/cache lookups are effectively immediate.
- Measured 2026-09-02: Antigravity answered five real text cards correctly in roughly 3.6–4.6 s when it returned before deadline. The local Qwen speed was 1–4 s but only 7/20 evaluated first answers were correct, so it is disabled on the workstation. A wasteful Developer API evaluation returned 0/24; batch evaluation is now blocked and the API is reserved for live fallback.
- Green→Enter for an answer already in hand: readiness OCR pass plus fast keystrokes (12–28 ms per character) plus the 60 ms slack.

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
