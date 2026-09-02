# Anime Trivia Automation

Windows client for Anime Soul trivia in Discord. It watches the calibrated primary-monitor region at 60 FPS, recognizes the newest card, uses reviewed history first, resolves genuinely new clues with Gemini 3.7 Flash and retrieval-grounded local Qwen3.8 in parallel, and atomically submits one complete answer only after the same card turns green.

Version 0.8.0 closes the failures proven by the 6 PM round. Slow inference no longer blocks the OCR/state thread, so a correct result survives the red-to-green transition instead of timing itself out. Ungrounded Gemini 3.7 Flash now runs beside local Qwen for text and cropped visual clues through a hard-deadline, structured-output provider; its key is read only from `GEMINI_API_KEY`. Answers stay in memory until green, then the complete value is committed to the exact empty composer in one UI Automation operation—partial prefixes such as the observed `Lego` can no longer be produced. Semantic bot reveals are now the primary learning path. The complete bot-confirmed 6 PM transcript expands reviewed history to 148 entries and the canonical answer catalog to 227.

```text
DXcam 60 FPS physical-pixel crop
  -> CUDA change/stability gate
  -> PaddleOCR GPU + newest-card/readiness extraction
  -> Discord UI Automation semantic clue read
       -> reviewed 148-pair history (exact text + exact emoji)
       -> fuzzy text cache / strict pHash cache
       -> unseen clue: concurrent Gemini 3.7 + hot Qwen3.8 retrieval/verifier
       -> visual clue: Gemini receives only the cropped clue image
       -> unresolved/low-confidence: no submission; wait for paired reveal
  -> claim one empty #💜anime-chat composer through UI Automation
  -> hold the complete answer in memory while red
  -> on green, atomically set and verify the complete composer value
  -> press Enter only under the same fingerprinted green-state lock
```

## What changed after the failed live runs

- Removed all Qwen-generated and incorrectly associated reveal entries from the mutable cache.
- Corrected the complete 6 PM round from the bot's own reveal messages.
- Mined all 120 cards from the four available days, including a 3-question mini-round, and paired them conflict-free with Anime Soul's bot reveals without using a Discord token or API.
- Added semantic accessibility lookup, which reads the actual Unicode emoji sequence instead of asking OCR or a vision model to identify it.
- Semantic clue and reveal reads can now inspect the one unambiguous Discord window in the background without activating, focusing, or typing into it.
- Novel Gemini/Qwen work runs outside the scene thread, so OCR continues to process readiness changes while models work.
- Gemini 3.7 Flash is preflighted at startup, uses low thinking and strict structured output, performs no live retry or Search grounding, and fails closed without delaying local Qwen.
- Proof boundary for v0.8.0: real Gemini text generation is verified, while the replayed Q10 inline-image request was accepted by the provider path but Google returned a free-tier `RateLimitError`; live multimodal answer accuracy remains unproven until quota resets, and that lane fails closed meanwhile.
- Replaced raw model guessing with retrieval-grounded Qwen3.8. Raw Qwen confidently missed several September 1 character clues; evidence retrieval plus verification resolved the five novel clues not promoted into history as Yuki Sohma, Dragon Ball Z, Akane Tsunemori, Initial D, and InuYasha in 3.4–3.7 seconds each.
- The app owns a loopback-only llama.cpp Q6 server, warms it before capture, and stops only that owned process on F12/Ctrl+C. Q4 KV cache keeps combined OCR/model desktop usage near 27.2 GiB on the 32 GiB RTX 5090.
- Added a locked-Q1 session-start latch so historical green cards cannot reactivate a completed or newly launched worker.
- Treats short accessible quotes such as “Sit, boy!” as text for reveal learning even when OCR geometry labels their crop visual.
- Replaced “Discord is foreground” with exact composer ownership: the only accepted editor is `Message #💜anime-chat`, it must be empty, and its content must equal the macro-owned prefix before each key.
- Discord remains untouched during the red reading window. At green the full answer is written atomically, verified, and submitted; character-by-character partial drafts were removed.
- If Chrome/Gemini is foreground when an answer resolves, the panel shows `WAITING DISCORD` with the answer and holds the task until Discord manually returns, the round closes, the clue changes, F12 is pressed, or the 55-second safety bound expires.
- Closed cards return before lookup or submission. Post-quiz scrolling is inert.
- Reveal learning is one transaction per live round: it records the pre-existing reveal baseline, requires a witnessed green state, same-card continuity, a nearby official result marker, and exactly one new answer.

## Repository layout

- `src/anime_trivia_automation/capture.py` — DXcam ownership and CUDA frame-change/stability gating.
- `src/anime_trivia_automation/ocr.py` — PaddleOCR GPU inference, active-card isolation, clue crop, and red/green/closed classification.
- `src/anime_trivia_automation/discord.py` — direct Windows UI Automation access to the semantic question and exact Discord composer.
- `src/anime_trivia_automation/cache.py` — authoritative history, fuzzy text, strict pHash, semantic clues, and atomic JSON persistence.
- `src/anime_trivia_automation/typing.py` — draft-before-green state machine, composer ownership, human-interference detection, and F12 stop.
- `src/anime_trivia_automation/status.py` — structured low-frequency operator events, counters, heartbeat, and atomic status persistence.
- `src/anime_trivia_automation/status_window.py` — passive top-right status panel using Windows no-activate/click-through styles.
- `src/anime_trivia_automation/novel.py` — managed llama.cpp lifecycle, diverse retrieval, Qwen3.8 synthesis, verification, canonicalization, and per-session caching.
- `src/anime_trivia_automation/gemini.py` — hard-deadline Gemini 3.7 structured text/image provider with secure key lookup and disabled-by-default Flash-Lite scout.
- `src/anime_trivia_automation/knowledge.py` — read-only exact-quote and FTS5 access to the local source-attributed anime index.
- `src/anime_trivia_automation/vlm.py` — experimental local-model resolver; live submission is disabled in config.
- `data/trivia_history.seed.json` — 148 reviewed clue→answer pairs with explicit provenance on the September 1 noon and 6 PM entries.
- `data/answer_catalog.seed.json` — 227 bot-observed or accepted-live canonical answer strings used for spelling canonicalization.
- `data/trivia_cache.seed.json` — reviewed text/pHash starter cache.
- `data/trivia_cache.json` — ignored mutable cache, repaired from the reviewed seed on launch.
- `scripts/replay_screenshots.py` — one-warmup offline replay of saved cards.
- `scripts/build_anime_knowledge.py` — streamed, atomic index build from the private local AniList, quote, and Manami source files.
- `docs/KNOWLEDGE_SOURCE_AUDIT.md` — detailed schema, size, freshness, license, and inclusion audit for all investigated datasets.

## Install

From the repository root in PowerShell:

```powershell
Set-Location D:\11_CS\00_REPOS\anime-trivia-automation
& .\scripts\install_windows.ps1
Copy-Item .\config.example.json .\config.json
```

The installed workstation config is already calibrated and should not be overwritten unless recalibrating. The Windows installer creates `.venv`, installs the checkout in editable mode, and uses the shared CUDA PyTorch runtime for PaddleOCR and the frame gate.

The installed workstation config enables the `novel` lane and points at its verified Qwen3.8-27B Q6 GGUF plus pinned llama.cpp CUDA runtime. It also enables Gemini 3.7 when `GEMINI_API_KEY` is present in the user environment; the key is never stored in JSON or the repository. The portable example still requires its own key and local-model paths. The older in-process `vlm` lane remains disabled.

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

Run the full app with typing disabled:

```powershell
.\.venv\Scripts\anime-trivia.exe --config .\config.json --dry-run
```

Replay saved cards through one warmed OCR/cache process:

```powershell
.\.venv\Scripts\python.exe .\scripts\replay_screenshots.py --config .\config.json `
  "C:\path\to\Screenshot 2026-08-31 180030.png" `
  "C:\path\to\Screenshot 2026-08-31 180059.png"
```

Run the targeted state/composer regressions:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Run live

Double-click `Start Anime Trivia.cmd`, or run:

```powershell
.\.venv\Scripts\anime-trivia.exe --config .\config.json
```

It is safe to start early. Leave the launcher open and keep Anime Soul visible. The Gemini API is integrated, so normal operation no longer needs browser screenshot/paste work. Manual research remains available as a fallback; if another app is foreground when an answer resolves, the panel retains it until Discord returns. The app refuses Discord search, another channel, a nonempty editor, or an editor modified by a human.

Manual text always wins: if you have already begun typing in Discord, the app leaves it untouched and does not press Enter. `WAITING GREEN` means the answer exists only in memory and Discord is untouched. `DRAFTING` now means one complete value has been atomically staged after green and Enter is imminent; any divergent user edit blocks Enter without being erased. Press F12 at any time to stop.

## Operator status panel

Live and dry-run launches now open a passive panel at the top-right of the primary monitor. It is topmost, click-through, excluded from the taskbar, and marked `WS_EX_NOACTIVATE`, so it cannot take keyboard focus from Discord. Its default placement is outside the calibrated left-side capture region.

The panel reports:

- `LOADING` and `ARMED` startup state;
- question number, clue, and red/green/closed readiness;
- `KNOWN`, `RESOLVING`, `NOVEL`, or `UNKNOWN`, proposed answer, confidence, and source;
- `WAITING DISCORD`, `MANUAL`, `DRAFTING`, `WAITING GREEN`, and `SUBMITTED` execution state;
- reveal learning, errors, and session counters.

`runtime/operator_status.json` is the atomic machine-readable snapshot. It receives a one-second heartbeat; if the worker stops updating unexpectedly, the panel changes to `STALE`. On a normal F12/Ctrl+C stop it shows `STOPPED` and closes after four seconds. Configure placement, opacity, polling, topmost, and click-through behavior in the `status` section of `config.json`.

## Trust and learning rules

Resolution order is:

1. exact semantic history match (including emoji/ZWJ sequences);
2. fuzzy verified-history/text match with threshold and runner-up margin;
3. strict pHash match with Hamming-distance and ambiguity margins;
4. concurrent ungrounded Gemini 3.7 structured resolution and retrieval-grounded Qwen3.8 verification;
5. enabled providers finalize together; agreement or one answer plus abstention may submit, while differing answers always abstain;
6. otherwise abstain.

No novel model answer is written directly to the durable cache. It can serve only its current live round. A clue becomes durable only after the same round has progressed red/green→closed and the bot posts exactly one newly observed official reveal. Discord's semantic bot result is primary; spatial OCR is the fallback. Text and semantic clues are persisted atomically, and visual hashes are stored only from locked/ready frames.

The mutable cache schema remains JSON and now also supports semantic clues:

```json
{
  "schema_version": 1,
  "text_questions": {"normalized clue": "Answer"},
  "image_hashes": {"64-hex pHash": "Answer"},
  "semantic_questions": {"anime_title:emoji or exact clue": "Answer"},
  "metadata": {
    "text_questions": {},
    "image_hashes": {},
    "semantic_questions": {}
  }
}
```

Writes use a same-directory temporary file, flush and `fsync`, then atomically replace the live file.

## Performance

- DXcam captures the calibrated region at 60 FPS.
- CUDA thumbnail/tile comparison suppresses duplicate frames and stabilizes three frames before OCR.
- The colored accent prelocator keeps warmed active-card OCR near the previously measured 60–100 ms range.
- Authoritative history/cache lookup is effectively immediate.
- The managed Qwen3.8 Q6 server cold-starts in roughly 8–11 seconds and remains hot. On the September 1 text/quote and emoji clue set, accepted novel answers completed in approximately 3.2–4.7 seconds including local/web retrieval and a verifier pass.
- Humanized typing retains the configured 0.4–1.1 second pre-delay and 0.03–0.08 second per-key delay, but performs that work during red. Green→Enter therefore needs only the configured 60 ms slack and final safety checks.
- Low-confidence or verifier-rejected novel answers remain `UNKNOWN` and never reach the composer.

## Implementation references

- [DXcam 0.3](https://github.com/ra1nty/DXcam/blob/v0.3.0/README.md)
- [PaddleOCR 3.x OCR pipeline](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [Microsoft UI Automation](https://learn.microsoft.com/en-us/windows/win32/winauto/entry-uiauto-win32)
- [Microsoft extended window styles (`WS_EX_NOACTIVATE`)](https://learn.microsoft.com/en-us/windows/win32/winmsg/extended-window-styles)
- [Python Tkinter event loop and `after`](https://docs.python.org/3/library/tkinter.html)
- [Python ImageHash](https://github.com/JohannesBuchner/imagehash)
- [Qwen3-VL official repository and generation guidance](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3.8 official repository](https://github.com/QwenLM/Qwen3.8)
- [llama.cpp server and schema-constrained responses](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [MediaWiki full-text Search API](https://www.mediawiki.org/wiki/API:Search)
