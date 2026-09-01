# Anime Trivia Automation

Windows client for Anime Soul trivia in Discord. It watches the calibrated primary-monitor region at 60 FPS, recognizes the newest card, resolves only from verified local data, drafts the answer while the card is red, and presses Enter only after the same card turns green.

Version 0.5.0 is the post-live-incident repair. The first 6 PM run proved that capture, OCR, and red/green recognition worked, but a single unverified model guess was not a safe answer source. Model-generated submissions are now disabled by default. The primary resolver is a local, server-verified history containing 97 exact clue→answer pairs (including Unicode emoji clues), plus the text and pHash caches.

```text
DXcam 60 FPS physical-pixel crop
  -> CUDA change/stability gate
  -> PaddleOCR GPU + newest-card/readiness extraction
  -> Discord UI Automation semantic clue read
       -> authoritative 97-pair history (exact text + exact emoji)
       -> fuzzy text cache / strict pHash cache
       -> unknown: no submission; wait for the bot's paired reveal
  -> claim one empty #💜anime-chat composer through UI Automation
  -> humanized draft while red
  -> verify exact macro-owned draft before every character and Enter
  -> press Enter only when the same card has the green outline
```

## What changed after the failed live run

- Removed all Qwen-generated and incorrectly associated reveal entries from the mutable cache.
- Corrected the complete 6 PM round from the bot's own reveal messages.
- Mined 97 conflict-free question/reveal pairs from Anime Soul's indexed Discord history without using a Discord token or API.
- Added semantic accessibility lookup, which reads the actual Unicode emoji sequence instead of asking OCR or a vision model to identify it.
- Disabled unverified model submissions. Qwen3-VL-32B, Qwen3.8-27B Q6, and Gemma 4 31B were tested against the exact failed round and were not accurate enough to authorize live answers.
- Replaced “Discord is foreground” with exact composer ownership: the only accepted editor is `Message #💜anime-chat`, it must be empty, and its content must equal the macro-owned prefix before each key.
- Drafts are typed during the five-second red reading window. Enter remains blocked until green, preserving the timing rule without losing long answers to the observed 0.8–2.0 second winners.
- Closed cards return before lookup or submission. Post-quiz scrolling is inert.
- Reveal learning is one transaction per live round: it records the pre-existing reveal baseline, requires a witnessed green state, same-card continuity, a nearby official result marker, and exactly one new answer.

## Repository layout

- `src/anime_trivia_automation/capture.py` — DXcam ownership and CUDA frame-change/stability gating.
- `src/anime_trivia_automation/ocr.py` — PaddleOCR GPU inference, active-card isolation, clue crop, and red/green/closed classification.
- `src/anime_trivia_automation/discord.py` — direct Windows UI Automation access to the semantic question and exact Discord composer.
- `src/anime_trivia_automation/cache.py` — authoritative history, fuzzy text, strict pHash, semantic clues, and atomic JSON persistence.
- `src/anime_trivia_automation/typing.py` — draft-before-green state machine, composer ownership, human-interference detection, and F12 stop.
- `src/anime_trivia_automation/vlm.py` — experimental local-model resolver; live submission is disabled in config.
- `data/trivia_history.seed.json` — 97 server-verified clue→answer pairs.
- `data/answer_catalog.seed.json` — 207 unique server-observed canonical answer strings for future constrained retrieval work.
- `data/trivia_cache.seed.json` — reviewed text/pHash starter cache.
- `data/trivia_cache.json` — ignored mutable cache, repaired from the reviewed seed on launch.
- `scripts/replay_screenshots.py` — one-warmup offline replay of saved cards.

## Install

From the repository root in PowerShell:

```powershell
Set-Location D:\11_CS\00_REPOS\anime-trivia-automation
& .\scripts\install_windows.ps1
Copy-Item .\config.example.json .\config.json
```

The installed workstation config is already calibrated and should not be overwritten unless recalibrating. The Windows installer creates `.venv`, installs the checkout in editable mode, and uses the shared CUDA PyTorch runtime for PaddleOCR and the frame gate.

The live default does **not** load a large language model. Optional model files may still be disk-warmed for experimentation, but `vlm.enabled` and `vlm.allow_unverified_submission` remain `false` for live use.

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

It is safe to start early. Leave the launcher open and keep Anime Soul's `#💜anime-chat` foregrounded during a round. The app locates and focuses the exact message composer itself; it refuses Discord search, another channel, a nonempty editor, or an editor modified by a human.

Do not type manually in the composer while the macro owns a draft. During red, other users may see Discord's normal “typing…” indicator, but nothing is submitted. On green, the app rechecks the round, foreground window, focused editor, and complete draft before pressing Enter. Press F12 at any time to stop.

## Trust and learning rules

Resolution order is:

1. exact semantic history match (including emoji/ZWJ sequences);
2. fuzzy verified-history/text match with threshold and runner-up margin;
3. strict pHash match with Hamming-distance and ambiguity margins;
4. otherwise abstain.

No model guess is written to the cache. An unknown clue can be learned only after the same live round has progressed red/green→closed and the bot posts one newly observed, spatially associated `Correct!` or `Time's up! ... the answer was ...` result. Text and semantic clues are persisted atomically; visual hashes are stored only from locked/ready frames.

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
- Humanized typing retains the configured 0.4–1.1 second pre-delay and 0.03–0.08 second per-key delay, but performs that work during red. Green→Enter therefore needs only the configured 60 ms slack and final safety checks.
- The local-model slow path is intentionally not part of live latency because none of the installed models met the required accuracy on the actual round.

## Implementation references

- [DXcam 0.3](https://github.com/ra1nty/DXcam/blob/v0.3.0/README.md)
- [PaddleOCR 3.x OCR pipeline](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [Microsoft UI Automation](https://learn.microsoft.com/en-us/windows/win32/winauto/entry-uiauto-win32)
- [Python ImageHash](https://github.com/JohannesBuchner/imagehash)
- [Qwen3-VL official repository and generation guidance](https://github.com/QwenLM/Qwen3-VL)
