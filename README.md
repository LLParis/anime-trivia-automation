# Anime Trivia Automation

This Windows client watches one fixed region of the primary display at 60 FPS, isolates the newest Anime Soul trivia hint, resolves it from a local text/pHash cache or a local Qwen3-VL model, and types the answer through `pynput` with humanized timing.

This is intentionally a repo-local Windows application, not a standalone wheel. The supported installation path is `scripts/install_windows.ps1`, which creates `.venv` and installs this checkout in editable mode so its config, seed cache, calibration tool, and model warmer stay together.

The implementation is tailored to the supplied examples. Emoji rounds still contain OCR-visible text such as “Anime Guessing Game” and “Answer with…”, so the code does **not** treat “any OCR text” as a text clue. It uses OCR boxes to crop only the band between the card header and answer instruction. If that band has meaningful words it follows the text path; otherwise it hashes and sends the emoji/image band to the visual path. Novel prose clues also use Qwen3-VL on a cache miss because randomized new clues cannot be answered by a finite starter cache.

```text
DXcam 60 FPS BGR crop
  -> CUDA change + 3-frame stability gate
  -> size-one latest-scene mailbox
  -> PaddleOCR GPU + newest-card/hint extraction
       -> solve/cache while red
       -> 6px green-outline component gate
       -> 0.4–1.1s human delay + guarded typing
            -> fuzzy text cache
            -> pHash image cache
            -> Qwen3-VL NF4 fallback -> atomic cache
```

The cache is seeded with all fourteen unambiguous prose/quote examples and the three supplied emoji rounds. Novel visual clues fail closed by default instead of submitting an unverified model guess; when the bot later posts “the answer was …”, the app learns that authoritative answer for the next recurrence.

## Repository layout

- `src/anime_trivia_automation/capture.py`: DXcam ownership and CUDA frame-change/stability gating.
- `src/anime_trivia_automation/ocr.py`: PaddleOCR 3.7 GPU inference and Anime Soul hint extraction.
- `src/anime_trivia_automation/cache.py`: RapidFuzz text matching, strict pHash matching, and atomic JSON writes.
- `src/anime_trivia_automation/vlm.py`: lazy, quantized Qwen3-VL fallback for text and visual clues.
- `src/anime_trivia_automation/typing.py`: Discord foreground guard, duplicate/stale-scene suppression, humanized typing, and F12 stop.
- `config.example.json`: every runtime/calibration setting.
- `data/trivia_cache.seed.json`: tracked starter answers; first launch copies it to the ignored mutable `data/trivia_cache.json`.

## 1. Install on the RTX 5090 workstation

Use PowerShell from the repository root:

```powershell
Set-Location D:\11_CS\00_REPOS\anime-trivia-automation
& .\scripts\install_windows.ps1
Copy-Item .\config.example.json .\config.json
```

The installer uses one shared GPU runtime:

- PyTorch 2.13 CUDA 13.0 for current stable Blackwell/sm_120 support.
- PaddleOCR 3.7 uses its supported `transformers` engine on that same PyTorch runtime, avoiding incompatible duplicate Windows cuDNN stacks.
- DXcam 0.3.0, PaddleX 3.7.2, Transformers 5.16.1, and the remaining pinned dependencies come from PyPI.

Pre-download and disk-warm the OCR and VLM models before a live round:

```powershell
.\.venv\Scripts\python.exe .\scripts\warm_models.py --config .\config.json
```

`Qwen/Qwen3-VL-32B-Instruct` is the accuracy-first local knowledge model. Its source weights are large, but NF4 lets the live model fit alongside PaddleOCR on the 32 GB RTX 5090. It resolves novel prose/character clues; raw novel emoji submissions remain gated off until locally verified or learned from an authoritative round reveal. The warm process exits after populating disk caches; the live process still loads the model into VRAM and reaches READY **before** arming capture. Set `vlm.local_files_only` to `true` after the download if live operation must never touch the network.

## 2. Calibrate the DXcam bounding box

1. Put Discord on the primary monitor and leave a complete Anime Soul card visible.
2. Run:

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\calibrate_region.py --config .\config.json
   ```

3. Drag a stable feed band large enough for the **tallest** expected newest card, including “Anime Guessing Game”, the colored left accent, the hint, “Answer with…”, and the question/status line. Exclude the chat input and unrelated sidebars. Normal chat messages may scroll above or below the card: the extractor anchors on the bottommost complete “Anime Guessing Game” card, validates its internal geometry, and ignores chatter.
4. The script atomically updates `capture.region`, sets `capture.calibrated` to `true`, and copies the bare `[left, top, right, bottom]` array. Live startup is blocked while `calibrated` is false.

DXcam regions are `(left, top, right, bottom)`, not `(x, y, width, height)`. `right` and `bottom` are exclusive. With `output_idx: null`, DXcam selects the primary output. To inspect all DXGI outputs:

```powershell
.\.venv\Scripts\anime-trivia.exe --print-outputs
```

For a non-primary output, set its index and use coordinates local to that output. The process enables physical-pixel DPI awareness so Windows display scaling does not silently shift the crop.

Normally leave `prompt.static_hint_roi` as `null`; OCR boxes dynamically find the variable-height hint band. If the card design changes and header/footer OCR becomes unreliable, it can be set to a second `[left, top, right, bottom]` region **relative to the capture crop**.

## 3. Prove recognition before enabling typing

Analyze one of the saved examples without sending keys:

```powershell
.\.venv\Scripts\anime-trivia.exe --config .\config.json --inspect-image "C:\Users\sirlo\OneDrive\Imágenes\Screenshots\Screenshot 2026-08-31 070005.png"
```

Then run the real 60 FPS pipeline in dry-run mode:

```powershell
.\.venv\Scripts\anime-trivia.exe --config .\config.json --dry-run
```

The console reports the extracted clue, expected answer type, cache/VLM source, OCR time, hash/lookup time, and the answer it would submit. Press F12 or Ctrl+C to stop.

## 4. Run live

After installation and calibration, Windows users can simply double-click `Start Anime Trivia.cmd`. Double-click `Test Anime Trivia.cmd` for a no-keystroke dry run.

The equivalent command-line launch is:

```powershell
.\.venv\Scripts\anime-trivia.exe --config .\config.json
```

Keep the Discord message box focused and reserve that editor for the macro during a round. Before every character and before Enter, the executor verifies that the foreground process is `Discord.exe`, the title contains `Discord`, and the same logical clue is still active. Native foreground APIs cannot distinguish Discord’s message editor from its search field, so caret placement remains a required operator boundary. The macro tracks and backspaces only its own partial characters after an interrupted answer. The macro and Discord must run at the same Windows integrity level; synthetic input generally cannot cross from a normal process into an elevated target.

The red/green card accent is authoritative. During the red `Get Ready` state the pipeline may OCR, solve, and cache, but keyboard execution remains blocked. It releases only when the **same question number/card** has a tall narrow green outline component and no dominant red component. “Answer Now!” and “answers OPEN · you have 60s” are logged as corroboration, not trusted by themselves. After green is confirmed, the required random 0.4–1.1 second pause begins, followed by 0.03–0.08 second per-character typing. The green state is rechecked through the delay, before every character, and immediately before Enter. Unknown, mixed, stale, or newly red state fails closed.

## Cache format

The mutable `data/trivia_cache.json` file intentionally keeps the requested string-to-string maps simple:

```json
{
  "schema_version": 1,
  "text_questions": {
    "a normalized question or clue": "Canonical Answer"
  },
  "image_hashes": {
    "64-hex-character-phash-for-hash-size-16": "Canonical Answer"
  },
  "metadata": {
    "text_questions": {},
    "image_hashes": {}
  }
}
```

Text is normalized with Unicode NFKC, case folding, punctuation removal, and whitespace collapse. RapidFuzz accepts a result only when it clears both the score threshold and the configured margin over the runner-up. Visual hints use `imagehash.phash(..., hash_size=16)` and Hamming distance; the default accepts at most 10 changed bits out of 256 and also requires an ambiguity margin. Only the isolated hint band is hashed, never the scrolling feed.

Valid Qwen answers are appended automatically. Writes go to a temporary file in the same directory, are flushed and `fsync`’d, then replace the live JSON atomically. Existing conflicting answers are preserved rather than silently overwritten.

## Performance and calibration notes

- The sub-0.3 second goal applies to the warmed capture/change/OCR/cache path. Submission intentionally waits for green and then the mandated 0.4–1.1 second human delay; a generative VLM miss is seconds-scale and becomes a pHash/text-cache fast hit next time.
- DXcam performs the region copy through Desktop Duplication, but exposes a CPU NumPy array. PaddleOCR, Qwen3-VL, and the frame-change gate run on CUDA. The required Python `imagehash` pHash is CPU/SciPy code and normally takes only a few milliseconds; claiming it is GPU-resident would be inaccurate.
- `video_mode: true` supplies an actual 60 FPS stream even when Discord is static. Duplicate frames are discarded by the CUDA comparison. A localized CUDA tile trigger makes the narrow red→green edit visible even inside a large chat band; three stable frames (about 50 ms at 60 FPS) prevent OCR from firing on a half-scrolled card. The same accent prelocates the active card before OCR, reducing the measured warmed red-card path from roughly 380 ms on the full chat band to about 60–62 ms.
- The supplied locked accent is RGB `#ED4245`; ready is `#2ECC70`. Detection also requires a narrow component at least 80 px tall, 3–14 px wide, aspect ratio at least 12, and 60% fill, preventing green/red emoji and chat text from opening the gate.
- If harmless animation repeatedly triggers OCR, crop more tightly first. If necessary, add relative rectangles to `change_detection.ignore_regions` or raise `mean_absolute_threshold` slightly.
- If a real card change is missed, lower `mean_absolute_threshold` and `changed_pixel_ratio` gradually.
- PaddleOCR is initialized once and runs a generated known-text smoke image. A CUDA runtime that loads but returns zero boxes fails startup instead of silently running blind.
- `ready_before_capture` is on by default: live capture will not arm until Qwen is resident. Background preload is available only as an explicit latency/availability tradeoff and can miss the first novel round.

## Current implementation basis

- [DXcam 0.3 capture API](https://github.com/ra1nty/DXcam/blob/v0.3.0/README.md)
- [PaddleOCR 3.x OCR pipeline and result fields](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [PaddleOCR Transformers inference engine](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [Qwen3-VL-32B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [Transformers multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)
- [Hugging Face bitsandbytes Windows/CUDA support](https://huggingface.co/docs/bitsandbytes/installation)
- [Python ImageHash pHash/Hamming-distance API](https://github.com/JohannesBuchner/imagehash)
