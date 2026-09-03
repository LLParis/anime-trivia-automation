# Anime Trivia Automation — research brief

A self-contained description of a working Windows desktop automation, written to
hand to an external deep-research model. Everything below is measured, not
estimated; where a number is a guess it says so. Commit `be3e571`, version
0.10.4, 141 tests passing.

**What we want from research:** how to make this class of system faster, more
accurate, and more robust. The specific open questions are in section 9. Please
do not re-propose anything in section 8 without new evidence — those are ideas
we already tried and measured.

---

## 1. The game we are playing

A Discord bot called **Anime Soul** runs a trivia quiz in a public channel three
times a day (07:00, 12:00, 18:00 local). Each quiz is 10 rounds. Per round:

1. The bot posts an embed card with a red left accent strip: *"Get Ready…
   🔒 Reading time — answers open in 5s"*. The clue is already visible.
2. About 3.4–5.8 s later the bot **edits the same message**: the strip turns
   green and the footer becomes *"🟢 answers OPEN · you have 60s"*.
3. Players race to type the answer as a normal chat message. **The first correct
   guess wins.** There is no penalty for a wrong guess and no limit on attempts.
4. The bot then edits the card to *"Round Over"* and posts a separate result
   message: *"✅ Correct! @user got it in 1.3s — the answer was Chainsaw Man."*
   or *"⏰ Time's up! Nobody got it — the answer was X."*

Clue types observed (n=186, the complete history since the quiz began
2026-08-28):

| Type | Count | Example | Answer |
|---|---|---|---|
| Quotation | 66 | `"A corpse is talking."` | Chainsaw Man |
| Emoji rebus | 58 | 🖌️ 🏫 🗼 🎨 | Blue Period |
| Prose description | 62 | "A young witch who leaves home at thirteen…" | Kiki |

Answer types are `anime_title` (124) or `character` (62); the card states which.

**Human competition is the benchmark.** Observed winning times on 2026-09-03
07:00, as credited by the bot: 1.2 s, 1.3 s, 1.4 s, 3.4 s, 4.6 s, 8.8 s, 14.6 s,
16.0 s, 18.2 s. Easy quotations go in about 1.3 s. Emoji rebuses take the room
14–18 s. One round went unanswered by everyone.

---

## 2. What the automation is

A single-process Python 3.14 Windows application, ~13,000 lines across 21
modules, that watches the Discord window, reads each card, decides an answer,
and types it into the real Discord message composer as genuine keystrokes.

It is not a Discord bot and does not touch the Discord API. There is no token,
no self-bot, no HTTP to Discord. It is screen capture plus UI Automation plus
synthetic keyboard input, which is the deliberate design constraint: it acts as
the human operator's hands, on their own logged-in desktop client.

Module sizes give a sense of where complexity lives:

```
app.py         2513   orchestration, round lifecycle, resolution routing
typing.py      1994   composer ownership, keystroke safety, guess dispatch
novel.py       1279   local retrieval-grounded solver (currently disabled)
config.py      1235   configuration + validation
antigravity.py  907   the live cloud solver (CLI subprocess)
gemini.py       757   an alternative cloud solver (disabled, quota exhausted)
cache.py        687   clue → answer lookup: exact, fuzzy, pHash, emoji affinity
ocr.py          616   card segmentation, OCR, red/green strip classification
discord.py      363   UI Automation reads of the card and result messages
capture.py      303   DXcam capture + CUDA change gate
```

---

## 3. The pipeline, with measured latencies

```
DXcam 60fps region capture
        │  ~16 ms/frame
        ▼
CUDA thumbnail change gate ─────────────────► unchanged frames go to:
        │  stable_frames = 3                   accent-strip watcher
        ▼                                              │
PaddleOCR (PP-OCRv6_small, GPU)                        │ classifies the card's
        │  60–500 ms depending on card                 │ left colour strip per
        ▼                                              │ frame; 2 agreeing
Card segmentation + red/green/closed                   │ frames = a flip
classification from the accent strip                   │
        │                                              ▼
        ├──────────────────────────────────►  force a re-read AND
        ▼                                      open the gate immediately
UI Automation read of Discord's                (see §5, shipped today)
accessibility tree for the exact
semantic clue (emoji survive here;
they do not survive OCR)
        │  ~127 ms
        ▼
Answer resolution, in order:
  1. exact history lookup           ~0 ms
  2. fuzzy history (RapidFuzz)      ~0 ms   (never fires for emoji: see §6)
  3. pHash image match              ~0 ms
  4. emoji affinity (new, §5)       ~0 ms
  5. cloud solver                   3.2–30+ s
        │
        ▼
Answer held in memory while the card is RED
        │
        ▼  at green
Composer ownership checks → one SendInput Unicode batch →
UIA read-back verification → Enter
        │  green → verified text: 0.17 s median (was 0.34 s)
        ▼
Post-Enter confirmation, learning the reveal, ledger
```

**The decisive timing relationship.** Answers open 3.4–5.8 s after the card
appears. The cloud solver returns 3.2–6.6 s after the card appears on easy
clues, and 15–40 s on hard ones. So on an easy clue the answer arrives within a
second or two of the gate opening, and on a hard clue it arrives long after
humans have won. Every 100 ms cut anywhere before the keystroke converts
directly into wins.

---

## 4. The solver

**Antigravity CLI** (`agy.exe`), Google's coding-agent CLI, invoked as a
subprocess with `--model gemini-3.8-flash-low --output-format json
--json-schema {answer, abstain, confidence} --sandbox`. Authenticated by
consumer Google account, not an API key. One process per round, spawned fresh.

Chosen because the Gemini Developer API key hit its free quota. This is a
constraint worth naming: **we are latency-bound on a CLI subprocess we do not
control**, including process start cost, and we have no streaming access to
first tokens.

The prompt is deliberately small: an instruction paragraph, the required answer
type, the clue, and for a rebus the emoji spelled out as Unicode names
(`"lower left paintbrush, school, tokyo tower, artist palette"`).

Two other solvers exist but are off: a local Qwen3.8-27B via llama.cpp
(fallback-only, never recovered a miss in testing, and competes for the GPU with
OCR), and the Gemini Developer API (quota exhausted).

---

## 5. What we shipped on 2026-09-03, with evidence

All three came from reading the round ledger rather than guessing.

**a. The budget was a timeout wearing an abstain's clothing.** The three rounds
where the app said nothing each abstained at *exactly* 12.0 s after the card
appeared, which was `antigravity.total_timeout_seconds`. Re-asking the CLI those
same clues answered in 5.5 s, 14.1 s and 21.8 s. One of them, Princess Mononoke,
was a round *nobody* won. Budget is now 40 s, inside the ~60 s answer window. A
late answer cannot reach a closed round: entry requires the green strip and
stale-round results are discarded, both proven live.

**b. Emoji rebuses answered with no model call.** The bot never repeats a clue —
186 clues, zero duplicates — so exact and fuzzy matching miss every emoji round.
But answers repeat (143 distinct answers over 186 rounds) and a returning answer
keeps part of its symbols:

| Answer | First rebus | Second rebus |
|---|---|---|
| Neon Genesis Evangelion | 🤖 🟣 🪽 🌇 | 🤖 🪽 🟣 🌇 |
| Delicious in Dungeon | 🥘 🐉 🗺️ 🍄 | 🥘 🐉 🛡️ 🗺️ |
| Blue Period | 🎨 🟦 🖼️ 🏫 | 🖌️ 🏫 🗼 🎨 |
| Initial D | 🚗 ⛰️ 🥤 💨 | 🚘 🍱 ⛰️ 🌙 🏁 |

So we score a rebus against every past rebus by cosine similarity over TF-IDF
weighted tokens: the glyphs (weight 2) plus the words of their Unicode names
(weight 1). The name words are what bridge Initial D's car being swapped for an
oncoming car, which share no codepoint but both carry "automobile". Scoring is
per *answer*, not per clue, or a title with two rebuses becomes its own closest
rival and hides a real tie.

Leave-one-out over all 58 real rebuses, sweeping threshold and margin:
**threshold 0.50, margin 0.06 → 80% precision, fires on 10 of 58.** Nothing
reached 85%. Verified live: three rebuses absent from history typed into the
real composer 0.17–0.23 s after green.

**c. Green now opens on the strip alone.** The strip watcher only inspects
frames the change gate judged *unchanged*, so when the strip flips the card
content is provably identical and the clue is already in hand. Previously the
flip forced a full OCR re-read and marked the round uncertain, pausing typing
for ~150 ms to re-derive known information. Now the gate opens on the flip and
the confirming read still lands and corrects if it disagrees. Median green → verified
text went 0.34 s → **0.17 s**.

---

## 6. Assets

- `data/trivia_history.seed.json` — 186 reviewed clue/answer/type triples, the
  complete quiz history. **Zero duplicate clues.** 143 distinct answers, 42 used
  more than once.
- `data/answer_catalog.seed.json` — 239 candidate answers. 8 of the 10 answers on
  2026-09-03 were already in it.
- `data/anime_knowledge.sqlite` — 589 MB local index: 61,960 anime, 87,664
  characters, 324,518 anime aliases, 201,473 character aliases, 158,232
  full-text-searchable records, and **8,586 quotes** with title and speaker.
- `runtime/round_ledger.jsonl` — append-only JSONL of every state transition
  with wall-clock and monotonic stamps. This is what made today's diagnoses
  possible and is the single most valuable piece of instrumentation in the
  system.

**An untapped lead, measured today.** Testing the local quote index against
every quotation clue in the history: **19 of 66 resolved, with zero wrong
answers.** That is 10% of all rounds answerable by a SQLite lookup in
milliseconds at 100% observed precision, versus a 6 s model call. The 47 misses
are famous lines the corpus simply lacks (Sailor Moon's transformation line,
Gurren Lagann's drill line, Team Rocket's motto), so this is a corpus-coverage
problem, not an approach problem.

---

## 7. Safety and operating constraints

These are hard requirements, not preferences. Any proposal must respect them.

- **The composer is shared with a human.** The operator uses the same Discord
  window while the app runs. Text they typed is never touched, cleared, or sent.
  Ownership is verified by reading the composer value back through UI Automation
  before every action.
- **One Enter per round, ever.** The submission outcome is treated as consumed
  from the first keydown, because an exception can still mean Windows accepted
  the input. Duplicate submission is the worst failure mode.
- **Rehearsal must never post.** A rehearsal mode types and verifies but withholds
  Enter. The card-painting harness refuses to run unless a live rehearsal worker
  owns the status file, so it cannot drive a real quiz session.
- **Nothing is claimed working until it is rehearsed on the real composer** with
  per-card answers and latencies. Unit tests are a smoke check only. This rule
  exists because a week of green tests accompanied a week of total live failure.
- The solver runs on a consumer Google account with no visible quota counter.
  Roughly 80–120 calls a day is the observed comfortable volume.
- Windows 11, RTX 5090, single machine, single monitor region captured.

---

## 8. Tried and measured — do not re-propose without new evidence

- **Mass-harvesting past clues into a lookup table.** Proposed twice by me,
  wrong both times. 186 clues contain zero duplicates; tomorrow's clue will be
  new. A scrollback harvester was built and works (8 of 8 rounds paired
  correctly from the live channel) but has near-zero value for prediction. Its
  real use is collecting *answers*, not clues.
- **Putting the 239-answer candidate pool in the prompt.** Tested on the three
  failed rounds with today's answers held out. It rescued Blue Period and turned
  Solo Leveling into a confidently wrong neighbour (Sword Art Online), and it
  measurably increased latency. Net unclear, not shipped.
- **Image hashing (pHash) of emoji cards.** Stored 3–5 hashes per visual card.
  Useless in practice: different emoji mean different pixels, so a returning
  answer with substituted symbols never matches.
- **Raising the model's confidence gate to reduce abstains.** The abstains were
  timeouts, not low confidence. The gate was never the problem.
- **A local Qwen3.8-27B fallback.** Never recovered a miss in testing and
  competes with OCR for the GPU.
- **A manual-Enter fallback launcher.** Built, then deleted at the operator's
  direction: one path, made reliable, beats two paths.

---

## 9. Open research questions

Ranked by expected value. Concrete, evidence-backed answers are far more useful
than surveys.

1. **Sub-second solving.** The core tension: easy clues need an answer within
   ~1 s of the card appearing to beat a 1.3 s human, and our cloud solver's
   floor is ~3.2 s including process spawn. What architectures get a
   general-knowledge answer in under a second on a single consumer machine?
   Small local models, embedding retrieval over a 61k-title index, distillation
   of the quiz's own answer distribution, speculative pre-computation while the
   card is still locked? What is the realistic accuracy/latency frontier at
   ~500 ms on an RTX 5090?

2. **Emoji rebus solving.** 31% of clues, and our weakest area: the model takes
   15–40 s and declines about half the time, and our similarity path covers only
   the 17% of rebuses whose answer has a prior rebus at 80% precision. How would
   you solve a 4-symbol pictographic rebus mapping to one of ~61,960 anime
   titles? Multimodal embedding of the glyph sequence against title/synopsis
   embeddings? Constrained decoding over the candidate set? Is there prior art
   on rebus solving we should read?

3. **Quote retrieval coverage.** 29% of quote clues resolve from a local corpus
   at 100% precision. Where do we get a materially larger, well-attributed anime
   quotation corpus, and what retrieval design handles paraphrase and
   translation variance? The same English line is often subtitled differently
   across releases.

4. **Answer-form matching.** The bot accepted "Mihael Keehl" for the reveal
   "Mello" but rejected "Digimon: Digital Monsters" for "Digimon Adventure". We
   do not know its matching rule. How would you infer an unknown string-matching
   rule from observed accept/reject pairs, and what answer form maximises
   acceptance probability under uncertainty? We have a 3-guess ladder with a
   5.2 s gap available and currently unused.

5. **Screen-to-decision latency.** We are at 0.17 s from our detection of green
   to verified text in the box. What is the floor for a capture → classify →
   synthetic-input loop on Windows, and where would you expect the remaining
   time to be? Note we still lose races where we had the answer early, implying
   our *detection* of green trails the server event by a few hundred ms that we
   have not yet decomposed.

6. **Anything structurally better.** We are a screen-scraping macro because the
   API path is off-limits. Given that constraint, is there a fundamentally
   better architecture than capture → OCR → accessibility-tree read → decide →
   synthetic keystrokes?

---

## 10. How to read our evidence

Every claim above traces to one of: `runtime/round_ledger.jsonl` (per-round state
transitions with timestamps), `runtime/logs/anime-trivia-*.log` (per-launch
detail), or a leave-one-out evaluation over the committed history. If a proposal
depends on a number we have not measured, say so explicitly — we would rather
run the measurement than accept an estimate.
