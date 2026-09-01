# Anime knowledge-source audit — 2026-09-01

The live resolver needs identity-bearing text: anime synopses, character biographies, aliases, media relationships, and quotes. Ratings, user profiles, and generic anime images are not useful evidence for this task.

## User-supplied leads

### Kaggle `dbdmobile/myanimelist-dataset`

- Updated 2023-07-28; ODbL 1.0 + DbCL 1.0.
- 7.35 GB raw / 1.93 GB ZIP across six files.
- The useful slice is only `anime-dataset-2023.csv`: 24,905 titles, aliases, genres, and synopses; 15,924,739 bytes raw / 5,895,377 bytes compressed.
- The other files are 731,290 user profiles and tens of millions of rating interactions. They are recommendation data, not trivia knowledge.
- Verdict: the synopsis slice is usable but superseded by the fresher and richer AniList source below.

### Deleted Reddit “1.77M users / 148M ratings” lead

The data survives at [GitHub](https://github.com/MRamazan/User-Animelist-Dataset), [Kaggle](https://www.kaggle.com/datasets/ramazanturann/user-animelist-dataset), and [Hugging Face](https://huggingface.co/datasets/mramazan/User-Animelist-Dataset).

- 1,774,522 users, 20,237 anime, and 148,170,496 ratings.
- `ratings.csv` is only `userID, animeID, rating`; `animes.csv` contains titles, genres, year, score, episodes, and links but no synopsis or character biography.
- License metadata conflicts: Kaggle currently reports CC BY-NC 4.0, while mirrors report CC BY 4.0, and GitHub has no license file.
- Verdict: exclude. It adds popularity priors but no clue-answer evidence.

### Hugging Face `lowres/anime-datasets` collection

| Dataset | Verified contents | Trivia value |
|---|---|---|
| `lowres/anime` | 1,454 images, five identity labels, 742 MB | Very low |
| `none-yet/anime-captions` | 337,038 generic image captions, 31.6 GB | Low; mostly unnamed |
| `p1atdev/niji-v5` | 3,000 synthetic style images, 9.44 GB | None |
| `lowres/anime-synthetics` | 2,186 synthetic/tag images, 3.98 GB | None |
| `mio/sukasuka-anime-vocal-dataset` | 3,495 clips for 26 speakers from one show | One-show voice ID only |
| `lowres/eggy` | 138 images with one label | None |
| `mohamed-khalil/AnimeQuotes` | 10,388 mostly Japanese quote/character/URL rows; no anime-title field | Low |
| `p1atdev/stackexchanges` | 12,318 Anime Stack Exchange Q&A rows in the simple split | Supplemental lore only |
| `mohamed-khalil/AnimeSongsLyrics` | 23,571 lyric/song/anime rows | Useful only for lyric clues |

The stronger quote source is [`ewgsta/animequotes`](https://huggingface.co/datasets/ewgsta/animequotes): 8,612 English `Anime, Character, Quote` rows, 1.5 MB, MIT, updated 2026-05-09. It directly contains several observed Anime Soul quotes, including “Sit, boy!”.

## Selected factual core

### Current AniList anime and character export

[`calebmwelsh/anilist-anime-dataset`](https://www.kaggle.com/datasets/calebmwelsh/anilist-anime-dataset), version 94, updated 2026-08-30:

- 20,425 anime; 19,180 descriptions.
- 87,664 unique characters; 49,728 character descriptions.
- English, romaji, native, preferred titles and synonyms.
- Character full/native/alternative names, roles, biographies, voice actors, and media relationships.
- 31,801 typed media-relation edges plus tags, studios, recommendations, and reviews.
- Selected CSV: 438,222,688 bytes raw / 91,746,659 bytes compressed.

Kaggle declares CC0, but that declaration may not override AniList upstream terms. The source and derived index remain private and ignored.

### Manami cross-provider normalization

[`manami-project/anime-offline-database`](https://github.com/manami-project/anime-offline-database), release 2026-27:

- 41,537 anime across MAL, AniList, AniDB, Kitsu, and other providers.
- Extensive synonyms, tags, and related-anime URLs.
- 6,034,492-byte compressed JSONL / 62,331,124 bytes expanded.
- ODbL 1.0 + DbCL 1.0.

It has no character biographies, but it is the best title-alias and cross-ID layer.

## Optional accuracy expansion

- [Anime Dataset 2025](https://www.kaggle.com/datasets/neelagiriaditya/anime-dataset-jan-1917-to-oct-2025): 28,955 anime, 23,828 synopses, 209,963 character rows, 112,987 biographies, nicknames, character→anime roles, and recommendations. The useful five-file slice is 31.13 MiB compressed; CC BY-NC-SA 4.0.
- Anime Stack Exchange simple parquet: 12,318 rows / 18,536,497 bytes; CC BY-SA 3.0. Treat it as attributed supplemental lore, never an automatic canonical answer source.

## Deployment decision

The installed core uses the current AniList CSV, English quote map, and Manami aliases—94.69 MiB compressed. A local SQLite FTS5 index provides exact quote lookup and millisecond anime/character BM25 retrieval; Wikipedia and web search remain the freshness and evidence fallback. The mixed-license source files and derived database are excluded from Git.
