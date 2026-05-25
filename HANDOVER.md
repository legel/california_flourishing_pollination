# Handover — California Flourishing & Pollination

> **Last-session state for the next agent / scientist / engineer.**
> Snapshot **2026-05-25**. See [`AGENTS_README.md`](./AGENTS_README.md) for the canonical full doc.

## TL;DR

The Phase 1 pipeline is **essentially complete**:

- **10.27 M of 10.30 M images embedded (99.77 %)** with DINOv3 ViT-L/16 + PhenoVision flowering/fruiting
- **1,275 shards / 4.15 TB** on HF at `deepearth/california-flourishing-pollination`
- **100 % per-photo CC license + creator** preserved in every row
- **95 % of HF shards** have correct PhenoVision column ordering (the other 5 % have no PhenoVision data — Phase 1.5 work)
- **Interactive Space** running at https://huggingface.co/spaces/deepearth/california-flourishing-pollination
- **Provenance**: 4 GBIF DOIs, sha256-attested DwC-A archives, every API query persisted in `provenance/`

## What's running right now

| Process | PID | What it's doing |
|---|---|---|
| watchdog | (see `outputs/watchdog.pid`) | 24h cycle; supervises download/embed/upload, relaunches any that die |
| download | varies — keeps exiting quickly because nothing left to download | the 22 remaining manifest URLs all 404 (deleted iNat photos) |
| embed | varies — usually idle in 60s poll | only ~24K corrupt JPEGs left in the gap; nothing to embed |
| upload | alive but nothing to push (local shard dir is empty) | will activate the moment any new shard lands |
| retry_backfill_fails.py | `nohup ... > logs/retry_backfill_fails.log 2>&1 &` | paced 30 s/commit to stay under HF 128/hour rate limit; ~90 min ETA for 178 shards |

## What's done

- **Master manifest**: 10,301,629 rows from 4 GBIF batches (DOIs in PROVENANCE.md §1c)
- **All 4 GBIF DwC-A zips archived** at `data/raw/gbif/` with `.meta.json` sidecars carrying the DOI + record count
- **PhenoVision label bug** ([B9](./AGENTS_README.md#13-key-bugs-found-and-fixed--read-this-before-changing-anything)) discovered, fixed in the extractor, and the 1,200 affected HF shards retroactively column-swap-fixed via `scripts/fix_phenovision_swap_on_hf.py`
- **Audio leak bug** ([B10](./AGENTS_README.md#13-key-bugs-found-and-fixed--read-this-before-changing-anything)) discovered (2,594 audio files masqueraded as .jpg from GBIF bird-pollinator batch), filtered out + bad-image-delete added to embedder
- **License + creator backfill** ([B0](./AGENTS_README.md#13-key-bugs-found-and-fixed--read-this-before-changing-anything)) for the 10.3M existing rows from DwC-A `multimedia.txt`
- **Clean taxon names** (no authority strings) backfilled into manifest + shards from DwC-A `species`/`genus`/`taxonRank` fields
- **HF Space** built and deployed: Gradio viewer with UMAP-RGB overlay, normalize-per-image, instant opacity slider, prefetch
- **All 6 lookups** uploaded to HF (`photo_attribution`, `taxon_clean_names`, `shard_index`, `umap_numpy`, `umap_encoder`, `global_pca`)

## What's left (Phase 1.5 / next session)

1. **Wait for `retry_backfill_fails.py` to finish** (~90 min). After that, 1,211 of 1,275 shards will have correct PhenoVision.
2. **Add PhenoVision to the 64 legacy shards** — re-fetch images for those rows, run PhenoVision only, patch the columns.
3. **(Optional) Train a ParametricUMAP** to replace the 206 MB UMAP-numpy encoder with a ~10 MB Keras encoder for faster Space cold-start.
4. **Push photo bytes to `hf://buckets/deepearth/cfp-raw-images`** (bucket exists, empty) — preserves photos beyond iNat's URL lifetime. ~2.4 TB.
5. **PhenoVision spatial localization** — DINOv3 patches × PhenoVision classifier → patch-level flowering probability.

See [AGENTS_README §16 Phase 1.5 backlog](./AGENTS_README.md#16-phase-15-backlog) for the full list.

## Critical pitfalls to avoid (paste from §17)

1. **DO NOT** introduce parallelism on HF commits ([B5](./AGENTS_README.md)). DO use `api.upload_large_folder`.
2. **DO NOT** revert to fp16 on DINOv3 ([B1](./AGENTS_README.md)). bf16 stays.
3. **DO NOT** use per-`gbif_id` keys for new checkpoints ([B4](./AGENTS_README.md)). URL-keyed.
4. **DO NOT** download shards without `_purge_cache()` ([B13](./AGENTS_README.md)). Disk WILL fill (1.3 TB → 100 % in ~3 hours at 6 workers).
5. **DO NOT** exceed 128 commits/hour to HF ([B14](./AGENTS_README.md)). Pace your scripts.
6. **DO NOT** parse DwC-A `multimedia.txt` without filtering `type='StillImage'` ([B12](./AGENTS_README.md)).
7. **PhenoVision label order**: `index 0 = fruiting, index 1 = flowering` ([B9](./AGENTS_README.md)). Triple-check before modifying the extractor.

## Where to look first when something breaks

| Symptom | First look |
|---|---|
| Disk full | `du -sh /home/legel/.cache/huggingface/hub/* /home/legel/cfp_shards /home/legel/cfp_images` — usually HF cache from a backfill script that forgot `_purge_cache()` |
| Embedder spinning but GPU idle | check `find /home/legel/cfp_images -name '*.jpg' \| head -5` then `cat <sample>.json` — likely audio leak (B12) |
| HF API errors `HfHubHTTPError` | check rate limit (B14) — 128 commits/hour |
| PhenoVision values look "off" | check shard run_id; if pre-`20260524T070916`, the shard had swapped labels (B9) and is awaiting backfill |
| Space load slow | shipped `species_list.json` should populate dropdown instantly; if not, `warm_caches_background()` is failing to load manifest from HF |
| Embedder produces no shards | check `tail -F logs/chain_embed.log` for "round N embedded X images" — if X > 0 but no shard files appear, see B10 (bad images clogging queue) |

## Final stats

```
manifest:   10,301,629 rows / 10,297,212 unique URLs
            5,244,656 observations / 16,446 species
            6,383 plant species + 10,063 pollinator species
            100% per-photo CC license + creator coverage

downloaded: 10,297,382 URLs (100.00%; 22 are iNat 404s)
embedded:   10,273,298 URLs (99.77%; gap is ~24K corrupt JPEGs)

HF:         1,275 embedding shards / 4.15 TB
            1,211 of 1,275 with correct PhenoVision labels (after retry completes)
            6 lookup files (photo_attribution, taxon_clean_names, shard_index,
                            umap_numpy, umap_encoder, global_pca)

Disk:       ~16% used on /home/legel (260 GB / 1.3 TB)
            HF cache bounded at ~12-25 GB (auto-purged after each backfill upload)

Pipeline:   embed throughput 260-340 img/s sustained, 585 img/s standalone
            (H200 ceiling for DINOv3-L/16 + PhenoVision)
            GPU util 47-59% during active embed; 0% during DataLoader spawn
            upload throughput 290 MB/s via api.upload_large_folder
```

## Repository state

All code on `main` at https://github.com/legel/california_flourishing_pollination.

Recent commit history (newest first):
- `535269a` Fix: shard-backfill script must purge HF cache after each upload
- `725e151` GPU saturation (batch nvJPEG + async I/O) + Space UX + HF shard backfill
- `8bb9246` Viewer: UMAP-only via numpy npz + default Arctostaphylos pallida + bigger embed batch
- `0dff7e2` Filter Sound media from GBIF DwC-A parses + delete bad images in embedder
- `978d92b` PhenoVision label swap + HTML-stacked viewer with JS-driven opacity slider
- `afb4fbd` Recover per-photo license/rights_holder/creator + clean taxon names (no authority)
- (full log: `git log --oneline`)

Next agent: just run `git pull && bash scripts/autonomous_watchdog.sh > logs/watchdog.log 2>&1 &` to keep the pipeline alive. Or pick up a Phase 1.5 backlog item.
