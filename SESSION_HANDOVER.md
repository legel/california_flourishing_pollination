# Session handover — 7-hour autonomous run starting 2026-05-22T07:17 UTC

> Lance: this is what's running while you're away. Auto-managed by `scripts/autonomous_watchdog.sh` (PID is in `outputs/watchdog.log`).

## What's running (4 critical processes, all watchdogged)

| Process | PID | Source | What it does |
|---|---|---|---|
| `download.pid` | 1288010 | `cfp.pipeline download` | Async-fetches images from the 8.5M-URL master manifest at concurrency=256, per-host=64, capped at 800 GB disk. Resumable via `outputs/checkpoint_downloaded.parquet`. |
| `embed.pid` | 1289347 | `cfp.pipeline embed --gpu-decode --with-phenovision` | **DINOv3 ViT-L/16 + PhenoVision (Dinnage 2025) in one GPU forward pass per image.** nvJPEG decode on GPU. Deletes images post-embed. Per-image output gains `phenovision_flowering_prob` + `phenovision_fruiting_prob`. |
| `upload.pid` | 1288157 | `cfp.pipeline upload --poll-seconds 300` | Tails the shard dir, pushes 4 GB shards to HF every 5 min, deletes local on ack. |
| watchdog | 1288003 | `scripts/autonomous_watchdog.sh` | Every 60 s: relaunches any of the above three that died. Every 5 min: writes a status snapshot to `logs/watchdog_status.log`. Runs for 7 hours then exits gracefully (workers continue). |

Plus three older processes still running independently:
- HF cleanup (`scripts/hf_cleanup_noncanonical.py`) — filters legacy HF shards to canonical Calscape only.
- iNat re-matcher (`cfp.cnps match match`) — enriches plants parquet with `inat_taxon_id`, `gbif_backbone_key`, `ca_observation_count`, `match_quality`.

## What's in the master manifest

**8,539,943 unique image URLs** in `data/processed/image_manifest.parquet`:
- 7,062,462 plant URLs (Calscape canonical filter applied — 6,969 species)
- 1,477,481 pollinator URLs (1,260 of 1,275 flying CA pollinators)
- 4,299,058 unique observations (some have multi-photo)
- Sourced from two GBIF Occurrence Downloads:
  - Plants: DOI `10.15468/dl.pbgs4h`, 3.6M occurrences
  - Pollinators: 766K occurrences (DOI in `data/raw/gbif/pollinators_ca_inat.meta.json`)

## Expected progress by hour 7

Throughput (verified):
- Download: ~200–500 img/sec sustained (network-bound)
- Embed: ~370 img/sec sustained (DINOv3+PhenoVision combined)
- Upload: ~30 s per 4 GB shard

Best case (7 hr × 370/s embed = 9.3 M images): the entire 8.5M manifest could finish embedding. Network-bound download is the slower factor; realistic estimate is **3–5 M images embedded + uploaded by hour 7**.

## How to inspect when you return

```bash
# Quick overall state
tail -50 logs/watchdog_status.log

# How many embedding shards on HF?
hf download deepearth/california-flourishing-pollination --repo-type dataset --list | grep embeddings | wc -l

# Per-process logs
tail -50 logs/chain_download.log
tail -50 logs/chain_embed.log
tail -50 logs/chain_upload.log
tail -50 logs/watchdog.log

# Live counters
python -c "import pandas as pd; print('downloaded:', len(pd.read_parquet('outputs/checkpoint_downloaded.parquet')))"
python -c "import pandas as pd; print('embedded:  ', len(pd.read_parquet('outputs/checkpoint_embedded.parquet')))"
```

## What's deferred to Phase 1.5

1. **Backfill PhenoVision onto the ~575K legacy DINOv3-only embeddings** that are on HF before the combined extractor went live. Re-process the same image URLs through PhenoVision only.
2. **PhenoVision spatial localization** of flowering (DINOv3 patches × PhenoVision classifier, or a flower-segmentation head). Currently we have image-level probability only.
3. **Raw-image bucket archive** to `hf://buckets/deepearth/cfp-raw-images` (created 2026-05-22, currently empty bar a few benchmark uploads). Bucket throughput benchmarked at ~36 img/sec for 400-image batches, ~150-200 img/sec extrapolated for 5K batches. Slower than embed rate of 245-370 img/sec, so adding it inline would bottleneck the pipeline. As a separate background job once embedding completes (~24 h sync to push 2.4 TB), this preserves all raw photos for re-processing with future models. iNat photo URLs remain in every embedding row, so the dataset is self-contained even without the bucket.
4. **Recover the rare ~8K tail** of CA-native plant taxa beyond Calscape's ~8.5K (would require Jepson MOU or family-level GBIF query splitting).

## Notable in-session bug fixes

1. **DINOv3 ViT-L/16 fp16 → NaN**: switched to bf16 (numerically equivalent to fp32 at fp16 speed). Discovered, scrubbed bad shards, re-ran cleanly. See `ASSUMPTIONS.md §E`.
2. **Shard-name collision after embedder restart**: embeddings_0000NN.parquet got skipped by uploader because remote already had that name from a prior restart. Fixed with `run_id` timestamp prefix per process start.
3. **nvJPEG / decode_image variably 3D vs 4D output** crashed `F.interpolate` with 5D input. Fixed with defensive shape normalization in `extractor_combined.py` / `extractor_gpu.py`. The crash loop wasted ~3 h of embedder throughput; watchdog kept the pipeline alive throughout but at ~34 img/sec instead of 245. Post-fix rate is 245-370 img/sec sustained.

## Safety

- All process PIDs are in `outputs/{download,embed,upload}.pid`. To stop the autonomous run cleanly:
  ```bash
  kill $(cat outputs/watchdog.pid) $(cat outputs/{download,embed,upload}.pid 2>/dev/null)
  ```
- Disk cap is 800 GB (currently ~30 GB used). Downloader pauses when hit; embedder drains the queue.
- `outputs/failed_downloads.parquet` accumulates any URLs that 404/timeout. Small (<0.1% of manifest historically).
- The HF dataset cleanup will continue independently and reach all ~68 shards.

## Files modified / added this session

- `src/cfp/dinov3/extractor_combined.py` — new
- `src/cfp/dinov3/extractor_gpu.py` — new
- `src/cfp/cnps/calscape.py` — new
- `src/cfp/cnps/match_inat.py` — new
- `src/cfp/gbif/batch_download.py` — new
- `scripts/autonomous_watchdog.sh` — new
- `scripts/analyze_calscape_globi.py` — new
- `scripts/hf_cleanup_noncanonical.py` — new
- `PROVENANCE.md` — updated (Calscape now PRIMARY, GBIF batch DOIs)
- `README.md` — updated
- `docs/FLOURISHING.md`, `docs/POLLINATION.md` — new
- `ASSUMPTIONS.md` — updated (bf16 finding, etc.)

All pushed to https://github.com/legel/california_flourishing_pollination main.
