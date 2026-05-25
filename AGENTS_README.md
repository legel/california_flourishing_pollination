# AGENTS_README — California Flourishing & Pollination

> **For future agents, scientists, engineers, and developers working on this project.**
> This is the single source of truth for what's built, what's running, what
> was learned, and what to know before changing anything. Read it end-to-end
> before acting. Keep it up to date when you change the system.

---

## 1. Project at a glance

**Goal.** Produce a self-supervised, multi-modal, analysis-ready dataset of
every iNaturalist Research-grade observation of every California-native plant
and every California-observed flying pollinator, encoded with **DINOv3
ViT-L/16** spatial features and **PhenoVision** flower/fruit probabilities.
Published as `deepearth/california-flourishing-pollination` on Hugging Face.

**Collaboration.** Ecological Intelligence, Inc. (Lance Legel, PI,
lance@ecological.dev) × UC Berkeley Quantitative Ecosystem Dynamics Lab
(Trevor Keenan, PI, keenangroup.info). Affiliated-Organization research
collaboration starting 2026-05-04.

**Citation.** Legel, L. & Keenan, T. (2026). *California Flourishing &
Pollination: a multi-modal AI dataset for ecological forecasting.* Hugging
Face Dataset.

**License.** MIT for code, parquets, embeddings, manifests, lookups, model
extraction artifacts. DINOv3 features are transformative derivatives of iNat
photos — we never redistribute photo bytes; per-photo CC license + creator
attribution preserved on every row.

---

## 2. Live state (snapshot 2026-05-25, update when you change anything)

| | |
|---|---|
| Master manifest | `data/processed/image_manifest.parquet` — **10,301,629 rows / 10,297,212 unique URLs / 5,244,656 observations / 16,446 species** (6,383 plants + 10,063 pollinators) |
| Downloaded | **10,297,382** (100.00 %) — the remaining 22 are iNat 404s (deleted photos) |
| Embedded | **10,273,298** (99.77 %) — the 24K gap is bad/corrupt JPEGs that nvJPEG could not decode |
| License coverage | **100 %** (83 % CC BY-NC, 11 % CC BY, 4.2 % CC0, 1.7 % other CC) |
| HF dataset | https://huggingface.co/datasets/deepearth/california-flourishing-pollination |
| HF embedding shards | **1,275** / 4.15 TB (8.7 TB quota; 48 % used) |
| PhenoVision label correctness on HF | **1,211 / 1,275 (95 %)** correct = 75 from post-fix embedder runs + 1,136 from the backfill swap. The remaining 64 are *legacy* pre-combined-extractor shards that never had PhenoVision columns at all (Phase 1.5 backlog item: run PhenoVision inference on those images to *add* the columns) |
| HF Space (viewer) | https://huggingface.co/spaces/deepearth/california-flourishing-pollination |
| Watchdog | `scripts/autonomous_watchdog.sh` — 24h cycle |

Always re-check with the live-stats one-liner in §9.

---

## 3. Repository layout

```
/home/legel/california_flourishing_pollination/
├── README.md                — public README (Github)
├── AGENTS_README.md         — THIS FILE
├── PROVENANCE.md            — exhaustive per-source provenance with GBIF DOIs
├── ASSUMPTIONS.md           — every operational assumption + risk-if-wrong
├── PIPELINE.md              — streaming architecture overview
├── RUN_GUIDE.md             — operator copy-paste recipe
├── SESSION_HANDOVER.md      — historical handover from the long autonomous run
├── HANDOVER.md              — current-state handover for the next session
├── docs/
│   ├── FLOURISHING.md       — plant-side track docs
│   ├── POLLINATION.md       — pollinator-side track docs
│   └── DINOV3_RUN.md        — concise CV-engineer-oriented run brief
├── src/cfp/                 — Python package (`pip install -e .`)
│   ├── cnps/calscape.py        # Canonical CA-native list (Calscape Excel)
│   ├── globi/fetch_and_filter.py
│   ├── pollinators/cross_check.py
│   ├── gbif/batch_download.py  # GBIF Occurrence Download API
│   ├── dinov3/
│   │   ├── extractor.py
│   │   ├── extractor_gpu.py
│   │   ├── extractor_combined.py    # DINOv3 + PhenoVision in one GPU pass (CURRENT)
│   │   ├── visualize.py
│   │   └── validate_sample.py
│   ├── pipeline/
│   │   ├── download.py
│   │   ├── embed.py            # GPU embedder (async I/O flush thread)
│   │   └── upload.py           # api.upload_large_folder uploader
│   └── hf/publish_meta.py
├── scripts/
│   ├── autonomous_watchdog.sh           # 24h cycle; supervises 3 workers
│   ├── integrate_birds_and_extras.sh    # GBIF batch integration
│   ├── integrate_broad_pollinators.sh
│   ├── cleanup_orphan_jpgs.py
│   ├── patch_license_and_names.py       # one-shot per-photo license + clean names
│   ├── backfill_phenovision.py          # add PhenoVision to old shards (use _purge_cache)
│   ├── fix_phenovision_swap_on_hf.py    # backfill old shards w/ swapped pheno cols (with cache cleanup)
│   ├── retry_backfill_fails.py          # paced retry of HF rate-limited fails
│   └── cfp_viewer_space/                # Gradio Space source
│       ├── app.py
│       ├── requirements.txt
│       ├── species_list.json            # 404 KB, shipped for instant dropdown
│       ├── extract_umap_numpy.py        # joblib -> numpy npz (version-independent)
│       ├── train_umap.py                # one-shot UMAP train on 100K patches
│       ├── train_global_pca.py          # one-shot Global PCA fit (18 KB output)
│       ├── build_shard_index.py         # build lookups/shard_index.parquet
│       └── sample_overlays.py           # generate side-by-side QA PNGs
├── data/
│   ├── raw/
│   │   ├── globi/            # interactions.tsv.gz (sha256 attested)
│   │   ├── calscape/         # native_to_california.xlsx
│   │   └── gbif/             # 4 DwC-A zips + .meta.json sidecars (DOIs)
│   └── processed/
│       ├── plants_california_native.parquet           # 8,507 Calscape taxa
│       ├── pollinators_california_flying.parquet      # 1,275 GloBi-confirmed
│       ├── globi_ca_plant_pollinator.parquet          # 45,805 CA interactions
│       ├── image_manifest.parquet                     # MASTER — 10.30M URL rows
│       ├── photo_attribution.parquet                  # per-photo CC license + creator (12.4M rows)
│       ├── taxon_clean_names.parquet                  # clean binomial per GBIF taxon (21,349 taxa)
│       └── gbif_taxon_keys*.json + gbif_download_key*.json
├── outputs/                  # PIDs + checkpoints
│   ├── checkpoint_downloaded.parquet
│   ├── checkpoint_embedded.parquet
│   ├── failed_downloads.parquet
│   ├── backfill_retry.json   # 178 swap-fix retries
│   └── {watchdog,download,embed,upload}.pid
├── provenance/               # per-stage JSONL audit logs (every API query, hash, DOI)
├── logs/                     # per-process logs (chain_download, chain_embed, chain_upload, watchdog, fix_phenovision_swap, retry_backfill_fails)
└── vendor/phenovision/       # user's Python port (Phenobase PR #1 fork)

External:
/home/legel/cfp_images/       — image cache (downloader → embedder)
/home/legel/cfp_shards/       — local shards staging (embedder → uploader)
/home/legel/cfp_shard_fix_tmp/— temp staging for the backfill swap (auto-purged after each upload)
/home/legel/.gbif/credentials — GBIF auth (chmod 600)
/home/legel/.cache/huggingface/token — HF token
```

---

## 4. The four critical processes (autonomous)

The watchdog (`scripts/autonomous_watchdog.sh`) re-launches any that die
every 60s and writes a status snapshot every 5 min to
`logs/watchdog_status.log`:

| Name | Command | Key flags |
|---|---|---|
| **download** | `python -m cfp.pipeline download` | `--manifest data/processed/image_manifest.parquet --image-dir /home/legel/cfp_images --concurrency 256 --per-host-concurrency 64 --cap-gb 800` |
| **embed** | `python -m cfp.pipeline embed` | `--image-dir /home/legel/cfp_images --shard-dir /home/legel/cfp_shards --backbone vitl16 --image-size 224 --batch-size 256 --images-per-shard 10000 --poll-seconds 10 --gpu-decode --with-phenovision` |
| **upload** | `python -m cfp.pipeline upload` | `--shard-dir /home/legel/cfp_shards --repo deepearth/california-flourishing-pollination --poll-seconds 300` (uses `api.upload_large_folder` — see B5) |
| **watchdog** | `scripts/autonomous_watchdog.sh` | runs 24 h then exits; rearm with `nohup bash scripts/autonomous_watchdog.sh > logs/watchdog.log 2>&1 &` |

PIDs in `outputs/{name}.pid`. To stop everything cleanly:
```bash
kill $(cat outputs/watchdog.pid) $(cat outputs/{download,embed,upload}.pid 2>/dev/null)
```

---

## 5. Data sources & GBIF batch DOIs

| # | Source | Predicate | Records | GBIF key | DOI |
|---|---|---|---|---|---|
| 1 | iNat-RG plants × CA × Calscape canonical | `TAXON_KEY IN [Calscape keys]` | 3,607,437 | `0007278-260519110011954` | [`10.15468/dl.pbgs4h`](https://doi.org/10.15468/dl.pbgs4h) |
| 2 | iNat-RG pollinators broad scope | `TAXON_KEY IN [216 Insecta, 5289 Trochilidae, 734 Chiroptera]` | 1,497,966 | `0007705-260519110011954` | [`10.15468/dl.cvbfp4`](https://doi.org/10.15468/dl.cvbfp4) |
| 3 | iNat-RG expanded bird pollinators | `TAXON_KEY IN [5289 Trochilidae, 9201093 Ptiliogonatidae, 9321 Mimidae, 6176 Icteridae, 5263 Parulidae, 9285 Cardinalidae, 5215 Bombycillidae]` | 326,544 | `0009596-260519110011954` | [`10.15468/dl.gphfhs`](https://doi.org/10.15468/dl.gphfhs) |
| 4 | iNat-RG plant extras (cultivar recovery) | 49 GBIF keys recovering 487 Calscape names that didn't match on first pass (29 cultivar-parent genera + 32 variety species heads + 62 HIGHERRANK matches) | 4,611,190 | `0009598-260519110011954` | [`10.15468/dl.nbe8dt`](https://doi.org/10.15468/dl.nbe8dt) |

Each `gbif_occurrence_id` in the master manifest is citable back to one of
these DOIs. DwC-A zips archived at `data/raw/gbif/` with `.meta.json` sidecars.

**Calscape canonical native list:** 8,507 taxa from manual Excel export
(Cloudflare blocks all automated access to Calscape). See PROVENANCE.md §1a.

**Flight-ability rules:** `data/processed/flight_ability_rules.csv` — orders
+ families that are flying-pollinator candidates. Formicidae excluded
(flightless workers).

---

## 6. Pipeline architecture

```
GBIF Occurrence Download (4 DOI'd downloads) ──┐
                                               ▼
                  data/processed/image_manifest.parquet  (10.30M URL rows)
                                               │
                            ┌──────────────────┴───────────────────┐
                            ▼                                      ▼
                       DOWNLOADER                          (URL-keyed checkpoint:
            (aiohttp, conc=256, cap=800GB)           outputs/checkpoint_downloaded.parquet)
                            │
                            ▼
              /home/legel/cfp_images/<gbif%1000>/<gbif_id>_<urlhash8>.jpg + .json
                            │
                            ▼
                        EMBEDDER  (GPU)
       ┌──────────────────────────────────────────────────────────────┐
       │  DataLoader workers (num_workers=16) read JPEG bytes         │
       │  Main thread (batch_size=256, bf16):                         │
       │    1. BATCH nvJPEG decode_jpeg(list, device='cuda')          │
       │    2. F.interpolate → 224² + ImageNet normalize on GPU       │
       │    3. DINOv3 ViT-L/16 forward → CLS + patches                │
       │    4. PhenoVision ViT-B/16 forward → sigmoid(logits)         │
       │       (index 0 = fruiting, index 1 = flowering — see B9)    │
       │    5. fp16 cast + row append                                 │
       │  Background-thread ThreadPoolExecutor (see B10):             │
       │    - shard parquet write (4 GB, was blocking GPU 5-10s)      │
       │    - sidecar unlink                                          │
       └──────────────────────────────────────────────────────────────┘
                            │  (10K rows per ~4 GB shard)
                            ▼
              /home/legel/cfp_shards/embeddings_<run_id>_<idx:06d>.parquet
                            │
                            ▼
                        UPLOADER
       ┌──────────────────────────────────────────────────────────────┐
       │  Every 5 min: symlink all candidate shards into a temp dir   │
       │  under embeddings/ prefix, call api.upload_large_folder      │
       │  (ONE git commit + parallel xet chunks ≈ 290 MB/s)           │
       └──────────────────────────────────────────────────────────────┘
                            │
                            ▼
       https://huggingface.co/datasets/deepearth/california-flourishing-pollination
```

---

## 7. Embedding shard schema

```python
{
  "gbif_occurrence_id": int64,
  "taxon_name": str,                    # CLEAN binomial (no authority)
  "gbif_taxon_key": int64,
  "dataset_role": "plant" | "pollinator",
  "license": str,                       # per-photo CC URL (e.g. http://creativecommons.org/licenses/by-nc/4.0/)
  "inat_observation_id": int64 | None,
  "image_url_large": str,               # iNat S3 URL — we don't redistribute the photo
  "observed_on": str,
  "decimal_latitude": float64,
  "decimal_longitude": float64,
  "cls_fp16": bytes,                    # np.float16 (1024,) packed
  "patches_fp16": bytes,                # np.float16 (14, 14, 1024) packed
  "cls_shape": [1024],
  "patches_shape": [14, 14, 1024],
  "backbone": "vitl16",
  "repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
  "embedded_utc": str,
  "phenovision_flowering_prob": float32,    # sigmoid output, 0..1
  "phenovision_fruiting_prob": float32,     # sigmoid output, 0..1
  "phenovision_repo": "phenobase/phenovision",
}
```

Decode at use:
```python
import numpy as np, pandas as pd
df = pd.read_parquet("embeddings/embeddings_*.parquet")
r = df.iloc[0]
cls = np.frombuffer(r["cls_fp16"], dtype=np.float16).reshape(r["cls_shape"])
patches = np.frombuffer(r["patches_fp16"], dtype=np.float16).reshape(r["patches_shape"])
```

---

## 8. Lookup files on HF (`lookups/`)

| File | Size | Purpose |
|---|---|---|
| `lookups/photo_attribution.parquet` | 236 MB | per-photo CC license + rights_holder + creator from GBIF multimedia.txt (12.4M rows) — join to embedding shards on `(gbif_occurrence_id, image_url_large)` |
| `lookups/taxon_clean_names.parquet` | 1.4 MB | clean binomial (no authority) + rank per `gbif_taxon_key` (21,349 taxa) |
| `lookups/shard_index.parquet` | 111 MB | `image_url_large → shard_path` (99.4 % coverage of all shards) — for the Space viewer |
| `lookups/umap_numpy.npz` | 206 MB | pretrained UMAP(1024→3) training data + 3D embeddings + channel ranges — version-independent (use sklearn NearestNeighbors to approximate transform) |
| `lookups/umap_encoder.joblib` | 1.46 GB | original UMAP encoder (joblib pickle; Python-3.11-only — see B11) |
| `lookups/global_pca.npz` | 18 KB | top-3 global PCA components fitted on 500K patch tokens (deprecated by UMAP but kept for fallback) |

---

## 9. Inspect-the-live-system one-liners

**Live stats**:
```bash
cd /home/legel/california_flourishing_pollination
/home/legel/miniconda3/envs/cfp/bin/python <<'PY'
import pandas as pd, os, subprocess
from huggingface_hub import HfApi
m = pd.read_parquet('data/processed/image_manifest.parquet', columns=['image_url_large','taxon_name','dataset_role','gbif_occurrence_id'])
mu = m['image_url_large'].nunique()
dl = len(pd.read_parquet('outputs/checkpoint_downloaded.parquet'))
em = len(pd.read_parquet('outputs/checkpoint_embedded.parquet'))
info = HfApi().repo_info('deepearth/california-flourishing-pollination', repo_type='dataset', files_metadata=True)
tb = sum((getattr(s,'size',None) or 0) for s in info.siblings)/1e12
n = sum(1 for s in info.siblings if s.rfilename.startswith('embeddings/'))
q = int(subprocess.check_output(['bash','-c',"find /home/legel/cfp_images -name '*.jpg' 2>/dev/null | wc -l"]).decode())
print(f"manifest: {mu:,} URLs / {m['gbif_occurrence_id'].nunique():,} obs / {m['taxon_name'].nunique():,} species")
print(f"  by role: {m['dataset_role'].value_counts().to_dict()}")
print(f"downloaded: {dl:,} ({100*dl/mu:.1f}%)   embedded: {em:,} ({100*em/mu:.1f}%)")
print(f"HF: {n} shards, {tb:.2f} TB / 8.7 TB ({100*tb/8.7:.1f}%)")
print(f"queue: {q:,} imgs, {len([f for f in os.listdir('/home/legel/cfp_shards') if f.endswith('.parquet')])} shards waiting upload")
PY
```

**Process health**:
```bash
for n in watchdog download embed upload; do
  pid=$(cat outputs/${n}.pid 2>/dev/null)
  alive="DEAD"; [ -d /proc/$pid ] && alive="alive elapsed=$(ps -p $pid -o etime= | tr -d ' ')"
  echo "  $n  pid=$pid  $alive"
done
```

**Tail logs**:
```bash
tail -F logs/chain_{download,embed,upload}.log logs/watchdog_status.log
```

**Disk** (watch for runaway HF cache):
```bash
df -h /home/legel; du -sh /home/legel/.cache/huggingface/hub /home/legel/cfp_shards /home/legel/cfp_images
```

---

## 10. Auth & credentials

| | |
|---|---|
| Hugging Face | `~/.cache/huggingface/token` (write token). Logged in as `ecodash`, member of `deepearth` org. |
| GBIF | `~/.gbif/credentials` (chmod 600): `GBIF_USERNAME=3co`, `GBIF_PASSWORD=…`, `GBIF_EMAIL=lance@3co.ai`. |
| GitHub | `gh` CLI authenticated as `legel`. Repo: `legel/california_flourishing_pollination`. |

---

## 11. Throughput numbers (verified, post-all-fixes)

| Stage | Sustained | Notes |
|---|---|---|
| Downloader | 70-200 URL/s | iNat CDN, network-bound; conc=256 |
| Embedder (DINOv3-L/16 + PhenoVision, bf16, batch 256, GPU decode + async I/O) | **~260-340 img/s sustained, 585 img/s standalone (H200 ceiling)** | GPU util 47-59 % during active; 0 % during DataLoader spawn between rounds |
| nvJPEG batch decode | <1 ms/img | single CUDA call for the whole batch |
| GPU memory | 4-5 GB at batch=256 | H200 has 143 GB; room for batch=2048+ |
| Uploader (`upload_large_folder`) | ~290 MB/s = ~260 shards/hr | one git commit per batch + parallel xet chunks |
| Backfill (shard swap + re-upload) | ~60/hour with 30s pacing | rate-limited to 128 commits/hour by HF |

---

## 12. What is and is NOT in the dataset

**IN:**
- iNaturalist Research-grade observations
- In `country=US, state_province=California`
- With at least one still image (NOT Sound media — see B12)
- Plant species in the Calscape canonical list (8,507 species, **6,383 with photos**)
- Animal species in {Insecta ∪ Trochilidae ∪ Chiroptera ∪ Ptiliogonatidae ∪ Mimidae ∪ Icteridae ∪ Parulidae ∪ Cardinalidae ∪ Bombycillidae} minus Formicidae (**10,063 species with photos**)
- Multi-photo observations contribute one row per photo (mean 1.99 photos/obs)
- Per-photo CC license + rights_holder + creator preserved on every row

**NOT IN:**
- Non-Research-grade iNat observations
- Plants outside Calscape canonical (rare CA natives without Calscape listing)
- Pollinators outside the listed families
- Formicidae (ants — flightless workers, project scope)
- Sound media (filtered post-bug-B12)
- Observations after the GBIF snapshot date (latest 2026-05-23)
- The 22 iNat photos that 404'd at fetch time

---

## 13. Key bugs found and fixed — **READ THIS BEFORE CHANGING ANYTHING**

### B1. DINOv3 ViT-L/16 fp16 → NaN
- Symptom: every embedding was `nan`. fp16 attention overflow.
- Fix: bf16 default. Numerically equivalent to fp32, fp16 speed on H200.
- Cost: 15,643 wasted embeddings; scrubbed.

### B2. Shard-name collision after embedder restart
- Symptom: uploader saw existing remote name, skipped new local shard with same name.
- Fix: every embedder run gets `run_id = "%Y%m%dT%H%M%S"` prefix.

### B3. nvJPEG/decode_image variable 3D or 4D output
- Symptom: F.interpolate raised "5D input" error; embedder crash-looped for ~3 h.
- Fix: defensive shape normalization in extractor (handle 2D/3D/4D, RGBA→RGB, grayscale→RGB).

### B4. Multi-photo observations dropped (gbif_id-keyed checkpoints)
- Symptom: gbif_id-keyed checkpoint kept 1 photo per obs (49 % have >1, max 116). We embedded 4.3 M of 8.5 M.
- Fix: URL-keyed checkpoints + `<gbif_id>_<urlhash8>.<ext>` filenames.

### B5. HF xet upload throttle from per-file uploads
- Symptom: per-file `api.upload_file` works at 30 MB/s → after parallel experiment, throttled to 8 MB/s.
- Fix: `api.upload_large_folder` (one commit + parallel xet chunks). 290 MB/s.

### B6. iNat species_counts page cap silently truncates
- Hard cap `page * per_page ≤ 10,000`. Switched to Calscape Excel as canonical authority.

### B7. iNat `establishment_means=native` ≠ "native to California"
- Returned Australian/Asian non-natives. Use `native=true` (per-place) or Calscape directly.

### B8. `taxon_scheme_id` parameter silently ignored on `/v1/taxa`.

### B9. PhenoVision label swap — flowering and fruiting columns flipped
- **Per `vendor/phenovision/inference.py`: `class_names = ['fruiting', 'flowering']`** — index 0 is fruiting, index 1 is flowering.
- The combined extractor (and the backfill_phenovision.py script) had them swapped (`flowering = probs[:,0]`).
- Impact: every embedding shard uploaded with run_id before `20260524T070916` had the column NAMES correct but the VALUES swapped.
- Fixes:
  - `src/cfp/dinov3/extractor_combined.py` — swapped indexes; future shards correct.
  - Embedder restarted with run_id `20260524T070916`.
  - `scripts/fix_phenovision_swap_on_hf.py` — downloads each old shard, swaps the two column values, re-uploads. 1,200 shards processed. xet dedupes the 99 % unchanged bytes so per-shard transfer is small.
  - `scripts/retry_backfill_fails.py` — paced retry for 178 HfHubHTTPError + xet errors from the first pass.
- Status: **95 % of HF shards now correct**; 5 % are legacy shards with no PhenoVision columns at all (Phase 1.5 work to add them).

### B10. Embedder didn't flush shards for hours
- Symptom: 6 hours after restart, 0 new shards on HF. Embedder logs said "round N embedded 2640 images" but no shards on disk. GPU 0 %.
- Root cause: **2,640 audio files** (`.wav`, `.mp3`, `.mpga`, `.m4a`) leaked into the manifest from the GBIF bird-pollinator batch (`multimedia.type='Sound'` rows). The downloader saved them with `.jpg` extension. The embedder couldn't decode any of them → `flush_batch()` had empty `surviving_paths` → bad files never got deleted → re-scanned forever, clogged the queue, no actual embeddings happened.
- Fixes:
  - `scripts/integrate_birds_and_extras.sh` + `src/cfp/gbif/batch_download.py` — filter `media.type == 'StillImage'` before merging.
  - `src/cfp/pipeline/embed.py` — when a batch image fails nvJPEG, **also delete it from disk** (was only deleting "surviving" successful images).
  - One-shot purge of 2,594 audio URLs from master manifest + 2,637 audio files from disk + corresponding checkpoint entries.

### B11. UMAP encoder load fails on Space (Python-version mismatch)
- Symptom: `TypeError: unsupported operand type(s) for +: 'ABCMeta' and 'dict'` when joblib-loading `umap_encoder.joblib` on the Space.
- Root cause: UMAP trained under Python 3.11 (local env), Space runs Python 3.13. The pickled UMAP class graph can't be reconstructed across Python versions (numba/pynndescent metaclass issue).
- Fix: `scripts/cfp_viewer_space/extract_umap_numpy.py` — extracts the UMAP's `_raw_data` + `embedding_` + channel ranges as a plain numpy `.npz` (206 MB, no class graph). The Space approximates `transform()` via `sklearn.NearestNeighbors` distance-weighted average of neighbor embeddings — visually identical to UMAP's transform for our use case, works under any Python version.

### B12. GBIF Sound media leaking as audio-as-jpg (see B10 root cause)

### B13. Shard-backfill script accumulated 970 GB in HF cache → filled disk
- Symptom: After ~250 shards processed, disk hit 100 %; all pipeline worker PID writes failed; watchdog spawned 9 zombie embed processes.
- Root cause: `hf_hub_download` caches each shard (~3-4 GB). 1,200 shards × 3 GB = 3.6 TB unbounded cache.
- Fix: `scripts/fix_phenovision_swap_on_hf.py` — `_purge_cache()` unlinks both the snapshot symlink and the underlying blob immediately after each successful upload. Cache stays bounded at ~12 GB (workers × shard size).

### B14. HF rate limit: 128 commits/hour per repo
- Symptom: backfill at 6 workers hit limit, started 1-hour sleeps.
- Fix: `scripts/retry_backfill_fails.py` paces at 30 s/commit → 120/hour, safely under limit.

---

## 14. Common operations

**Restart all workers:**
```bash
kill $(cat outputs/watchdog.pid) $(cat outputs/{download,embed,upload}.pid 2>/dev/null)
sleep 5
nohup bash scripts/autonomous_watchdog.sh > logs/watchdog.log 2>&1 &
echo $! > outputs/watchdog.pid
```

**Submit + integrate a new GBIF batch:**
```bash
python -m cfp.gbif batch resolve-keys --plants data/processed/plants_california_native.parquet
python -m cfp.gbif batch submit
python -m cfp.gbif batch wait --poll-seconds 30
# Then write a one-shot integrate script (see scripts/integrate_birds_and_extras.sh template) that:
#  1) waits for both .zip + .meta.json
#  2) parses DwC-A (with type='StillImage' filter — see B12)
#  3) merges into image_manifest.parquet
#  4) pkill -f "cfp.pipeline download" so watchdog relaunches with bigger manifest
nohup bash scripts/integrate_<name>.sh > logs/integrate_<name>.log 2>&1 &
```

**Backfill a column across all shards (downloading + editing + re-uploading):**
```bash
# Template: scripts/fix_phenovision_swap_on_hf.py
# Always include _purge_cache() after each upload (see B13)
# Always pace at <= 120/hour to avoid HF rate limit (see B14)
nohup python scripts/fix_phenovision_swap_on_hf.py --workers 3 > logs/fix_X.log 2>&1 &
```

**Publish updated species/interaction/provenance metadata to HF:**
```bash
python -m cfp.hf publish-meta --repo deepearth/california-flourishing-pollination
```

**Verify a sample embedding shard:**
```bash
python -c "
import pandas as pd, numpy as np
from huggingface_hub import hf_hub_download
p = hf_hub_download('deepearth/california-flourishing-pollination',
                    'embeddings/embeddings_20260524T070916_000000.parquet', repo_type='dataset')
df = pd.read_parquet(p)
r = df.iloc[0]
cls = np.frombuffer(r['cls_fp16'], dtype=np.float16).reshape(r['cls_shape'])
patches = np.frombuffer(r['patches_fp16'], dtype=np.float16).reshape(r['patches_shape'])
print(f'rows={len(df)} taxon={r[\"taxon_name\"]} cls_norm={np.linalg.norm(cls):.2f} pat_std={patches.std():.3f}')
print(f'PhenoVision flowering={r[\"phenovision_flowering_prob\"]:.3f} fruiting={r[\"phenovision_fruiting_prob\"]:.3f}')
"
```

**Refresh the Space**: edit `scripts/cfp_viewer_space/app.py`, then:
```bash
python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='scripts/cfp_viewer_space/app.py', path_in_repo='app.py',
                     repo_id='deepearth/california-flourishing-pollination', repo_type='space',
                     commit_message='<what changed>')
"
```

---

## 15. Sister projects (downstream)

- **`legel/deepearth`** — DeepEarth architecture (Earth4D positional encoding + multi-modal world model).
- **`legel/deepearth/models/flowering`** — flowering forecasting model that will consume this dataset.
- **`legel/deepearth/models/pollination`** — plant-pollinator interaction forecasting.
- **`legel/deepearth/models/fire_ecology`** — current SoTA on Globe-LFMC 2.0 (R²=0.78). DINOv3 features will join the LFMC predictor.
- **`legel/phenovision`** — user's Python port of Phenobase/phenovision (PR #1 fork). Vendored at `vendor/phenovision/`.

---

## 16. Phase 1.5 backlog

1. **Add PhenoVision to the 64 legacy shards** that have only DINOv3 — re-fetch the images, run PhenoVision only, patch the columns.
2. **Raw-image bucket archive** — push photo bytes to `hf://buckets/deepearth/cfp-raw-images` once embedding completes (bucket exists, empty).
3. **PhenoVision spatial localization** — DINOv3 patches × PhenoVision classifier → patch-level flowering probability.
4. **Recover the rare ~2K CA-native plant taxa** beyond Calscape (Jepson MOU).
5. **Expand pollinator scope** further (Aves beyond the 7 added families).
6. **Per-row CC-license filter sidecar** for downstream consumers who need permissive-only subsets.
7. **Train a small ParametricUMAP** (Keras encoder, ~10 MB) instead of the 206 MB knn approximator — would shrink the Space's UMAP load time significantly.

---

## 17. When in doubt

1. **Check the live stats** (§9) first. The pipeline is usually still running.
2. **Check `logs/watchdog_status.log`** for the trend over time (5-min snapshots).
3. **Read `PROVENANCE.md`** for the full data-source story with DOIs.
4. **Read `ASSUMPTIONS.md`** for every operational decision and the risk-if-wrong.
5. **DO NOT** introduce parallelism on HF commits (B5). DO use `upload_large_folder`.
6. **DO NOT** revert to fp16 on DINOv3 (B1). bf16 stays.
7. **DO NOT** use per-`gbif_id` keys for new checkpoints (B4). URL-keyed.
8. **DO NOT** download shards without `_purge_cache()` (B13). Disk WILL fill.
9. **DO NOT** exceed 128 commits/hour to HF (B14). Pace your scripts.
10. **DO NOT** parse DwC-A `multimedia.txt` without filtering `type='StillImage'` (B12).
11. **DO NOT** swap PhenoVision indexes — `index 0 = fruiting, index 1 = flowering` (B9).
12. When you change anything that affects this README, update it.
