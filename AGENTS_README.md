# AGENTS_README — California Flourishing & Pollination

> **For future agents working on this project.** This document is the single source of truth for what's running, what's been built, and what to know before changing anything. Read it end-to-end before acting. Keep it up to date when you change the system.

---

## 1. Project at a glance

**Goal.** Produce a self-supervised, multi-modal, analysis-ready dataset of every Research-grade iNaturalist observation of every California-native plant and every California-observed flying pollinator, encoded with DINOv3 ViT-L/16 spatial features and PhenoVision flower/fruit probabilities. Published as `deepearth/california-flourishing-pollination` on Hugging Face. Powers downstream forecasting models in [`legel/deepearth/models/`](https://github.com/legel/deepearth/tree/main/models).

**Collaboration.** Ecological Intelligence, Inc. (Lance Legel, PI; lance@ecological.dev; ecological.dev) × Quantitative Ecosystem Dynamics Lab at UC Berkeley (Trevor Keenan, PI; keenangroup.info). Affiliated-Organization research collaboration starting 2026-05-04.

**Citation (planned).** Legel, L. & Keenan, T. (2026). _California Flourishing & Pollination: a multi-modal AI dataset for ecological forecasting_. Hugging Face Dataset & companion manuscript.

**License.** MIT for code, generated parquets, embeddings, manifests. DINOv3 features are transformative derivatives of iNat photos — we never redistribute the photo bytes; per-photo license + URL are preserved per row.

---

## 2. Live state (snapshot, update when you change anything)

| | |
|---|---|
| Master manifest | `data/processed/image_manifest.parquet` — **9,851,832 URLs / 5,000,043 observations / 16,400 species** (6,432 plants + 9,968 pollinators) |
| Embedded so far | ~6.5M (66 %); HF has ~780 shards / 2.6 TB |
| HF dataset | https://huggingface.co/datasets/deepearth/california-flourishing-pollination |
| Storage quota | 8.7 TB total; ~30 % used; projected ~3.8 TB final |
| Hardware | 1 × NVIDIA H200 (143 GB VRAM); ~1.3 TB local disk on `/home/legel`; 800 GB image-cache cap |
| Pipeline ETA | ~5-8 h to embed all remaining; download is now the slow side |

Always re-check with the live-stats one-liner in §9.

---

## 3. Repository layout

```
/home/legel/california_flourishing_pollination/
├── README.md                — public README
├── PROVENANCE.md            — exhaustive per-source provenance (CNPS, GloBi, DINOv3, etc.)
├── ASSUMPTIONS.md           — every operational assumption + risk-if-wrong
├── PIPELINE.md              — streaming architecture overview
├── RUN_GUIDE.md             — operator copy-paste recipe to rerun from scratch
├── SESSION_HANDOVER.md      — historical handover from the long autonomous run
├── AGENTS_README.md         — THIS FILE
├── docs/
│   ├── FLOURISHING.md       — plant-side track docs
│   ├── POLLINATION.md       — pollinator-side track docs
│   └── DINOV3_RUN.md        — concise CV-engineer-oriented run brief
├── src/cfp/                 — Python package (`pip install -e .`)
│   ├── cnps/
│   │   ├── calscape.py        # ingest Calscape Excel export (canonical CA-native list)
│   │   ├── fetch.py           # legacy iNat species_counts fetch (superseded)
│   │   └── match_inat.py      # robust iNat+GBIF cross-reference
│   ├── globi/
│   │   └── fetch_and_filter.py # GloBi download + RO-ontology pollination filter
│   ├── pollinators/
│   │   └── cross_check.py     # GloBi candidates × GBIF CA-presence × flight-ability
│   ├── gbif/
│   │   ├── build_manifest.py  # legacy per-species GBIF iteration (slow, deprecated)
│   │   └── batch_download.py  # GBIF Occurrence Download API (CURRENT path)
│   ├── dinov3/
│   │   ├── extractor.py        # PIL-path baseline (CPU decode, fallback)
│   │   ├── extractor_gpu.py    # nvJPEG GPU-decode DINOv3
│   │   ├── extractor_combined.py  # DINOv3 + PhenoVision in one GPU session (CURRENT)
│   │   ├── visualize.py        # UMAP→RGB overlay (validation)
│   │   └── validate_sample.py  # n-image DINOv3 sanity-check CLI
│   ├── pipeline/
│   │   ├── download.py         # async iNat photo downloader (URL-keyed checkpoint)
│   │   ├── embed.py            # GPU embedder (DataLoader workers, --gpu-decode, --with-phenovision)
│   │   └── upload.py           # batched upload_large_folder uploader (CURRENT)
│   └── hf/
│       └── publish_meta.py    # one-shot push of species lists + dataset card
├── scripts/
│   ├── autonomous_watchdog.sh           # keeps download+embed+upload alive 7h
│   ├── integrate_broad_pollinators.sh   # wait + parse + merge broad-pollinator DwC-A
│   ├── analyze_calscape_globi.py        # plant×pollinator network analysis
│   ├── hf_cleanup_noncanonical.py       # one-shot scrub HF shards to canonical Calscape
│   └── merge_manifests.py               # legacy merge helper (superseded by batch_download)
├── configs/pipeline.example.yaml
├── data/
│   ├── raw/                  # source snapshots (gitignored, kept as sha256-attested archives)
│   │   ├── globi/            # GloBi interactions.tsv.gz + refuted
│   │   ├── calscape/         # archived "Native To California.xlsx"
│   │   └── gbif/             # plants_ca_inat.zip + pollinators_*.zip DwC-A archives
│   └── processed/            # canonical parquets (gitignored except flight_ability_rules.csv)
│       ├── plants_california_native.parquet      # 8,507 Calscape canonical
│       ├── pollinators_california_flying.parquet # 1,275 GloBi-cross-checked
│       ├── pollinators_candidates.parquet
│       ├── pollinators_excluded.parquet
│       ├── globi_ca_plant_pollinator.parquet     # 45,805 CA interactions
│       ├── image_manifest.parquet                # **MASTER manifest (URL-grain)**
│       ├── image_manifest_plants_gbif_batch.parquet
│       ├── image_manifest_pollinators_broad.parquet  # broad pollinator GBIF batch
│       ├── gbif_taxon_keys.json                  # name → GBIF taxon key (plants)
│       ├── gbif_taxon_keys_pollinators.json
│       ├── gbif_download_key.json                # GBIF batch download key (plants)
│       ├── gbif_download_key_pollinators_broad.json
│       └── flight_ability_rules.csv              # tracked in git
├── outputs/                  # checkpoints + PIDs (gitignored except chain_pids.json)
│   ├── checkpoint_downloaded.parquet    # URL-keyed
│   ├── checkpoint_embedded.parquet      # URL-keyed
│   ├── checkpoint_*.legacy_gbifid.parquet  # pre-multi-photo-fix backups
│   ├── failed_downloads.parquet
│   ├── chain_pids.json
│   └── {watchdog,download,embed,upload,upload2}.pid
├── provenance/               # per-stage JSONL audit logs (every API query, sha256, DOI)
├── logs/                     # per-process logs
└── vendor/phenovision/       # user's Python port of phenobase/phenovision (PR #1 fork)

External:
/home/legel/cfp_images/       — image-cache buckets (gitignored; downloader → embedder pulls from here)
/home/legel/cfp_shards/       — local embedding shards (gitignored; uploader pushes from here)
/home/legel/.gbif/credentials — GBIF account creds (chmod 600)
/home/legel/.cache/huggingface/token — HF auth token
```

---

## 4. The four critical long-running processes

Watchdog (`scripts/autonomous_watchdog.sh`) re-launches any of these on death every 60 s and writes a status snapshot to `logs/watchdog_status.log` every 5 min:

| Name | Command | Key flags |
|---|---|---|
| **download** | `python -m cfp.pipeline download` | `--manifest data/processed/image_manifest.parquet --image-dir /home/legel/cfp_images --concurrency 256 --per-host-concurrency 64 --cap-gb 800` |
| **embed** | `python -m cfp.pipeline embed` | `--image-dir /home/legel/cfp_images --shard-dir /home/legel/cfp_shards --backbone vitl16 --image-size 224 --batch-size 32 --gpu-decode --with-phenovision --poll-seconds 60` |
| **upload** | `python -m cfp.pipeline upload` | `--shard-dir /home/legel/cfp_shards --repo deepearth/california-flourishing-pollination --poll-seconds 300` |
| **watchdog** | `scripts/autonomous_watchdog.sh` | runs 7 h then exits; rearm with `nohup bash scripts/autonomous_watchdog.sh > logs/watchdog.log 2>&1 &` |

PIDs are in `outputs/{name}.pid`. To stop everything cleanly:
```bash
kill $(cat outputs/watchdog.pid) $(cat outputs/{download,embed,upload}.pid 2>/dev/null)
```

---

## 5. Data sources and the GBIF batch download path

### 5a. Plants — CNPS Calscape (canonical) + GBIF Occurrence Download

- Calscape export `~/Native To California.xlsx` (manual UI download from `calscape.org/search` because Cloudflare Turnstile blocks all automated access). Archived at `data/raw/calscape/native_to_california.xlsx` (sha256 attested).
- Ingested by `cfp.cnps calscape ingest` → `data/processed/plants_california_native.parquet` (8,507 taxa × 50 trait fields).
- GBIF batch download: predicate `{DATASET_KEY=50c9509d-... (iNat Research-grade), COUNTRY=US, STATE_PROVINCE=California, MEDIA_TYPE=StillImage, TAXON_KEY IN <7,556 keys resolved via /species/match>}`. **DOI `10.15468/dl.pbgs4h`**; 3.6M occurrences; DwC-A at `data/raw/gbif/plants_ca_inat.zip` (2.0 GB); parsed → `data/processed/image_manifest_plants_gbif_batch.parquet`.
- CNPS definition of native (verbatim, https://cnps.org/gardening/why-natives/what-are-native-plants):
  > Our native plants grew here prior to European contact. California's native plants evolved here over a very long period, and are the plants which the first Californians knew and depended on for their livelihood.

### 5b. Pollinators — broad GBIF batch (Insecta + Trochilidae + Chiroptera)

- Same GBIF predicate, but `TAXON_KEY IN [216 Insecta, 5289 Trochilidae, 734 Chiroptera]`. **DOI `10.15468/dl.cvbfp4`**; 1.5M occurrences; DwC-A at `data/raw/gbif/pollinators_broad_ca_inat.zip` (0.83 GB); parsed → `data/processed/image_manifest_pollinators_broad.parquet`.
- Formicidae (ants — flightless workers) **excluded at parse time** per project scope (see `scripts/integrate_broad_pollinators.sh`).

### 5c. Plant × pollinator interaction graph — GloBi (label/auxiliary)

- Source: `https://depot.globalbioticinteractions.org/snapshot/target/data/tsv/interactions.tsv.gz` (concept DOI `10.5281/zenodo.3950589`, Poelen et al. 2014).
- 2.6 GB compressed; archived at `data/raw/globi/interactions.tsv.gz` (sha256 `85723ad5d7d86bf6f8e9ccd325a68a67d2e3fb61c4c32d3a15634fe49bf819ee`).
- Filter via RO ontology IRIs (apply to `interactionTypeId`, **not** label):
  - `RO_0002455` pollinates / `RO_0002456` pollinatedBy
  - `RO_0002622` visitsFlowersOf / `RO_0002623` flowersVisitedBy
- CA geographic filter: bbox `[-124.55,-114.13] × [32.53, 42.01]` OR locality regex `(?i)\b(California|Calif\.|\bCA\b)\b` excluding `Baja\s+California`.
- Output: `data/processed/globi_ca_plant_pollinator.parquet` (45,805 rows). Auxiliary signal for downstream pollination-network models.

### 5d. PhenoVision (flower / fruit labels)

- Model: `phenobase/phenovision` (ViT-B/16, MIT, Dinnage 2025 — Methods in Ecology and Evolution 16(8):1763).
- Loaded alongside DINOv3 in the combined extractor; output `flowering_prob` + `fruiting_prob` ∈ [0,1] per image.
- Image processor: ImageNet mean/std at 224² (same as DINOv3 — they share the preprocessed tensor).

### 5e. DINOv3

- Model: `facebook/dinov3-vitl16-pretrain-lvd1689m` (300M params, gated). License accepted under HF account `ecodash`.
- Precision: **bf16** (fp16 → NaN in attention end-to-end; see ASSUMPTIONS.md §E).
- Input: 224² RGB. Output: CLS `(1024,)` + 4 register tokens (discarded) + 14×14 patch tokens `(14, 14, 1024)`.
- Stored as `np.float16` bytes in parquet (`cls_fp16`, `patches_fp16` columns; sizes in `cls_shape`, `patches_shape`).

---

## 6. Pipeline architecture

```
GBIF Occurrence Download (one-shot per role) ─┐
                                              ▼
                              data/processed/image_manifest.parquet  (9.85M URL rows)
                                              │
                            ┌─────────────────┴─────────────────┐
                            ▼                                   ▼
                       DOWNLOADER                            (resumable via
            (async aiohttp, conc=256, cap=800GB)        outputs/checkpoint_downloaded.parquet
                            │                              keyed by image_url_large)
                            ▼
              /home/legel/cfp_images/<gbif%1000>/<gbif_id>_<urlhash8>.jpg + .json
                            │
                            ▼
                        EMBEDDER  (GPU)
       ┌──────────────────────────────────────────────────────────────┐
       │  DataLoader workers (num_workers=12-16) read raw JPEG bytes  │
       │  Main thread:                                                │
       │    1. nvJPEG decode_jpeg(device='cuda')   ~2 ms/img          │
       │    2. F.interpolate to 224² (bilinear+antialias)             │
       │    3. ImageNet normalize on GPU tensor                       │
       │    4. DINOv3 ViT-L/16 forward  → CLS + patches               │
       │    5. PhenoVision ViT-B/16 forward → sigmoid(logits)         │
       │    6. fp16 cast, parquet shard write, image unlink           │
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
       │  Re-query remote, delete local shards confirmed on HF        │
       └──────────────────────────────────────────────────────────────┘
                            │
                            ▼
       https://huggingface.co/datasets/deepearth/california-flourishing-pollination
```

Backpressure: each stage runs at its own pace; the disk acts as the queue. If downloader gets too far ahead it self-throttles at `--cap-gb 800`. If embedder lags, queue grows; once embedder catches up, queue drains.

---

## 7. Embedding shard schema

```python
{
  "gbif_occurrence_id": int64,
  "inat_observation_id": int64 | None,
  "inat_observation_uuid": str | None,
  "taxon_name": str,                # scientificName from GBIF
  "gbif_taxon_key": int64,
  "inat_taxon_id": int64 | None,
  "dataset_role": "plant" | "pollinator",
  "kingdom": "Plantae" | "Animalia",
  "family": str | None,
  "image_url_large": str,           # iNat S3 URL — we don't redistribute the photo
  "image_url_original": str | None,
  "license": str | None,            # per-photo iNat license
  "rights_holder": str | None,
  "observed_on": str | None,
  "decimal_latitude": float64 | None,
  "decimal_longitude": float64 | None,
  "locality": str | None,
  "recorder_login": str | None,
  "snapshot_utc": str,
  "cls_fp16": bytes,                # np.float16 (1024,) packed
  "patches_fp16": bytes,            # np.float16 (14, 14, 1024) packed
  "cls_shape": [1024],
  "patches_shape": [14, 14, 1024],
  "backbone": "vitl16",
  "repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
  "phenovision_flowering_prob": float32,
  "phenovision_fruiting_prob": float32,
  "phenovision_repo": "phenobase/phenovision",
  "embedded_utc": str,
}
```

Decode at use time:
```python
import numpy as np, pandas as pd
df = pd.read_parquet("embeddings/embeddings_*.parquet")
r = df.iloc[0]
cls = np.frombuffer(r["cls_fp16"], dtype=np.float16).reshape(r["cls_shape"])
patches = np.frombuffer(r["patches_fp16"], dtype=np.float16).reshape(r["patches_shape"])
```

---

## 8. Key bugs found and fixed (DO NOT regress these)

### B1. DINOv3 ViT-L/16 fp16 → NaN
- Symptom: every embedding cls/patch was `nan` end-to-end. fp16 attention overflow.
- Fix: bf16 default in `extractor*.py`. Numerically equivalent to fp32 at fp16 speed.
- Discovery cost: 15,643 wasted embeddings; scrubbed.

### B2. Shard-name collision after embedder restart
- Symptom: uploader saw existing remote name, skipped local shard with same name but different content.
- Fix: every embedder run gets a `run_id = "%Y%m%dT%H%M%S"` prefix; shard names are `embeddings_<run_id>_<idx>.parquet`.

### B3. nvJPEG/decode_image variably 3D or 4D output
- Symptom: F.interpolate raised "5D input" error; embedder crash-looped for ~3 h. Watchdog kept relaunching at 1/min.
- Fix: defensive shape normalization in `extractor_combined.py` and `extractor_gpu.py` (handle 2D/3D/4D, RGBA→RGB, grayscale→RGB).
- Lesson: every decoder error path needs explicit `failed_idx` tracking — never crash a batch on one bad image.

### B4. Multi-photo observations dropped (gbif_id-keyed checkpoints)
- Symptom: download/embed checkpoints were `gbif_occurrence_id`-keyed. iNat observations average 1.99 photos (49% have >1, max 116). We embedded only one photo per obs ⇒ 4.3M instead of 8.5M.
- Fix: checkpoints + image filenames switched to URL-keyed (`<gbif_id>_<urlhash8>.<ext>` + `image_url_large` checkpoint column). Legacy gbif_id-keyed checkpoints backed up to `*.legacy_gbifid.parquet`.

### B5. HF xet upload throttle from parallel per-file uploads
- Symptom: per-file `api.upload_file` works at 30 MB/s → after parallel-uploader experiment, throttled to 8 MB/s and stalls entirely.
- Fix: rewrite uploader to use `api.upload_large_folder` (ONE commit per batch + parallel xet chunks). Measured 290 MB/s effective = 260 shards/hour vs 2/hour throttled.
- Lesson: HF Hub rate-limits commit creation; one large commit is dramatically better than many small ones for bulk LFS work.

### B6. iNat species_counts page cap silently truncates
- Symptom: `/v1/observations/species_counts` hard-caps `page * per_page ≤ 10,000`. We thought we had all CA-native taxa; we had only top-by-obs-count.
- Resolution: switched to Calscape Excel (manual export) as canonical authority. iNat is now a secondary cross-reference. Note that `not_in_taxon_id` does NOT continue past the cap on this endpoint (verified).

### B7. iNat `establishment_means=native` ≠ "native to California"
- Symptom: returned 18,676 taxa including Australian (`Melaleuca`), African (`Encephalartos`), Asian (`Camptotheca`) species — community-curation noise.
- Resolution: use `native=true` shortcut instead (the per-place strict flag). 8,869 clean taxa. But: Calscape Excel is the truer source — use that, with iNat as a join for taxon_id + obs counts.

### B8. `taxon_scheme_id` parameter silently ignored on `/v1/taxa`
- Don't try to use iNat taxon scheme #12 (Calflora mirror) as a query filter; the API ignores it.

---

## 9. Inspect-the-live-system one-liners

Latest stats:
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

Process health:
```bash
for n in watchdog download embed upload; do
  pid=$(cat outputs/${n}.pid 2>/dev/null)
  alive="DEAD"; [ -d /proc/$pid ] && alive="alive elapsed=$(ps -p $pid -o etime= | tr -d ' ') cpu=$(ps -p $pid -o pcpu= | tr -d ' ')%"
  echo "  $n  pid=$pid  $alive"
done
```

GPU duty cycle (rapid sample):
```bash
PEAK=0; SUM=0; for i in $(seq 1 50); do U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits); SUM=$((SUM+U)); [ "$U" -gt "$PEAK" ] && PEAK=$U; sleep 0.1; done
echo "PEAK: ${PEAK}%   AVG: $((SUM/50))%"
```

Tail logs:
```bash
tail -F logs/chain_{download,embed,upload}.log logs/watchdog_status.log
```

---

## 10. Auth & credentials

| | |
|---|---|
| Hugging Face | `~/.cache/huggingface/token` (write token). Logged in as `ecodash`, member of `deepearth` org. `hf` CLI at `/home/legel/miniconda3/bin/hf` (also in `cfp` env). |
| GBIF | `~/.gbif/credentials` (chmod 600): `GBIF_USERNAME=3co`, `GBIF_PASSWORD=…`, `GBIF_EMAIL=lance@3co.ai`. Required for Occurrence Download submissions. |
| GitHub | `gh` CLI authenticated as `legel`. Code repos: `legel/california_flourishing_pollination` (this), `legel/deepearth` (downstream models), `legel/phenovision` (Python port of PhenoVision). |
| Bucket | `hf://buckets/deepearth/cfp-raw-images` exists but currently empty (Phase 1.5 raw-image archive target). |

---

## 11. Throughput numbers (verified)

| Stage | Sustained | Notes |
|---|---|---|
| Downloader | ~100-200 img/sec | iNat CDN, network-bound; conc=256 |
| Embedder (DINOv3 ViT-L/16 + PhenoVision, bf16, batch 32, gpu_decode) | ~170 img/sec (170-370 peak with no PhenoVision) | H200 with nvJPEG decode |
| Embed without PhenoVision | ~500 img/sec | PhenoVision adds ~30% per-batch overhead |
| nvJPEG single-image decode | ~2 ms | 5× faster than PIL on H200 |
| GPU util | duty cycle ~20-40% avg, 100% peak | network/decoder-bound, GPU has headroom |
| Uploader (upload_large_folder, batched commit) | ~290 MB/s = ~260 shards/hr | xet's batch path |
| Uploader (per-file upload_file, throttled) | 8-30 MB/s = 2-72 shards/hr | DO NOT USE |

---

## 12. What is and is NOT in the dataset

**IN:**
- iNaturalist Research-grade observations
- In `country=US, state_province=California`
- With at least one still image
- Plant species in the Calscape canonical list (8,507 species, 6,432 with photos)
- Animal species in {Insecta ∪ Trochilidae ∪ Chiroptera} minus Formicidae (9,968 species with photos)
- Multi-photo observations contribute one row per photo (mean 1.99 photos/obs)

**NOT IN:**
- Non-Research-grade iNat observations (Casual + Verifiable)
- Plants outside Calscape canonical (rare CA natives without Calscape listing)
- Pollinators outside Insecta+Trochilidae+Chiroptera (excludes some flower-visiting birds, e.g. Mimidae)
- Formicidae (ants — flightless workers, per project scope)
- Observations recorded after the GBIF snapshot date (2026-05-22)

Coverage is honest and Nature-paper-defensible; expanding scope is a Phase 1.5 task.

---

## 13. Phase 1.5 backlog (deferred work)

1. **Raw-image bucket archive** — push photo bytes to `hf://buckets/deepearth/cfp-raw-images` as a ~24 h background sync once embedding completes. Bucket exists, empty. ~2.4 TB.
2. **PhenoVision backfill** — the first ~575K embeddings on HF predate the combined extractor (DINOv3 only, no PhenoVision). Re-process those specific image URLs through PhenoVision only and patch in the columns.
3. **PhenoVision spatial localization** — DINOv3 patches × PhenoVision classifier → patch-level flowering probability. Or train a flower-segmentation head on the DINOv3 features.
4. **Recover the rare ~8K tail** of CA-native plant taxa beyond Calscape's 8,507. Path: Jepson MOU or family-level GBIF query splitting.
5. **Expand pollinator scope** further (Aves beyond Trochilidae — orioles, tanagers, warblers as occasional flower visitors).
6. **Re-fetch iNat establishment_means with the proper `native=true` filter** for the 35-row leakage in the current Calscape parquet's `inat_taxon_id` / `ca_observation_count` columns.
7. **Add per-row CC-license filter** as a sidecar parquet so downstream consumers can quickly subset to permissive licenses.

---

## 14. Common operations

**Restart all workers (after manifest change, code change, or accidental kill):**
```bash
# kill running watchdog & workers
kill $(cat outputs/watchdog.pid) $(cat outputs/{download,embed,upload}.pid 2>/dev/null)
sleep 5
# relaunch watchdog (it will spawn the three workers)
nohup bash scripts/autonomous_watchdog.sh > logs/watchdog.log 2>&1 &
echo $! > outputs/watchdog.pid
```

**Resubmit a GBIF batch download** (when you want to expand species/predicate scope):
```bash
# 1. Update keys
python -m cfp.gbif batch resolve-keys --plants data/processed/plants_california_native.parquet
# 2. Submit
python -m cfp.gbif batch submit
# 3. Wait + download
python -m cfp.gbif batch wait --poll-seconds 30
# 4. Parse to manifest parquet
python -m cfp.gbif batch parse
# 5. Merge into master manifest (see scripts/integrate_broad_pollinators.sh for the pattern)
```

**Publish the species/interaction/provenance metadata to HF** (run after manifest changes):
```bash
python -m cfp.hf publish-meta --repo deepearth/california-flourishing-pollination
```

**Quick GPU verification** (test a single batch through DINOv3+PhenoVision):
```bash
python -c "
import sys; sys.path.insert(0,'src')
from cfp.dinov3.extractor_combined import DINOv3PhenoVisionExtractor
from pathlib import Path
import time, torch
e = DINOv3PhenoVisionExtractor()
imgs = list(Path('/home/legel/cfp_images').rglob('*.jpg'))[:64]
bufs = [p.read_bytes() for p in imgs]
e.embed_from_bytes(bufs[:4])  # warmup
torch.cuda.synchronize()
t0 = time.time()
out, fail = e.embed_from_bytes(bufs)
torch.cuda.synchronize()
print(f'{len(bufs)/(time.time()-t0):.0f} img/sec  has_nan={(out.cls!=out.cls).any() or (out.patches!=out.patches).any()}  failed={len(fail)}')
"
```

---

## 15. Sister projects (downstream)

- **`legel/deepearth`** — the AI model architecture (Earth4D positional encoding + multi-modal world model).
- **`legel/deepearth/models/flowering`** — flowering forecasting model that will consume this dataset.
- **`legel/deepearth/models/pollination`** — plant-pollinator interaction forecasting.
- **`legel/deepearth/models/fire_ecology`** — current SoTA on Globe-LFMC 2.0 (R²=0.78). DINOv3 features will join the LFMC predictor.
- **`legel/phenovision`** — the user's Python port of `Phenobase/phenovision` (upstream PR #1, unmerged). Vendored at `vendor/phenovision/`.

---

## 16. When in doubt

1. **Check the live stats** (§9) first. The pipeline is usually still running — don't restart what's already working.
2. **Check `logs/watchdog_status.log`** for the trend over time (5-min snapshots).
3. **Read `PROVENANCE.md`** for the full data-source story with DOIs.
4. **Read `ASSUMPTIONS.md`** for every operational decision and the risk-if-wrong.
5. **DO NOT** introduce parallelism on HF commits (B5). DO use `upload_large_folder`.
6. **DO NOT** revert to fp16 on DINOv3 (B1). bf16 stays.
7. **DO NOT** use per-`gbif_id` keys for new checkpoints (B4). URL-keyed.
8. When you change anything that affects this README, update it.
