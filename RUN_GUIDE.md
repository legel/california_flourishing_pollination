# Operator Run Guide

Copy-pasteable sequence to materialize the Phase 1 dataset from a fresh clone. Each stage emits a JSONL provenance log under `provenance/` and a parquet/PNG artifact under `data/` or `outputs/`. Every command is **resumable** — re-running a stage skips work already in its checkpoint.

```bash
# ------------------------------------------------------------------------
# 0. ONE-TIME SETUP
# ------------------------------------------------------------------------
conda activate cfp
pip install -e .                      # editable install of the cfp package
hf auth login                         # write token from huggingface.co/settings/tokens
# Visit and accept DINOv3 licences (gated):
#   https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
#   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

# ------------------------------------------------------------------------
# 1. SPECIES LISTS
# ------------------------------------------------------------------------
# 1a. California native plants
python -m cfp.cnps fetch \
    --outdir data/processed --prov-dir provenance
# -> data/processed/plants_california_native.parquet

# 1b. GloBi snapshot (~2.6 GB)
python -m cfp.globi fetch \
    --raw-dir data/raw/globi --prov-dir provenance
# -> data/raw/globi/interactions.tsv.gz + refuted-interactions.tsv.gz

# 1c. GloBi → CA-native plant × pollinator interactions
python -m cfp.globi filter \
    --raw-dir data/raw/globi \
    --natives data/processed/plants_california_native.parquet \
    --out data/processed/globi_ca_plant_pollinator.parquet \
    --out-candidates data/processed/pollinators_candidates.parquet
# -> two parquets above + provenance JSONL

# 1d. Cross-check: flight ability + ≥1 CA iNat observation per candidate
python -m cfp.pollinators cross-check \
    --candidates data/processed/pollinators_candidates.parquet \
    --rules data/processed/flight_ability_rules.csv \
    --out-kept data/processed/pollinators_california_flying.parquet \
    --out-excluded data/processed/pollinators_excluded.parquet
# -> the final pollinator list

# ------------------------------------------------------------------------
# 2. IMAGE MANIFEST
# ------------------------------------------------------------------------
python -m cfp.gbif build-manifest \
    --plants data/processed/plants_california_native.parquet \
    --pollinators data/processed/pollinators_california_flying.parquet \
    --out data/processed/image_manifest.parquet \
    --stats-path data/processed/image_manifest_stats.parquet
# -> data/processed/image_manifest.parquet (one row per photo)

# ------------------------------------------------------------------------
# 3. DINOV3 SANITY CHECK (REQUIRED BEFORE SCALING)
# ------------------------------------------------------------------------
python -m cfp.dinov3 validate-sample \
    --n 10 --backbone vitb16 --image-size 448 \
    --outdir data/validation/dinov3_sanity
# Open data/validation/dinov3_sanity/dinov3_sanity_vitb16_448.zip
# REVIEW the *_deep_features.png overlays — do they segment the plant coherently?
# If YES, proceed. If NO, debug before burning GPU on the production pass.

# ------------------------------------------------------------------------
# 4. STREAMING PIPELINE (three concurrent stages)
# ------------------------------------------------------------------------
# Each can run in its own tmux/screen pane. They share the filesystem as
# the queue; disk pressure provides natural backpressure.

# 4a. Downloader — pulls iNat photos to /home/legel/cfp_images
python -m cfp.pipeline download \
    --manifest data/processed/image_manifest.parquet \
    --image-dir /home/legel/cfp_images \
    --cap-gb 800 --concurrency 64

# 4b. Embedder — DINOv3 ViT-L/16 on GPU, deletes images on success
python -m cfp.pipeline embed \
    --image-dir /home/legel/cfp_images \
    --shard-dir /home/legel/cfp_shards \
    --backbone vitl16 --image-size 224 --batch-size 64

# 4c. Uploader — pushes shards to HF, deletes local on ack
python -m cfp.pipeline upload \
    --shard-dir /home/legel/cfp_shards \
    --repo deepearth/california-flourishing-pollination \
    --poll-seconds 60       # tail-style: keep watching for new shards

# ------------------------------------------------------------------------
# 5. DATASET CARD + METADATA PUBLICATION
# ------------------------------------------------------------------------
python -m cfp.hf publish-meta \
    --repo deepearth/california-flourishing-pollination
# -> commits README.md (dataset card), PROVENANCE.md, species lists,
#    interactions, manifest, and every provenance JSONL to HF.
```

## Pre-DINOv3-access mode

If DINOv3 access is still pending Meta approval, run stages 1–2 + 4a only. The downloader will fill `--image-dir` up to `--cap-gb` then pause. When DINOv3 lands, start 4b and 4c in parallel and the downloader will resume as space frees.

## Monitoring

Every stage writes Rich progress bars to stdout. For long-running stages, append `2>&1 | tee logs/<stage>.log` so you can inspect after the fact. The JSONL provenance under `provenance/` is the audit trail — every row is one event.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `hf: command not found` | Wrong env active | `conda activate cfp` (or use `/home/legel/miniconda3/bin/hf`) |
| DINOv3 401 / gated | License not accepted on your HF account | Visit the HF model page, click "Agree and access repository" |
| `hf auth login` succeeds but `hf upload` 403s | Token is read-only | Regenerate as **write** token |
| iNat API `422` after 10K rows | Hit page×per_page=10000 cap | Already handled — we cursor-paginate with `id_above` |
| GloBi 404 on supporting files | Optional file moved upstream | Tolerated — only `interactions.tsv.gz` is required |
| Disk fills before embedder runs | `--cap-gb` set too high vs available disk | Lower `--cap-gb` or free space elsewhere |
| Embedder GPU OOM | Batch too large for fp16 + image_size | Lower `--batch-size` (default 64) or `--image-size` |
