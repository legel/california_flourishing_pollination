#!/bin/bash
# Chained Phase 1 orchestrator.
#
# Waits for the two GBIF manifest builds to finish (they must already be
# running), then merges them, then starts the streaming download → embed →
# upload pipeline concurrently. Designed to be invoked once and left running.
#
# Usage:
#   bash scripts/run_phase1_chain.sh
#
# Logs go to logs/chain_<stage>.log.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/legel/miniconda3/envs/cfp/bin/python
ROOT="$(pwd)"
PLANTS_MANIFEST="$ROOT/data/processed/image_manifest_plants.parquet"
POLL_MANIFEST="$ROOT/data/processed/image_manifest_pollinators.parquet"
MERGED_MANIFEST="$ROOT/data/processed/image_manifest.parquet"
IMAGE_DIR=/home/legel/cfp_images
SHARD_DIR=/home/legel/cfp_shards
HF_REPO=deepearth/california-flourishing-pollination

mkdir -p logs "$IMAGE_DIR" "$SHARD_DIR"

# 1. Wait for both manifest builds to finish — sentinel is the final parquet
#    being non-empty (the builders create it at the very end).
echo "[$(date -Is)] waiting for both manifests to land…"
until [ -s "$PLANTS_MANIFEST" ] && [ -s "$POLL_MANIFEST" ]; do
    sleep 30
done
echo "[$(date -Is)] both manifests present"

# 2. Merge.
echo "[$(date -Is)] merging manifests…"
$PY scripts/merge_manifests.py 2>&1 | tee logs/chain_merge.log

# 3. Start the three streaming stages concurrently.
echo "[$(date -Is)] starting download stage (cap 800 GB)…"
nohup $PY -m cfp.pipeline download \
    --manifest "$MERGED_MANIFEST" \
    --image-dir "$IMAGE_DIR" \
    --checkpoint outputs/checkpoint_downloaded.parquet \
    --failed outputs/failed_downloads.parquet \
    --concurrency 64 --cap-gb 800 \
    >> logs/chain_download.log 2>&1 &
DL_PID=$!
echo "[$(date -Is)] downloader started pid=$DL_PID"

echo "[$(date -Is)] starting embed stage (ViT-L/16, 224, batch 64)…"
nohup $PY -m cfp.pipeline embed \
    --image-dir "$IMAGE_DIR" \
    --shard-dir "$SHARD_DIR" \
    --checkpoint outputs/checkpoint_embedded.parquet \
    --backbone vitl16 --image-size 224 --batch-size 64 \
    --images-per-shard 10000 \
    >> logs/chain_embed.log 2>&1 &
EMB_PID=$!
echo "[$(date -Is)] embedder started pid=$EMB_PID"

echo "[$(date -Is)] starting upload stage (poll every 5 min)…"
nohup $PY -m cfp.pipeline upload \
    --shard-dir "$SHARD_DIR" \
    --repo "$HF_REPO" \
    --poll-seconds 300 \
    >> logs/chain_upload.log 2>&1 &
UP_PID=$!
echo "[$(date -Is)] uploader started pid=$UP_PID"

# Persist the PIDs so the operator can later kill / wait individually.
cat > outputs/chain_pids.json <<EOF
{
  "downloader": $DL_PID,
  "embedder":   $EMB_PID,
  "uploader":   $UP_PID,
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[$(date -Is)] chain orchestrator finished setup; the three stages run independently in background"
echo "[$(date -Is)] PIDs persisted to outputs/chain_pids.json"
