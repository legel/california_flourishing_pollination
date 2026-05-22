#!/bin/bash
# Autonomous 7-hour watchdog: keeps the 4 critical pipeline processes alive.
#
# Re-launches any that exit. Writes a status snapshot every 5 min to
# logs/watchdog_status.log. Reports throughput, queue depth, HF shard count,
# and disk usage so a human (or future agent) can audit progress at any time.
#
# Designed to be `nohup`-launched and left alone for hours.

set -uo pipefail
cd /home/legel/california_flourishing_pollination

PY=/home/legel/miniconda3/envs/cfp/bin/python
IMG=/home/legel/cfp_images
SHD=/home/legel/cfp_shards
mkdir -p logs "$IMG" "$SHD"

DOWNLOAD_CMD="$PY -m cfp.pipeline download --manifest data/processed/image_manifest.parquet \
    --image-dir $IMG \
    --checkpoint outputs/checkpoint_downloaded.parquet \
    --failed outputs/failed_downloads.parquet \
    --concurrency 256 --per-host-concurrency 64 --cap-gb 800"

# Use --with-phenovision: DINOv3 + PhenoVision in one GPU pass per image.
EMBED_CMD="$PY -m cfp.pipeline embed --image-dir $IMG --shard-dir $SHD \
    --checkpoint outputs/checkpoint_embedded.parquet \
    --backbone vitl16 --image-size 224 --batch-size 32 \
    --images-per-shard 10000 --poll-seconds 60 \
    --gpu-decode --with-phenovision"

UPLOAD_CMD="$PY -m cfp.pipeline upload --shard-dir $SHD \
    --repo deepearth/california-flourishing-pollination \
    --poll-seconds 300"

# Helper: launch $1 under nohup, write its PID to a tracker
launch() {
  local name="$1"; shift
  echo "[$(date -Is)] launching $name" >> logs/watchdog.log
  nohup "$@" > "logs/chain_${name}.log" 2>&1 &
  echo $! > "outputs/${name}.pid"
}

alive() {
  local name="$1"
  local p
  p=$(cat "outputs/${name}.pid" 2>/dev/null) || return 1
  [ -n "$p" ] && [ -d "/proc/$p" ]
}

# Launch any process that isn't currently alive.
ensure() {
  local name="$1"; shift
  if ! alive "$name"; then
    echo "[$(date -Is)] $name not alive — relaunching" >> logs/watchdog.log
    launch "$name" "$@"
    sleep 5
  fi
}

# Status snapshot (writes to logs/watchdog_status.log)
snapshot() {
  $PY <<'PYEOF' >> logs/watchdog_status.log 2>&1
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd, subprocess, json

t = datetime.now(timezone.utc).isoformat()
out = [f"=== {t} ==="]
def kv(k, v): out.append(f"  {k}: {v}")
try:
    dl = len(pd.read_parquet("outputs/checkpoint_downloaded.parquet"))
    kv("downloaded", f"{dl:,}")
except Exception: kv("downloaded", "—")
try:
    em = len(pd.read_parquet("outputs/checkpoint_embedded.parquet"))
    kv("embedded", f"{em:,}")
except Exception: kv("embedded", "—")
try:
    m = pd.read_parquet("data/processed/image_manifest.parquet", columns=["gbif_occurrence_id"])
    kv("manifest_unique", f"{m['gbif_occurrence_id'].nunique():,}")
except Exception: kv("manifest_unique", "—")
imgs = subprocess.run(["bash","-c","find /home/legel/cfp_images -name '*.jpg' 2>/dev/null | wc -l"], capture_output=True, text=True).stdout.strip()
kv("img_queue", imgs)
shards = subprocess.run(["bash","-c","ls /home/legel/cfp_shards 2>/dev/null | wc -l"], capture_output=True, text=True).stdout.strip()
kv("local_shards_to_upload", shards)
disk = subprocess.run(["df","-h","/home/legel"], capture_output=True, text=True).stdout.strip().split("\n")[-1]
kv("disk", disk)
gpu = subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
kv("gpu", gpu)
try:
    from huggingface_hub import HfApi
    n = len([f for f in HfApi().list_repo_files("deepearth/california-flourishing-pollination", repo_type="dataset") if f.startswith("embeddings/")])
    kv("hf_embedding_shards", n)
except Exception: kv("hf_embedding_shards", "—")
for name in ("download", "embed", "upload"):
    p = Path(f"outputs/{name}.pid").read_text().strip() if Path(f"outputs/{name}.pid").exists() else ""
    alive = bool(p and Path(f"/proc/{p}").exists())
    kv(f"{name}_pid", f"{p} {'alive' if alive else 'DEAD'}")
print("\n".join(out))
PYEOF
}

# Initial launch
ensure download $DOWNLOAD_CMD
ensure embed $EMBED_CMD
ensure upload $UPLOAD_CMD

# Main loop: every 60 s, check; every 5 min, snapshot
deadline=$(( $(date +%s) + 7*3600 ))
last_snapshot=0
echo "[$(date -Is)] watchdog start — running for 7 hours" >> logs/watchdog.log
while [ "$(date +%s)" -lt "$deadline" ]; do
  ensure download $DOWNLOAD_CMD
  ensure embed $EMBED_CMD
  ensure upload $UPLOAD_CMD
  now=$(date +%s)
  if [ $((now - last_snapshot)) -ge 300 ]; then
    snapshot
    last_snapshot=$now
  fi
  sleep 60
done
echo "[$(date -Is)] watchdog deadline reached — stopping watchdog (workers continue)" >> logs/watchdog.log
