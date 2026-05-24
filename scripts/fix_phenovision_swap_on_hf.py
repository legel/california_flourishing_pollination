"""Properly fix the swapped PhenoVision columns in the 1,072 old HF shards.

For each shard with run_id < 20260524T070916:
  1. download (HF cache hit if already local)
  2. swap phenovision_flowering_prob ↔ phenovision_fruiting_prob *values*
     (column names stay the same; the data moves)
  3. write to local temp
  4. upload back to the same path (xet dedupes the unchanged 99% of bytes —
     CLS, patches, metadata — so the network transfer is small even though
     the file is 3-4 GB)
  5. delete local temp

Runs ThreadPoolExecutor with N workers. Progress logged every shard.
"""
from __future__ import annotations
import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "deepearth/california-flourishing-pollination"
FIX_THRESHOLD = "embeddings_20260524T070916"  # any shard < this needs swapping
TMP = Path("/home/legel/cfp_shard_fix_tmp")
TMP.mkdir(exist_ok=True)


def needs_fix(shard_path: str) -> bool:
    """Old shards with run_id < FIX_THRESHOLD have swapped labels.
    Also the 4 earliest shards `embeddings_00000{0..3}.parquet` had labels
    fixed by my earlier backfill_phenovision.py — but that backfill used the
    BUGGY extractor too, so they're ALSO wrong. Include them."""
    name = shard_path.split("/")[-1]
    m = re.match(r"embeddings_(\d{8}T\d{6})_", name)
    if m:
        return m.group(1) < "20260524T070916"
    # No run_id (legacy): always old/wrong
    return True


def fix_one(shard_path: str, api: HfApi) -> str:
    t0 = time.time()
    try:
        local = hf_hub_download(REPO, shard_path, repo_type="dataset")
        df = pd.read_parquet(local)
    except Exception as e:
        return f"FAIL download {shard_path}: {type(e).__name__}: {e}"

    if "phenovision_flowering_prob" not in df.columns:
        return f"SKIP {shard_path}: no phenovision columns"

    # Swap the values
    fl = df["phenovision_flowering_prob"].copy()
    df["phenovision_flowering_prob"] = df["phenovision_fruiting_prob"]
    df["phenovision_fruiting_prob"] = fl

    out = TMP / Path(shard_path).name
    df.to_parquet(out, index=False)

    try:
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=shard_path,
            repo_id=REPO, repo_type="dataset",
            commit_message=f"PhenoVision swap fix: {Path(shard_path).name}",
        )
    except Exception as e:
        return f"FAIL upload {shard_path}: {type(e).__name__}: {e}"
    finally:
        try: out.unlink()
        except OSError: pass

    return f"OK {shard_path} in {time.time()-t0:.0f}s ({len(df):,} rows)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api = HfApi()
    all_shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                        if f.startswith("embeddings/") and f.endswith(".parquet"))
    to_fix = [s for s in all_shards if needs_fix(s)]
    if args.limit:
        to_fix = to_fix[:args.limit]
    print(f"total shards: {len(all_shards)}", flush=True)
    print(f"to fix:       {len(to_fix)}", flush=True)
    print(f"workers:      {args.workers}", flush=True)

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fix_one, s, api): s for s in to_fix}
        for fut in as_completed(futs):
            n_done += 1
            msg = fut.result()
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta_s = (len(to_fix) - n_done) / max(rate, 1e-6)
            print(f"[{n_done:>4}/{len(to_fix)}] {msg}  · {rate:.2f}/s · ETA {eta_s/60:.0f} min",
                  flush=True)


if __name__ == "__main__":
    main()
