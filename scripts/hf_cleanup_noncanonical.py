#!/usr/bin/env python3
"""Scrub embedding shards on the HF dataset of any non-canonical-Calscape plant rows.

For each ``embeddings/*.parquet`` shard on the HF dataset:
  1. Download to a temp file.
  2. Load it.
  3. Filter: keep all dataset_role='pollinator' rows; keep dataset_role='plant'
     rows only if taxon_name is in the canonical Calscape native list.
  4. If any rows were dropped, write the cleaned parquet and upload (overwrite).
  5. If no rows were dropped, do nothing for that shard.

Idempotent. Safe to re-run.
"""

import argparse
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "deepearth/california-flourishing-pollination"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-parquet", default="/home/legel/california_flourishing_pollination/data/processed/plants_california_native.parquet")
    ap.add_argument("--cache-dir", default="/home/legel/_hf_cleanup_cache")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    canonical = set(
        pd.read_parquet(args.canonical_parquet)["scientific_name"].dropna().str.strip()
    )
    print(f"canonical Calscape names: {len(canonical):,}")

    files = sorted([f for f in api.list_repo_files(REPO, repo_type="dataset")
                    if f.startswith("embeddings/")])
    print(f"shards on HF: {len(files)}")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    prov_path = Path("/home/legel/california_flourishing_pollination/provenance") / (
        f"hf_cleanup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.jsonl"
    )
    prov = prov_path.open("w")

    total_rows_before = 0
    total_rows_after = 0
    cleaned_shards = 0
    untouched_shards = 0
    for i, remote_path in enumerate(files, 1):
        local = hf_hub_download(
            repo_id=REPO, repo_type="dataset", filename=remote_path,
            cache_dir=args.cache_dir, force_download=False,
        )
        df = pd.read_parquet(local)
        n_before = len(df)
        plant_mask = df["dataset_role"] == "plant"
        keep = ~plant_mask | df["taxon_name"].isin(canonical)
        dropped = (~keep).sum()

        total_rows_before += n_before
        prov.write({"shard": remote_path, "n_before": int(n_before), "dropped": int(dropped)}.__repr__() + "\n")

        if dropped == 0:
            untouched_shards += 1
            total_rows_after += n_before
            print(f"  [{i}/{len(files)}] {remote_path.split('/')[-1]:55} clean ({n_before:,} rows)")
            continue

        df_clean = df[keep].reset_index(drop=True)
        total_rows_after += len(df_clean)
        if args.dry_run:
            print(f"  [{i}/{len(files)}] {remote_path.split('/')[-1]:55} WOULD drop {dropped:,}/{n_before:,}")
            continue

        # Write cleaned shard to a temp file + upload
        tmp = Path(local).parent / ("_cleaned_" + Path(local).name)
        df_clean.to_parquet(tmp, index=False)
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo=remote_path,
            repo_id=REPO, repo_type="dataset",
            commit_message=f"cleanup: drop {dropped} non-canonical plant rows from {remote_path.split('/')[-1]}",
        )
        tmp.unlink(missing_ok=True)
        Path(local).unlink(missing_ok=True)  # free disk
        cleaned_shards += 1
        print(f"  [{i}/{len(files)}] {remote_path.split('/')[-1]:55} CLEANED dropped {dropped:,}/{n_before:,} → {len(df_clean):,}")

    prov.write(f"SUMMARY total_before={total_rows_before} total_after={total_rows_after} cleaned_shards={cleaned_shards} untouched_shards={untouched_shards}\n")
    prov.close()
    shutil.rmtree(cache, ignore_errors=True)
    print(f"\n=== DONE ===")
    print(f"total rows before: {total_rows_before:,}")
    print(f"total rows after:  {total_rows_after:,}  (removed {total_rows_before-total_rows_after:,})")
    print(f"shards cleaned:    {cleaned_shards}")
    print(f"shards untouched:  {untouched_shards}")
    print(f"provenance:        {prov_path}")


if __name__ == "__main__":
    main()
