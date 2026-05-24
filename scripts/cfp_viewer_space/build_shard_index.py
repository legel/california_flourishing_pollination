"""Build `lookups/shard_index.parquet` for the CFP Viewer Space.

Streams parquet metadata from each HF shard (only the `image_url_large`
column, not the embedding bytes), tags each row with its shard path, and
writes the union as a single parquet. Result is ~200 MB compressed —
small enough to keep loaded in a Gradio Space worker so per-request
lookups are O(1).

Total transfer: ~10M rows × tens of bytes per row ≈ a few GB streamed
once (vs 3.5 TB of full shards). Runtime ~20-30 min on a fast link.
"""
from __future__ import annotations
import pandas as pd
from huggingface_hub import HfApi, hf_hub_url
import pyarrow.parquet as pq
import io
import requests

REPO = "deepearth/california-flourishing-pollination"


def main() -> None:
    api = HfApi()
    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                    if f.startswith("embeddings/") and f.endswith(".parquet"))
    print(f"shards: {len(shards)}")

    parts = []
    for i, s in enumerate(shards):
        url = hf_hub_url(REPO, s, repo_type="dataset")
        # Stream only one column — pyarrow + HTTP byte-range
        # (simpler: just download the parquet and read one column; HF caches it
        # so re-runs are fast)
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(REPO, s, repo_type="dataset")
        df = pd.read_parquet(local, columns=["image_url_large"])
        df["shard_path"] = s
        parts.append(df)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(shards)} shards — {sum(len(p) for p in parts):,} rows so far")

    out = pd.concat(parts, ignore_index=True)
    out.to_parquet("shard_index.parquet", index=False)
    print(f"wrote shard_index.parquet ({len(out):,} rows)")
    api.upload_file(
        path_or_fileobj="shard_index.parquet",
        path_in_repo="lookups/shard_index.parquet",
        repo_id=REPO, repo_type="dataset",
        commit_message="shard_index.parquet: image_url_large -> shard_path (for the CFP Viewer Space)",
    )
    print("uploaded to HF")


if __name__ == "__main__":
    main()
