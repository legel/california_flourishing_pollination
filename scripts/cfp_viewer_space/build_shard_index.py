"""Build `lookups/shard_index.parquet` for the CFP Viewer Space.

Uses HfFileSystem + pyarrow column projection so we transfer only the
`image_url_large` column from each shard (HTTP byte-range reads via the
parquet footer + the relevant column chunks), not the full 3 GB shard.

Total transfer: ~10M rows × tens of bytes per row ≈ <2 GB streamed (vs
3.5 TB of full shards). Runtime ~5-15 min depending on bandwidth.

Parallelized with a ThreadPoolExecutor to keep the HTTP pipeline full.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "deepearth/california-flourishing-pollination"


def _one(fs: HfFileSystem, shard_path: str) -> pd.DataFrame:
    """Stream just the image_url_large column from one HF parquet shard."""
    p = f"datasets/{REPO}/{shard_path}"
    with fs.open(p, "rb") as f:
        tbl = pq.read_table(f, columns=["image_url_large"])
    df = tbl.to_pandas()
    df["shard_path"] = shard_path
    return df


def main() -> None:
    api = HfApi()
    fs = HfFileSystem()
    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                    if f.startswith("embeddings/") and f.endswith(".parquet"))
    print(f"shards: {len(shards)}", flush=True)

    t0 = time.time()
    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_one, fs, s): s for s in shards}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                parts.append(fut.result())
            except Exception as e:
                print(f"  ! {futs[fut]}: {e}", flush=True)
            if i % 25 == 0:
                rows = sum(len(p) for p in parts)
                rate = i / max(1, time.time() - t0)
                eta = (len(shards) - i) / max(1e-6, rate)
                print(f"  {i:>4}/{len(shards)} shards — {rows:,} rows — "
                      f"{rate:.1f} shard/s — ETA {eta:.0f}s", flush=True)

    out = pd.concat(parts, ignore_index=True)
    print(f"\nbuilt index: {len(out):,} rows in {time.time()-t0:.0f}s", flush=True)
    out.to_parquet("shard_index.parquet", index=False)
    print(f"wrote shard_index.parquet ({len(out):,} rows)", flush=True)
    api.upload_file(
        path_or_fileobj="shard_index.parquet",
        path_in_repo="lookups/shard_index.parquet",
        repo_id=REPO, repo_type="dataset",
        commit_message="shard_index.parquet: image_url_large -> shard_path (for the CFP Viewer Space)",
    )
    print("uploaded to HF")


if __name__ == "__main__":
    main()
