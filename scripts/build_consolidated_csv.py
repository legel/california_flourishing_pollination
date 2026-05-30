"""Build a single consolidated CSV from all HF embedding shards, EXCLUDING
the heavy DINOv3 embedding columns (cls_fp16, patches_fp16, cls_shape,
patches_shape, backbone, repo) — just the iNaturalist metadata + PhenoVision
flowering/fruiting probabilities.

Strategy:
  - Stream just the metadata columns from each shard via HfFileSystem +
    pyarrow column projection (HTTP byte-range reads); ~2 MB/shard not 4 GB.
  - Detect which shards have been PhenoVision-swap-corrected by parsing
    commit history; apply value swap for any old-run-id shards that have
    NOT yet been swapped on HF.
  - Join with the master manifest for additional fields (family, kingdom,
    locality, taxon_name_verbatim, rights_holder, creator) not present in
    every shard.
  - Write CSV to disk; upload to HF as a lookup.

Output: lookups/observations_consolidated.csv on HF.
"""
from __future__ import annotations
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "deepearth/california-flourishing-pollination"
FIX_TS = "20260524T070916"  # any shard with run_id >= this has correct PhenoVision labels
OUT_PATH = Path("observations_consolidated.csv")

# Columns to read from each shard (skip the heavy bytes columns)
SHARD_COLS = [
    "gbif_occurrence_id", "taxon_name", "gbif_taxon_key",
    "dataset_role", "license", "inat_observation_id", "image_url_large",
    "observed_on", "decimal_latitude", "decimal_longitude",
    "phenovision_flowering_prob", "phenovision_fruiting_prob",
    "embedded_utc",
]


def _swap_pheno_if_needed(df: pd.DataFrame, shard_path: str,
                           swapped_on_hf: set[str]) -> pd.DataFrame:
    """Return df with phenovision_flowering_prob / fruiting_prob set to their
    semantically-correct values.

    Logic:
      - run_id >= FIX_TS: embedder produced correct labels.
      - run_id <  FIX_TS AND shard is in swapped_on_hf: values on HF are
        correct (column values were swapped by the backfill).
      - run_id <  FIX_TS AND NOT swapped_on_hf: columns are still swapped
        on HF; flip values for the CSV.
    """
    if "phenovision_flowering_prob" not in df.columns:
        return df  # legacy shards with no PhenoVision at all
    m = re.match(r"embeddings/embeddings_(\d{8}T\d{6})_", shard_path)
    new_run = m and m.group(1) >= FIX_TS
    swap_done = shard_path in swapped_on_hf
    needs_swap_here = (not new_run) and (not swap_done)
    if needs_swap_here:
        fl = df["phenovision_flowering_prob"].copy()
        df["phenovision_flowering_prob"] = df["phenovision_fruiting_prob"]
        df["phenovision_fruiting_prob"] = fl
    return df


def _read_one(fs: HfFileSystem, shard_path: str, swapped_on_hf: set[str]) -> Optional[pd.DataFrame]:
    p = f"datasets/{REPO}/{shard_path}"
    try:
        with fs.open(p, "rb") as f:
            present = set(pq.ParquetFile(f).schema.names)
        cols = [c for c in SHARD_COLS if c in present]
        with fs.open(p, "rb") as f:
            tbl = pq.read_table(f, columns=cols)
    except Exception as e:
        print(f"  ! {shard_path}: {type(e).__name__}: {e}", flush=True)
        return None
    df = tbl.to_pandas()
    df = _swap_pheno_if_needed(df, shard_path, swapped_on_hf)
    return df


def _shards_swapped_on_hf(api: HfApi) -> set[str]:
    """Parse commit history to find which old shards have been swap-fixed.
    Commit titles look like 'PhenoVision swap fix: embeddings_xxx.parquet'."""
    swapped = set()
    for c in api.list_repo_commits(REPO, repo_type="dataset"):
        if "PhenoVision swap fix" in c.title:
            m = re.search(r"embeddings_[\d\w_T]+\.parquet", c.title)
            if m:
                swapped.add(f"embeddings/{m.group()}")
    return swapped


def main() -> None:
    t0 = time.time()
    api = HfApi()
    fs = HfFileSystem()

    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                    if f.startswith("embeddings/") and f.endswith(".parquet"))
    print(f"shards: {len(shards):,}", flush=True)

    swapped = _shards_swapped_on_hf(api)
    print(f"PhenoVision-swap-fix commits found: {len(swapped):,}", flush=True)

    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_read_one, fs, s, swapped): s for s in shards}
        n = 0
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None:
                parts.append(df)
            n += 1
            if n % 50 == 0:
                rows = sum(len(p) for p in parts)
                rate = n / max(time.time() - t0, 1)
                print(f"  {n:>4}/{len(shards)} shards · {rows:,} rows · "
                      f"{rate:.1f} shards/s · ETA {(len(shards)-n)/max(rate,1e-6):.0f}s",
                      flush=True)

    print(f"\nconcat {sum(len(p) for p in parts):,} rows…", flush=True)
    df = pd.concat(parts, ignore_index=True)
    print(f"  total rows: {len(df):,}", flush=True)

    # Enrich with manifest fields (family, kingdom, locality, etc.)
    print("\njoining with master manifest…", flush=True)
    m = pd.read_parquet("data/processed/image_manifest.parquet", columns=[
        "gbif_occurrence_id", "image_url_large", "kingdom", "family",
        "taxon_name_verbatim", "rights_holder", "creator", "locality",
        "inat_observation_uuid",
    ])
    df = df.merge(m, on=["gbif_occurrence_id", "image_url_large"], how="left")
    print(f"  after join: {len(df):,} rows · {len(df.columns)} cols", flush=True)
    print(f"  cols: {list(df.columns)}", flush=True)

    # Reorder to put identifiers + PhenoVision first
    front = ["gbif_occurrence_id", "inat_observation_id", "inat_observation_uuid",
             "taxon_name", "taxon_name_verbatim", "gbif_taxon_key",
             "dataset_role", "kingdom", "family",
             "phenovision_flowering_prob", "phenovision_fruiting_prob",
             "observed_on", "decimal_latitude", "decimal_longitude", "locality",
             "image_url_large", "license", "rights_holder", "creator",
             "embedded_utc"]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df = df[cols]

    # Drop the swapped-flag columns we don't need
    df.to_csv(OUT_PATH, index=False)
    sz_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nwrote {OUT_PATH}: {sz_mb:.0f} MB", flush=True)

    # Upload to HF
    api.upload_file(
        path_or_fileobj=str(OUT_PATH),
        path_in_repo="lookups/observations_consolidated.csv",
        repo_id=REPO, repo_type="dataset",
        commit_message=(f"observations_consolidated.csv: every embedded row "
                        f"({len(df):,}) minus the DINOv3 embedding bytes — "
                        f"iNat metadata + PhenoVision flowering/fruiting + "
                        f"taxonomy + per-photo CC license + creator + lat/lng"),
    )
    print(f"uploaded · total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
