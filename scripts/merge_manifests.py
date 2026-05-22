#!/usr/bin/env python3
"""Merge the per-role GBIF manifest builds into one image manifest.

Reads:
    data/processed/image_manifest_plants.parquet
    data/processed/image_manifest_pollinators.parquet

Writes:
    data/processed/image_manifest.parquet      (deduped on (gbif_occurrence_id, image_url_large))
    data/processed/image_manifest_stats.parquet (concatenation of the per-role stats)

The two manifests come from independent runs of `cfp.gbif build-manifest` —
one with `--pollinators` absent (plants-only) and one with `--plant-limit 0`
(pollinators-only).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/home/legel/california_flourishing_pollination")


def main() -> None:
    plants = ROOT / "data/processed/image_manifest_plants.parquet"
    pollinators = ROOT / "data/processed/image_manifest_pollinators.parquet"
    out = ROOT / "data/processed/image_manifest.parquet"
    stats_out = ROOT / "data/processed/image_manifest_stats.parquet"

    dfs = []
    for p in (plants, pollinators):
        if not p.exists():
            print(f"missing {p}; skipping")
            continue
        d = pd.read_parquet(p)
        print(f"read {p.name}: {len(d):,} rows")
        dfs.append(d)
    if not dfs:
        sys.exit("nothing to merge")

    merged = pd.concat(dfs, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=["gbif_occurrence_id", "image_url_large"], keep="first").reset_index(drop=True)
    print(f"merged: {n_before:,} → {len(merged):,} after dedup")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    # Concatenate stats.
    stats_dfs = []
    for p in (
        ROOT / "data/processed/image_manifest_plants_stats.parquet",
        ROOT / "data/processed/image_manifest_pollinators_stats.parquet",
    ):
        if p.exists():
            stats_dfs.append(pd.read_parquet(p))
    if stats_dfs:
        pd.concat(stats_dfs, ignore_index=True).to_parquet(stats_out, index=False)

    # Quick summary
    print()
    print("by role:")
    print(merged["dataset_role"].value_counts().to_string())
    print()
    print(f"per-role unique species:")
    for role, g in merged.groupby("dataset_role"):
        print(f"  {role}: {g['taxon_name'].nunique():,} species, {len(g):,} images")
    print(f"\nwrote: {out}")
    print(f"stats: {stats_out}")

    # Update provenance.
    prov = ROOT / "provenance" / f"manifest_merge_{datetime.now(timezone.utc).isoformat().replace(':','').replace('-','')[:15]}.jsonl"
    prov.parent.mkdir(parents=True, exist_ok=True)
    prov.write_text(json.dumps({
        "type": "merge",
        "stage": "gbif.merge_manifests",
        "merged_utc": datetime.now(timezone.utc).isoformat(),
        "n_plants_rows": int(len(dfs[0])) if len(dfs) > 0 else 0,
        "n_pollinators_rows": int(len(dfs[1])) if len(dfs) > 1 else 0,
        "n_unique_rows": int(len(merged)),
        "out_path": str(out),
        "stats_path": str(stats_out),
    }) + "\n")
    print(f"prov:  {prov}")


if __name__ == "__main__":
    main()
