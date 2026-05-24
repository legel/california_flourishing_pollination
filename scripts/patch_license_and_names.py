"""One-shot remediation: rebuild the per-photo license + rights_holder
+ clean-taxon-name columns from the 4 archived GBIF DwC-A zips, patch
the master manifest, and write two sidecar parquets that HF dataset
consumers can join to existing embedding shards.

Why this exists:
  The original parser at `cfp.gbif.batch_download:parse_` and
  `scripts/integrate_birds_and_extras.sh` did `media.merge(occ[..., "license",
  "rightsHolder", ...])`. Both DwC-A tables expose those column names, so
  pandas suffixed them to `license_x` / `license_y`. The parser then read
  `r.get("license")` and got `None`. All 10.3M manifest rows + all 1,072
  uploaded embedding shards lost the per-photo CC license + creator name.

  Separately, taxon_name came from GBIF's `scientificName` which includes
  the taxonomic authority (e.g. "Apis mellifera Linnaeus, 1758"). The
  DwC-A also exposes a clean `species` field plus `genus`/`taxonRank`/
  `infraspecificEpithet`/`cultivarEpithet` that we can use to construct
  a clean canonical name without authority.

Inputs (already on disk, no re-download):
  data/raw/gbif/plants_ca_inat.zip
  data/raw/gbif/pollinators_broad_ca_inat.zip
  data/raw/gbif/bird_pollinators_ca_inat.zip
  data/raw/gbif/plants_extras_ca_inat.zip

Outputs:
  data/processed/image_manifest.parquet                  -- patched in place
  data/processed/photo_attribution.parquet              -- (url -> license, rights_holder, creator)
  data/processed/taxon_clean_names.parquet              -- (gbif_taxon_key -> clean_name, rank)
  (these two are uploaded to HF as sidecar lookups for existing shards)
"""
from __future__ import annotations
import re
import time
import zipfile
from pathlib import Path
import pandas as pd


ZIPS = [
    "data/raw/gbif/plants_ca_inat.zip",
    "data/raw/gbif/pollinators_broad_ca_inat.zip",
    "data/raw/gbif/bird_pollinators_ca_inat.zip",
    "data/raw/gbif/plants_extras_ca_inat.zip",
]


_URL_LARGE_RX = re.compile(r"/(small|medium|large|original|square)\.(jpe?g|png|gif|webp)", re.I)


def _to_large(url: str) -> str:
    """Normalize an iNat photo URL to the /large.* variant we manifest on."""
    if not isinstance(url, str):
        return url
    return _URL_LARGE_RX.sub(lambda m: f"/large.{m.group(2)}", url)


def _s(v) -> str:
    """Coerce NaN/None to empty string; pass through strings stripped."""
    if isinstance(v, str):
        return v.strip()
    return ""


def _build_clean_name(rank: str, species: str, genus: str, infra: str,
                       cultivar: str, fallback: str) -> str:
    """Construct a clean canonical taxon name (no authority) from DwC-A fields."""
    rank = rank.upper()
    if rank == "VARIETY" and species and infra:
        return f"{species} var. {infra}"
    if rank == "SUBSPECIES" and species and infra:
        return f"{species} subsp. {infra}"
    if rank in ("FORM", "FORMA") and species and infra:
        return f"{species} f. {infra}"
    if cultivar and genus:
        return f"{genus} '{cultivar}'"
    if rank == "SPECIES" and species:
        return species
    if rank == "GENUS" and genus:
        return genus
    return species or genus or fallback or ""


def parse_zip(zip_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (photo_attribution_df, taxon_clean_df) from one DwC-A zip."""
    t0 = time.time()
    print(f"\n=== {zip_path} ===", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        occ = pd.read_csv(
            z.open("occurrence.txt"), sep="\t", low_memory=False, on_bad_lines="skip",
            usecols=[
                "gbifID", "scientificName", "taxonKey", "taxonRank",
                "species", "genus", "specificEpithet", "infraspecificEpithet",
                "cultivarEpithet",
            ],
        )
        media = pd.read_csv(
            z.open("multimedia.txt"), sep="\t", low_memory=False, on_bad_lines="skip",
            usecols=["gbifID", "identifier", "license", "rightsHolder", "creator"],
        )
    print(f"  occurrences: {len(occ):,}   media: {len(media):,}   "
          f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Photo-grain attribution: (gbifID, url_large) -> (license, rights_holder, creator)
    media = media[media["identifier"].astype(str).str.contains("inaturalist", na=False)]
    media["image_url_large"] = media["identifier"].astype(str).map(_to_large)
    photo = media[["gbifID", "image_url_large", "license", "rightsHolder", "creator"]].rename(
        columns={"gbifID": "gbif_occurrence_id", "rightsHolder": "rights_holder"}
    )

    # Taxon-key-grain clean name (dedupe first → one entry per taxon, then build name)
    taxon = occ[["taxonKey", "taxonRank", "scientificName",
                 "species", "genus", "specificEpithet", "infraspecificEpithet",
                 "cultivarEpithet"]].drop_duplicates(subset=["taxonKey"]).reset_index(drop=True)
    taxon["clean_taxon_name"] = [
        _build_clean_name(_s(r), _s(s), _s(g), _s(i), _s(c), _s(sn))
        for r, s, g, i, c, sn in zip(
            taxon["taxonRank"], taxon["species"], taxon["genus"],
            taxon["infraspecificEpithet"], taxon["cultivarEpithet"], taxon["scientificName"],
        )
    ]
    taxon = taxon.rename(columns={
        "taxonKey": "gbif_taxon_key", "taxonRank": "taxon_rank",
        "scientificName": "scientific_name_with_authority",
    })
    print(f"  → photo rows: {len(photo):,}   unique taxa: {len(taxon):,}", flush=True)
    return photo, taxon


def main() -> None:
    photos, taxa = [], []
    for zp in ZIPS:
        p = Path(zp)
        if not p.exists():
            print(f"SKIP missing: {zp}", flush=True)
            continue
        ph, tx = parse_zip(zp)
        photos.append(ph)
        taxa.append(tx)

    photo_df = pd.concat(photos, ignore_index=True).drop_duplicates(
        subset=["gbif_occurrence_id", "image_url_large"]
    ).reset_index(drop=True)
    taxon_df = pd.concat(taxa, ignore_index=True).drop_duplicates(
        subset=["gbif_taxon_key"]
    ).reset_index(drop=True)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    photo_df.to_parquet("data/processed/photo_attribution.parquet", index=False)
    taxon_df.to_parquet("data/processed/taxon_clean_names.parquet", index=False)
    print(f"\nwrote data/processed/photo_attribution.parquet ({len(photo_df):,} rows)")
    print(f"wrote data/processed/taxon_clean_names.parquet  ({len(taxon_df):,} rows)")

    # === Patch master manifest ===
    print("\n=== patching master manifest ===", flush=True)
    master = pd.read_parquet("data/processed/image_manifest.parquet")
    print(f"  master rows: {len(master):,}")

    # Photo-grain join
    join_keys = ["gbif_occurrence_id", "image_url_large"]
    photo_lk = photo_df[join_keys + ["license", "rights_holder", "creator"]]
    before = len(master)
    if "license" in master.columns:
        master = master.drop(columns=["license"])
    if "rights_holder" in master.columns:
        master = master.drop(columns=["rights_holder"])
    if "creator" in master.columns:
        master = master.drop(columns=["creator"])
    master = master.merge(photo_lk, on=join_keys, how="left")
    assert len(master) == before, f"merge changed row count: {before} → {len(master)}"
    print(f"  license non-null: {master['license'].notna().sum():,} ({100*master['license'].notna().mean():.1f}%)")
    print(f"  rights_holder non-null: {master['rights_holder'].notna().sum():,}")
    print(f"  creator non-null:       {master['creator'].notna().sum():,}")

    # Taxon clean name join
    taxon_lk = taxon_df[["gbif_taxon_key", "clean_taxon_name", "taxon_rank"]]
    master = master.merge(taxon_lk, on="gbif_taxon_key", how="left")
    # Keep the original `taxon_name` (verbatim scientificName w/ authority) and add a clean column.
    print(f"  clean_taxon_name non-null: {master['clean_taxon_name'].notna().sum():,} "
          f"({100*master['clean_taxon_name'].notna().mean():.1f}%)")
    # Stats on authority-suffix reduction
    auth_rx = re.compile(r"(\d{4}\)?|[A-Z]\.\s|Linn|et al\.|ex\s+[A-Z])")
    n_dirty_before = master["taxon_name"].dropna().astype(str).map(
        lambda s: bool(auth_rx.search(s))).sum()
    n_dirty_after = master["clean_taxon_name"].dropna().astype(str).map(
        lambda s: bool(auth_rx.search(s))).sum()
    print(f"  authority-pattern matches: {n_dirty_before:,} (original) → {n_dirty_after:,} (clean)")

    master.to_parquet("data/processed/image_manifest.parquet", index=False)
    print(f"  wrote patched master ({len(master):,} rows)")


if __name__ == "__main__":
    import os
    os.chdir("/home/legel/california_flourishing_pollination")
    main()
