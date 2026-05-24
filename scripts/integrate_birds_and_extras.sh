#!/bin/bash
# Wait for the two GBIF batches (bird-pollinators + plant-extras), parse them,
# merge into the master manifest (with Calscape filter for plants, no filter for
# birds), and signal the downloader to relaunch with the bigger manifest.
#
# Bird-pollinator predicate: Trochilidae + Ptiliogonatidae + Mimidae + Icteridae +
#   Parulidae + Cardinalidae + Bombycillidae  (CA × iNat-RG × StillImage)
# Plant-extras predicate:  49 GBIF taxon keys from Calscape unmatched recovery
#   (29 cultivar genera + 32 variety species heads + 62 HIGHERRANK matches,
#   deduplicated). Genus-level expansions over-include — Calscape canonical name
#   filter at parse time keeps only natives in the master manifest.

set -uo pipefail
cd /home/legel/california_flourishing_pollination
PY=/home/legel/miniconda3/envs/cfp/bin/python

LOG=logs/integrate_birds_extras.log
echo "[$(date -Is)] waiting for bird-pollinator + plant-extras batches..." >> $LOG

BIRDS=data/raw/gbif/bird_pollinators_ca_inat.zip
EXTRAS=data/raw/gbif/plants_extras_ca_inat.zip
BIRDS_META=${BIRDS%.zip}.meta.json
EXTRAS_META=${EXTRAS%.zip}.meta.json

# Wait until BOTH zips are present with their meta.json sidecar
until [ -s "$BIRDS" ] && [ -s "$BIRDS_META" ] && [ -s "$EXTRAS" ] && [ -s "$EXTRAS_META" ]; do
    sleep 60
done
echo "[$(date -Is)] both zips ready: birds=$(du -sh $BIRDS|awk '{print $1}'), extras=$(du -sh $EXTRAS|awk '{print $1}')" >> $LOG

$PY <<'PYEOF' >> $LOG 2>&1
import zipfile, re, pandas as pd
from datetime import datetime, timezone

calscape = set(pd.read_parquet('data/processed/plants_california_native.parquet')
               ['scientific_name'].dropna().str.strip())
print(f"[{datetime.utcnow().isoformat()}] canonical Calscape: {len(calscape):,}")

def _clean_name(rank, species, genus, infra, cultivar, fallback):
    rank = str(rank or "").upper()
    species = species if isinstance(species, str) and species else None
    genus = genus if isinstance(genus, str) and genus else None
    infra = infra if isinstance(infra, str) and infra else None
    cultivar = cultivar if isinstance(cultivar, str) and cultivar else None
    if rank == "VARIETY" and species and infra: return f"{species} var. {infra}"
    if rank == "SUBSPECIES" and species and infra: return f"{species} subsp. {infra}"
    if rank in ("FORM","FORMA") and species and infra: return f"{species} f. {infra}"
    if cultivar and genus: return f"{genus} '{cultivar}'"
    if rank == "SPECIES" and species: return species
    if rank == "GENUS" and genus: return genus
    return species or genus or fallback

def parse_dwca(path, role, kingdom, calscape_filter=False):
    with zipfile.ZipFile(path) as z:
        occ = pd.read_csv(z.open("occurrence.txt"), sep="\t", low_memory=False, on_bad_lines="skip")
        media = pd.read_csv(z.open("multimedia.txt"), sep="\t", low_memory=False, on_bad_lines="skip")
    print(f"  {path}: {len(occ):,} occurrences, {len(media):,} media (pre-filter)")
    # Filter media to StillImage only — bird-pollinator batches include
    # `type=Sound` rows (.wav/.mp3 bird recordings) that, if downloaded,
    # masquerade as .jpg and crash the embedder forever.
    if "type" in media.columns:
        before = len(media)
        media = media[media["type"].astype(str).str.lower() == "stillimage"]
        print(f"    StillImage filter: {before:,} -> {len(media):,}")
    if calscape_filter:
        before = len(occ)
        occ = occ[occ["scientificName"].isin(calscape)]
        print(f"    Calscape filter: {before:,} -> {len(occ):,}")
    # IMPORTANT: both DwC-A tables expose `license` and `rightsHolder`. Rename the
    # occurrence-side cols BEFORE merge so pandas doesn't silently suffix them
    # to `license_x`/`license_y` (root cause of the earlier 100%-null bug).
    occ_renamed = occ[["gbifID","scientificName","taxonKey","taxonRank","species","genus",
                       "specificEpithet","infraspecificEpithet","cultivarEpithet","family",
                       "decimalLatitude","decimalLongitude","eventDate","license","rightsHolder",
                       "recordedBy","occurrenceID","verbatimLocality"]].rename(
        columns={"license":"license_occ","rightsHolder":"rightsHolder_occ"}
    )
    m = media.merge(occ_renamed, on="gbifID", how="inner")
    rows = []
    snap = datetime.now(timezone.utc).isoformat()
    for _, r in m.iterrows():
        url = r.get("identifier") or r.get("references")
        if not url or "inaturalist" not in str(url): continue
        large = re.sub(r"/(small|medium|large|original|square)\.(jpe?g|png|gif|webp)",
                       lambda mm: f"/large.{mm.group(2)}", str(url), flags=re.I)
        rows.append({
            "gbif_occurrence_id": int(r["gbifID"]),
            "inat_observation_id": None,
            "inat_observation_uuid": r.get("occurrenceID"),
            "taxon_name": _clean_name(r.get("taxonRank"), r.get("species"), r.get("genus"),
                                       r.get("infraspecificEpithet"), r.get("cultivarEpithet"),
                                       r["scientificName"]),
            "taxon_name_verbatim": r["scientificName"],
            "gbif_taxon_key": int(r["taxonKey"]),
            "taxon_rank": str(r.get("taxonRank") or "").upper(),
            "inat_taxon_id": None,
            "dataset_role": role, "kingdom": kingdom,
            "family": r.get("family"),
            "image_url_large": large, "image_url_original": None,
            "photo_id": None,
            # per-photo license + creator (multimedia.txt — kept as `license`/`rightsHolder`)
            "license": r.get("license"), "rights_holder": r.get("rightsHolder"),
            "creator": r.get("creator"),
            "observed_on": r.get("eventDate"),
            "decimal_latitude": r.get("decimalLatitude"),
            "decimal_longitude": r.get("decimalLongitude"),
            "locality": r.get("verbatimLocality"),
            "recorder_login": r.get("recordedBy"),
            "snapshot_utc": snap,
        })
    df = pd.DataFrame(rows)
    print(f"    -> {len(df):,} image-grain rows, {df['taxon_name'].nunique() if len(df) else 0} unique species")
    return df

print("\n=== BIRDS (no Calscape filter — pollinator role) ===")
df_birds = parse_dwca('data/raw/gbif/bird_pollinators_ca_inat.zip',
                      role='pollinator', kingdom='Animalia', calscape_filter=False)
df_birds.to_parquet('data/processed/image_manifest_birds.parquet', index=False)

print("\n=== PLANT EXTRAS (Calscape filter applied) ===")
df_extras = parse_dwca('data/raw/gbif/plants_extras_ca_inat.zip',
                       role='plant', kingdom='Plantae', calscape_filter=True)
df_extras.to_parquet('data/processed/image_manifest_plants_extras.parquet', index=False)

print("\n=== MERGE INTO MASTER ===")
master = pd.read_parquet('data/processed/image_manifest.parquet')
print(f"  master before: {len(master):,}")
combined = pd.concat([master, df_birds, df_extras], ignore_index=True)
combined = combined.drop_duplicates(subset=['gbif_occurrence_id','image_url_large']).reset_index(drop=True)
print(f"  master after:  {len(combined):,}   delta=+{len(combined)-len(master):,}")
print(f"  by role: {combined['dataset_role'].value_counts().to_dict()}")
print(f"  unique species: {combined['taxon_name'].nunique():,}")
combined.to_parquet('data/processed/image_manifest.parquet', index=False)
print(f"  wrote master manifest")
PYEOF

echo "[$(date -Is)] killing downloader so watchdog relaunches with bigger manifest" >> $LOG
pkill -f "cfp.pipeline download" 2>&1 || true
echo "[$(date -Is)] done" >> $LOG
