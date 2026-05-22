#!/bin/bash
# Wait for the broad-pollinator GBIF batch to finish, then parse + filter +
# merge into the master image_manifest.parquet, then kill the downloader so
# the watchdog relaunches it with the bigger manifest.
#
# Excludes Formicidae (ants, flightless workers) per project scope.

set -uo pipefail
cd /home/legel/california_flourishing_pollination
PY=/home/legel/miniconda3/envs/cfp/bin/python

echo "[$(date -Is)] waiting for broad-pollinator GBIF batch..." >> logs/integrate_broad.log

ZIP=data/raw/gbif/pollinators_broad_ca_inat.zip
META=${ZIP%.zip}.meta.json
until [ -s "$ZIP" ] && [ -s "$META" ]; do
    sleep 60
done
echo "[$(date -Is)] zip ready: $(du -sh $ZIP | awk '{print $1}')" >> logs/integrate_broad.log

$PY <<'PYEOF' >> logs/integrate_broad.log 2>&1
import zipfile, re, pandas as pd
from datetime import datetime, timezone

print(f"[{datetime.utcnow().isoformat()}] parsing DwC-A...")
zp = "data/raw/gbif/pollinators_broad_ca_inat.zip"
with zipfile.ZipFile(zp) as z:
    occ = pd.read_csv(z.open("occurrence.txt"), sep="\t", low_memory=False, on_bad_lines="skip")
    media = pd.read_csv(z.open("multimedia.txt"), sep="\t", low_memory=False, on_bad_lines="skip")
print(f"  occurrence rows: {len(occ):,}, multimedia rows: {len(media):,}")

# Drop Formicidae (ants — flightless workers per project scope)
before = len(occ)
occ_filt = occ[~(occ["family"].fillna("") == "Formicidae")].copy()
print(f"  excluded Formicidae: {before - len(occ_filt):,}")

m = media.merge(
    occ_filt[["gbifID","scientificName","taxonKey","family","decimalLatitude",
              "decimalLongitude","eventDate","license","rightsHolder","recordedBy",
              "occurrenceID","verbatimLocality"]],
    on="gbifID", how="inner"
)

rows = []
snapshot = datetime.now(timezone.utc).isoformat()
for _, r in m.iterrows():
    url = r.get("identifier") or r.get("references")
    if not url or "inaturalist" not in str(url):
        continue
    large = re.sub(r"/(small|medium|large|original|square)\.(jpe?g|png|gif|webp)",
                   lambda mm: f"/large.{mm.group(2)}", str(url), flags=re.I)
    rows.append({
        "gbif_occurrence_id": int(r["gbifID"]),
        "inat_observation_id": None,
        "inat_observation_uuid": r.get("occurrenceID"),
        "taxon_name": r["scientificName"],
        "gbif_taxon_key": int(r["taxonKey"]),
        "inat_taxon_id": None,
        "dataset_role": "pollinator",
        "kingdom": "Animalia",
        "family": r.get("family"),
        "image_url_large": large,
        "image_url_original": None,
        "photo_id": None,
        "license": r.get("license"),
        "rights_holder": r.get("rightsHolder"),
        "observed_on": r.get("eventDate"),
        "decimal_latitude": r.get("decimalLatitude"),
        "decimal_longitude": r.get("decimalLongitude"),
        "locality": r.get("verbatimLocality"),
        "recorder_login": r.get("recordedBy"),
        "snapshot_utc": snapshot,
    })

df_new = pd.DataFrame(rows)
out = "data/processed/image_manifest_pollinators_broad.parquet"
df_new.to_parquet(out, index=False)
print(f"  parsed {len(df_new):,} image-grain rows; {df_new['taxon_name'].nunique()} unique species")

master_path = "data/processed/image_manifest.parquet"
master = pd.read_parquet(master_path)
plants = master[master["dataset_role"] == "plant"]
combined = pd.concat([plants, df_new], ignore_index=True)
combined = combined.drop_duplicates(subset=["gbif_occurrence_id","image_url_large"]).reset_index(drop=True)
combined.to_parquet(master_path, index=False)
print(f"  master manifest now: {len(combined):,} rows ({combined['dataset_role'].value_counts().to_dict()})")
print(f"  unique species: {combined['taxon_name'].nunique():,}")
PYEOF

echo "[$(date -Is)] manifest merged; killing downloader so watchdog relaunches it" >> logs/integrate_broad.log
pkill -f "cfp.pipeline download" 2>&1 || true
echo "[$(date -Is)] done" >> logs/integrate_broad.log
