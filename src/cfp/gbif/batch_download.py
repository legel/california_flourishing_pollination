"""GBIF Occurrence Download — batch path.

Instead of per-species pagination (which costs ~6 GBIF API calls × 8,108 plant
species at GBIF rate limits = 60+ min), this submits a SINGLE async download
request whose predicate contains every Calscape taxon key. GBIF processes it
server-side, returns one Darwin Core Archive zip with every matching
observation, and assigns a DOI to the download.

Stages:
  1. ``cfp.gbif batch resolve-keys``  — batch /species/match Calscape names to
                                        GBIF backbone taxon keys (concurrency,
                                        retry-on-429). Writes taxon_keys.json.
  2. ``cfp.gbif batch submit``        — submit the download with the predicate.
                                        Writes download_key.json.
  3. ``cfp.gbif batch wait``          — poll the download status every 30 s
                                        until SUCCEEDED, then download the zip.
  4. ``cfp.gbif batch parse``         — parse the DwC-A into our manifest
                                        parquet schema (dataset_role='plant').
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import requests
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

app = typer.Typer(add_completion=False, help="GBIF batch download (Occurrence Download API).")
console = Console()

GBIF_API = "https://api.gbif.org/v1"
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
USER_AGENT = "DeepEarth-CFP/0.1 (GBIF batch download; lance@ecological.dev)"


def _load_creds() -> tuple[str, str, str]:
    p = Path.home() / ".gbif" / "credentials"
    if not p.exists():
        raise typer.BadParameter(f"missing {p} (write GBIF_USERNAME=, GBIF_PASSWORD=, GBIF_EMAIL=)")
    kv: dict[str, str] = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    return kv["GBIF_USERNAME"], kv["GBIF_PASSWORD"], kv["GBIF_EMAIL"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: batch resolve Calscape names → GBIF backbone taxon keys
# ─────────────────────────────────────────────────────────────────────────────


async def _match_one(client: httpx.AsyncClient, name: str, kingdom: str = "Plantae") -> Optional[int]:
    for _ in range(5):
        try:
            r = await client.get(f"{GBIF_API}/species/match",
                                 params={"name": name, "kingdom": kingdom, "strict": "false"})
            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(2.0)
                continue
            r.raise_for_status()
            j = r.json()
            if "usageKey" in j and j.get("matchType") in {"EXACT", "FUZZY"}:
                return int(j["usageKey"])
            return None
        except httpx.HTTPError:
            await asyncio.sleep(2.0)
    return None


async def _resolve_all(names: list[str], concurrency: int, kingdom: str = "Plantae") -> dict[str, Optional[int]]:
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, Optional[int]] = {}
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            tid = prog.add_task("GBIF match", total=len(names))

            async def w(n: str) -> None:
                async with sem:
                    out[n] = await _match_one(client, n, kingdom=kingdom)
                    prog.update(tid, advance=1)

            await asyncio.gather(*(w(n) for n in names))
    return out


@app.command("resolve-keys")
def resolve_keys(
    plants_parquet: Path = typer.Option(Path("data/processed/plants_california_native.parquet"), "--plants"),
    pollinators_parquet: Optional[Path] = typer.Option(None, "--pollinators",
                                                       help="If provided, resolve pollinator names (column 'animal_name') instead."),
    out: Path = typer.Option(Path("data/processed/gbif_taxon_keys.json"), "--out"),
    concurrency: int = typer.Option(16, "--concurrency"),
    kingdom: str = typer.Option("Plantae", "--kingdom"),
) -> None:
    if pollinators_parquet is not None:
        df = pd.read_parquet(pollinators_parquet)
        names = sorted(set(df["animal_name"].dropna().str.strip().tolist()))
    else:
        df = pd.read_parquet(plants_parquet)
        names = sorted(set(df["scientific_name"].dropna().str.strip().tolist()))
    console.print(f"resolving {len(names):,} names → GBIF taxon keys (kingdom={kingdom}, concurrency={concurrency})")
    mapping = asyncio.run(_resolve_all(names, concurrency, kingdom=kingdom))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "n_names": len(names),
        "n_matched": sum(1 for v in mapping.values() if v is not None),
        "name_to_taxon_key": mapping,
    }, indent=2))
    n_match = sum(1 for v in mapping.values() if v is not None)
    console.print(f"matched: {n_match:,} / {len(names):,}  ({100*n_match/len(names):.1f}%)")
    console.print(f"unique GBIF taxon keys: {len({v for v in mapping.values() if v}):,}")
    console.print(f"wrote: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: submit the download with the predicate
# ─────────────────────────────────────────────────────────────────────────────


@app.command("submit")
def submit(
    keys_json: Path = typer.Option(Path("data/processed/gbif_taxon_keys.json"), "--keys"),
    out: Path = typer.Option(Path("data/processed/gbif_download_key.json"), "--out"),
    format_: str = typer.Option("DWCA", "--format", help="DWCA | SIMPLE_CSV | SPECIES_LIST"),
) -> None:
    user, pw, email = _load_creds()
    keys = sorted({v for v in json.loads(keys_json.read_text())["name_to_taxon_key"].values() if v})
    console.print(f"submitting GBIF Occurrence Download for {len(keys):,} taxon keys (format={format_})")

    predicate = {
        "creator": user,
        "notificationAddresses": [email],
        "sendNotification": True,
        "format": format_,
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "DATASET_KEY", "value": INAT_DATASET_KEY},
                {"type": "equals", "key": "COUNTRY", "value": "US"},
                {"type": "equals", "key": "STATE_PROVINCE", "value": "California"},
                {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"},
                {"type": "in", "key": "TAXON_KEY", "values": [str(k) for k in keys]},
            ],
        },
    }
    r = requests.post(f"{GBIF_API}/occurrence/download/request",
                      json=predicate, auth=(user, pw),
                      headers={"User-Agent": USER_AGENT}, timeout=120)
    r.raise_for_status()
    download_key = r.text.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "download_key": download_key,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "format": format_,
        "n_taxon_keys": len(keys),
    }, indent=2))
    console.print(f"download key: [bold]{download_key}[/]")
    console.print(f"saved:        {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: wait for ready + download the zip
# ─────────────────────────────────────────────────────────────────────────────


@app.command("wait")
def wait_(
    key_json: Path = typer.Option(Path("data/processed/gbif_download_key.json"), "--key"),
    out_zip: Path = typer.Option(Path("data/raw/gbif/plants_ca_inat.zip"), "--out-zip"),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds"),
) -> None:
    user, pw, _ = _load_creds()
    key = json.loads(key_json.read_text())["download_key"]
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"polling GBIF for download {key}…")
    while True:
        r = requests.get(f"{GBIF_API}/occurrence/download/{key}", auth=(user, pw),
                         headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        meta = r.json()
        status = meta.get("status")
        console.print(f"  status={status}  records={meta.get('totalRecords')}  size={meta.get('size')}")
        if status == "SUCCEEDED":
            url = meta["downloadLink"]
            console.print(f"[green]ready[/]: {url}")
            with requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=300) as r2:
                r2.raise_for_status()
                total = int(r2.headers.get("Content-Length") or 0)
                with out_zip.open("wb") as f:
                    n = 0
                    for chunk in r2.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        n += len(chunk)
                console.print(f"[green]downloaded[/] {n/1e9:.2f} GB → {out_zip}")
            doi = meta.get("doi")
            (out_zip.with_suffix(".meta.json")).write_text(json.dumps({
                "doi": doi, "status": status, "records": meta.get("totalRecords"),
                "size": meta.get("size"), "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "key": key,
            }, indent=2))
            console.print(f"DOI: {doi}")
            return
        if status in ("KILLED", "CANCELLED", "FAILED"):
            console.print(f"[red]download terminal status: {status}[/]")
            raise typer.Exit(2)
        time.sleep(poll_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: parse DwC-A → manifest parquet
# ─────────────────────────────────────────────────────────────────────────────


@app.command("parse")
def parse_(
    zip_path: Path = typer.Option(Path("data/raw/gbif/plants_ca_inat.zip"), "--zip"),
    keys_json: Path = typer.Option(Path("data/processed/gbif_taxon_keys.json"), "--keys"),
    out: Path = typer.Option(Path("data/processed/image_manifest_plants_gbif_batch.parquet"), "--out"),
) -> None:
    """Parse the GBIF DwC-A into our image-grain manifest parquet."""
    if not zip_path.exists():
        raise typer.BadParameter(f"missing {zip_path}")
    # The DwC-A includes occurrence.txt + multimedia.txt.
    with zipfile.ZipFile(zip_path) as z:
        with z.open("occurrence.txt") as f:
            occ = pd.read_csv(f, sep="\t", low_memory=False, on_bad_lines="skip")
        try:
            with z.open("multimedia.txt") as f:
                media = pd.read_csv(f, sep="\t", low_memory=False, on_bad_lines="skip")
        except KeyError:
            media = pd.DataFrame()

    console.print(f"occurrence rows: {len(occ):,}")
    console.print(f"multimedia rows (pre-filter): {len(media):,}")
    # Filter media to StillImage only — Sound rows leak audio that the
    # downloader saves as .jpg and the embedder can't decode.
    if not media.empty and "type" in media.columns:
        before = len(media)
        media = media[media["type"].astype(str).str.lower() == "stillimage"]
        console.print(f"  StillImage filter: {before:,} -> {len(media):,}")

    # Build name → taxon_key lookup for our Calscape canonical names.
    keys_map = json.loads(keys_json.read_text())["name_to_taxon_key"]
    # Reverse: taxon_key → canonical_calscape_name (for join back)
    rev = {v: k for k, v in keys_map.items() if v}

    # Each row of multimedia is one photo. Inner-join to occurrences for metadata.
    # IMPORTANT: both DwC-A tables expose `license` and `rightsHolder`. Rename the
    # occurrence-side columns BEFORE merge so pandas doesn't silently suffix them
    # to `license_x`/`license_y` (which is how the per-photo license + creator name
    # got lost across 10M manifest rows in earlier runs).
    if not media.empty:
        occ_renamed = occ[["gbifID", "scientificName", "taxonKey", "taxonRank",
                           "species", "genus", "specificEpithet", "infraspecificEpithet",
                           "cultivarEpithet", "decimalLatitude", "decimalLongitude",
                           "eventDate", "license", "rightsHolder", "recordedBy",
                           "occurrenceID", "family", "verbatimLocality"]].rename(columns={
            "license": "license_occ", "rightsHolder": "rightsHolder_occ",
        })
        m = media.merge(occ_renamed, on="gbifID", how="inner")
    else:
        m = pd.DataFrame()

    rows = []
    snapshot = datetime.now(timezone.utc).isoformat()
    import re as _re
    _URL_RX = _re.compile(r"/(small|medium|large|original|square)\.(jpe?g|png|gif|webp)", _re.I)
    for _, r in m.iterrows():
        url = r.get("identifier") or r.get("references")
        if not url or "inaturalist" not in str(url):
            continue
        large = _URL_RX.sub(lambda m_: f"/large.{m_.group(2)}", str(url))
        # Construct a clean canonical taxon name (no authority) from DwC-A fields.
        rank = str(r.get("taxonRank") or "").upper()
        species, genus = r.get("species"), r.get("genus")
        infra, cultivar = r.get("infraspecificEpithet"), r.get("cultivarEpithet")
        if rank == "VARIETY" and species and isinstance(infra, str) and infra:
            clean = f"{species} var. {infra}"
        elif rank == "SUBSPECIES" and species and isinstance(infra, str) and infra:
            clean = f"{species} subsp. {infra}"
        elif rank in ("FORM", "FORMA") and species and isinstance(infra, str) and infra:
            clean = f"{species} f. {infra}"
        elif isinstance(cultivar, str) and cultivar and isinstance(genus, str):
            clean = f"{genus} '{cultivar}'"
        elif rank == "SPECIES" and isinstance(species, str) and species:
            clean = species
        elif rank == "GENUS" and isinstance(genus, str) and genus:
            clean = genus
        else:
            clean = species or genus or r["scientificName"]
        rows.append({
            "gbif_occurrence_id": int(r["gbifID"]),
            "inat_observation_id": None,
            "inat_observation_uuid": r.get("occurrenceID"),
            "taxon_name": rev.get(int(r["taxonKey"]), clean),  # canonical Calscape name when known
            "taxon_name_verbatim": r["scientificName"],         # GBIF scientificName with authority
            "gbif_taxon_key": int(r["taxonKey"]),
            "taxon_rank": rank,
            "inat_taxon_id": None,
            "dataset_role": "plant",
            "kingdom": "Plantae",
            "family": r.get("family"),
            "image_url_large": large,
            "image_url_original": None,
            "photo_id": None,
            # per-photo license + creator (multimedia.txt — kept as `license`/`rightsHolder`)
            "license": r.get("license"),
            "rights_holder": r.get("rightsHolder"),
            "creator": r.get("creator"),
            "observed_on": r.get("eventDate"),
            "decimal_latitude": r.get("decimalLatitude"),
            "decimal_longitude": r.get("decimalLongitude"),
            "locality": r.get("verbatimLocality"),
            "recorder_login": r.get("recordedBy"),
            "snapshot_utc": snapshot,
        })

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    console.print(f"[green]wrote[/] {len(df):,} image-grain rows → {out}")
    console.print(f"unique species: {df['taxon_name'].nunique():,}")
    console.print(f"unique occurrences: {df['gbif_occurrence_id'].nunique():,}")
