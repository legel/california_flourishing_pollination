"""Construct the per-image iNaturalist manifest via the GBIF Occurrence API.

For every species in ``--plants`` ∪ ``--pollinators``:

  1. Resolve the GBIF backbone taxonKey (cached).
  2. Page through ``/v1/occurrence/search`` with:
        country=US, stateProvince=California,
        datasetKey=50c9509d-22c7-4a22-a47d-8c48425ef4a7  (iNat Research-grade)
        taxonKey=<key>, hasCoordinate=true (optional), limit=300
     GBIF caps offset+limit at 100 000 — for species above that, we record the
     count and fall back to GBIF Occurrence Download (issued separately).
  3. For every record, extract every photo URL (``media`` list) at "large" size.
  4. Append row(s) to a parquet shard.

The output schema (image-grain):

    gbif_occurrence_id (int64)
    inat_observation_id (int64 | null)
    inat_observation_uuid (utf8 | null)
    taxon_name (utf8)
    gbif_taxon_key (int64)
    inat_taxon_id (int64 | null)
    dataset_role (utf8)              -- 'plant' | 'pollinator'
    kingdom (utf8)
    family (utf8 | null)
    image_url_large (utf8)
    image_url_original (utf8 | null)
    photo_id (int64 | null)
    license (utf8 | null)
    rights_holder (utf8 | null)
    observed_on (date32 | null)
    decimal_latitude (float64 | null)
    decimal_longitude (float64 | null)
    locality (utf8 | null)
    recorder_login (utf8 | null)
    snapshot_utc (utf8)

Per-species image counts and any species that exceed the 100k offset cap are
written to ``data/processed/image_manifest_stats.parquet``.

Run:
    python -m cfp.gbif build-manifest \\
        --plants data/processed/plants_california_native.parquet \\
        --pollinators data/processed/pollinators_california_flying.parquet \\
        --out data/processed/image_manifest.parquet
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

app = typer.Typer(add_completion=False, help="GBIF / iNat image-manifest builder.")
console = Console()

GBIF_API = "https://api.gbif.org/v1"
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
USER_AGENT = "DeepEarth-CFP/0.1 (image-manifest builder; lance@3co.ai)"
PAGE_LIMIT = 300              # GBIF max per page
OFFSET_CAP = 100_000           # GBIF hard cap on offset+limit


# iNaturalist photo URL patterns. We rewrite *all* size suffixes to "large".
_RE_INAT_SIZE = re.compile(r"/(small|medium|large|original|square)\.(jpe?g|png|gif|webp)", re.I)


def _to_large(url: str) -> Optional[str]:
    """Normalize an iNat photo URL to its 'large' variant; return None if not iNat."""
    if not url:
        return None
    if "inaturalist" not in url and "inaturalist-open-data" not in url:
        return None
    return _RE_INAT_SIZE.sub(lambda m: f"/large.{m.group(2)}", url)


def _to_original(url: str) -> Optional[str]:
    """Best-effort 'original' variant for archival reference (we do NOT download it)."""
    if not url:
        return None
    if "inaturalist" not in url and "inaturalist-open-data" not in url:
        return None
    return _RE_INAT_SIZE.sub(lambda m: f"/original.{m.group(2)}", url)


async def _get_with_retry(client: httpx.AsyncClient, url: str, params: dict,
                          max_attempts: int = 5, max_wait: float = 10.0) -> httpx.Response:
    """GBIF-courteous GET — bounded exponential backoff on 429 / 5xx / connect / timeout."""
    wait = 1.0
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = await client.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                if ra is not None:
                    try:
                        wait = max(wait, min(float(ra), max_wait))
                    except ValueError:
                        pass
                await asyncio.sleep(wait)
                wait = min(wait * 2, max_wait)
                continue
            r.raise_for_status()
            return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_exc = e
            await asyncio.sleep(wait)
            wait = min(wait * 2, max_wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"giving up after {max_attempts} retries on {url}")


async def _gbif_match(client: httpx.AsyncClient, name: str, kingdom: Optional[str]) -> Optional[dict]:
    params = {"name": name, "strict": "false"}
    if kingdom:
        params["kingdom"] = kingdom
    try:
        r = await _get_with_retry(client, f"{GBIF_API}/species/match", params)
        j = r.json()
        if "usageKey" in j and j.get("matchType") in {"EXACT", "FUZZY"}:
            return j
    except httpx.HTTPError:
        return None
    return None


async def _gbif_count(client: httpx.AsyncClient, taxon_key: int) -> int:
    r = await _get_with_retry(
        client,
        f"{GBIF_API}/occurrence/search",
        {
            "country": "US",
            "stateProvince": "California",
            "datasetKey": INAT_DATASET_KEY,
            "taxonKey": taxon_key,
            "limit": 0,
        },
    )
    return int(r.json().get("count", 0))


async def _gbif_search_page(client: httpx.AsyncClient, taxon_key: int, offset: int) -> dict:
    r = await _get_with_retry(
        client,
        f"{GBIF_API}/occurrence/search",
        {
            "country": "US",
            "stateProvince": "California",
            "datasetKey": INAT_DATASET_KEY,
            "taxonKey": taxon_key,
            "limit": PAGE_LIMIT,
            "offset": offset,
            "mediaType": "StillImage",
        },
    )
    return r.json()


def _records_to_rows(records: list, *, taxon_name: str, gbif_key: int,
                     inat_taxon_id: Optional[int], role: str, kingdom: str,
                     snapshot: str) -> Iterable[dict]:
    for rec in records:
        media = rec.get("media") or []
        for m in media:
            url = m.get("identifier") or m.get("references")
            large = _to_large(url)
            if not large:
                continue
            # GBIF parses iNat observation refs into references like ".../observations/<id>"
            inat_obs_id = None
            ref = rec.get("references") or ""
            mref = re.search(r"observations/(\d+)", ref)
            if mref:
                try:
                    inat_obs_id = int(mref.group(1))
                except ValueError:
                    pass
            yield {
                "gbif_occurrence_id": int(rec["key"]),
                "inat_observation_id": inat_obs_id,
                "inat_observation_uuid": rec.get("occurrenceID") or rec.get("catalogNumber"),
                "taxon_name": taxon_name,
                "gbif_taxon_key": int(gbif_key),
                "inat_taxon_id": int(inat_taxon_id) if inat_taxon_id is not None else None,
                "dataset_role": role,
                "kingdom": kingdom,
                "family": rec.get("family"),
                "image_url_large": large,
                "image_url_original": _to_original(url),
                "photo_id": None,  # iNat photo_id not surfaced in GBIF response; populated by downloader if needed
                "license": m.get("license") or rec.get("license"),
                "rights_holder": m.get("rightsHolder") or rec.get("rightsHolder"),
                "observed_on": rec.get("eventDate"),
                "decimal_latitude": rec.get("decimalLatitude"),
                "decimal_longitude": rec.get("decimalLongitude"),
                "locality": rec.get("locality") or rec.get("verbatimLocality"),
                "recorder_login": rec.get("recordedBy"),
                "snapshot_utc": snapshot,
            }


async def _process_taxon(
    client: httpx.AsyncClient,
    name: str,
    role: str,
    kingdom_hint: str,
    inat_taxon_id: Optional[int],
    snapshot: str,
) -> tuple[list[dict], dict]:
    """Resolve GBIF key + paginate all iNat-CA records → rows for this taxon."""
    stats = {
        "taxon_name": name, "role": role, "kingdom_hint": kingdom_hint,
        "gbif_key": None, "match_type": None, "total_count": 0,
        "image_rows": 0, "capped_by_gbif_offset": False, "error": None,
    }
    rows: list[dict] = []
    try:
        m = await _gbif_match(client, name, kingdom_hint)
        if not m:
            stats["error"] = "no GBIF match"
            return rows, stats
        key = int(m["usageKey"])
        stats["gbif_key"] = key
        stats["match_type"] = m.get("matchType")
        stats["total_count"] = await _gbif_count(client, key)

        if stats["total_count"] == 0:
            return rows, stats

        offset = 0
        while offset < min(stats["total_count"], OFFSET_CAP):
            page = await _gbif_search_page(client, key, offset)
            recs = page.get("results", []) or []
            for r in _records_to_rows(
                recs, taxon_name=name, gbif_key=key,
                inat_taxon_id=inat_taxon_id, role=role,
                kingdom=kingdom_hint, snapshot=snapshot,
            ):
                rows.append(r)
            offset += PAGE_LIMIT
            if page.get("endOfRecords"):
                break

        if stats["total_count"] > OFFSET_CAP:
            stats["capped_by_gbif_offset"] = True
        stats["image_rows"] = len(rows)
    except httpx.HTTPError as e:
        stats["error"] = f"HTTPError: {e}"
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
    return rows, stats


async def _run(
    species: pd.DataFrame,
    snapshot: str,
    concurrency: int,
    shard_dir: Path,
    shard_rows: int,
    stats_path: Path,
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    timeout = httpx.Timeout(60.0, connect=10.0)
    semaphore = asyncio.Semaphore(concurrency)
    buffer: list[dict] = []
    shard_idx = 0
    all_stats: list[dict] = []

    def _flush() -> None:
        nonlocal buffer, shard_idx
        if not buffer:
            return
        out = shard_dir / f"manifest_shard_{shard_idx:05d}.parquet"
        pd.DataFrame(buffer).to_parquet(out, index=False)
        console.print(f"[green]wrote shard[/] {out}  ({len(buffer)} rows)")
        buffer = []
        shard_idx += 1

    async with httpx.AsyncClient(headers=headers, timeout=timeout, http2=False) as client:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            tid = prog.add_task("species", total=len(species))

            async def worker(row: dict) -> None:
                async with semaphore:
                    try:
                        rows, stats = await asyncio.wait_for(
                            _process_taxon(
                                client,
                                name=row["taxon_name"],
                                role=row["role"],
                                kingdom_hint=row["kingdom_hint"],
                                inat_taxon_id=row.get("inat_taxon_id"),
                                snapshot=snapshot,
                            ),
                            timeout=600,  # cap one species at 10 minutes
                        )
                    except asyncio.TimeoutError:
                        rows = []
                        stats = {
                            "taxon_name": row["taxon_name"], "role": row["role"],
                            "kingdom_hint": row["kingdom_hint"], "gbif_key": None,
                            "match_type": None, "total_count": 0, "image_rows": 0,
                            "capped_by_gbif_offset": False, "error": "TimeoutError: species exceeded 600s",
                        }
                    all_stats.append(stats)
                    buffer.extend(rows)
                    if len(buffer) >= shard_rows:
                        _flush()
                    prog.update(tid, advance=1)

            # return_exceptions so one stuck worker can't strand the whole gather.
            await asyncio.gather(*(worker(r) for _, r in species.iterrows()), return_exceptions=True)

    _flush()
    pd.DataFrame(all_stats).to_parquet(stats_path, index=False)
    console.print(f"[bold green]stats[/] → {stats_path}")


def _coalesce_shards(shard_dir: Path, out: Path) -> int:
    """Concatenate all shards into a single deduplicated manifest parquet."""
    shards = sorted(shard_dir.glob("manifest_shard_*.parquet"))
    if not shards:
        return 0
    tables = [pq.read_table(s) for s in shards]
    table = pa.concat_tables(tables, promote_options="default")
    df = table.to_pandas()
    before = len(df)
    df = df.drop_duplicates(subset=["gbif_occurrence_id", "image_url_large"])
    df.to_parquet(out, index=False)
    return len(df)


@app.command("build-manifest")
def build_manifest(
    plants: Path = typer.Option(Path("data/processed/plants_california_native.parquet"), "--plants"),
    pollinators: Path = typer.Option(Path("data/processed/pollinators_california_flying.parquet"), "--pollinators"),
    out: Path = typer.Option(Path("data/processed/image_manifest.parquet"), "--out"),
    stats_path: Path = typer.Option(Path("data/processed/image_manifest_stats.parquet"), "--stats-path"),
    shard_dir: Path = typer.Option(Path("data/processed/_manifest_shards"), "--shard-dir"),
    shard_rows: int = typer.Option(100_000, "--shard-rows"),
    concurrency: int = typer.Option(8, "--concurrency"),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
    plant_limit: Optional[int] = typer.Option(None, "--plant-limit"),
    pollinator_limit: Optional[int] = typer.Option(None, "--pollinator-limit"),
) -> None:
    """Build the iNaturalist image manifest from plant + pollinator species lists."""
    if not plants.exists():
        raise typer.BadParameter(f"missing {plants}; run `cfp.cnps fetch` first.")

    # Normalize the species inputs to a single (taxon_name, role, kingdom_hint, inat_taxon_id) view.
    p_df = pd.read_parquet(plants)
    species_rows = [
        {"taxon_name": r["scientific_name"], "role": "plant", "kingdom_hint": "Plantae",
         "inat_taxon_id": int(r["inat_taxon_id"]) if pd.notna(r.get("inat_taxon_id")) else None}
        for _, r in p_df.iterrows()
        if pd.notna(r.get("scientific_name")) and (r.get("rank") in ("species", "subspecies", "variety", "form") if "rank" in r else True)
    ]
    if plant_limit is not None:
        species_rows = species_rows[:plant_limit]

    if pollinators.exists():
        a_df = pd.read_parquet(pollinators)
        a_rows = [
            {"taxon_name": r["animal_name"], "role": "pollinator", "kingdom_hint": "Animalia",
             "inat_taxon_id": None}
            for _, r in a_df.iterrows()
            if pd.notna(r.get("animal_name"))
        ]
        if pollinator_limit is not None:
            a_rows = a_rows[:pollinator_limit]
        species_rows += a_rows
    else:
        console.print(f"[yellow]no pollinators parquet at {pollinators} — manifest will be plants-only[/]")

    species = pd.DataFrame(species_rows)
    console.print(f"will query [bold]{len(species)}[/] species across {species['role'].value_counts().to_dict()}")

    snapshot = datetime.now(timezone.utc).isoformat()
    prov_dir.mkdir(parents=True, exist_ok=True)
    prov_path = prov_dir / f"image_manifest_{snapshot.replace(':', '').replace('-','')[:15]}.jsonl"
    with prov_path.open("w") as prov:
        prov.write(json.dumps({
            "type": "run_meta", "stage": "gbif.build_manifest",
            "started_utc": snapshot, "gbif_api": GBIF_API,
            "inat_dataset_key": INAT_DATASET_KEY,
            "page_limit": PAGE_LIMIT, "offset_cap": OFFSET_CAP,
            "n_species": int(len(species)),
            "plants": str(plants), "pollinators": str(pollinators),
            "shard_rows": shard_rows, "concurrency": concurrency,
        }) + "\n")

    asyncio.run(_run(species, snapshot, concurrency, shard_dir, shard_rows, stats_path))

    out.parent.mkdir(parents=True, exist_ok=True)
    n = _coalesce_shards(shard_dir, out)
    with prov_path.open("a") as prov:
        prov.write(json.dumps({
            "type": "result", "manifest_rows": n,
            "manifest_path": str(out), "shard_dir": str(shard_dir),
            "stats_path": str(stats_path),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
    console.rule("[bold green]Manifest complete")
    console.print(f"rows: [bold]{n}[/]  →  {out}")
    console.print(f"per-species stats: {stats_path}")
    console.print(f"provenance: {prov_path}")


if __name__ == "__main__":
    app()
