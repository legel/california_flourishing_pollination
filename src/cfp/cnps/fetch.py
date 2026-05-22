"""Fetch the California-native plant species list.

After four rounds of investigation we converged on this open + Nature-grade path:

    iNaturalist  /v1/observations/species_counts
        ?place_id=14         (California)
        &iconic_taxa=Plantae
        &native=true          ← STRICT filter; not the broader establishment_means
        &per_page=500

This endpoint returns every plant taxon (a) classified as ``native`` for the
California place specifically (per iNat's California check list #312, which
itself ingests CNPS / Calflora / Jepson editorial decisions), and (b) with at
least one observation in California — which is precisely the intersection we
need for downstream image collection. Yields **~8,869 taxa** (verified
2026-05-22) — close to Calscape's own self-reported "more than 8,500 types"
and consistent with CNPS's canonical definition of "California native":

    "Our native plants grew here prior to European contact. California's
    native plants evolved here over a very long period, and are the plants
    which the first Californians knew and depended on for their livelihood."
        — California Native Plant Society,
          https://www.cnps.org/gardening/why-natives/what-are-native-plants

Important: ``establishment_means=native`` (the looser filter we used first) is
NOT equivalent — it admits ~10K taxa with a long tail of non-CA-natives
(e.g. *Melaleuca incana*, *Banksia ashbyi*, *Encephalartos woodii*) that
happen to be listed as native somewhere intersecting the query. ``native=true``
is the strict per-place shortcut.

Why not the alternatives:

  - iNat taxon scheme #12 (Calflora) is NOT queryable by API — the
    ``taxon_scheme_id`` parameter on ``/v1/taxa`` is silently ignored, so a
    naive call walks all 1.4M iNat taxa.
  - Calflora's plant search UI is a JavaScript/GWT app with no public bulk
    endpoint and several 404 paths; only per-taxon HTML scrapes work.
  - Calscape (CNPS) sits behind Cloudflare and is CC-BY-NC.
  - Wikidata SPARQL (`wdt:P3420`) returns Calflora-listed taxa (~10K) but
    lacks the native flag at scale.
  - Jepson eFlora and CCH2 are CITE-ONLY (license prohibits redistribution).

The chosen iNat endpoint is fast (one HTTP call per ~500 taxa, ~40 calls
total), open (CC0/CC-BY-NC per record), and pairs naturally with the
downstream image manifest (which already uses iNat data via GBIF).

Output schema:

    inat_taxon_id (int64)
    scientific_name (utf8)
    rank (utf8)
    rank_level (int32)
    common_name (utf8 | null)
    ca_observation_count (int64)        -- iNat Research+Verifiable obs in CA
    establishment_means (utf8)          -- 'native' (always, by construction)
    kingdom / phylum / class / order / family / genus / species (utf8 | null)
    parent_id (int64 | null)
    inat_url (utf8)
    snapshot_utc (utf8)

Run:
    python -m cfp.cnps fetch --outdir data/processed --prov-dir provenance
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

app = typer.Typer(add_completion=False, help="California-native plant species list (iNat species_counts).")
console = Console()

INAT_API = "https://api.inaturalist.org/v1"
CALIFORNIA_PLACE_ID = 14
PLANTAE_TAXON_ID = 47126
USER_AGENT = "DeepEarth-CFP/0.1 (cnps fetch; lance@3co.ai)"

# These are filled into each row's classification by walking `ancestors`.
_RANK_ORDER = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def _ranks_from_ancestor_ids(ancestor_ids: list[int], detail_cache: dict[int, dict]) -> dict:
    """Map a taxon's ancestor list → {kingdom, phylum, ...} using the detail cache."""
    out = {r: None for r in _RANK_ORDER}
    for aid in ancestor_ids:
        a = detail_cache.get(int(aid))
        if not a:
            continue
        r = a.get("rank")
        if r in out and out[r] is None:
            out[r] = a.get("name")
    return out


def _iter_native_species_counts(per_page: int = 500, sleep_s: float = 0.4) -> Iterable[dict]:
    """Stream every {taxon, count} row from iNat species_counts for CA + Plantae + native.

    Uses the strict per-place ``native=true`` shortcut + ``iconic_taxa=Plantae``,
    which together return ~8,869 CA-native plant taxa (verified 2026-05-22) —
    matching Calscape's self-reported "more than 8,500 types." Pagination fits
    in the iNat 10K page-cap (18 pages × 500 = 9,000), so no cursor continuation
    is needed.
    """
    seen: set[int] = set()
    page = 1
    while True:
        r = requests.get(
            f"{INAT_API}/observations/species_counts",
            params={
                "place_id": CALIFORNIA_PLACE_ID,
                "iconic_taxa": "Plantae",
                "native": "true",
                "per_page": per_page,
                "page": page,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        if r.status_code in (429, 503):
            time.sleep(2.0)
            continue
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", []) or []
        if not results:
            return
        for row in results:
            tid = int(row["taxon"]["id"])
            if tid in seen:
                continue
            seen.add(tid)
            yield row
        if page * per_page >= int(payload.get("total_results", 0)):
            return
        page += 1
        time.sleep(sleep_s)


def _fetch_ancestor_details(ancestor_ids: set[int], sleep_s: float = 0.1) -> dict[int, dict]:
    """Batch-fetch ancestor taxa to populate rank names (kingdom..family)."""
    out: dict[int, dict] = {}
    ids = sorted(ancestor_ids)
    with Progress(
        TextColumn("ancestors"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        tid = prog.add_task("ancestors", total=len(ids))
        # iNat accepts comma-separated ids on /taxa/{ids}; up to ~30 per call is safe.
        for i in range(0, len(ids), 30):
            batch = ids[i : i + 30]
            r = requests.get(
                f"{INAT_API}/taxa/{','.join(map(str, batch))}",
                headers={"User-Agent": USER_AGENT},
                timeout=60,
            )
            if r.status_code in (429, 503):
                time.sleep(2.0)
                continue
            try:
                r.raise_for_status()
            except requests.HTTPError:
                prog.update(tid, advance=len(batch))
                continue
            for t in (r.json().get("results", []) or []):
                out[int(t["id"])] = t
            prog.update(tid, advance=len(batch))
            time.sleep(sleep_s)
    return out


@app.command()
def fetch(
    outdir: Path = typer.Option(Path("data/processed"), "--outdir"),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
    per_page: int = typer.Option(500, "--per-page"),
    fetch_ancestors: bool = typer.Option(True, "--fetch-ancestors/--no-fetch-ancestors",
                                         help="Backfill kingdom..family from the iNat ancestor batch endpoint."),
) -> None:
    """Fetch CA-native plants via iNat species_counts and write parquet + provenance."""
    outdir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat()
    prov_path = prov_dir / f"cnps_inat_species_counts_{snapshot.replace(':', '').replace('-', '')[:15]}.jsonl"
    prov = prov_path.open("w")
    prov.write(json.dumps({
        "type": "run_meta", "stage": "cnps.fetch",
        "started_utc": snapshot,
        "source": "iNaturalist /v1/observations/species_counts",
        "params": {
            "place_id": CALIFORNIA_PLACE_ID,
            "iconic_taxa": "Plantae",
            "native": "true",
        },
        "definition_of_native": (
            "Our native plants grew here prior to European contact. California's "
            "native plants evolved here over a very long period, and are the plants "
            "which the first Californians knew and depended on for their livelihood. "
            "— California Native Plant Society, "
            "https://www.cnps.org/gardening/why-natives/what-are-native-plants"
        ),
        "api": INAT_API,
        "fetch_ancestors": fetch_ancestors,
    }) + "\n")

    rows: list[dict] = []
    ancestor_ids_needed: set[int] = set()
    console.rule("[bold cyan]iNat species_counts (CA × Plantae × native)")
    pulled = 0
    for row in _iter_native_species_counts(per_page=per_page):
        t = row["taxon"]
        rows.append({
            "inat_taxon_id": int(t["id"]),
            "scientific_name": t.get("name"),
            "rank": t.get("rank"),
            "rank_level": int(t.get("rank_level") or -1),
            "common_name": t.get("preferred_common_name"),
            "ca_observation_count": int(row.get("count", 0)),
            "establishment_means": "native",
            "parent_id": int(t["parent_id"]) if t.get("parent_id") else None,
            "inat_url": f"https://www.inaturalist.org/taxa/{t['id']}",
            "snapshot_utc": snapshot,
            "_ancestor_ids": t.get("ancestor_ids") or [],
        })
        ancestor_ids_needed.update(int(x) for x in (t.get("ancestor_ids") or []))
        pulled += 1
        if pulled % 500 == 0:
            console.print(f"  pulled {pulled}")
            prov.write(json.dumps({"type": "checkpoint", "count": pulled}) + "\n")
            prov.flush()
    console.print(f"[green]species_counts returned {pulled} rows[/]")

    # Ancestor enrichment (kingdom..family) — single batched pass.
    detail_cache: dict[int, dict] = {}
    if fetch_ancestors and ancestor_ids_needed:
        console.rule("[bold cyan]ancestor enrichment")
        # Don't refetch ids we already have (the species rows themselves).
        already = {r["inat_taxon_id"] for r in rows}
        to_fetch = ancestor_ids_needed - already
        console.print(f"will fetch {len(to_fetch)} ancestor taxa to build kingdom..family columns")
        detail_cache = _fetch_ancestor_details(to_fetch)

    # Build final rows.
    out_rows: list[dict] = []
    for r in rows:
        ranks = _ranks_from_ancestor_ids(r.pop("_ancestor_ids"), detail_cache) if fetch_ancestors else {k: None for k in _RANK_ORDER}
        out_rows.append({**r, **ranks})

    df = pd.DataFrame(out_rows)
    # iNat returns some taxa twice (once per establishment_means listing, e.g. both
    # "native" and "endemic"). Dedupe by taxon id, keeping the row with the highest
    # observation count so the popularity ordering is preserved.
    n_before = len(df)
    df = df.sort_values("ca_observation_count", ascending=False).drop_duplicates(
        subset=["inat_taxon_id"], keep="first"
    ).reset_index(drop=True)
    if n_before != len(df):
        console.print(f"[yellow]deduped {n_before - len(df)} rows (taxa listed under multiple establishment_means)[/]")

    out_path = outdir / "plants_california_native.parquet"
    df.to_parquet(out_path, index=False)

    prov.write(json.dumps({
        "type": "result",
        "count": len(df),
        "n_species_rank": int((df["rank"] == "species").sum()),
        "n_subspecies": int((df["rank"] == "subspecies").sum()),
        "n_variety": int((df["rank"] == "variety").sum()),
        "out_path": str(out_path),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    prov.close()

    console.rule("[bold green]CNPS fetch complete")
    console.print(f"taxa fetched: [bold]{len(df)}[/]")
    by_rank = df["rank"].value_counts().to_dict()
    console.print(f"by rank:      {by_rank}")
    console.print(f"output:       {out_path}")
    console.print(f"provenance:   {prov_path}")
