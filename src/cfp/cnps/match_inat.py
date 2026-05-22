"""Robust iNat + GBIF cross-reference for the Calscape canonical plant list.

The first-pass enrichment matched only 5% of Calscape taxa to iNat because:
  1. ``rank=species`` filter excluded 2,295 variety/subspecies/hybrid rows.
  2. iNat's text search doesn't handle "var." / "subsp." infix patterns well.
  3. Silent 429s in parallel async calls were swallowed by the exception handler.

This rewrite addresses all three:
  * No rank filter — accept any active taxon.
  * Multi-attempt name normalization: exact → strip subspecific epithet → species head.
  * Retry-on-429 with bounded exponential backoff (max 5 attempts, 10 s cap).
  * GBIF ``/species/match`` fallback (handles synonyms + name authority differences).
  * Explicit ``match_provenance`` column so we can audit how each row resolved.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

app = typer.Typer(add_completion=False, help="Robust iNat+GBIF matcher for Calscape taxa.")
console = Console()

INAT_API = "https://api.inaturalist.org/v1"
GBIF_API = "https://api.gbif.org/v1"
USER_AGENT = "DeepEarth-CFP/0.1 (calscape iNat+GBIF matcher; lance@ecological.dev)"

_RE_INFRA = re.compile(r"\s+(var\.|subsp\.|ssp\.|f\.|forma)\s+", re.I)


def _name_variants(name: str) -> list[str]:
    """Yield successively more permissive search forms of a Calscape name."""
    name = (name or "").strip()
    out = [name]
    # Drop the subspecific epithet
    m = _RE_INFRA.search(name)
    if m:
        head, _, _ = name.partition(m.group(0))
        out.append(head.strip())
    # Drop hybrid markers
    if "×" in name or " x " in name:
        out.append(name.replace("×", "").replace(" x ", " ").strip())
    # Bare binomial (drop everything after the first 2 tokens)
    parts = name.split()
    if len(parts) > 2:
        out.append(" ".join(parts[:2]))
    # De-dupe preserving order
    seen, dedup = set(), []
    for n in out:
        if n and n not in seen:
            seen.add(n)
            dedup.append(n)
    return dedup


async def _get_with_retry(client: httpx.AsyncClient, url: str, params: dict,
                          max_attempts: int = 5, max_wait: float = 10.0) -> Optional[httpx.Response]:
    """Bounded exponential backoff on 429 / 5xx / network errors. Returns None on final failure."""
    wait = 1.0
    for _ in range(max_attempts):
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
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.HTTPError):
            await asyncio.sleep(wait)
            wait = min(wait * 2, max_wait)
    return None


async def _inat_match(client: httpx.AsyncClient, name: str) -> Optional[dict]:
    """Try iNat /v1/taxa with multiple name variants; return best match or None."""
    for variant in _name_variants(name):
        r = await _get_with_retry(client, f"{INAT_API}/taxa",
                                  {"q": variant, "is_active": "true"})
        if r is None:
            continue
        results = r.json().get("results", []) or []
        if not results:
            continue
        # Prefer exact (case-insensitive) name match
        for res in results:
            if (res.get("name") or "").lower() == name.lower():
                return {**res, "matched_variant": variant, "match_quality": "exact_full"}
        # Then exact match on the variant
        for res in results:
            if (res.get("name") or "").lower() == variant.lower():
                return {**res, "matched_variant": variant, "match_quality": "exact_variant"}
        # Then first result (iNat's relevance ordering)
        first = results[0]
        return {**first, "matched_variant": variant, "match_quality": "first_result"}
    return None


async def _gbif_match(client: httpx.AsyncClient, name: str) -> Optional[dict]:
    """GBIF /species/match — handles synonyms + name authority differences."""
    r = await _get_with_retry(client, f"{GBIF_API}/species/match",
                              {"name": name, "kingdom": "Plantae", "strict": "false"})
    if r is None:
        return None
    j = r.json()
    if j.get("matchType") not in {"EXACT", "FUZZY"} or "usageKey" not in j:
        return None
    return j


async def _inat_ca_obs_count(client: httpx.AsyncClient, taxon_id: int) -> Optional[int]:
    """CA native observation count for a given iNat taxon id."""
    r = await _get_with_retry(client, f"{INAT_API}/observations/species_counts",
                              {"place_id": 14, "taxon_id": taxon_id, "native": "true", "per_page": 1})
    if r is None:
        return None
    results = r.json().get("results", []) or []
    return int(results[0].get("count", 0)) if results else 0


async def _resolve_one(client: httpx.AsyncClient, name: str) -> dict:
    out = {
        "inat_taxon_id": None, "ca_observation_count": None,
        "gbif_backbone_key": None,
        "match_quality": None, "matched_variant": None, "matched_name": None,
    }
    inat = await _inat_match(client, name)
    if inat is not None:
        out["inat_taxon_id"] = int(inat["id"])
        out["matched_variant"] = inat["matched_variant"]
        out["matched_name"] = inat.get("name")
        out["match_quality"] = f"inat:{inat['match_quality']}"
        out["ca_observation_count"] = await _inat_ca_obs_count(client, int(inat["id"]))
    # GBIF backbone (independent — useful for downstream image manifest builder)
    gbif = await _gbif_match(client, name)
    if gbif is not None:
        out["gbif_backbone_key"] = int(gbif["usageKey"])
        if out["match_quality"] is None:
            out["match_quality"] = f"gbif:{gbif.get('matchType','?').lower()}"
            out["matched_name"] = gbif.get("scientificName")
    return out


async def _run(df: pd.DataFrame, concurrency: int) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(60.0, connect=10.0)
    headers = {"User-Agent": USER_AGENT}
    out: dict[int, dict] = {}

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            tid = prog.add_task("match", total=len(df))

            async def worker(idx, name):
                async with semaphore:
                    out[idx] = await _resolve_one(client, name)
                    prog.update(tid, advance=1)

            await asyncio.gather(*(worker(i, n) for i, n in df["scientific_name"].items()))

    df = df.copy()
    for col in ("inat_taxon_id", "ca_observation_count", "gbif_backbone_key"):
        df[col] = pd.Series({i: v.get(col) for i, v in out.items()}).astype("Int64")
    for col in ("match_quality", "matched_variant", "matched_name"):
        df[col] = pd.Series({i: v.get(col) for i, v in out.items()})
    return df


@app.command()
def match(
    parquet: Path = typer.Option(Path("data/processed/plants_california_native.parquet"), "--parquet"),
    out: Path = typer.Option(Path("data/processed/plants_california_native.parquet"), "--out"),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
    concurrency: int = typer.Option(3, "--concurrency", help="iNat is sensitive to >5 concurrent — keep low."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process only the first N (debug)."),
    only_unmatched: bool = typer.Option(True, "--only-unmatched/--all",
                                        help="Skip rows that already have inat_taxon_id set; default true."),
) -> None:
    """Robust iNat + GBIF cross-reference for every Calscape taxon."""
    df = pd.read_parquet(parquet)
    console.print(f"loaded {len(df):,} taxa from {parquet}")

    if only_unmatched and "inat_taxon_id" in df.columns:
        mask = df["inat_taxon_id"].isna()
        console.print(f"{mask.sum():,} unmatched; {len(df)-mask.sum():,} already have inat_taxon_id")
        work = df[mask].copy()
    else:
        work = df.copy()
    if limit:
        work = work.head(limit)
    console.print(f"matching {len(work):,} taxa…")

    snapshot = datetime.now(timezone.utc).isoformat()
    enriched = asyncio.run(_run(work, concurrency=concurrency))

    # Merge enriched columns back into the full dataframe.
    if only_unmatched and "inat_taxon_id" in df.columns:
        full = df.copy()
        for col in ("inat_taxon_id", "ca_observation_count", "gbif_backbone_key",
                    "match_quality", "matched_variant", "matched_name"):
            if col not in full.columns:
                full[col] = pd.Series(dtype="object" if col.startswith("match") else "Int64")
            full.loc[enriched.index, col] = enriched[col]
    else:
        full = enriched

    full.to_parquet(out, index=False)

    n_inat = int(full["inat_taxon_id"].notna().sum())
    n_gbif = int(full["gbif_backbone_key"].notna().sum())
    n_either = int((full["inat_taxon_id"].notna() | full["gbif_backbone_key"].notna()).sum())
    console.rule("[bold green]Match complete")
    console.print(f"iNat matched:  [bold]{n_inat:,}[/] / {len(full):,} = {100*n_inat/len(full):.1f}%")
    console.print(f"GBIF matched:  [bold]{n_gbif:,}[/] / {len(full):,} = {100*n_gbif/len(full):.1f}%")
    console.print(f"Either:        [bold]{n_either:,}[/] / {len(full):,} = {100*n_either/len(full):.1f}%")
    if "match_quality" in full.columns:
        console.print("\nmatch_quality breakdown:")
        console.print(full["match_quality"].value_counts(dropna=False).to_string())

    prov_dir.mkdir(parents=True, exist_ok=True)
    prov_path = prov_dir / f"calscape_match_{snapshot.replace(':','').replace('-','')[:15]}.jsonl"
    with prov_path.open("w") as p:
        p.write(json.dumps({
            "type": "run_meta", "stage": "cnps.calscape.match",
            "started_utc": snapshot, "concurrency": concurrency,
            "inat_api": INAT_API, "gbif_api": GBIF_API,
            "n_total": int(len(full)), "n_inat_matched": n_inat,
            "n_gbif_matched": n_gbif, "n_either_matched": n_either,
        }) + "\n")
    console.print(f"provenance: {prov_path}")
