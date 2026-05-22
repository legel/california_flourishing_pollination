"""Cross-check candidate pollinators: flight ability + GBIF CA observation.

Inputs:
    data/processed/pollinators_candidates.parquet     (from cfp.globi filter)
    data/processed/flight_ability_rules.csv           (versioned in this repo)

Outputs:
    data/processed/pollinators_california_flying.parquet
    data/processed/pollinators_excluded.parquet       (with reason column)

Per candidate:
  1. Resolve the GBIF backbone taxonKey (with retries + cache).
  2. Pull its higher classification (kingdom/phylum/class/order/family).
  3. Apply flight rule lookup (family overrides order overrides class).
  4. Verify ≥1 Research-grade observation exists in California
     (`country=US`, `stateProvince=California`, iNat dataset).
  5. Persist both kept and excluded rows with provenance.

Concurrency: 8 async workers against GBIF (no published rate limit; we self-cap).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

app = typer.Typer(add_completion=False, help="Pollinator candidate cross-check (flight + CA presence).")
console = Console()

GBIF_API = "https://api.gbif.org/v1"
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
USER_AGENT = "DeepEarth-CFP/0.1 (pollinator cross-check; lance@3co.ai)"


def _load_flight_rules(path: Path) -> dict:
    """Build {level: {name_lower: (flying|None, notes)}} from the rules CSV."""
    rules = {"class": {}, "order": {}, "family": {}}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        lvl = str(row["level"]).strip().lower()
        if lvl not in rules:
            continue
        flying_str = str(row["flying"]).strip().lower()
        if flying_str == "true":
            flying: Optional[bool] = True
        elif flying_str == "false":
            flying = False
        else:
            flying = None
        rules[lvl][str(row["name"]).strip().lower()] = (flying, str(row.get("notes", "")))
    return rules


def _classify_flying(rules: dict, row: dict) -> tuple[Optional[bool], str]:
    """Family > order > class precedence; first hit wins."""
    for lvl, key in (("family", "family"), ("order", "order"), ("class", "class")):
        v = (row.get(key) or "").strip().lower()
        if v and v in rules[lvl]:
            flying, note = rules[lvl][v]
            return flying, f"{lvl}={v}; {note}"
    return None, "no rule matched; default exclude"


async def _get_with_retry(client: httpx.AsyncClient, url: str, params: dict,
                          max_attempts: int = 6) -> httpx.Response:
    """GET with exponential-backoff retry on 429 / 5xx / connection error."""
    wait = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            r = await client.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                if ra is not None:
                    try:
                        wait = max(wait, float(ra))
                    except ValueError:
                        pass
                await asyncio.sleep(wait)
                wait = min(wait * 2, 60.0)
                continue
            r.raise_for_status()
            return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
            await asyncio.sleep(wait)
            wait = min(wait * 2, 60.0)
    # Final attempt — let the exception propagate.
    r = await client.get(url, params=params)
    r.raise_for_status()
    return r


async def _resolve_one(client: httpx.AsyncClient, name: str) -> dict:
    """One GBIF backbone match + classification + CA-iNat presence (with 429 retry)."""
    out = {"resolved_name": name, "gbif_key": None, "gbif_match_type": None,
           "class": None, "order": None, "family": None, "ca_inat_count": None,
           "error": None}
    try:
        m = await _get_with_retry(client, f"{GBIF_API}/species/match",
                                  {"name": name, "strict": "false"})
        j = m.json()
        out["gbif_match_type"] = j.get("matchType")
        if "usageKey" not in j:
            out["error"] = "no GBIF match"
            return out
        key = int(j["usageKey"])
        out["gbif_key"] = key
        out["class"] = j.get("class")
        out["order"] = j.get("order")
        out["family"] = j.get("family")

        c = await _get_with_retry(
            client,
            f"{GBIF_API}/occurrence/search",
            {
                "country": "US",
                "stateProvince": "California",
                "datasetKey": INAT_DATASET_KEY,
                "taxonKey": key,
                "limit": 0,
            },
        )
        out["ca_inat_count"] = int(c.json().get("count", 0))
    except httpx.HTTPError as e:
        out["error"] = f"HTTPError: {e}"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


async def _run_async(candidates: pd.DataFrame, concurrency: int, sleep_s: float) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    headers = {"User-Agent": USER_AGENT}
    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(headers=headers, timeout=timeout, http2=False) as client:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task_id = prog.add_task("GBIF resolve+verify", total=len(candidates))

            async def worker(idx: int, row: dict) -> None:
                async with semaphore:
                    res = await _resolve_one(client, row["animal_name"])
                    res["candidate_idx"] = idx
                    res["animal_name"] = row["animal_name"]
                    results.append(res)
                    prog.update(task_id, advance=1)
                    await asyncio.sleep(sleep_s)

            await asyncio.gather(*(worker(i, r) for i, r in candidates.iterrows()))

    return pd.DataFrame(results).sort_values("candidate_idx").reset_index(drop=True)


@app.callback(invoke_without_command=True)
def main(
    candidates: Path = typer.Option(
        Path("data/processed/pollinators_candidates.parquet"),
        "--candidates",
        help="Output of `cfp.globi filter` (column `animal_name` required).",
    ),
    rules_csv: Path = typer.Option(
        Path("data/processed/flight_ability_rules.csv"), "--rules"
    ),
    out_kept: Path = typer.Option(
        Path("data/processed/pollinators_california_flying.parquet"), "--out-kept"
    ),
    out_excluded: Path = typer.Option(
        Path("data/processed/pollinators_excluded.parquet"), "--out-excluded"
    ),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
    concurrency: int = typer.Option(8, "--concurrency"),
    sleep_s: float = typer.Option(0.0, "--sleep-s", help="Per-worker sleep after each request."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process only the first N candidates (debug)."),
) -> None:
    """Run the flight + CA-iNat cross-check on every candidate pollinator."""
    if not candidates.exists():
        raise typer.BadParameter(f"missing {candidates} — run `cfp.globi filter` first.")
    rules = _load_flight_rules(rules_csv)

    cand_df = pd.read_parquet(candidates)
    if limit:
        cand_df = cand_df.head(limit)
    console.print(f"loaded [bold]{len(cand_df)}[/] candidates from {candidates}")

    snapshot = datetime.now(timezone.utc).isoformat()
    prov_dir.mkdir(parents=True, exist_ok=True)
    prov_path = prov_dir / f"pollinators_cross_check_{snapshot.replace(':','').replace('-','')[:15]}.jsonl"

    started = time.time()
    enriched = asyncio.run(_run_async(cand_df, concurrency=concurrency, sleep_s=sleep_s))
    elapsed = time.time() - started

    # Merge enrichment back; resolve flight ability per row.
    merged = cand_df.reset_index().rename(columns={"index": "candidate_idx"}).merge(
        enriched, on=["candidate_idx", "animal_name"], how="left"
    )
    flights = merged.apply(
        lambda r: _classify_flying(rules, {"class": r["class"], "order": r["order"], "family": r["family"]}),
        axis=1,
    )
    merged["is_flying"] = [bool(f[0]) if f[0] is True else False for f in flights]
    merged["flight_rule"] = [f[1] for f in flights]
    merged["ca_observed"] = merged["ca_inat_count"].fillna(0).astype(int) >= 1
    merged["keep"] = merged["is_flying"] & merged["ca_observed"]

    kept = merged[merged["keep"]].copy()
    excluded = merged[~merged["keep"]].copy()
    excluded["exclusion_reason"] = [
        "; ".join(
            r
            for r in (
                (None if row["is_flying"] else "not_flying"),
                (None if row["ca_observed"] else "no_CA_iNat_observation"),
                (None if not row["error"] else f"error: {row['error']}"),
            )
            if r
        )
        for _, row in excluded.iterrows()
    ]

    out_kept.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out_kept, index=False)
    excluded.to_parquet(out_excluded, index=False)

    with prov_path.open("w") as prov:
        prov.write(json.dumps({
            "type": "run_meta", "stage": "pollinators.cross_check",
            "started_utc": snapshot, "elapsed_s": elapsed,
            "gbif_api": GBIF_API, "inat_dataset_key": INAT_DATASET_KEY,
            "candidates_file": str(candidates), "rules_file": str(rules_csv),
            "n_candidates": len(cand_df), "n_kept": len(kept), "n_excluded": len(excluded),
        }) + "\n")
        for _, r in merged.iterrows():
            prov.write(json.dumps({
                "type": "row",
                "animal_name": r["animal_name"],
                "gbif_key": int(r["gbif_key"]) if pd.notna(r["gbif_key"]) else None,
                "gbif_match_type": r["gbif_match_type"],
                "class": r["class"], "order": r["order"], "family": r["family"],
                "ca_inat_count": int(r["ca_inat_count"]) if pd.notna(r["ca_inat_count"]) else None,
                "is_flying": bool(r["is_flying"]),
                "flight_rule": r["flight_rule"],
                "keep": bool(r["keep"]),
            }) + "\n")

    console.rule("[bold green]Cross-check complete")
    console.print(f"kept (flying + CA-observed): [bold]{len(kept)}[/]  →  {out_kept}")
    console.print(f"excluded: [bold]{len(excluded)}[/]  →  {out_excluded}")
    console.print(f"provenance: {prov_path}")


if __name__ == "__main__":
    app()
