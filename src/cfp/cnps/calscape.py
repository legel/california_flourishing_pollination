"""Ingest the canonical CNPS / Calscape California native plant list.

Source: a CNPS Calscape "Options → Export list to Excel" download with the
"Native to California" filter applied. The Calscape website itself sits behind
Cloudflare Turnstile and exposes no public bulk API, so the file is fetched
manually by the project PI and stored locally; we then normalize the 50-column
Excel into a stable parquet schema and cross-reference each taxon to iNat for
downstream observation counts + iNat taxon ids.

Calscape is the CNPS-curated authoritative source. Yields ~8,507 taxa at the
2026-05-22 snapshot — matching CNPS's stated "more than 8,500 types."

CNPS canonical definition of "California native plant" (from
https://www.cnps.org/gardening/why-natives/what-are-native-plants):

    Our native plants grew here prior to European contact. California's native
    plants evolved here over a very long period, and are the plants which the
    first Californians knew and depended on for their livelihood.

Output schema (`data/processed/plants_california_native.parquet`):

    inat_taxon_id (int64 | null)         -- joined from iNat for downstream queries
    scientific_name (utf8)               -- 'Botanical Name'
    common_name (utf8 | null)            -- 'Common Name'
    rank (utf8)                          -- derived ('species' | 'subspecies' | 'variety' | …)
    ca_observation_count (int64 | null)  -- joined from iNat species_counts
    establishment_means (utf8)           -- always 'native' (by construction)
    is_cultivar (bool | null)
    rarity (utf8 | null)
    plant_type / form / seasonality / flower_color / flowering_season (utf8 | null)
    fragrance / sun / soil_drainage / water_requirement / summer_irrigation / ease_of_care
    nursery_availability / companions / special_uses
    communities_simplified / communities / hardiness / sunset_zones
    soil / soil_texture / soil_ph / soil_toxicity / mulch / site_type
    elevation_min / elevation_max / rainfall_min / rainfall_max (numeric where parseable)
    height_min / height_max / width_min / width_max (numeric where parseable)
    butterflies_and_moths_supported (int64 | null)  -- direct pollination signal
    attracts_wildlife (utf8 | null)
    other_names / alternative_common_names / obsolete_names (utf8 | null)
    jepson_link / calscape_url (utf8 | null)
    snapshot_utc (utf8)

Cross-reference to iNat (one call per taxon, cached):
    inat_taxon_id        ← /v1/taxa?q=<name>&rank=species
    ca_observation_count ← /v1/observations/species_counts?place_id=14&taxon_id=<id>&native=true
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

app = typer.Typer(add_completion=False, help="Ingest Calscape Excel export.")
console = Console()

INAT_API = "https://api.inaturalist.org/v1"
USER_AGENT = "DeepEarth-CFP/0.1 (calscape ingest; lance@ecological.dev)"


# Map Calscape Excel column names → our snake_case parquet schema.
COLMAP = {
    "Botanical Name": "scientific_name",
    "Common Name": "common_name",
    "Butterflies and Moths Supported": "butterflies_and_moths_supported",
    "Attracts Wildlife": "attracts_wildlife",
    "Plant Type": "plant_type",
    "Form": "form",
    "Height": "height_text",
    "Width": "width_text",
    "Growth Rate": "growth_rate",
    "Seasonality": "seasonality",
    "Flower Color": "flower_color",
    "Flowering Season": "flowering_season",
    "Fragrance": "fragrance",
    "Sun": "sun",
    "Soil Drainage": "soil_drainage",
    "Water Requirement": "water_requirement",
    "Summer Irrigation": "summer_irrigation",
    "Ease of Care": "ease_of_care",
    "Nursery Availability": "nursery_availability",
    "Companions": "companions",
    "Special Uses": "special_uses",
    "Communities (simplified)": "communities_simplified",
    "Communities": "communities",
    "Hardiness": "hardiness",
    "Sunset Zones": "sunset_zones",
    "Soil": "soil",
    "Soil Texture": "soil_texture",
    "Soil pH": "soil_ph",
    "Soil Toxicity": "soil_toxicity",
    "Mulch": "mulch",
    "Site Type": "site_type",
    "Elevation (min)": "elevation_min_text",
    "Elevation (max)": "elevation_max_text",
    "Rainfall (min)": "rainfall_min_text",
    "Rainfall (max)": "rainfall_max_text",
    "Tips": "tips",
    "Pests": "pests",
    "Propagation": "propagation",
    "Height (min)": "height_min_text",
    "Height (max)": "height_max_text",
    "Width (min)": "width_min_text",
    "Width (max)": "width_max_text",
    "Other Names": "other_names",
    "Alternative Common Names": "alternative_common_names",
    "Obsolete Names": "obsolete_names",
    "Rarity": "rarity",
    "Is Cultivar": "is_cultivar",
    "Jepson Link": "jepson_link",
    "Plant Url": "calscape_url",
}


def _to_float(x) -> Optional[float]:
    """Parse Calscape numeric-looking strings ('20 - 236 ft', '5 cm') leniently."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _derive_rank(name: str) -> str:
    """Derive taxonomic rank from the binomial / trinomial pattern."""
    name = (name or "").strip()
    if " var. " in name:
        return "variety"
    if " subsp. " in name or " ssp. " in name:
        return "subspecies"
    if " f. " in name or " forma " in name:
        return "form"
    if " × " in name or " x " in name or "×" in name:
        return "hybrid"
    parts = name.split()
    if len(parts) >= 2 and parts[1][0].islower():
        return "species"
    if len(parts) == 1:
        return "genus"
    return "unknown"


async def _resolve_inat(client: httpx.AsyncClient, name: str) -> dict:
    """One iNat /taxa/match call + one /observations/species_counts call for CA count."""
    out = {"inat_taxon_id": None, "ca_observation_count": None}
    try:
        # iNat taxa match
        r = await client.get(f"{INAT_API}/taxa", params={"q": name, "rank": "species", "is_active": "true"})
        r.raise_for_status()
        results = r.json().get("results", []) or []
        match = None
        for res in results:
            if res.get("name", "").lower() == name.lower():
                match = res
                break
        if match is None and results:
            match = results[0]
        if match is None:
            return out
        out["inat_taxon_id"] = int(match["id"])
        # iNat CA observation count for this taxon (native filter retained)
        r = await client.get(
            f"{INAT_API}/observations/species_counts",
            params={"place_id": 14, "taxon_id": match["id"], "native": "true", "per_page": 1},
        )
        r.raise_for_status()
        results = r.json().get("results", []) or []
        if results:
            out["ca_observation_count"] = int(results[0].get("count", 0))
    except httpx.HTTPError:
        pass
    return out


async def _enrich(df: pd.DataFrame, concurrency: int, sleep_s: float) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(30.0, connect=10.0)
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
            tid = prog.add_task("iNat lookup", total=len(df))

            async def worker(idx, name):
                async with semaphore:
                    out[idx] = await _resolve_inat(client, name)
                    prog.update(tid, advance=1)
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

            await asyncio.gather(*(worker(i, n) for i, n in df["scientific_name"].items()))

    df = df.copy()
    df["inat_taxon_id"] = pd.Series({i: v.get("inat_taxon_id") for i, v in out.items()}).astype("Int64")
    df["ca_observation_count"] = pd.Series({i: v.get("ca_observation_count") for i, v in out.items()}).astype("Int64")
    return df


@app.command()
def ingest(
    xlsx: Path = typer.Option(Path("/home/legel/Native To California.xlsx"), "--xlsx"),
    out: Path = typer.Option(Path("data/processed/plants_california_native.parquet"), "--out"),
    raw_archive: Path = typer.Option(Path("data/raw/calscape/native_to_california.xlsx"), "--raw-archive"),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
    enrich_with_inat: bool = typer.Option(True, "--enrich/--no-enrich",
                                          help="Join iNat taxon_id + CA obs count per taxon (slow, ~30 min for 8.5K taxa)."),
    concurrency: int = typer.Option(4, "--concurrency"),
    sleep_s: float = typer.Option(0.0, "--sleep-s"),
) -> None:
    """Parse the Calscape Excel export → canonical CA-native plants parquet."""
    if not xlsx.exists():
        raise typer.BadParameter(f"missing {xlsx}")
    snapshot = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_archive.parent.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)

    # Archive the raw file (sha256-tagged) so the snapshot is auditable.
    import hashlib, shutil
    sha = hashlib.sha256(xlsx.read_bytes()).hexdigest()
    if not raw_archive.exists() or hashlib.sha256(raw_archive.read_bytes()).hexdigest() != sha:
        shutil.copy2(xlsx, raw_archive)
    console.print(f"raw archive: {raw_archive}  sha256={sha[:12]}…")

    # Header is at row 5 (0-indexed) in Calscape exports.
    df = pd.read_excel(xlsx, sheet_name=0, header=5)
    console.print(f"parsed {len(df):,} rows × {len(df.columns)} columns from {xlsx.name}")

    # Rename columns to our schema, drop rows with no botanical name.
    df = df.rename(columns=COLMAP)
    df = df[df["scientific_name"].notna() & (df["scientific_name"].str.strip() != "")].reset_index(drop=True)

    # Numeric parses (lenient — keep original text too).
    for stem in ("height_min", "height_max", "width_min", "width_max",
                 "elevation_min", "elevation_max", "rainfall_min", "rainfall_max"):
        text_col = f"{stem}_text"
        if text_col in df.columns:
            df[stem] = df[text_col].apply(_to_float).astype("Float64")

    # Derived rank from name pattern.
    df["rank"] = df["scientific_name"].apply(_derive_rank)
    df["establishment_means"] = "native"
    df["snapshot_utc"] = snapshot

    # Coerce numeric fields where appropriate.
    if "butterflies_and_moths_supported" in df.columns:
        df["butterflies_and_moths_supported"] = pd.to_numeric(
            df["butterflies_and_moths_supported"], errors="coerce"
        ).astype("Int64")
    if "is_cultivar" in df.columns:
        df["is_cultivar"] = df["is_cultivar"].map(
            lambda x: True if str(x).strip().lower() in {"yes", "true", "1"}
            else (False if str(x).strip().lower() in {"no", "false", "0"} else None)
        )

    # Enrich with iNat
    if enrich_with_inat:
        console.rule("[bold cyan]iNat enrichment")
        df = asyncio.run(_enrich(df, concurrency=concurrency, sleep_s=sleep_s))
    else:
        df["inat_taxon_id"] = pd.Series(dtype="Int64")
        df["ca_observation_count"] = pd.Series(dtype="Int64")

    df.to_parquet(out, index=False)

    prov_path = prov_dir / f"cnps_calscape_{snapshot.replace(':','').replace('-','')[:15]}.jsonl"
    with prov_path.open("w") as prov:
        prov.write(json.dumps({
            "type": "run_meta", "stage": "cnps.calscape.ingest",
            "started_utc": snapshot,
            "source": "CNPS Calscape — Options → Export list to Excel (Native to California filter)",
            "raw_file": str(raw_archive),
            "raw_sha256": sha,
            "definition_of_native": (
                "Our native plants grew here prior to European contact. California's "
                "native plants evolved here over a very long period, and are the plants "
                "which the first Californians knew and depended on for their livelihood. "
                "— California Native Plant Society, "
                "https://www.cnps.org/gardening/why-natives/what-are-native-plants"
            ),
            "n_rows": int(len(df)),
            "rank_counts": df["rank"].value_counts().to_dict(),
            "enriched_with_inat": enrich_with_inat,
            "inat_matched": int(df["inat_taxon_id"].notna().sum()) if enrich_with_inat else None,
            "out_path": str(out),
        }) + "\n")

    console.rule("[bold green]Calscape ingest complete")
    console.print(f"taxa fetched: [bold]{len(df)}[/]")
    console.print(f"by rank:      {df['rank'].value_counts().to_dict()}")
    console.print(f"output:       {out}")
    console.print(f"provenance:   {prov_path}")
    if enrich_with_inat:
        n_match = df["inat_taxon_id"].notna().sum()
        console.print(f"iNat matched: {n_match:,} / {len(df):,} = {100*n_match/len(df):.1f}%")
