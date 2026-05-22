"""Download GloBi interactions and filter to CA-native plant × pollinator rows.

Two stages:

1. **fetch**  — download ``interactions.tsv.gz`` + ``refuted-interactions.tsv.gz``
                from the GloBi depot, write to ``data/raw/globi/``, hash the file,
                and emit a provenance JSONL with the resolved version metadata.
2. **filter** — DuckDB streaming filter (no full load) producing two parquet outputs:
                ``data/processed/globi_ca_plant_pollinator.parquet`` and
                ``data/processed/pollinators_candidates.parquet``.

The pollination filter uses RO ontology IRIs (`interactionTypeId`):

    RO_0002455 pollinates           (animal → plant)
    RO_0002456 pollinatedBy         (plant  → animal)
    RO_0002622 visitsFlowersOf      (animal → plant)
    RO_0002623 flowersVisitedBy     (plant  → animal)

California geographic filter is a logical OR over:
    (a) coords in bbox [-124.55,-114.13] × [32.53, 42.01]
    (b) locality regex /\\b(California|Calif\\.|\\bCA\\b)\\b/i (with Baja guard)

Run:
    python -m cfp.globi fetch --raw-dir data/raw/globi
    python -m cfp.globi filter --raw-dir data/raw/globi \\
        --natives data/processed/plants_california_native.parquet \\
        --out data/processed/globi_ca_plant_pollinator.parquet
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, help="GloBi ingestion + pollination filter.")
console = Console()

DEPOT = "https://depot.globalbioticinteractions.org/snapshot/target/data"
INTERACTIONS_URL = f"{DEPOT}/tsv/interactions.tsv.gz"
REFUTED_URL = f"{DEPOT}/tsv/refuted-interactions.tsv.gz"
TAXON_MAP_URL = f"{DEPOT}/tsv/taxonMap.tsv.gz"
CITATIONS_URL = f"{DEPOT}/tsv/citations.tsv.gz"

POLLINATION_IRIS = [
    "http://purl.obolibrary.org/obo/RO_0002455",  # pollinates
    "http://purl.obolibrary.org/obo/RO_0002456",  # pollinatedBy
    "http://purl.obolibrary.org/obo/RO_0002622",  # visitsFlowersOf
    "http://purl.obolibrary.org/obo/RO_0002623",  # flowersVisitedBy
]

PLANT_TARGET_IRIS = {"http://purl.obolibrary.org/obo/RO_0002455",   # pollinates: animal→plant (plant=target)
                     "http://purl.obolibrary.org/obo/RO_0002622"}   # visitsFlowersOf: animal→plant
PLANT_SOURCE_IRIS = {"http://purl.obolibrary.org/obo/RO_0002456",   # pollinatedBy: plant→animal (plant=source)
                     "http://purl.obolibrary.org/obo/RO_0002623"}   # flowersVisitedBy: plant→animal

# California bounding box (decimal degrees).
CA_BBOX = {"lon_min": -124.55, "lon_max": -114.13, "lat_min": 32.53, "lat_max": 42.01}


def _stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> dict:
    """Stream a URL to dest, hashing as we go. Returns size + sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    size = 0
    req = urllib.request.Request(url, headers={"User-Agent": "DeepEarth-CFP/0.1"})
    started = datetime.now(timezone.utc)
    with urllib.request.urlopen(req) as r, dest.open("wb") as f:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            sha.update(buf)
            size += len(buf)
    return {
        "url": url,
        "dest": str(dest),
        "bytes": size,
        "sha256": sha.hexdigest(),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.command()
def fetch(
    raw_dir: Path = typer.Option(Path("data/raw/globi"), "--raw-dir"),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
    include_supporting: bool = typer.Option(True, "--include-supporting/--no-include-supporting",
                                            help="Also pull taxonMap.tsv.gz + citations.tsv.gz."),
) -> None:
    """Download GloBi snapshot files to ``raw_dir`` and write provenance."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat()
    prov_path = prov_dir / f"globi_snapshot_{snapshot.replace(':', '').replace('-', '')[:15]}.jsonl"
    prov_f = prov_path.open("w")
    prov_f.write(
        json.dumps(
            {
                "type": "run_meta",
                "stage": "globi.fetch",
                "started_utc": snapshot,
                "depot": DEPOT,
                "concept_doi": "10.5281/zenodo.3950589",
                "note": "Pin the version DOI in provenance/globi_version.json once the latest Zenodo record is resolved.",
            }
        )
        + "\n"
    )

    urls_required = [INTERACTIONS_URL, REFUTED_URL]
    urls_optional = [TAXON_MAP_URL, CITATIONS_URL] if include_supporting else []

    for url in urls_required + urls_optional:
        dest = raw_dir / Path(url).name
        console.print(f"[cyan]downloading[/] {url} → {dest}")
        try:
            info = _stream_download(url, dest)
        except urllib.error.HTTPError as e:
            if url in urls_optional and e.code in (403, 404):
                console.print(f"  [yellow]{e.code} — skipping optional file[/]")
                prov_f.write(json.dumps({"type": "download_skipped", "url": url, "http_status": e.code}) + "\n")
                continue
            raise
        prov_f.write(json.dumps({"type": "download", **info}) + "\n")
        prov_f.flush()
        console.print(f"  [green]{info['bytes']/1e9:.2f} GB[/] sha256={info['sha256'][:12]}…")
    prov_f.close()
    console.rule("[bold green]GloBi download complete")
    console.print(f"Files in: {raw_dir}")
    console.print(f"Provenance: {prov_path}")


@app.command("filter")
def filter_(
    raw_dir: Path = typer.Option(Path("data/raw/globi"), "--raw-dir"),
    natives: Path = typer.Option(
        Path("data/processed/plants_california_native.parquet"),
        "--natives",
        help="Parquet produced by `cfp.cnps fetch` — column `scientific_name`.",
    ),
    out_interactions: Path = typer.Option(
        Path("data/processed/globi_ca_plant_pollinator.parquet"),
        "--out",
    ),
    out_candidates: Path = typer.Option(
        Path("data/processed/pollinators_candidates.parquet"),
        "--out-candidates",
    ),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
) -> None:
    """DuckDB filter: pollination interactions × CA-native plants × CA geography."""
    # DuckDB is optional in requirements.txt; install lazily if missing.
    try:
        import duckdb  # type: ignore
    except ImportError as e:
        raise typer.BadParameter(
            "duckdb not installed. pip install duckdb (≥1.1) — chosen for streaming TSV filter."
        ) from e

    interactions_file = raw_dir / "interactions.tsv.gz"
    refuted_file = raw_dir / "refuted-interactions.tsv.gz"
    if not interactions_file.exists():
        raise typer.BadParameter(f"missing {interactions_file} — run `cfp.globi fetch` first.")

    out_interactions.parent.mkdir(parents=True, exist_ok=True)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat()
    prov_path = prov_dir / f"globi_filter_{snapshot.replace(':', '').replace('-', '')[:15]}.jsonl"
    prov_f = prov_path.open("w")
    prov_f.write(json.dumps({
        "type": "run_meta",
        "stage": "globi.filter",
        "started_utc": snapshot,
        "pollination_iris": POLLINATION_IRIS,
        "ca_bbox": CA_BBOX,
        "interactions_file": str(interactions_file),
        "refuted_file": str(refuted_file),
        "natives_parquet": str(natives),
    }) + "\n")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    natives_table = con.execute(
        f"CREATE TEMP VIEW v_natives AS SELECT DISTINCT scientific_name FROM '{natives}'"
    )
    iri_list = "(" + ", ".join(f"'{x}'" for x in POLLINATION_IRIS) + ")"
    plant_target_list = "(" + ", ".join(f"'{x}'" for x in PLANT_TARGET_IRIS) + ")"
    plant_source_list = "(" + ", ".join(f"'{x}'" for x in PLANT_SOURCE_IRIS) + ")"

    base_query = f"""
        WITH refuted AS (
            SELECT
                sourceTaxonId, targetTaxonId, interactionTypeId, referenceDoi
            FROM read_csv_auto('{refuted_file}', delim='\\t', header=True, ignore_errors=True)
        ),
        filtered AS (
            SELECT
                i.sourceTaxonId, i.sourceTaxonName, i.sourceTaxonRank, i.sourceTaxonPathNames,
                i.sourceTaxonKingdomName, i.sourceTaxonOrderName, i.sourceTaxonFamilyName,
                i.interactionTypeId, i.interactionTypeName,
                i.targetTaxonId, i.targetTaxonName, i.targetTaxonRank, i.targetTaxonPathNames,
                i.targetTaxonKingdomName, i.targetTaxonOrderName, i.targetTaxonFamilyName,
                i.decimalLatitude, i.decimalLongitude, i.localityId, i.localityName,
                i.eventDate, i.referenceCitation, i.referenceDoi, i.sourceCitation, i.sourceDOI
            FROM read_csv_auto('{interactions_file}', delim='\\t', header=True, ignore_errors=True) i
            WHERE i.interactionTypeId IN {iri_list}
              AND (
                  (i.interactionTypeId IN {plant_target_list} AND i.targetTaxonName IN (SELECT scientific_name FROM v_natives))
                  OR
                  (i.interactionTypeId IN {plant_source_list} AND i.sourceTaxonName IN (SELECT scientific_name FROM v_natives))
              )
              AND (
                  (i.decimalLatitude BETWEEN {CA_BBOX['lat_min']} AND {CA_BBOX['lat_max']}
                   AND i.decimalLongitude BETWEEN {CA_BBOX['lon_min']} AND {CA_BBOX['lon_max']})
                  OR (
                      regexp_matches(coalesce(i.localityName, ''), '(?i)\\b(California|Calif\\.|\\bCA\\b)\\b')
                      AND NOT regexp_matches(coalesce(i.localityName, ''), '(?i)Baja\\s+California')
                  )
              )
        )
        SELECT * FROM filtered f
        WHERE NOT EXISTS (
            SELECT 1 FROM refuted r
            WHERE r.sourceTaxonId = f.sourceTaxonId
              AND r.targetTaxonId = f.targetTaxonId
              AND r.interactionTypeId = f.interactionTypeId
        )
    """
    console.print("[cyan]running DuckDB filter (this may take 5–15 min the first time)…")
    con.execute(f"COPY ({base_query}) TO '{out_interactions}' (FORMAT PARQUET)")
    n_rows = con.execute(f"SELECT COUNT(*) FROM '{out_interactions}'").fetchone()[0]
    console.print(f"[green]wrote[/] {n_rows} interaction rows → {out_interactions}")

    candidates_query = f"""
        WITH src AS (SELECT * FROM '{out_interactions}'),
        animals AS (
            SELECT
                CASE WHEN interactionTypeId IN {plant_target_list} THEN sourceTaxonName
                     ELSE targetTaxonName END AS animal_name,
                CASE WHEN interactionTypeId IN {plant_target_list} THEN sourceTaxonId
                     ELSE targetTaxonId END AS animal_taxon_id,
                CASE WHEN interactionTypeId IN {plant_target_list} THEN sourceTaxonOrderName
                     ELSE targetTaxonOrderName END AS animal_order,
                CASE WHEN interactionTypeId IN {plant_target_list} THEN sourceTaxonFamilyName
                     ELSE targetTaxonFamilyName END AS animal_family,
                CASE WHEN interactionTypeId IN {plant_target_list} THEN sourceTaxonPathNames
                     ELSE targetTaxonPathNames END AS animal_path,
                CASE WHEN interactionTypeId IN {plant_target_list} THEN sourceTaxonKingdomName
                     ELSE targetTaxonKingdomName END AS animal_kingdom,
                CASE WHEN interactionTypeId IN {plant_target_list} THEN targetTaxonName
                     ELSE sourceTaxonName END AS plant_name
            FROM src
        )
        SELECT
            animal_name, animal_taxon_id, animal_kingdom, animal_order, animal_family, animal_path,
            COUNT(DISTINCT plant_name) AS n_ca_native_plants,
            COUNT(*) AS n_interactions,
            list(DISTINCT plant_name) AS plant_examples
        FROM animals
        WHERE animal_kingdom IN ('Animalia') OR animal_kingdom IS NULL
        GROUP BY animal_name, animal_taxon_id, animal_kingdom, animal_order, animal_family, animal_path
        ORDER BY n_ca_native_plants DESC, n_interactions DESC
    """
    con.execute(f"COPY ({candidates_query}) TO '{out_candidates}' (FORMAT PARQUET)")
    n_cand = con.execute(f"SELECT COUNT(*) FROM '{out_candidates}'").fetchone()[0]
    console.print(f"[green]wrote[/] {n_cand} pollinator-candidate rows → {out_candidates}")

    prov_f.write(json.dumps({
        "type": "result",
        "interactions_rows": int(n_rows),
        "candidate_pollinators": int(n_cand),
        "interactions_parquet": str(out_interactions),
        "candidates_parquet": str(out_candidates),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    prov_f.close()
    console.rule("[bold green]GloBi filter complete")
    console.print(f"Provenance: {prov_path}")


if __name__ == "__main__":
    app()
