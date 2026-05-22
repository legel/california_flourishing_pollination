"""Async downloader stage.

Streams iNaturalist photo URLs from the manifest, fetches each at
``image_url_large``, writes the bytes to disk, and persists per-image metadata
(license/attribution/observation_id) alongside. Pauses when the image directory
crosses ``--cap-gb`` and resumes once embedding clears space.

Disk layout (flat with hash-sharded subdirs to keep inode counts manageable):

    <image_dir>/<gbif_id % 1000>/<gbif_id>.<ext>            -- the image file
    <image_dir>/<gbif_id % 1000>/<gbif_id>.json             -- companion metadata

Checkpoint (resumable):
    outputs/checkpoint_downloaded.parquet                   -- gbif_occurrence_id present iff written

Failures (logged, retried with backoff, finally dropped):
    outputs/failed_downloads.parquet
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

app = typer.Typer(add_completion=False, help="iNaturalist image downloader.")
console = Console()

USER_AGENT = "DeepEarth-CFP/0.1 (image downloader; lance@3co.ai; +https://huggingface.co/deepearth)"


def _disk_usage(path: Path) -> int:
    """Recursive `du -sb` equivalent. Lazy: we cache between checks."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _path_for(image_dir: Path, gbif_id: int, ext: str) -> tuple[Path, Path]:
    bucket = image_dir / f"{gbif_id % 1000:03d}"
    return bucket / f"{gbif_id}.{ext}", bucket / f"{gbif_id}.json"


def _ext_from_url(url: str) -> str:
    tail = url.rsplit(".", 1)[-1].lower().split("?")[0]
    if tail not in {"jpg", "jpeg", "png", "gif", "webp"}:
        return "jpg"
    return "jpg" if tail == "jpeg" else tail


async def _fetch_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: dict,
    image_dir: Path,
) -> dict:
    """Download one image. Returns a result dict (status='ok'|'fail', size_bytes, path...)."""
    gbif_id = int(row["gbif_occurrence_id"])
    url = row["image_url_large"]
    ext = _ext_from_url(url)
    img_path, meta_path = _path_for(image_dir, gbif_id, ext)
    if img_path.exists():
        return {"gbif_occurrence_id": gbif_id, "status": "skipped_exists", "size_bytes": img_path.stat().st_size,
                "path": str(img_path)}

    img_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        async with semaphore:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
                reraise=True,
            ):
                with attempt:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status >= 500 or r.status == 429:
                            raise aiohttp.ClientResponseError(
                                request_info=r.request_info, history=r.history,
                                status=r.status, message=f"HTTP {r.status}", headers=r.headers,
                            )
                        if r.status >= 400:
                            return {"gbif_occurrence_id": gbif_id, "status": "fail", "http_status": r.status,
                                    "url": url, "error": f"HTTP {r.status}"}
                        data = await r.read()
                        tmp = img_path.with_suffix(img_path.suffix + ".tmp")
                        tmp.write_bytes(data)
                        os.replace(tmp, img_path)
                        meta_path.write_text(json.dumps({
                            "gbif_occurrence_id": gbif_id,
                            "url": url,
                            "license": row.get("license"),
                            "rights_holder": row.get("rights_holder"),
                            "inat_observation_id": row.get("inat_observation_id"),
                            "taxon_name": row.get("taxon_name"),
                            "gbif_taxon_key": row.get("gbif_taxon_key"),
                            "dataset_role": row.get("dataset_role"),
                            "observed_on": row.get("observed_on"),
                            "decimal_latitude": row.get("decimal_latitude"),
                            "decimal_longitude": row.get("decimal_longitude"),
                            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                            "elapsed_s": time.time() - started,
                        }))
                        return {"gbif_occurrence_id": gbif_id, "status": "ok",
                                "size_bytes": len(data), "path": str(img_path)}
    except RetryError as e:
        return {"gbif_occurrence_id": gbif_id, "status": "fail", "url": url, "error": f"retry exhausted: {e}"}
    except Exception as e:
        return {"gbif_occurrence_id": gbif_id, "status": "fail", "url": url, "error": f"{type(e).__name__}: {e}"}


def _load_checkpoint(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return set(pq.read_table(path, columns=["gbif_occurrence_id"]).to_pandas()["gbif_occurrence_id"].astype(int).tolist())


def _append_checkpoint(path: Path, ids: list[int]) -> None:
    if not ids:
        return
    df = pd.DataFrame({"gbif_occurrence_id": ids,
                       "downloaded_utc": datetime.now(timezone.utc).isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path)
        df = pd.concat([existing.to_pandas(), df], ignore_index=True)
    df.to_parquet(path, index=False)


def _append_failures(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path).to_pandas()
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)


@app.command()
def download(
    manifest: Path = typer.Option(Path("data/processed/image_manifest.parquet"), "--manifest"),
    image_dir: Path = typer.Option(Path("/home/legel/cfp_images"), "--image-dir"),
    checkpoint: Path = typer.Option(Path("outputs/checkpoint_downloaded.parquet"), "--checkpoint"),
    failed_path: Path = typer.Option(Path("outputs/failed_downloads.parquet"), "--failed"),
    concurrency: int = typer.Option(64, "--concurrency"),
    per_host_concurrency: int = typer.Option(32, "--per-host-concurrency"),
    cap_gb: float = typer.Option(800.0, "--cap-gb", help="Pause when image_dir usage exceeds this many GB."),
    cap_check_every: int = typer.Option(500, "--cap-check-every",
                                        help="Recheck disk usage every N images."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process only the first N undone images."),
    shuffle: bool = typer.Option(True, "--shuffle/--no-shuffle", help="Shuffle manifest order so species aren't fetched serially."),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
) -> None:
    """Download images from the manifest to ``image_dir`` (cap-aware, resumable)."""
    if not manifest.exists():
        raise typer.BadParameter(f"missing manifest {manifest}; run `cfp.gbif build-manifest` first.")

    image_dir.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat()
    prov_path = prov_dir / f"download_{snapshot.replace(':','').replace('-','')[:15]}.jsonl"
    prov_dir.mkdir(parents=True, exist_ok=True)
    prov = prov_path.open("w")
    prov.write(json.dumps({
        "type": "run_meta", "stage": "pipeline.download",
        "started_utc": snapshot, "manifest": str(manifest),
        "image_dir": str(image_dir), "cap_gb": cap_gb,
        "concurrency": concurrency, "per_host_concurrency": per_host_concurrency,
    }) + "\n")
    prov.flush()

    df = pd.read_parquet(manifest)
    done = _load_checkpoint(checkpoint)
    df = df[~df["gbif_occurrence_id"].isin(done)]
    if shuffle:
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    if limit:
        df = df.head(limit)
    console.print(f"to download: [bold]{len(df)}[/]  (checkpoint already covers {len(done)})")

    cap_bytes = int(cap_gb * 1024**3)
    connector_limit = max(concurrency, per_host_concurrency)
    timeout = aiohttp.ClientTimeout(total=120, connect=15)

    successes_buffer: list[int] = []
    failures_buffer: list[dict] = []
    n_done = 0
    last_disk_check = 0
    cur_disk = _disk_usage(image_dir)

    async def runner() -> None:
        nonlocal n_done, last_disk_check, cur_disk
        semaphore = asyncio.Semaphore(concurrency)
        connector = aiohttp.TCPConnector(
            limit=connector_limit, limit_per_host=per_host_concurrency, ttl_dns_cache=600
        )
        async with aiohttp.ClientSession(connector=connector, timeout=timeout,
                                         headers={"User-Agent": USER_AGENT}) as session:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as prog:
                tid = prog.add_task("downloading", total=len(df))
                rows_iter = (row.to_dict() for _, row in df.iterrows())
                tasks: set[asyncio.Task] = set()

                for row in rows_iter:
                    # Backpressure on disk cap.
                    if cur_disk >= cap_bytes:
                        console.print(f"[yellow]disk cap reached ({cur_disk/1e9:.1f} GB ≥ {cap_gb} GB) — waiting 60s[/]")
                        await asyncio.sleep(60)
                        cur_disk = _disk_usage(image_dir)
                        continue

                    # Spawn one fetch task.
                    tasks.add(asyncio.create_task(_fetch_one(session, semaphore, row, image_dir)))

                    # Drain when too many in flight.
                    if len(tasks) >= concurrency * 4:
                        done_set, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for t in done_set:
                            res = await t
                            n_done += 1
                            prog.update(tid, advance=1)
                            if res.get("status") == "ok":
                                successes_buffer.append(int(res["gbif_occurrence_id"]))
                                cur_disk += int(res.get("size_bytes", 0))
                            elif res.get("status") == "skipped_exists":
                                successes_buffer.append(int(res["gbif_occurrence_id"]))
                            else:
                                failures_buffer.append(res)
                            # Periodic flush.
                            if len(successes_buffer) >= 1000:
                                _append_checkpoint(checkpoint, successes_buffer)
                                successes_buffer.clear()
                            if len(failures_buffer) >= 500:
                                _append_failures(failed_path, failures_buffer)
                                failures_buffer.clear()
                            # Periodic disk recheck.
                            if n_done - last_disk_check >= cap_check_every:
                                cur_disk = _disk_usage(image_dir)
                                last_disk_check = n_done

                # Drain remaining.
                if tasks:
                    for coro in asyncio.as_completed(tasks):
                        res = await coro
                        n_done += 1
                        prog.update(tid, advance=1)
                        if res.get("status") == "ok":
                            successes_buffer.append(int(res["gbif_occurrence_id"]))
                        elif res.get("status") == "skipped_exists":
                            successes_buffer.append(int(res["gbif_occurrence_id"]))
                        else:
                            failures_buffer.append(res)

    asyncio.run(runner())
    _append_checkpoint(checkpoint, successes_buffer)
    _append_failures(failed_path, failures_buffer)
    prov.write(json.dumps({
        "type": "result",
        "downloaded_total": n_done,
        "disk_usage_bytes": _disk_usage(image_dir),
        "image_dir": str(image_dir),
        "checkpoint": str(checkpoint),
        "failed_path": str(failed_path),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    prov.close()
    console.rule("[bold green]download stage complete (or paused)")
    console.print(f"checkpoint: {checkpoint}")
    console.print(f"failures:   {failed_path}")
    console.print(f"provenance: {prov_path}")
