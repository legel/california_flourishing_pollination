"""GPU embedding stage.

Consumes images from ``--image-dir`` (one per file, plus a companion `.json`
metadata sidecar written by the downloader), runs DINOv3 in batches, persists
embedding shards to ``--shard-dir``, then **deletes the image** on success.

Embedding shard layout (one parquet per shard):
    embeddings_<shard_id>.parquet  with columns:
        gbif_occurrence_id (int64)
        taxon_name (utf8)
        gbif_taxon_key (int64)
        dataset_role (utf8)
        license (utf8)
        inat_observation_id (int64)
        image_url_large (utf8)
        cls_fp16 (binary)            -- shape (D,) packed via numpy.tobytes
        patches_fp16 (binary)        -- shape (H_p, W_p, D) packed via numpy.tobytes
        cls_shape (list<int32>)
        patches_shape (list<int32>)
        backbone (utf8)
        repo (utf8)
        embedded_utc (utf8)

Storage trade-off: we keep both CLS and spatial patches because (a) the user
plans to train both global-classification and spatial-inference models, and
(b) recomputing spatial later means re-downloading the image. With ViT-L/16 at
224 (14×14 patches × 1024 dim × 2 B = ~400 KB per image including CLS), 5M
images ≈ 2 TB of embeddings. At 448 (28×28 = ~1.6 MB/image) → 8 TB. We default
to 224 for production; flag --image-size to override.

Resumable: any image whose `gbif_occurrence_id` is in
``outputs/checkpoint_embedded.parquet`` is skipped.
"""

from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import typer
from PIL import Image, UnidentifiedImageError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.utils.data import DataLoader, Dataset

from cfp.dinov3.extractor import DINOv3Extractor
from cfp.dinov3.extractor_gpu import DINOv3ExtractorGPU
from cfp.dinov3.extractor_combined import DINOv3PhenoVisionExtractor


class _PendingImageDataset(Dataset):
    """Decodes images + reads metadata in parallel worker processes (PIL path)."""

    def __init__(self, items: List[Tuple[int, Path, Path]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        gbif_id, img_path, meta_path = self.items[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return {"gbif_id": gbif_id, "image": None, "meta": None,
                    "img_path": str(img_path), "ok": False}
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        return {"gbif_id": gbif_id, "image": img, "meta": meta,
                "img_path": str(img_path), "ok": True}


class _RawBytesDataset(Dataset):
    """Workers only read raw JPEG bytes from disk; GPU decodes them later (nvJPEG path)."""

    def __init__(self, items: List[Tuple[int, Path, Path]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        gbif_id, img_path, meta_path = self.items[idx]
        try:
            buf = img_path.read_bytes()
        except OSError:
            return {"gbif_id": gbif_id, "bytes": None, "meta": None,
                    "img_path": str(img_path), "ok": False}
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        return {"gbif_id": gbif_id, "bytes": buf, "meta": meta,
                "img_path": str(img_path), "ok": True}


def _collate_passthrough(batch):
    """Default collate would try to stack PIL Images — keep them as a list."""
    return batch

app = typer.Typer(add_completion=False, help="DINOv3 embedding worker.")
console = Console()


def _scan_pending(image_dir: Path, done: set) -> Iterator[tuple[int, Path, Path]]:
    """Yield (gbif_id, image_path, meta_path) for every image whose URL is not in `done`.

    Image filenames are ``<gbif_id>_<url_hash8>.<ext>`` for multi-photo support.
    Legacy ``<gbif_id>.<ext>`` files (from before the URL-keyed downloader) are
    still handled — we read their URL from the sidecar JSON. ``done`` is a set
    of image_url_large strings.
    """
    for bucket in sorted(image_dir.glob("*")):
        if not bucket.is_dir():
            continue
        for img in sorted(bucket.glob("*")):
            if img.suffix == ".json" or img.suffix.endswith(".tmp"):
                continue
            stem = img.stem
            try:
                gbif_id = int(stem.split("_", 1)[0])
            except ValueError:
                continue
            meta_path = img.with_suffix(".json")
            url = None
            if meta_path.exists():
                try:
                    url = json.loads(meta_path.read_text()).get("image_url_large") or \
                          json.loads(meta_path.read_text()).get("url")
                except Exception:
                    pass
            if url and url in done:
                continue
            yield gbif_id, img, meta_path


def _load_checkpoint(path: Path) -> set[str]:
    """Checkpoint is keyed by image_url_large (one entry per photo). Legacy
    checkpoints that only tracked gbif_occurrence_id are read defensively —
    those entries don't carry URL info so we treat the checkpoint as empty
    under the new scheme (legacy IDs may get re-embedded as URL-keyed rows)."""
    if not path.exists():
        return set()
    tbl = pq.read_table(path)
    if "image_url_large" in tbl.column_names:
        return set(tbl.to_pandas()["image_url_large"].dropna().astype(str).tolist())
    return set()


def _flush_shard(rows: list[dict], shard_dir: Path, shard_idx: int, run_id: str = "") -> Path:
    name = f"embeddings_{run_id}_{shard_idx:06d}.parquet" if run_id else f"embeddings_{shard_idx:06d}.parquet"
    out = shard_dir / name
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def _append_checkpoint(path: Path, urls: list[str]) -> None:
    if not urls:
        return
    df = pd.DataFrame({"image_url_large": urls,
                       "embedded_utc": datetime.now(timezone.utc).isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path).to_pandas()
        if "image_url_large" in existing.columns:
            df = pd.concat([existing[["image_url_large", "embedded_utc"]], df], ignore_index=True)
    df.to_parquet(path, index=False)


@app.command()
def embed(
    image_dir: Path = typer.Option(Path("/home/legel/cfp_images"), "--image-dir"),
    shard_dir: Path = typer.Option(Path("/home/legel/cfp_shards"), "--shard-dir"),
    checkpoint: Path = typer.Option(Path("outputs/checkpoint_embedded.parquet"), "--checkpoint"),
    backbone: str = typer.Option("vitl16", "--backbone"),
    image_size: int = typer.Option(224, "--image-size"),
    batch_size: int = typer.Option(64, "--batch-size"),
    images_per_shard: int = typer.Option(10_000, "--images-per-shard"),
    delete_after: bool = typer.Option(True, "--delete-after/--keep-images"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    poll_seconds: float = typer.Option(
        0.0, "--poll-seconds",
        help="If >0, after each pass-through sleep this long and re-scan for new images. "
             "Use for tail-the-disk concurrent operation alongside the downloader.",
    ),
    gpu_decode: bool = typer.Option(
        False, "--gpu-decode/--cpu-decode",
        help="If true, workers only read raw JPEG bytes and the GPU decodes "
             "them via nvJPEG (5-10× throughput vs PIL path).",
    ),
    with_phenovision: bool = typer.Option(
        False, "--with-phenovision",
        help="Also run PhenoVision (Dinnage 2025) to label each image with "
             "flowering + fruiting probability. Requires --gpu-decode.",
    ),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
) -> None:
    """Run the DINOv3 embedding pass over all unembedded images in `image_dir`."""
    if not image_dir.exists():
        raise typer.BadParameter(f"missing {image_dir}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat()
    prov_path = prov_dir / f"embed_{snapshot.replace(':','').replace('-','')[:15]}.jsonl"

    done = _load_checkpoint(checkpoint)
    console.print(f"checkpoint already has {len(done)} embedded ids")

    # Unique run_id avoids shard-name collisions across embed-process restarts
    # (which would otherwise clobber each other on the HF dataset).
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    console.print(f"this run's shard prefix: embeddings_{run_id}_")
    if gpu_decode:
        console.print("[bold cyan]GPU decode path (nvJPEG) enabled[/]")
    if with_phenovision:
        console.print("[bold cyan]PhenoVision flower+fruit classifier enabled (Dinnage 2025)[/]")
        if not gpu_decode:
            raise typer.BadParameter("--with-phenovision requires --gpu-decode")

    if with_phenovision:
        extractor = DINOv3PhenoVisionExtractor(
            dinov3_backbone=backbone, image_size=image_size,
        )
    elif gpu_decode:
        extractor = DINOv3ExtractorGPU(backbone=backbone, image_size=image_size)
    else:
        extractor = DINOv3Extractor(backbone=backbone, image_size=image_size)
    console.print(
        f"DINOv3 loaded: backbone={backbone}  repo={extractor.repo}  embed_dim={extractor.embed_dim}  "
        f"patch={extractor.patch_size}  grid={extractor.grid_hw}  device={extractor.device}  dtype={extractor.dtype}"
    )

    prov = prov_path.open("w")
    prov.write(json.dumps({
        "type": "run_meta", "stage": "pipeline.embed",
        "started_utc": snapshot, "image_dir": str(image_dir), "shard_dir": str(shard_dir),
        "backbone": backbone, "repo": extractor.repo, "image_size": image_size,
        "batch_size": batch_size, "delete_after": delete_after,
        "embed_dim": extractor.embed_dim, "grid_hw": list(extractor.grid_hw),
    }) + "\n")
    prov.flush()

    # Start at 0 — combined with run_id prefix, names are globally unique.
    shard_idx = 0
    shard_rows: list[dict] = []
    new_ids: list[int] = []
    n_done = 0

    batch_imgs: list = []  # PIL.Image for cpu path, raw bytes for gpu path
    batch_meta: list[dict] = []
    batch_paths: list[Path] = []

    def flush_batch() -> None:
        nonlocal shard_idx, shard_rows, new_ids
        if not batch_imgs:
            return
        if gpu_decode:
            out, failed_idx = extractor.embed_from_bytes(batch_imgs)
            if failed_idx:
                bad = set(failed_idx)
                for i in sorted(bad, reverse=True):
                    meta = batch_meta[i]
                    prov.write(json.dumps({"type": "bad_image",
                                           "gbif_occurrence_id": meta["gbif_occurrence_id"],
                                           "path": str(batch_paths[i])}) + "\n")
                surviving_meta = [m for i, m in enumerate(batch_meta) if i not in bad]
                surviving_paths = [p for i, p in enumerate(batch_paths) if i not in bad]
            else:
                surviving_meta = batch_meta
                surviving_paths = batch_paths
        else:
            out = extractor.embed(batch_imgs)
            surviving_meta = batch_meta
            surviving_paths = batch_paths
        cls_np = out.cls.astype(np.float16)
        patches_np = out.patches.astype(np.float16)
        # PhenoVision outputs are present only on the combined extractor.
        flowering = getattr(out, "flowering_prob", None)
        fruiting = getattr(out, "fruiting_prob", None)
        repo = getattr(extractor, "repo", getattr(extractor, "dinov3_repo", "?"))
        for i, meta in enumerate(surviving_meta):
            cls_bytes = cls_np[i].tobytes()
            patches_bytes = patches_np[i].tobytes()
            row = {
                **meta,
                "cls_fp16": cls_bytes,
                "patches_fp16": patches_bytes,
                "cls_shape": list(cls_np[i].shape),
                "patches_shape": list(patches_np[i].shape),
                "backbone": backbone,
                "repo": repo,
                "embedded_utc": datetime.now(timezone.utc).isoformat(),
            }
            if flowering is not None:
                row["phenovision_flowering_prob"] = float(flowering[i])
                row["phenovision_fruiting_prob"] = float(fruiting[i])
                row["phenovision_repo"] = getattr(extractor, "phenovision_repo", "phenobase/phenovision")
            shard_rows.append(row)
            new_ids.append(str(meta.get("image_url_large") or meta.get("url") or meta["gbif_occurrence_id"]))
        if delete_after:
            for p in surviving_paths:
                try:
                    p.unlink(missing_ok=True)
                    p.with_suffix(".json").unlink(missing_ok=True)
                except OSError:
                    pass
        if len(shard_rows) >= images_per_shard:
            out_path = _flush_shard(shard_rows, shard_dir, shard_idx, run_id=run_id)
            console.print(f"[green]flushed shard[/] {out_path}  ({len(shard_rows)} rows)")
            shard_idx += 1
            _append_checkpoint(checkpoint, new_ids)
            shard_rows = []
            new_ids = []
        batch_imgs.clear()
        batch_meta.clear()
        batch_paths.clear()

    import time as _time
    rounds = 0
    num_workers = int(os.environ.get("CFP_EMBED_WORKERS", "8"))
    while True:
        rounds += 1
        round_done = 0
        # Materialize the list of pending items, then hand to a DataLoader that
        # decodes + reads metadata in parallel worker processes.
        pending = list(_scan_pending(image_dir, done))
        if limit:
            pending = pending[: max(0, limit - n_done)]
        console.print(f"[cyan]round {rounds}: {len(pending)} pending images to embed[/]")
        if not pending:
            if poll_seconds <= 0:
                break
            console.print(f"[cyan]nothing to do; sleeping {poll_seconds:g}s[/]")
            _time.sleep(poll_seconds)
            done = _load_checkpoint(checkpoint)
            continue

        ds = _RawBytesDataset(pending) if gpu_decode else _PendingImageDataset(pending)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_passthrough,
            prefetch_factor=4,
            persistent_workers=False,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            tid = prog.add_task(f"embed pass {rounds}", total=len(pending))
            for batch in loader:
                batch_imgs.clear()
                batch_meta.clear()
                batch_paths.clear()
                for item in batch:
                    if not item["ok"]:
                        prov.write(json.dumps({"type": "bad_image",
                                               "gbif_occurrence_id": item["gbif_id"],
                                               "path": item["img_path"]}) + "\n")
                        continue
                    meta_obj = item["meta"] or {}
                    batch_imgs.append(item["bytes"] if gpu_decode else item["image"])
                    batch_meta.append({
                        "gbif_occurrence_id": item["gbif_id"],
                        "taxon_name": meta_obj.get("taxon_name"),
                        "gbif_taxon_key": meta_obj.get("gbif_taxon_key"),
                        "dataset_role": meta_obj.get("dataset_role"),
                        "license": meta_obj.get("license"),
                        "inat_observation_id": meta_obj.get("inat_observation_id"),
                        "image_url_large": meta_obj.get("url"),
                        "observed_on": meta_obj.get("observed_on"),
                        "decimal_latitude": meta_obj.get("decimal_latitude"),
                        "decimal_longitude": meta_obj.get("decimal_longitude"),
                    })
                    batch_paths.append(Path(item["img_path"]))
                if batch_imgs:
                    flush_batch()
                n_done += len(batch)
                round_done += len(batch)
                prog.update(tid, advance=len(batch))
            if shard_rows:
                out_path = _flush_shard(shard_rows, shard_dir, shard_idx, run_id=run_id)
                console.print(f"[green]flushed final shard[/] {out_path}  ({len(shard_rows)} rows)")
                _append_checkpoint(checkpoint, new_ids)
                shard_idx += 1
                shard_rows = []
                new_ids = []

        if poll_seconds <= 0:
            break
        console.print(f"[cyan]round {rounds} embedded {round_done} images; sleeping {poll_seconds:g}s before re-scan[/]")
        _time.sleep(poll_seconds)
        # Refresh done set so we skip what we just embedded (and any concurrent runs).
        done = _load_checkpoint(checkpoint)

    prov.write(json.dumps({
        "type": "result", "n_embedded": n_done,
        "checkpoint": str(checkpoint), "shard_dir": str(shard_dir),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    prov.close()
    console.rule("[bold green]embed stage complete (or idle)")
    console.print(f"checkpoint: {checkpoint}")
    console.print(f"provenance: {prov_path}")
