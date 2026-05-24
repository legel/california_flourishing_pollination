"""Backfill PhenoVision flowering+fruiting probabilities into the 4 oldest
HF embedding shards (embeddings_000000..000003.parquet, ~40K rows total)
that predate the combined DINOv3+PhenoVision extractor.

We do NOT re-run DINOv3 — the existing CLS+patches columns stay byte-identical.
We only download the image, run PhenoVision (ViT-B/16) once, and append the
two new columns + phenovision_repo. Result is uploaded back to HF.

This works without preserving the original images on disk because the image
URL is in each row (`image_url_large`).
"""
from __future__ import annotations
import asyncio
import io
import time
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from transformers import ViTForImageClassification


REPO = "deepearth/california-flourishing-pollination"
PHENOVISION_REPO = "phenobase/phenovision"
USER_AGENT = "DeepEarth-CFP/0.1 (phenovision backfill; lance@ecological.dev)"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


async def fetch_one(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                return None
            return await r.read()
    except Exception:
        return None


async def fetch_all(urls: list[str], concurrency: int = 64) -> list[bytes | None]:
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency, limit_per_host=32, ttl_dns_cache=600)
    out: list[bytes | None] = [None] * len(urls)

    async def w(i: int, u: str) -> None:
        async with sem:
            out[i] = await fetch_one(session, u)

    async with aiohttp.ClientSession(connector=conn, headers={"User-Agent": USER_AGENT}) as session:
        await asyncio.gather(*(w(i, u) for i, u in enumerate(urls)))
    return out


def main() -> None:
    api = HfApi()
    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                    if f.startswith("embeddings/") and f.endswith(".parquet"))
    # old format: no run_id between embeddings_ and the index
    old = [s for s in shards if "T" not in s.split("/")[-1].split("_", 2)[1]]
    print(f"shards to backfill: {len(old)}", flush=True)
    for s in old:
        print(f"  {s}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nloading PhenoVision on {device} {dtype}", flush=True)
    model = ViTForImageClassification.from_pretrained(PHENOVISION_REPO, torch_dtype=dtype).to(device).eval()
    mean = _IMAGENET_MEAN.to(device, dtype)
    std = _IMAGENET_STD.to(device, dtype)

    out_dir = Path("/home/legel/cfp_shards_backfill")
    out_dir.mkdir(exist_ok=True)

    for shard_name in old:
        t0 = time.time()
        print(f"\n=== {shard_name} ===", flush=True)
        local = hf_hub_download(REPO, shard_name, repo_type="dataset")
        df = pd.read_parquet(local)
        print(f"  rows: {len(df):,}", flush=True)

        if "phenovision_flowering_prob" in df.columns:
            print("  already has PhenoVision — skipping", flush=True)
            continue

        urls = df["image_url_large"].astype(str).tolist()
        print(f"  fetching {len(urls):,} images…", flush=True)
        bufs = asyncio.run(fetch_all(urls, concurrency=128))
        ok = sum(1 for b in bufs if b)
        print(f"  fetched ok: {ok:,}/{len(bufs):,}  ({time.time()-t0:.0f}s)", flush=True)

        # Inference in batches
        flowering = np.full(len(df), np.nan, dtype=np.float32)
        fruiting = np.full(len(df), np.nan, dtype=np.float32)
        batch_size = 64
        with torch.inference_mode():
            for start in range(0, len(bufs), batch_size):
                chunk = bufs[start:start + batch_size]
                tensors = []
                idxs = []
                for j, b in enumerate(chunk):
                    if not b:
                        continue
                    try:
                        pil = Image.open(io.BytesIO(b)).convert("RGB").resize(
                            (224, 224), Image.BILINEAR)
                    except Exception:
                        continue
                    t = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).to(
                        device, dtype, non_blocking=True) / 255.0
                    tensors.append(t.unsqueeze(0))
                    idxs.append(start + j)
                if not tensors:
                    continue
                batch = (torch.cat(tensors, dim=0) - mean) / std
                logits = model(pixel_values=batch).logits.float()
                probs = torch.sigmoid(logits).cpu().numpy()
                for j, i in enumerate(idxs):
                    flowering[i] = probs[j, 0]
                    fruiting[i] = probs[j, 1] if probs.shape[1] > 1 else 0.0

        df["phenovision_flowering_prob"] = flowering
        df["phenovision_fruiting_prob"] = fruiting
        df["phenovision_repo"] = PHENOVISION_REPO
        n_ok = int(np.sum(~np.isnan(flowering)))
        print(f"  PhenoVision ok: {n_ok:,}/{len(df):,}  ({time.time()-t0:.0f}s total)", flush=True)

        out_path = out_dir / Path(shard_name).name
        df.to_parquet(out_path, index=False)
        # Upload (replace)
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=shard_name,
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"PhenoVision backfill: {Path(shard_name).name}",
        )
        print(f"  uploaded {shard_name}  ({time.time()-t0:.0f}s)", flush=True)
        out_path.unlink()

    print("\nbackfill complete.", flush=True)


if __name__ == "__main__":
    main()
