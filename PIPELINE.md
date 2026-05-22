# Streaming Pipeline Architecture

The production embedding pass processes millions of iNaturalist images without persisting them. Stages run concurrently with bounded queues so each step exerts backpressure on its upstream — at steady state the GPU is the bottleneck and the disk footprint of in-flight images is ~`O(batch × workers)`.

```
       ┌───────────────────────────────────────────────────────────────────────────┐
       │                                                                           │
       │   PRODUCER                BUFFER       GPU WORKER       UPLOADER          │
       │   (download)              (RAM)        (DINOv3)         (HF dataset)      │
       │                                                                           │
       │   M async HTTP        ─►  Queue   ─►   N CUDA streams ─► HTTP push  ─►    │
       │   workers fetch           (cap K)      one model copy    parquet shards   │
       │   image bytes from        bytes        per GPU device    appended to HF   │
       │   iNaturalist CDN                      → embedding (np)                   │
       │                                                                           │
       │   ◄─ backpressure ─                                                       │
       │     (block on full queue)                                                 │
       │                                                                           │
       │   Image bytes deleted as soon as the GPU worker consumes them.            │
       │                                                                           │
       └───────────────────────────────────────────────────────────────────────────┘
                            ▲                                       │
                            │                                       │
                            └────────── resumable checkpoint ◄──────┘
                                       (parquet manifest of done IDs)
```

## Components

### Producer
- `asyncio` + `aiohttp` (M concurrent workers, default M=64).
- Pull image URLs from `data/processed/image_manifest.parquet`, skip IDs already in the resumable checkpoint, fetch bytes from `https://static.inaturalist.org/photos/<id>/large.<ext>`.
- Retry on 429/5xx with exponential backoff via `tenacity`.
- Push `(observation_id, taxon_id, image_id, bytes)` onto an `asyncio.Queue(maxsize=K)`; this is the backpressure point.

### GPU worker
- Single process per GPU (1 on this H200), preloads DINOv3 ViT-L/16 in fp16 (~600 MB).
- Pulls batches from the queue, decodes PIL → tensor on CPU workers (`torch.utils.data.DataLoader` with `prefetch_factor=4`), pushes to GPU.
- Forward pass returns CLS token + spatial patch tokens (`[N_patches × D]`).
- Stores `np.float16` arrays in an in-memory shard buffer; flushes to a parquet file per ~10 000 images.
- Drops image bytes immediately after embedding.

### Uploader
- Watches the shard directory; on each completed shard:
  - Append to `manifest_embeddings.parquet` (image_id, taxon_id, observation_id, image_url, license, embedding_path, shard_id, embedding_dtype, n_patches, patch_dim).
  - `hf upload deepearth/california-flourishing-pollination shards/<shard>.parquet --repo-type=dataset`.
- After successful upload, mark shard IDs as resumable-done in the local checkpoint.

### Resumable checkpoint
- Single parquet file `outputs/checkpoint_done_image_ids.parquet`.
- Updated atomically (write to `.tmp` then rename) after each shard's upload acknowledged.

## Throughput model

On a single H200 (143 GB VRAM, ~1980 TFLOPs fp16):
- DINOv3 ViT-L/16 at 224² in fp16 → ~5 ms/image at batch 64 → **~200 img/s sustained** (network-bound in practice).
- iNaturalist CDN with 64 concurrent fetches → ~100–250 img/s (sample size + region dependent).
- ⇒ pipeline is roughly balanced; will scale linearly with additional CDN concurrency until rate-limited.
- 5M images × 5 ms = ~7 GPU-hours; wall-clock dominated by network at ~4–10 days continuous.

## Failure modes & handling

| Failure | Handling |
|---|---|
| 429 from iNaturalist CDN | exponential backoff + reduce M; respect `Retry-After` |
| Image is HTML / 404 | log, skip, record in `failed.parquet` |
| Corrupt JPEG | catch `UnidentifiedImageError`, log, skip |
| GPU OOM | catch, halve batch, retry that batch only |
| Process crash | resume from checkpoint on restart |
| HF push failure | shard stays in pending dir; retry until success |
| Disk pressure | uploader fsync + delete shard parquet after HF ack |

## Configuration

All knobs live in `configs/pipeline.yaml` — see [`configs/pipeline.example.yaml`](configs/pipeline.example.yaml) for defaults. The streaming pipeline is invoked via `python -m cfp.pipeline.stream`.
