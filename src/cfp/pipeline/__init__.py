"""Streaming GBIF → image → DINOv3 → HF pipeline.

Three independently-runnable, independently-resumable stages connected by the
local filesystem as a bounded queue:

  download  — async HTTP fetch of iNaturalist large-size photos from a manifest;
              writes JPGs/PNGs to ``--image-dir``; pauses when disk usage
              approaches ``--cap-gb``.
  embed     — GPU worker (DINOv3 ViT-L/16 by default); consumes images, writes
              embedding shards, **deletes images** on success.
  upload    — pushes embedding shards to the HF dataset repo; deletes local
              shards on ack.

All three can run concurrently — the disk acts as the queue and applies natural
backpressure. Each stage maintains its own resumable checkpoint parquet under
``outputs/``.

Run:
    python -m cfp.pipeline download --manifest data/processed/image_manifest.parquet \\
        --image-dir /home/legel/cfp_images --cap-gb 800
    python -m cfp.pipeline embed   --image-dir /home/legel/cfp_images --shard-dir /home/legel/cfp_shards
    python -m cfp.pipeline upload  --shard-dir /home/legel/cfp_shards \\
        --repo deepearth/california-flourishing-pollination
"""
