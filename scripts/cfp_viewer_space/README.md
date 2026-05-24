---
title: California Flourishing & Pollination Viewer
emoji: 🌼
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Browse 10M iNat × DINOv3 × PhenoVision records
---

# CFP Viewer Space

An interactive browser for the `deepearth/california-flourishing-pollination`
Hugging Face dataset. Shows each iNaturalist photo alongside its
**PhenoVision flowering/fruiting probabilities** and a **DINOv3 14×14×1024
PCA-to-RGB overlay at 50% opacity**.

## Storage strategy (no 3.5 TB on the Space)

- The Space ships only the ~200 MB master manifest + a ~200 MB shard index.
- Embedding shards (~3 GB each, 1,072 shards = 3.5 TB total) are pulled
  on demand from the parent dataset via `huggingface_hub.hf_hub_download`.
- HF caches each downloaded shard on the Space's persistent disk
  (~50 GB), so recently-viewed shards stay fast.
- Photos are fetched directly from iNaturalist at request time
  (`https://inaturalist-open-data.s3.amazonaws.com/photos/...`).

## One-time setup

```bash
# Build the shard index (image_url_large -> shard_path) and upload to HF.
# Runs in ~30 min on a fast link; needed before the Space can locate
# embeddings per row.
python build_shard_index.py
```

## Local dev

```bash
pip install gradio pandas pillow numpy huggingface_hub requests pyarrow
python app.py
```

## Deploying to a HF Space

```bash
hf upload-folder . --repo-id deepearth/cfp-viewer --repo-type space
```

The Space's `requirements.txt` should list:

```
gradio
pandas
pillow
numpy
huggingface_hub
requests
pyarrow
```

## What the user sees

- **Search by species** (autocomplete on the clean `taxon_name`)
- **Random observation** button for serendipity
- **Direct iNat URL** input for direct linking
- For each selected record:
  - the iNaturalist photo (right column, top-left)
  - the DINOv3 RGB overlay (right column, top-right)
  - PhenoVision flowering + fruiting probabilities
  - clean + verbatim scientific name, observation date, locality, lat/lng
  - photo creator + CC license

## Future enhancements

- Map view with leaflet/deck.gl clustering on the 5M observation lat/lngs
- Phenology browser: filter by month + flowering probability ≥ threshold
- Pollinator co-occurrence search: given a plant species, list pollinators
  observed nearby (lat/lng × date window) and show their photos
