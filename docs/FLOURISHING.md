# Flourishing

> The plant-side track of the California Flourishing & Pollination pipeline.

**Problem statement.** When, where, and at what intensity do California's native plant species flower? Flowering is the central phenological event that drives downstream pollination, fruiting, seed dispersal, and ecosystem function. Climate change, drought, fire, and land-use change are perturbing flowering phenology at scales we cannot yet measure or forecast continuously across the state.

**Goal of this track.** Produce an analysis-ready, self-supervised, spatial-feature dataset of every California-native plant observation suitable for training continuous, per-species, per-location flowering forecasters. Downstream model lives at [`legel/deepearth/models/flowering`](https://github.com/legel/deepearth/tree/main/models/flowering).

## Inputs (this track)

| Source | Role | Output |
|---|---|---|
| **iNaturalist `/v1/observations/species_counts`** (`place_id=14`, `taxon_id=47126`, `establishment_means=native`) | Authoritative CA-native plant species list — community-curated through Calflora editors. **The full CNPS canonical list ([~12K taxa via Calscape](https://calscape.org/our-data)) is the Phase 1.5 target** (currently limited to the top 10,000 by iNat's pagination cap; see [`PROVENANCE.md`](../PROVENANCE.md) §1a for the honest disclosure). | `data/processed/plants_california_native.parquet` |
| **GBIF iNaturalist Research-grade dataset** (`datasetKey=50c9509d-22c7-4a22-a47d-8c48425ef4a7`) filtered to `country=US, stateProvince=California` | Citable per-image manifest with DOIs | `data/processed/image_manifest_plants.parquet` |
| **iNaturalist photo CDN** (URL only — we never redistribute the photo bytes) | Source pixels for DINOv3 feature extraction | streamed → embedded → deleted |
| **PhenoVision** ([Dinnage 2025](https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.70081)) | Per-image flower-presence + fruiting probability label (ViT classifier, MIT-licensed); user's Python port at [`legel/phenovision`](https://github.com/legel/phenovision) vendored at `vendor/phenovision/` | Phase 1.5 label join |

## Stages (this track)

```
cfp.cnps fetch          → CA-native plant species list (10K taxa, see §1a caveat)
cfp.gbif build-manifest → per-image iNat URL manifest, dataset_role='plant'
cfp.pipeline download   → fetch large-size JPGs (resumable, 800 GB cap)
cfp.pipeline embed      → DINOv3 ViT-L/16 bf16 spatial features, delete image
cfp.pipeline upload     → HF shard publication (dataset_role='plant' rows)
cfp.phenovision label   → [Phase 1.5] flower-presence labels per row
```

Every stage emits a provenance JSONL under `provenance/` tagged with the source URL, snapshot UTC, sha256, and resolved DOI.

## Companion track

→ [`POLLINATION.md`](POLLINATION.md) — the animal-side track sharing this pipeline.

The two tracks are decoupled in source (plant species ≠ pollinator species) but share the GBIF manifest builder, DINOv3 extractor, and streaming pipeline.

## Open issues (Phase 1.5)

- Acquire full CNPS canonical native list (~12K taxa) via Calscape data partnership or family-level GBIF query splitting (currently iNat's `/observations/species_counts` caps at 10K rows; see [`PROVENANCE.md`](../PROVENANCE.md) §1a).
- Apply PhenoVision per-image flower-presence labels and join to embedding manifest.
- Filter community-curation noise from the long tail (≤5 obs taxa include some non-CA-natives mistakenly listed).
