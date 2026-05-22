# Pollination

> The animal-side track of the California Flourishing & Pollination pipeline.

**Problem statement.** Which animals pollinate which California-native plants, where, and when? Pollination is a multi-species *interaction* network, not a per-species trait — and that network is being restructured by climate change, pesticide use, and habitat loss faster than we can measure it.

**Goal of this track.** Produce an analysis-ready, self-supervised, spatial-feature dataset of every California-observed flying pollinator with a documented interaction with a California-native plant, plus the interaction graph itself. Downstream model lives at [`legel/deepearth/models/pollination`](https://github.com/legel/deepearth/tree/main/models/pollination).

## Inputs (this track)

| Source | Role | Output |
|---|---|---|
| **GloBi (Global Biotic Interactions)** — full `interactions.tsv.gz` snapshot from [depot.globalbioticinteractions.org](https://depot.globalbioticinteractions.org/snapshot/target/data/), concept DOI [`10.5281/zenodo.3950589`](https://doi.org/10.5281/zenodo.3950589) | Pollination interaction graph (CA-native plants ⨯ animal pollinators) | `data/raw/globi/interactions.tsv.gz`, `data/processed/globi_ca_plant_pollinator.parquet` |
| **RO ontology IRIs** `RO_0002455` (pollinates), `RO_0002456` (pollinatedBy), `RO_0002622` (visitsFlowersOf), `RO_0002623` (flowersVisitedBy) | Controlled vocabulary for pollination interaction filtering | filter clause in `cfp.globi filter` |
| **GBIF backbone classification** for each candidate pollinator | Kingdom/Phylum/Class/Order/Family hierarchy | `data/processed/pollinators_candidates.parquet` |
| **GBIF iNaturalist Research-grade CA observations** per candidate | Existence cross-check (only keep pollinators with ≥1 CA iNat observation) | `data/processed/pollinators_california_flying.parquet` |
| **Curated `flight_ability_rules.csv`** at order/family level | Excludes flightless taxa (Formicidae workers, Arachnida, etc.) per project scope | `data/processed/flight_ability_rules.csv` |
| **iNaturalist photo CDN** (URL only) | Source pixels for DINOv3 feature extraction | streamed → embedded → deleted |

## Stages (this track)

```
cfp.globi fetch            → download interactions.tsv.gz (2.6 GB, sha256-recorded)
cfp.globi filter           → CA × native-plant × pollination-IRI rows (~45,805)
                             + candidate pollinator list (~1,598)
cfp.pollinators cross-check → flight-ability + GBIF-CA-presence gating (~1,275 kept)
cfp.gbif build-manifest    → per-image iNat URL manifest, dataset_role='pollinator'
cfp.pipeline download/embed/upload  → same streaming pipeline as flourishing
```

Every stage emits a provenance JSONL under `provenance/` tagged with the source URL, snapshot UTC, sha256, and resolved DOI.

## Snapshot statistics (2026-05-22)

| Metric | Value |
|---|---|
| GloBi pollination rows globally | 1,924,124 (`pollinates` + `visitsFlowersOf`) |
| GloBi rows in CA bbox or locality | 62,090 |
| GloBi rows × CA-native plants | **45,805** |
| Candidate pollinators (unique animals on the non-plant side) | **1,598** |
| Kept after flight + CA-iNat cross-check | **1,275** (top: *Bombus vosnesenskii*, *Calypte anna*, *Danaus plexippus*, *Vanessa cardui*) |
| Order distribution | 432 Hymenoptera, 247 Lepidoptera, 201 Coleoptera, 115 Diptera, 92 Hemiptera, 7 Apodiformes (hummingbirds), 28 Orthoptera+others |
| Total CA iNat observations across kept pollinators | ~1.16M |

## Companion track

→ [`FLOURISHING.md`](FLOURISHING.md) — the plant-side track sharing this pipeline.

The two tracks are decoupled in source (plant species ≠ pollinator species) but share the GBIF manifest builder, DINOv3 extractor, and streaming pipeline.

## Open issues (Phase 1.5)

- Replace bbox + locality-string CA filter with precise CA TIGER-polygon clipping (geopandas).
- Add per-row interaction-type provenance (e.g. tag which IRI matched) for downstream Bayesian uncertainty modeling.
- Cross-reference pollinator life-stage from GloBi `sourceLifeStageName` to filter immatures.
- Pull GloBi `argumentTypeId` to distinguish supporting vs. refuting evidence.
