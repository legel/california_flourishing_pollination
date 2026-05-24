# Scientific Provenance

This document records — for every input dataset, every query, every model checkpoint, and every transformation — the exact URL, query string, snapshot date, license, and citation. Updated as the pipeline runs. Every artifact under `data/` and `outputs/` is traceable back to an entry here.

> Format: each section follows the W3C PROV pattern (Source → Activity → Entity), augmented with a `Snapshot date` (UTC), a `Citation` block, and a `License` line. All raw API responses are persisted under `provenance/` so the queries themselves are auditable.

---

## 1. California native plant species list

### 1a. CNPS Calscape — PRIMARY (canonical, authoritative)

**CNPS canonical definition of "California native plant"** (quoted verbatim, see https://www.cnps.org/gardening/why-natives/what-are-native-plants):

> Our native plants grew here prior to European contact. California's native plants evolved here over a very long period, and are the plants which the first Californians knew and depended on for their livelihood.

- **Source:** https://calscape.org/search (Native to California filter) → **Options → Export list to Excel**. Calscape is the CNPS-maintained authoritative plant database.
- **Mechanism:** The Calscape website sits behind Cloudflare Turnstile and exposes no public bulk API or CSV endpoint (verified: `/sitemap.xml`, `/export/search/`, `/app/taxon_search?nat=t&format=csv`, `/our-data`, and other paths all return HTTP 403 with a JS challenge wall). Automated bypass (cloudscraper, playwright-stealth + xvfb + headed Chromium with anti-detection args) likewise fails. The export is therefore acquired manually by the project PI through the Calscape UI and archived locally with sha256 attestation under `data/raw/calscape/`.
- **Snapshot date:** _set per fetch — recorded in `provenance/cnps_calscape_*.jsonl`_
- **Count at snapshot 2026-05-22:** **8,507 taxa** — matching CNPS's own self-reported "more than 8,500 types."
- **Per-taxon fields (50-column Calscape export):** Botanical Name, Common Name, **Butterflies and Moths Supported** (count — direct pollinator-network signal), Attracts Wildlife, Plant Type, Form, Height/Width (text + parsed min/max), Growth Rate, Seasonality, **Flower Color, Flowering Season**, Fragrance, Sun, Soil Drainage, Water Requirement, Summer Irrigation, Ease of Care, **Nursery Availability**, Companions, Special Uses, Communities (Jepson ecoregions), Hardiness, Sunset Zones, Soil characteristics, Mulch, Site Type, Elevation (min/max), Rainfall (min/max), Tips, Pests, Propagation, Other Names / Alternative Common Names / Obsolete Names, **Rarity**, **Is Cultivar**, Jepson Link, Calscape URL.
- **Cross-reference to iNat (added by `cfp.cnps calscape ingest`):** `inat_taxon_id` (resolved via `/v1/taxa?q=<name>&rank=species`), `ca_observation_count` (resolved via `/v1/observations/species_counts?place_id=14&taxon_id=<id>&native=true`). These are required for the downstream image manifest builder.
- **License:** Calscape data is **CC-BY-NC 4.0** per Calscape's terms; the derived parquet redistributes Calscape's per-taxon fields under the same NC restriction (i.e., for non-commercial research use only). Note: this is a per-row license metadata field added to the parquet; the DINOv3 embeddings derived downstream are transformative derivatives of *iNaturalist photos* (separate licensing chain) and are MIT-licensed.
- **Citation:** California Native Plant Society (2026). *Calscape: Native Plants for California.* https://calscape.org/. "Native to California" export snapshotted _YYYY-MM-DD_.

### 1b. iNaturalist species_counts (CA × Plantae × native=true) — secondary cross-reference

**CNPS canonical definition of "California native plant"** (quoted verbatim, see https://www.cnps.org/gardening/why-natives/what-are-native-plants):

> Our native plants grew here prior to European contact. California's native plants evolved here over a very long period, and are the plants which the first Californians knew and depended on for their livelihood.

- **Endpoint:** `https://api.inaturalist.org/v1/observations/species_counts?place_id=14&iconic_taxa=Plantae&native=true&per_page=500`
- **Returns:** every plant taxon (a) classified as `native` for the **California place specifically** (per iNat's California Check List #312, which ingests CNPS / Calflora / Jepson editorial decisions), and (b) with at least one observation in California.
- **Snapshot date:** _set per fetch — recorded in `provenance/cnps_inat_species_counts_*.jsonl`_
- **Count at snapshot 2026-05-22:** **8,869 taxa** (8,783 species + 72 hybrids + 11 genera + 3 complex).
- **Why `native=true`, not `establishment_means=native`.** The looser `establishment_means=native` admits ~10,000 taxa, but the long tail contains a community-curation noise band of non-CA-natives (e.g., *Melaleuca incana* — AU; *Banksia ashbyi* — WA; *Encephalartos woodii* — SA; *Camptotheca acuminata* — CN) that happen to be listed native somewhere intersecting the query. `native=true` is the strict per-place shortcut and yields a clean 8,869 — consistent with Calscape's self-reported "more than 8,500 types," with Jepson eFlora's ~6,500 native minimum-rank taxa, and with CNPS's 6,300 native-plants estimate.
- **Why this endpoint and not the bulk Calscape feed.** Calscape is the CNPS-maintained authoritative database but has no public bulk endpoint: `https://calscape.org/sitemap.xml`, `/our-data`, `/app/taxon_search?nat=t&format=csv`, and other paths all hit Cloudflare's bot-wall (HTTP 403 with JS challenge). The iNat `native=true` filter is the closest open + Nature-publishable mirror of Calscape, since iNat's California Check List #312 is itself curated from Calflora/CNPS/Jepson. A future Phase 1.5 task is a CNPS data-sharing MOU for the canonical Calscape feed, which carries the additional horticultural traits (sun/water/soil/bloom) and nursery availability fields.
- **Other paths we tried (all dead ends):**
  - `/v1/taxa?taxon_scheme_id=12` (Calflora mirror) — the `taxon_scheme_id` parameter is silently ignored; returns all 1.4M iNat taxa.
  - Wikidata SPARQL on `wdt:P3420` (Calflora ID) returns 10,744 taxa but lacks a native flag at scale (only 1,357 have a `wdt:P9714` native-range claim).
  - GBIF has no Calflora-published checklist dataset (verified via `/v1/dataset/search?q=calflora&type=CHECKLIST`).
  - Jepson eFlora and CCH2 explicitly prohibit redistribution (cite-only).
  - `https://www.inaturalist.org/check_lists/312.csv` returns HTTP 401 (auth required).
- **License:** iNat data is CC0/CC-BY-NC per the iNaturalist Terms; the derived parquet is redistributed under **MIT** (transformative derivative — species names, ranks, observation counts; no photo bytes).
- **Citation:** iNaturalist (2026). *Observations of native California plants (place_id=14, iconic_taxa=Plantae, native=true).* https://api.inaturalist.org/v1/observations/species_counts. Accessed _YYYY-MM-DD_.

### 1b. CNPS Rare Plant Inventory (rarity overlay)
- **URL:** https://rareplants.cnps.org/Search/Advanced/
- **Export:** built-in CSV "Export Results" button (≈2,400 rare CA taxa)
- **Snapshot date:** _pending_
- **Fields:** scientific name, common name, family, CRPR rank, state/federal status, county distribution, habitat, bloom period, elevation.
- **License:** citation required.
- **Citation:** California Native Plant Society, Rare Plant Program (2026). *Rare Plant Inventory (online edition, v9.5.1).* https://www.rareplants.cnps.org.

### 1c. GBIF occurrence backbone (citable observations)
- **URL:** https://api.gbif.org/v1/occurrence/search
- **Filter:** `country=US`, `stateProvince=California`, `kingdom=Plantae`, taxon ∈ Calscape-canonical set.
- **Bulk download:** `https://api.gbif.org/v1/occurrence/download/request` → DOI per download.
- **License:** per-record `license` field (CC0 / CC-BY / CC-BY-NC); per-photo `license`/`rightsHolder` in `multimedia.txt` (preferred — that is the creator-set photo license).
- **Citable downloads used in the master manifest** (all on iNat Research-grade × CA × StillImage):

  | # | Predicate | Records | GBIF key | DOI | Snapshot (UTC) |
  |---|---|---|---|---|---|
  | 1 | `TAXON_KEY IN [Calscape canonical plant keys]` | 3,607,437 | `0007278-260519110011954` | [`10.15468/dl.pbgs4h`](https://doi.org/10.15468/dl.pbgs4h) | 2026-05-22 |
  | 2 | `TAXON_KEY IN [216 Insecta, 5289 Trochilidae, 734 Chiroptera]` (broad pollinator scope; supersedes earlier narrow batch `10.15468/dl.yr3ud7`) | 1,497,966 | `0007705-260519110011954` | [`10.15468/dl.cvbfp4`](https://doi.org/10.15468/dl.cvbfp4) | 2026-05-22 |
  | 3 | `TAXON_KEY IN [5289 Trochilidae, 9201093 Ptiliogonatidae, 9321 Mimidae, 6176 Icteridae, 5263 Parulidae, 9285 Cardinalidae, 5215 Bombycillidae]` (added 2026-05-23 to broaden flower-visiting bird coverage) | 326,544 | `0009596-260519110011954` | [`10.15468/dl.gphfhs`](https://doi.org/10.15468/dl.gphfhs) | 2026-05-23 |
  | 4 | `TAXON_KEY IN [49 keys recovering 487 Calscape names unmatched on first pass — 29 cultivar-parent genera, 32 variety species heads, 62 HIGHERRANK matches]` (added 2026-05-23) | 4,611,190 | `0009598-260519110011954` | [`10.15468/dl.nbe8dt`](https://doi.org/10.15468/dl.nbe8dt) | 2026-05-23 |

  All four DwC-A zips are archived at `data/raw/gbif/` with `.meta.json` sidecars carrying the download key + DOI + record count. Each `gbif_occurrence_id` in the master manifest is traceable back to one of these DOIs.
- **Citation (composite — cite all four downloads + the iNat dataset):**
  *iNaturalist contributors / California Native Plant Society Calscape (2026). iNaturalist Research-grade observations (GBIF dataset key `50c9509d-22c7-4a22-a47d-8c48425ef4a7`), California-scoped subsets via GBIF Occurrence Download:* https://doi.org/10.15468/dl.pbgs4h, https://doi.org/10.15468/dl.cvbfp4, https://doi.org/10.15468/dl.gphfhs, https://doi.org/10.15468/dl.nbe8dt.

### 1d. Jepson eFlora (taxonomic authority, **cite only**)
- **URL:** https://ucjeps.berkeley.edu/eflora/
- **Use:** name reconciliation for ambiguous taxa; per-taxon Jepson link is preserved in the Calscape export for direct deep-linking.
- **License:** explicitly prohibits redistribution. Cite only; do not republish data.
- **Citation:** Jepson Flora Project (eds.) (2026). *Jepson eFlora.* University and Jepson Herbaria, UC Berkeley. https://ucjeps.berkeley.edu/eflora/.

---

## 2. GloBi — Global Biotic Interactions

- **Concept DOI** (resolves to latest snapshot): `10.5281/zenodo.3950589`
- **Latest versioned snapshot to be pinned in `provenance/globi_version.json`** at fetch time (e.g. `10.5281/zenodo.17118569`, v9, Sep 2025).
- **Bulk file used (interpreted, fully resolved taxon paths + lat/lon + citations):**
  `https://depot.globalbioticinteractions.org/snapshot/target/data/tsv/interactions.tsv.gz` (~2 GB compressed, ~6.5 GB uncompressed)
  plus `refuted-interactions.tsv.gz` (subtracted from filter).
- **API base** (spot lookups only — not used for bulk traversal): https://api.globalbioticinteractions.org/
- **Snapshot date:** _set at fetch time → `provenance/globi_snapshot_<UTC>.json`_

### Pollination interaction-type filter (OBO/RO IRIs)
Filter on the IRI (`interactionTypeId`), never on the label (label drift across snapshots).

| IRI | Label | Direction |
|---|---|---|
| `http://purl.obolibrary.org/obo/RO_0002455` | pollinates | animal → plant |
| `http://purl.obolibrary.org/obo/RO_0002456` | pollinatedBy | plant → animal |
| `http://purl.obolibrary.org/obo/RO_0002622` | visitsFlowersOf | animal → plant |
| `http://purl.obolibrary.org/obo/RO_0002623` | flowersVisitedBy | plant → animal |

(Broader `visits`/`visitedBy` — `RO_0002618`/`RO_0002619` — deliberately excluded: noisy without floral-structure post-filtering.)

### California geographic filter (applied as an OR)
- Coords: `-124.55 ≤ decimalLongitude ≤ -114.13` AND `32.53 ≤ decimalLatitude ≤ 42.01`, then clipped to the CA TIGER polygon (`gadm`/Census shapefile).
- Locality regex: `(?i)\b(California|Calif\.|\bCA\b)\b`, with a guard against `Baja\s+California`.
- Each retained row is tagged `geo_match ∈ {"polygon", "bbox", "locality", "both"}` for downstream provenance.

### License
- GloBI data: per Zenodo record — typically CC0; verified at snapshot pin time.
- **Citation:** Poelen, J. H., Simons, J. D., & Mungall, C. J. (2014). *Global Biotic Interactions: An open infrastructure to share and analyze species-interaction datasets.* Ecological Informatics, 24, 148–159. https://doi.org/10.1016/j.ecoinf.2014.08.005 — plus the resolved version DOI of the pinned snapshot.

---

## 3. iNaturalist (via GBIF index)

- **GBIF iNaturalist Research-grade dataset key:** `50c9509d-22c7-4a22-a47d-8c48425ef4a7`
- **GBIF API base:** https://api.gbif.org/v1/
- **Pollinator CA-presence cross-check pattern (one query per candidate animal taxon):**
  ```
  https://api.gbif.org/v1/occurrence/search?country=US&stateProvince=California
    &datasetKey=50c9509d-22c7-4a22-a47d-8c48425ef4a7
    &taxonKey=<gbif_backbone_key>&limit=0
  ```
  Read `count`. `count >= 1` ⇒ taxon is observed in CA on iNaturalist.
- **Bulk pull for >1k taxa:** use GBIF Occurrence Download with predicate `{country: US, stateProvince: California, datasetKey: 50c9509d-..., taxonKey: IN [...]}` to get a single DOI-cited dump.
- **Image-size policy:** request `large` size from iNaturalist's static CDN (`https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/large.<ext>` or `https://static.inaturalist.org/photos/<id>/large.<ext>`); do **not** redistribute the raw photo — store only the URL alongside the DINOv3 embedding.
- **License notes:** iNaturalist photos carry per-photo licenses (CC0, CC-BY, CC-BY-NC, etc.) plus All Rights Reserved. We persist the license string per image in the manifest and downstream filter where required.
- **Citation:** iNaturalist contributors, iNaturalist Research-grade Observations [Data set]. GBIF.org. https://doi.org/10.15468/ab3s5x (verify current DOI at snapshot time).

## 3b. Flight-ability lookup

Pollinator candidates from GloBI are gated by a curated `is_flying` rule applied at the order/family level using the GBIF backbone classification (`/v1/species/{key}`):

- **Include:** Lepidoptera, Diptera, Hymenoptera (all Apoidea + winged Vespidae/Crabronidae/etc.) **except Formicidae** (workers flightless — per project scope), Coleoptera (case-by-case at family level; default-include common pollinator families: Cantharidae, Cerambycidae, Cleridae, Meloidae, Mordellidae, Scarabaeidae, Buprestidae, Nitidulidae), Hemiptera, Thysanoptera, Neuroptera, Mecoptera.
- **Include (vertebrate):** Trochilidae (hummingbirds — only NA pollinator bird family of note); Chiroptera (mainly Phyllostomidae in CA: *Choeronycteris mexicana*, *Leptonycteris* spp.).
- **Exclude:** Formicidae, Arachnida, Gastropoda, non-Chiropteran Mammalia, reptiles.

Lookup table source: `data/processed/flight_ability_rules.csv` (versioned in this repo; reviewed by collaborators).

---

## 4. PhenoVision

- **Upstream repo:** https://github.com/Phenobase/phenovision (R-primary, MIT-licensed per Dinnage 2025)
- **Python port (used here):** https://github.com/legel/phenovision (fork, branch `main`); originated as upstream PR #1 ("Add simple inference script for flowering and fruiting prediction"), opened 2025-09-23, closed unmerged 2026-03-16. Cloned locally to `vendor/phenovision/`.
- **Inference entry point:** `vendor/phenovision/inference.py` — `PhenoVisionClassifier` class wrapping `transformers.ViTForImageClassification`; multi-label sigmoid outputs for flowering + fruiting.
- **Reproductive model weights:** https://huggingface.co/phenobase/phenovision (v1.1.0, trained 2025-10-27, MIT license, HF DOI `10.57967/hf/7952`). Loaded via `ViTForImageClassification.from_pretrained("phenobase/phenovision")`.
- **Leaves model weights (not used in Phase 1):** https://huggingface.co/phenobase/phenovisionL (HF DOI `10.57967/hf/5785`).
- **Image processor:** `google/vit-base-patch16-224`.
- **Code archive:** Zenodo DOI `10.5281/zenodo.15182888` — https://zenodo.org/records/15182889
- **Citation:** Dinnage, R., et al. (2025). *PhenoVision: A framework for automating and delivering research-ready plant phenology data from field images.* Methods in Ecology and Evolution, 16(8), 1763–1780. https://doi.org/10.1111/2041-210X.70081

---

## 5. DINOv3

- **Model:** Meta AI DINOv3 (self-supervised vision transformer, LVD-1689M pretrain).
- **Validation backbone:** `facebook/dinov3-vitb16-pretrain-lvd1689m` (10-image sanity check at 448², 28×28 patches × 768 dim).
- **Production backbone:** `facebook/dinov3-vitl16-pretrain-lvd1689m` (224², 14×14 patches × 1024 dim, 300M params).
- **Checkpoint source:** Hugging Face — gated; access granted to user `ecodash` on 2026-05-22.
- **Precision:** **bf16** (fp16 → NaN in attention end-to-end; bf16 is numerically equivalent to fp32 at fp16 throughput). See `ASSUMPTIONS.md` §E.
- **Token layout per image:** `[CLS] [4 register] [14×14 patch tokens]`. The 4 register tokens are discarded in our extractor; we persist CLS (1024,) and patches (14, 14, 1024).
- **License:** DINOv3 License Agreement (Meta AI; commercial use permitted under Meta's terms; we redistribute only the *transformative-derivative* embeddings, not weights).
- **Citation:** Siméoni, O., et al. (2025). *DINOv3.* arXiv:2508.10104. https://doi.org/10.48550/arXiv.2508.10104

---

## 6. UMAP (validation visualization)

- **Library:** `umap-learn` (pinned version in `requirements.txt`)
- **Use:** project 16×16×D DINOv3 spatial tokens to 16×16×3 → normalize to RGB → 50% opacity overlay on source image.
- **Citation:** McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426.

---

## Activity log

| Date (UTC) | Stage | Source | Snapshot artifact | Notes |
|---|---|---|---|---|
| 2026-05-22T01:46Z | `globi.fetch` | https://depot.globalbioticinteractions.org/snapshot/target/data | `data/raw/globi/interactions.tsv.gz` (2.69 GB, sha256 `85723ad5…`) + `refuted-interactions.tsv.gz` (24.7 MB, sha256 `d50e5d7e…`) | Concept DOI `10.5281/zenodo.3950589`; supporting files (`taxonMap`, `citations`) absent at depot — tolerated. |
| 2026-05-22T02:03Z | `dinov3.validate_sample` | `facebook/dinov3-vitb16-pretrain-lvd1689m` + iNat v1 | `data/validation/dinov3_sanity/dinov3_sanity_vitb16_448.zip` (24 MB) | 10 iconic CA-native plant photos at iNat large size; ViT-B/16 @ 448², 28×28 patch grid, embed_dim=768; UMAP→RGB overlays show coherent unsupervised segmentation of plant/sky/background. Cleared for production scale. |
| 2026-05-22T02:08Z | `cnps.fetch` | iNaturalist `/v1/observations/species_counts?place_id=14&taxon_id=47126&establishment_means=native` | `data/processed/plants_california_native.parquet` | 10,000 unique CA-native plant taxa (9,738 species + 257 hybrids + smaller ranks; 31 dupe rows from multi-establishment listings deduped on `inat_taxon_id`). Top by observation count: California poppy (67K), coast live oak (61K), buckwheat (60K), toyon (52K). |
| 2026-05-22T02:09Z | `globi.filter` | `data/raw/globi/interactions.tsv.gz` ⨯ `plants_california_native.parquet` | `data/processed/globi_ca_plant_pollinator.parquet` (45,805 rows) + `data/processed/pollinators_candidates.parquet` (1,598 rows) | RO IRIs `RO_0002455` + `RO_0002456` + `RO_0002622` + `RO_0002623`; CA geo filter via bbox OR locality regex; refuted rows subtracted. |
| 2026-05-22T02:12Z | `pollinators.cross_check` (v2) | GBIF backbone + iNaturalist Research-grade dataset (`50c9509d-22c7-4a22-a47d-8c48425ef4a7`) × flight_ability_rules.csv | `data/processed/pollinators_california_flying.parquet` (1,275 rows) + `data/processed/pollinators_excluded.parquet` (323 rows) | 1,598 candidates × {GBIF /species/match → classification, GBIF /occurrence/search CA-iNat presence}; bounded retry-on-429 (max 5 attempts, 10s cap) — v1 misclassified 1,500+ as non-CA due to GBIF rate-limiting, v2 confirms 1,275 flying-and-CA-observed. Top kept: Bombus vosnesenskii (15,754 CA iNat obs), Calypte anna / Anna's hummingbird (55,424), Danaus plexippus / monarch (42,330), Vanessa cardui / painted lady (17,514). |
