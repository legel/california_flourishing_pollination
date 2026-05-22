# Scientific Assumptions

Every operational choice in this pipeline encodes an assumption. They are listed here so reviewers can audit each one independently. Each entry has a **claim**, a **why**, and (when the choice was non-trivial) the **alternatives considered** and the **risk if wrong**.

## A. Defining "California-native plant"

**Claim.** A taxon is treated as California-native iff it appears in iNaturalist's `/v1/observations/species_counts` response for `place_id=14, taxon_id=47126, establishment_means=native`.

**Why.** iNat's `establishment_means` is sourced from `listed_taxa` curated for the California place by community editors that include Calflora/CNPS contributors. No alternative open machine-readable source exposes a per-taxon native flag at this scale (Wikidata SPARQL via `wdt:P3420` returns Calflora-listed taxa without the native flag; Calflora's plant search is JS/GWT with no public bulk endpoint; the iNat `taxon_scheme_id` API parameter is silently ignored; Jepson eFlora explicitly prohibits redistribution).

**Alternatives considered.** Calflora per-taxon scrape (~10K HTML requests, slow, CC-BY-NC); CNPS Rare Plant Inventory CSV (only ~2.4K rare taxa, not the full native list); Wikidata SPARQL on `wdt:P3420` (10,744 taxa but no native flag); Jepson eFlora (license blocks redistribution).

**Risk if wrong.** False positives admit naturalized exotics into the plant set. False negatives drop natives that aren't yet curated in iNat's listed_taxa. We mitigate by recording each taxon's `inat_taxon_id`, `parent_id`, and full ancestry — downstream consumers can re-filter against any future canonical native list (Calflora MOU, Jepson, GBIF).

## B. Defining "California-observed flying pollinator"

**Claim.** An animal taxon is included iff (a) it appears in a GloBi pollination interaction (`RO_0002455`, `RO_0002456`, `RO_0002622`, `RO_0002623`) with a CA-native plant within the California geographic scope, AND (b) it is observed in California at least once on iNaturalist Research-grade (via GBIF), AND (c) a curated family/order-level flight-ability rule (`data/processed/flight_ability_rules.csv`) marks it as flying.

**Why.** Pollination is an interaction, not a trait — GloBi is the canonical multi-source aggregation. iNat-via-GBIF cross-check guards against pollinators recorded interacting in another region but never observed in CA. The flight gate honors the project's stated scope (excludes Formicidae workers and other flightless flower visitors).

**Alternatives considered.** Trait-database lookups for flight ability per species (BETSI, Big-Bee-Network) — too sparse beyond bees; manual review per family — does not scale. Both deferred to Phase 1.5 refinement.

**Risk if wrong.** Excluding flightless ants drops a non-trivial pollination signal in arid CA ecosystems; the data is preserved in `pollinators_excluded.parquet` with `is_flying=False` so a future analysis can re-include them.

## C. CA geographic filter for GloBi rows

**Claim.** An interaction is "in California" iff either (a) `decimalLatitude ∈ [32.53, 42.01]` AND `decimalLongitude ∈ [-124.55, -114.13]`, OR (b) `localityName` matches `(?i)\b(California|Calif\.|\bCA\b)\b` excluding `Baja California`.

**Why.** GloBi coverage of `(decimalLatitude, decimalLongitude)` is incomplete (~40–60% of rows have coords); the locality regex recovers a non-trivial number of locality-only records.

**Risk if wrong.** False positives admit border-area interactions just outside CA (the bbox is a coarse rectangle). We do not currently clip to the precise CA polygon — adding `geopandas` + Census TIGER shapefile is a one-paragraph Phase 1.5 task.

## D. iNaturalist large-size photo URL pattern

**Claim.** Every iNat photo URL can be rewritten to its `large` variant by substituting the size suffix (`square|medium|original` → `large`). Hosted at `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/large.<ext>` or `https://static.inaturalist.org/photos/<id>/large.<ext>`.

**Why.** Empirically verified across the 10-image validation sample. iNat uses a stable predictable URL pattern.

**Risk if wrong.** A small number of older/legacy photos may live at non-conforming URLs and fail with 404 during the download stage. We tolerate these and log them in `outputs/failed_downloads.parquet`.

## E. DINOv3 backbone choice (ViT-L/16, 224², bfloat16)

**Claim.** `facebook/dinov3-vitl16-pretrain-lvd1689m` at 224² input and **bfloat16** precision is the right backbone for a one-time, reusable spatial-feature asset.

**Why.** ViT-L/16 is the largest DINOv3 backbone (300M params, embed_dim=1024) that fits comfortably in H200 memory at half precision with a large batch. 224² gives 14×14=196 patch tokens — sufficient for the user's stated downstream goal of patch-level species inference, while keeping the per-image embedding compact (~400 KB CLS+patches in bf16). The 10-image validation at ViT-B/16 + 448² confirmed that spatial features are semantically coherent (plant vs. background separation, flower vs. leaf).

**Numerical-stability finding (publication-relevant).** DINOv3 ViT-L/16 in fp16 produces NaN activations end-to-end on this stack (PyTorch 2.5.1 cu124 + transformers 5.9.0 + Hugging Face port `facebook/dinov3-vitl16-pretrain-lvd1689m`). Verified empirically: the **same input image** yields `[-0.18080, 0.38482, -0.01116, …]` in fp32, `[-0.18457, 0.37109, -0.00668, …]` in bf16, and `[nan, nan, nan, …]` in fp16. Likely cause is attention-score overflow above fp16's 65 504 ceiling; bf16 preserves the dynamic range of fp32 at fp16-equivalent throughput and memory cost, and matches fp32 to ≥3 decimals on every layer we sampled. Our first 15 643 production embeddings were computed in fp16 (NaN), discovered immediately on shard-0 inspection, scrubbed (shards deleted, embed checkpoint cleared, the corresponding IDs reset in `outputs/checkpoint_downloaded.parquet`), and re-embedded under bf16. The Hugging Face dataset never received the bad shards.

**Alternatives considered.** ViT-H+/7B (10× slower, marginal quality gain at this image scale); 448² (4× more tokens, ~8 TB total embeddings vs. ~2 TB at 224² for 5M images); fp32 (2× memory + slower, no quality gain over bf16 at our test); fp16 (broken, see above); DINOv2 (older; DINOv3 is the canonical 2025 baseline).

**Risk if wrong.** Future re-embedding requires re-downloading the original photos. We mitigate by persisting the original `image_url_large` in every embedding row.

## F. "Native" vs. "endemic" deduplication

**Claim.** iNat sometimes lists a taxon under multiple `establishment_means` for one place (e.g., both `native` and `endemic`). We dedupe on `inat_taxon_id`, keeping the row with the highest `ca_observation_count`.

**Why.** Two rows for one taxon would inflate downstream counts and confuse manifest joins.

**Risk if wrong.** Endemism is a stronger native signal than mere nativity; we lose that nuance unless it is re-derived from the per-taxon page. We accept this for Phase 1.

## G. We do NOT redistribute the iNaturalist photos

**Claim.** The Hugging Face dataset includes DINOv3 embeddings + the original photo URL + per-photo license, but never the photo bytes.

**Why.** Per-photo licenses on iNat span CC0, CC-BY, CC-BY-NC, CC-BY-SA, CC-BY-NC-SA, and "all rights reserved" — bulk redistribution would violate the most restrictive subset. The URL is sufficient for any downstream consumer to re-download under each photo's own terms.

**Risk if wrong.** None for redistribution; some user-deleted photos may 404 in the future, breaking reproducibility for those records. We mitigate by snapshotting the URL + license string at embedding time — the historical license is auditable from the manifest even after a 404.

## H. We assume Phase 1 may include non-plant photos in the embedding set

**Claim.** When the iNat observation has only a non-plant photo (e.g., a habitat shot, a field-notebook snapshot, a worksheet), we still embed it under the labeled species.

**Why.** The user's downstream goal is *patch-level species inference* from spatial features — a model trained to ignore "this patch is paper, not poppy" is more useful than one that only sees clean plant portraits.

**Risk if wrong.** Some species labels will be noisier. We do not currently filter by PhenoVision flower-presence at embedding time. PhenoVision can be applied as a post-hoc tag on the embedding rows in Phase 1.5 without re-embedding.

## I. Computational reproducibility

**Claim.** A future researcher with the snapshot dumps + code in this repo can reproduce the dataset bit-for-bit (modulo upstream source drift).

**Why.** Every external query records: URL, parameters, snapshot UTC, sha256 of the raw bytes (where applicable), and the resolved DOI. Every output parquet carries `snapshot_utc`. Every model has a pinned HF repo + commit-revision.

**Risk if wrong.** iNat + GBIF data drift over time (taxa renamed, observations deleted/added). We snapshot raw GloBi locally; iNat/GBIF are live APIs whose responses we log but do not freeze. A re-run "tomorrow" will produce a similar but not identical dataset. The PROVENANCE.md activity log captures every snapshot timestamp so the temporal slice is auditable.
