# California Flourishing & Pollination

> **AI for Scientific Modeling of Ecosystems — Dynamic Forecasting of Flowering and Pollination across California with DeepEarth**
>
> A research collaboration between **[Ecological Intelligence, Inc.](https://ecological.dev)** (Lance Legel, PI) and the **[Quantitative Ecosystem Dynamics (QED) Lab](https://www.keenangroup.info/)** at UC Berkeley (Trevor Keenan, PI; ESPM).

This repository holds the Phase 1 pipeline: **data collection, pre-processing, and self-supervised spatial-feature embedding** of every iNaturalist Research-grade observation of every California-native plant and every California-observed flying pollinator. Outputs are published as the [`deepearth/california-flourishing-pollination`](https://huggingface.co/datasets/deepearth/california-flourishing-pollination) dataset on Hugging Face. An interactive viewer is at the [companion Space](https://huggingface.co/spaces/deepearth/california-flourishing-pollination).

Downstream forecasting models live in [`legel/deepearth/models/flowering`](https://github.com/legel/deepearth/tree/main/models/flowering) and [`legel/deepearth/models/pollination`](https://github.com/legel/deepearth/tree/main/models/pollination).

## At a glance

| | |
|---|---|
| Images embedded | **10.27 M** (99.77 % of 10.30 M URL manifest) |
| Observations | **5.24 M** iNaturalist Research-grade × California × StillImage |
| Species | **16,446** (6,383 CA-native plants + 10,063 flying pollinators) |
| Per-image features | DINOv3 ViT-L/16 CLS (1024,) + 14×14×1024 spatial patches + PhenoVision flowering/fruiting probabilities |
| Geographic scope | California, USA — coordinates required, Research-grade only |
| Storage on HF | ~4.15 TB across **1,275 parquet shards** |
| License | **MIT** for code/parquets/embeddings; per-photo CC license preserved per row (83 % CC BY-NC) |

## Two coordinated tracks

The pipeline addresses two decoupled-but-coordinated problems. See the per-track docs:

- **[Flourishing](docs/FLOURISHING.md)** — when, where, and at what intensity do California's native plants flower? Plant-side data, PhenoVision flower/fruit labels, downstream `flowering` model.
- **[Pollination](docs/POLLINATION.md)** — which animals pollinate which native plants, where, and when? Animal-side data, GloBi interaction graph, downstream `pollination` model.

The two tracks share infrastructure (GBIF batch manifest builder, DINOv3 + PhenoVision combined extractor, async download → embed → upload pipeline) but produce distinct species lists, distinct manifest rows (tagged `dataset_role='plant'|'pollinator'`), and distinct downstream models.

The dataset supports three downstream research thrusts (Phases 2-4, separate repositories):

1. **Ecological forecasting of flowering & pollination** — PhenoVision flower detection + GloBi pollination records + AmeriFlux + remote sensing.
2. **Dense field reconstruction with Bayesian uncertainty** — Senseiver-style sparse-to-dense inference across space, time, and modalities.
3. **Physics-inspired graph neural networks for ecosystem causality** — GraphCast/GenCast-inspired GNN forecasting.

---

## Data sources (citable GBIF DOIs)

Every `gbif_occurrence_id` in the manifest is traceable to one of four GBIF Occurrence Downloads:

| # | Predicate | Records | DOI |
|---|---|---|---|
| 1 | iNat-RG plants × CA × Calscape canonical (`TAXON_KEY ∈ Calscape keys`) | 3.6 M | [`10.15468/dl.pbgs4h`](https://doi.org/10.15468/dl.pbgs4h) |
| 2 | iNat-RG broad pollinators (`Insecta ∪ Trochilidae ∪ Chiroptera`) | 1.5 M | [`10.15468/dl.cvbfp4`](https://doi.org/10.15468/dl.cvbfp4) |
| 3 | iNat-RG expanded bird pollinators (6 flower-visiting families) | 0.3 M | [`10.15468/dl.gphfhs`](https://doi.org/10.15468/dl.gphfhs) |
| 4 | iNat-RG plant cultivar/variety recovery (49 GBIF keys for 487 unmatched Calscape names) | 4.6 M | [`10.15468/dl.nbe8dt`](https://doi.org/10.15468/dl.nbe8dt) |

Native plant list: **CNPS Calscape** ("Native to California" filter), 8,507 canonical taxa. Pollinator scope gated by [`flight_ability_rules.csv`](./data/processed/flight_ability_rules.csv) — Insecta + Trochilidae + Chiroptera + Ptiliogonatidae + Mimidae + Icteridae + Parulidae + Cardinalidae + Bombycillidae, minus Formicidae (flightless workers). PhenoVision flower/fruit classifier from [Dinnage et al. 2025](https://doi.org/10.1111/2041-210X.70081). DINOv3 from [Siméoni et al. 2025](https://arxiv.org/abs/2508.10104).

Full details in [`PROVENANCE.md`](./PROVENANCE.md). Per-stage JSONL provenance logs in [`provenance/`](./provenance/).

---

## Repository layout

```
california_flourishing_pollination/
├── README.md                     ← you are here
├── AGENTS_README.md              ← canonical onboarding for future agents (READ FIRST)
├── PROVENANCE.md                 ← exhaustive data + query provenance (GBIF DOIs, citations, hashes)
├── ASSUMPTIONS.md                ← every operational assumption + risk-if-wrong
├── PIPELINE.md                   ← streaming architecture overview
├── RUN_GUIDE.md                  ← operator copy-paste recipe
├── HANDOVER.md                   ← latest-session state for the next agent
├── docs/{FLOURISHING,POLLINATION,DINOV3_RUN}.md
├── src/cfp/                      ← Python package
│   ├── cnps/calscape.py          # Calscape canonical native ingest
│   ├── globi/fetch_and_filter.py
│   ├── pollinators/cross_check.py
│   ├── gbif/batch_download.py    # GBIF Occurrence Download (batch path)
│   ├── dinov3/extractor_combined.py  # DINOv3 + PhenoVision in one GPU pass
│   ├── pipeline/{download,embed,upload}.py
│   └── hf/publish_meta.py
├── scripts/
│   ├── autonomous_watchdog.sh           # 24h cycle worker supervisor
│   ├── integrate_*.sh                   # GBIF batch integrators
│   ├── patch_license_and_names.py       # per-photo license + clean names
│   ├── fix_phenovision_swap_on_hf.py    # backfill old shards (with cache cleanup)
│   ├── retry_backfill_fails.py          # paced retry of HF rate-limited fails
│   └── cfp_viewer_space/                # Gradio Space source (deployed on HF)
├── data/raw/                     ← snapshots (GloBi, Calscape, GBIF DwC-A zips)
├── data/processed/               ← canonical parquets (manifest, lookups)
├── outputs/                      ← PIDs + checkpoints (gitignored)
├── provenance/                   ← per-stage JSONL audit logs
├── logs/                         ← per-process logs
└── vendor/phenovision/           ← user's Python port of phenobase/phenovision
```

## Interactive viewer

[**deepearth/california-flourishing-pollination** Space](https://huggingface.co/spaces/deepearth/california-flourishing-pollination) — Gradio app:
- Browse 10 M observations by species (autocomplete on 16K taxa) or click random
- Per-record view: iNat photo + DINOv3 patch-grid overlay (UMAP-projected to 3D RGB) + PhenoVision flowering/fruiting probabilities + full per-photo CC license + creator attribution
- Opacity slider is JS-driven (instant); resolution slider and normalize-per-image checkbox recompute the overlay server-side
- Loads in ~5 s (404 KB static species list ships with the Space; manifest + UMAP encoder lazy-load in background)

## Reproducibility

```bash
conda create -n cfp python=3.11 -y
conda activate cfp
pip install -e .

# 1. Acquire CNPS Calscape canonical native plant list (manual Excel export — see PROVENANCE.md §1a)
python -m cfp.cnps calscape ingest --xlsx ~/Native\ To\ California.xlsx

# 2. GloBi pollination interactions (CA-scoped)
python -m cfp.globi.fetch_and_filter

# 3. Pollinator flight-ability cross-check
python -m cfp.pollinators.cross_check

# 4. GBIF batch downloads (4 separate Occurrence Downloads with DOIs)
python -m cfp.gbif batch resolve-keys
python -m cfp.gbif batch submit
python -m cfp.gbif batch wait
python -m cfp.gbif batch parse

# 5. Launch the autonomous pipeline (download → embed → upload)
nohup bash scripts/autonomous_watchdog.sh > logs/watchdog.log 2>&1 &

# 6. Publish species lists + provenance + dataset card to HF
python -m cfp.hf publish-meta
```

## Citation

```bibtex
@dataset{legel_keenan_2026_cfp,
  title = {California Flourishing \& Pollination: a multi-modal AI dataset for ecological forecasting},
  author = {Legel, Lance and Keenan, Trevor},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/deepearth/california-flourishing-pollination},
  doi = {via GBIF Occurrence Downloads: 10.15468/dl.pbgs4h, 10.15468/dl.cvbfp4, 10.15468/dl.gphfhs, 10.15468/dl.nbe8dt}
}
```

Also cite the upstream sources: PhenoVision (Dinnage 2025, *Methods in Ecology and Evolution* 16(8):1763 — https://doi.org/10.1111/2041-210X.70081); DINOv3 (Siméoni et al. 2025, arXiv:2508.10104); GloBi (Poelen et al. 2014, *Ecological Informatics* 24:148-159); iNaturalist Research-grade observations; CNPS Calscape.

## License

**MIT** for code, parquets, embeddings, manifests, lookups, model extraction artifacts.

iNaturalist photos are not redistributed — only the `image_url_large` + per-photo CC license string + creator attribution are preserved per row (83 % CC BY-NC, 11 % CC BY, 4 % CC0, 2 % other CC variants). The DINOv3 spatial features are transformative derivatives that cannot be reversed to recover the source image.

DINOv3 model weights: per Meta's DINOv3 license (gated on HF, access granted to user `ecodash`). PhenoVision model weights: MIT (Dinnage 2025).

## Contact

Lance Legel — `lance@ecological.dev` · [@deepearth on HF](https://huggingface.co/deepearth) · [github.com/legel/deepearth](https://github.com/legel/deepearth)
