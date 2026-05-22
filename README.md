# California Flourishing & Pollination

> **AI for Scientific Modeling of Ecosystems — Dynamic Forecasting of Flowering and Pollination across California with DeepEarth**
>
> A research collaboration between **[Ecological Intelligence, Inc.](https://ecological.dev)** (Lance Legel, PI) and the **[Quantitative Ecosystem Dynamics (QED) Lab](https://www.keenangroup.info/)** at UC Berkeley (Trevor Keenan, PI; ESPM). Affiliated Organization research collaboration, hosted at the QED lab starting **May 4, 2026**.

This repository hosts the Phase 1 pipeline: **data collection, pre-processing, and self-supervised embedding** of every iNaturalist observation of every California-native plant and every California-observed flying pollinator. Outputs are published as the [`deepearth/california-flourishing-pollination`](https://huggingface.co/datasets/deepearth/california-flourishing-pollination) dataset on Hugging Face. Downstream forecasting models live in [`legel/deepearth/models/flowering`](https://github.com/legel/deepearth/tree/main/models/flowering) and [`legel/deepearth/models/pollination`](https://github.com/legel/deepearth/tree/main/models/pollination).

## Two decoupled tracks

The pipeline addresses two decoupled-but-coordinated problems. See the per-track docs:

- **[Flourishing](docs/FLOURISHING.md)** — when, where, and at what intensity do California's native plants flower? Plant-side data, PhenoVision flower-presence labels, downstream `flowering` model.
- **[Pollination](docs/POLLINATION.md)** — which animals pollinate which native plants, where, and when? Animal-side data, GloBi interaction graph, downstream `pollination` model.

The two tracks share infrastructure (GBIF manifest builder, DINOv3 extractor, streaming download → embed → upload pipeline) but produce distinct species lists, distinct manifest rows (tagged `dataset_role='plant'|'pollinator'`), and distinct downstream models.

The dataset is designed to support three downstream research thrusts (Phases 2–4, separate repositories):

1. **Ecological forecasting of flowering & pollination** — PhenoVision-based flower detection + GloBi pollination records + AmeriFlux + remote sensing.
2. **Dense field reconstruction with Bayesian uncertainty** — Senseiver-style sparse-to-dense inference across space, time, and modalities.
3. **Physics-inspired graph neural networks for ecosystem causality** — GraphCast/GenCast-inspired GNN forecasting.

---

## Scientific Provenance

Every input dataset, every query, every model checkpoint, and every transformation in this pipeline is documented in [`PROVENANCE.md`](./PROVENANCE.md) with a citable URL, query string, snapshot date, and license. This repository is built to a **Nature/Science submission-grade** standard of reproducibility.

## Repository Layout

```
california_flourishing_pollination/
├── README.md                          ← you are here
├── PROVENANCE.md                      ← exhaustive data & query provenance
├── PIPELINE.md                        ← architecture of the streaming pipeline
├── requirements.txt                   ← pinned Python dependencies
├── configs/                           ← run configs (DINOv3 size, batch, paths)
├── src/cfp/                           ← Python package
│   ├── cnps/                          ← CNPS / Calscape native plant ingestion
│   ├── globi/                         ← GloBi download + interaction filtering
│   ├── pollinators/                   ← flying-pollinator cross-check
│   ├── gbif/                          ← GBIF / iNaturalist image discovery
│   ├── phenovision/                   ← flower-presence classifier wrapper
│   ├── dinov3/                        ← spatial-embedding extractor
│   ├── pipeline/                      ← async stream orchestrator
│   └── hf/                            ← Hugging Face publishing
├── scripts/                           ← entry-point CLIs
├── notebooks/                         ← exploratory notebooks
├── data/
│   ├── raw/                           ← snapshots (CNPS, GloBi, GBIF queries)
│   ├── processed/                     ← filtered species lists, manifests
│   └── validation/                    ← 10-image DINOv3 sanity check artifacts
├── outputs/
│   ├── embeddings/                    ← DINOv3 spatial features
│   └── manifests/                     ← parquet/jsonl image+embedding manifests
├── provenance/                        ← raw query+response logs (per source)
└── logs/
```

## Pipeline Stages

| # | Track | Stage | Source | Output |
|---|---|---|---|---|
| 1 | flourishing | CA-native plant species list | iNat species_counts (Phase 1.5: CNPS Calscape) | `data/processed/plants_california_native.parquet` |
| 2 | pollination | GloBi interaction download | GloBi `interactions.tsv.gz` (DOI 10.5281/zenodo.3950589) | `data/raw/globi/` |
| 3 | pollination | GloBi pollination filter | RO IRIs × CA bbox/locality × CA natives | `data/processed/globi_ca_plant_pollinator.parquet` + `pollinators_candidates.parquet` |
| 4 | pollination | Pollinator cross-check | GBIF backbone + iNat-CA-presence + flight rules | `data/processed/pollinators_california_flying.parquet` |
| 5 | flourishing + pollination | Image manifest | GBIF iNaturalist Research-grade CA dataset | `data/processed/image_manifest_{plants,pollinators}.parquet` |
| 6 | flourishing + pollination | DINOv3 sanity (n=10) | DINOv3 ViT-B/16 + UMAP → RGB overlay | `data/validation/dinov3_sanity_*.zip` |
| 7 | flourishing + pollination | Streaming production embedding | DINOv3 **ViT-L/16 bf16** over all images, **delete on success** | HF shards `embeddings_*.parquet` |
| 8 | flourishing + pollination | Hugging Face publication | `deepearth/california-flourishing-pollination` | metadata + species lists + embeddings + provenance |

## Reproducibility

```bash
conda create -n cfp python=3.11 -y
conda activate cfp
pip install -r requirements.txt

# 1. Acquire native plant + pollinator species lists (downloads + filters)
python -m cfp.cnps.fetch
python -m cfp.globi.fetch_and_filter
python -m cfp.pollinators.cross_check

# 2. Build the iNaturalist image manifest from GBIF
python -m cfp.gbif.build_manifest

# 3. Validation pass (n=10 sample images, DINOv3 ViT-B/16, UMAP → RGB)
python -m cfp.dinov3.validate_sample --n 10 --backbone vitb16

# 4. Production streaming embedding pass
python -m cfp.pipeline.stream --backbone vitl16

# 5. Publish to Hugging Face
python -m cfp.hf.publish
```

## Citation (planned)

Legel, L., Keenan, T., et al. (2026). *California Flourishing & Pollination: a multi-modal AI dataset for ecological forecasting.* Hugging Face Dataset & companion manuscript.

## License

**MIT** for the entire repository — code, generated parquets, embeddings, manifests. The DINOv3 spatial features are transformative derivatives of the source photos (cannot be reversed to recover the image); the per-photo license string from iNaturalist is preserved in every manifest row so downstream consumers re-fetch photos under each photo's own terms.

## Contact

Lance Legel — `lance@ecological.dev` · [@deepearth on HF](https://huggingface.co/deepearth) · [github.com/legel/deepearth](https://github.com/legel/deepearth)
