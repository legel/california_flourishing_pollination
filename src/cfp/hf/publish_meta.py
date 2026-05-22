"""Publish the non-embedding artifacts (species lists, interactions, provenance, README)."""

from __future__ import annotations

import json
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import HfApi, CommitOperationAdd
from rich.console import Console

app = typer.Typer(add_completion=False, help="HF metadata publisher.")
console = Console()


def _render_dataset_card(stats: dict) -> str:
    return textwrap.dedent(f"""\
    ---
    license: mit
    pretty_name: California Flourishing & Pollination
    tags:
      - ecology
      - biodiversity
      - phenology
      - pollination
      - california
      - dinov3
      - inaturalist
      - gbif
      - globi
      - cnps
      - calflora
      - deepearth
    task_categories:
      - image-classification
      - feature-extraction
    size_categories:
      - 1M<n<10M
    language:
      - en
    annotations_creators:
      - expert-generated
      - crowdsourced
    source_datasets:
      - inaturalist-research-grade
      - gbif
      - global-biotic-interactions
      - calflora
    ---

    # California Flourishing & Pollination

    [DeepEarth](https://github.com/legel/deepearth) × UC Berkeley QED Lab — a
    self-supervised spatial-feature dataset of every iNaturalist Research-grade
    observation of every **California-native plant** and every
    **California-observed flying pollinator**, encoded with **DINOv3 ViT-L/16**.

    Maintained by [Ecological Intelligence, Inc.](https://ecological.dev) (Lance Legel, PI)
    in collaboration with the [Quantitative Ecosystem Dynamics Lab](https://www.keenangroup.info/)
    at UC Berkeley (Trevor Keenan, PI). The data-collection / DINOv3-inference /
    publication pipeline (with full scientific provenance) lives in
    [`legel/california_flourishing_pollination`](https://github.com/legel/california_flourishing_pollination).
    The downstream DeepEarth models trained on this dataset live in
    [`legel/deepearth/models/flowering`](https://github.com/legel/deepearth/tree/main/models/flowering)
    and [`legel/deepearth/models/pollination`](https://github.com/legel/deepearth/tree/main/models/pollination).

    ## Why this dataset exists

    California is a global biodiversity hotspot whose native flora and pollinator
    fauna are under accelerating stress from climate change, land-use change, and
    invasive species. There is no single, openly-licensed, AI-ready dataset that
    binds plant *flowering* to its pollinator *visitation* at scale across the
    state. This dataset is a first step: it precomputes Meta AI's DINOv3 ViT-L/16
    spatial features for every Research-grade iNaturalist observation of every
    California-native plant and every flying pollinator with at least one
    California record, freezing those features as a reusable scientific asset.

    ## What it contains

    - `species/plants_california_native.parquet` — every CA-native plant taxon
      from the open Calflora / Jepson / GBIF stack (see `PROVENANCE.md` §1).
      Stats: **{stats.get('n_plants', '—')}** taxa.
    - `species/pollinators_california_flying.parquet` — every animal taxon that
      (a) is documented in GloBi as pollinating or visiting flowers of a CA
      native, (b) has ≥1 Research-grade iNaturalist observation in California,
      and (c) is gated by a curated flight-ability rule table. Stats:
      **{stats.get('n_pollinators', '—')}** taxa.
    - `interactions/globi_ca_plant_pollinator.parquet` — every GloBi pollination
      interaction (RO_0002455 / RO_0002456 / RO_0002622 / RO_0002623) between a
      CA-native plant and an animal, geographically scoped to California.
      Stats: **{stats.get('n_interactions', '—')}** rows.
    - `manifests/image_manifest.parquet` — image-grain manifest of every photo
      we embedded: `gbif_occurrence_id`, `inat_observation_id`, `taxon_name`,
      `gbif_taxon_key`, `image_url_large`, `license`, etc. **We do not
      redistribute the photos**; iNaturalist remains the source of record.
    - `embeddings/embeddings_*.parquet` — one row per image: CLS token +
      spatial patch tokens (`H_p × W_p × D` grid), in fp16. Decode with the
      `cls_shape`/`patches_shape` columns. Backbone:
      `facebook/dinov3-vitl16-pretrain-lvd1689m`. Stats:
      **{stats.get('n_embeddings', '—')}** images across
      **{stats.get('n_shards', '—')}** parquet shards.
    - `provenance/*.jsonl` — every API query, every file hash, every snapshot
      timestamp, every model checkpoint, copied verbatim from the pipeline run.
    - `PROVENANCE.md` — human-readable provenance document.

    ## How to use it

    ```python
    from datasets import load_dataset

    ds = load_dataset("deepearth/california-flourishing-pollination",
                      data_files="embeddings/embeddings_000000.parquet")
    import numpy as np
    row = ds["train"][0]
    cls = np.frombuffer(row["cls_fp16"], dtype=np.float16).reshape(row["cls_shape"])
    patches = np.frombuffer(row["patches_fp16"], dtype=np.float16).reshape(row["patches_shape"])
    ```

    The `taxon_name` column joins to `species/plants_california_native.parquet`
    or `species/pollinators_california_flying.parquet`; `gbif_taxon_key` joins
    to the GBIF backbone for higher classification.

    ## Licensing

    | Component | License |
    |---|---|
    | This dataset (DINOv3 embeddings + manifests + species lists + interaction tables + code) | **MIT** |
    | Source iNaturalist photos (we store the URL + license string only — never the photo bytes) | per-photo license recorded in every manifest row |
    | DINOv3 model weights | per Meta's DINOv3 license (gated on HF) |
    | PhenoVision model weights | MIT (Dinnage 2025) |
    | GloBi interaction data | CC0 (per concept DOI `10.5281/zenodo.3950589`) |

    DINOv3 spatial features are transformative derivatives of the source photos — the embeddings cannot be reversed to recover the image. Each row of the embedding shards carries the original `image_url_large` plus the per-photo `license` string from iNaturalist, so downstream consumers re-fetch photos under each photo's own terms.

    ## Citation

    ```bibtex
    @dataset{{legel_keenan_2026_cfp,
      title = {{California Flourishing \\& Pollination: a multi-modal AI dataset for ecological forecasting}},
      author = {{Legel, Lance and Keenan, Trevor}},
      year = {{2026}},
      publisher = {{Hugging Face}},
      url = {{https://huggingface.co/datasets/deepearth/california-flourishing-pollination}},
    }}
    ```

    Also cite the upstream sources (PhenoVision Dinnage 2025; DINOv3 Meta 2024;
    GloBi Poelen 2014; iNaturalist; GBIF; Calflora).

    ## Contact

    Lance Legel — `lance@ecological.dev` · [@deepearth on HF](https://huggingface.co/deepearth)

    _Generated {datetime.now(timezone.utc).isoformat()}_.
    """)


@app.callback(invoke_without_command=True)
def main(
    repo: str = typer.Option("deepearth/california-flourishing-pollination", "--repo"),
    repo_type: str = typer.Option("dataset", "--repo-type"),
    project_root: Path = typer.Option(Path("."), "--project-root"),
    plants_parquet: Path = typer.Option(Path("data/processed/plants_california_native.parquet"), "--plants"),
    pollinators_parquet: Path = typer.Option(Path("data/processed/pollinators_california_flying.parquet"), "--pollinators"),
    interactions_parquet: Path = typer.Option(Path("data/processed/globi_ca_plant_pollinator.parquet"), "--interactions"),
    manifest_parquet: Path = typer.Option(Path("data/processed/image_manifest.parquet"), "--manifest"),
    provenance_dir: Path = typer.Option(Path("provenance"), "--provenance-dir"),
    provenance_md: Path = typer.Option(Path("PROVENANCE.md"), "--provenance-md"),
) -> None:
    """Push species lists + interactions + manifest + provenance + README to the dataset repo."""
    api = HfApi()
    me = api.whoami()
    console.print(f"HF user: [bold]{me.get('name')}[/]")

    # Best-effort stats for the README.
    stats = {}
    try:
        import pandas as pd
        if plants_parquet.exists():
            stats["n_plants"] = len(pd.read_parquet(plants_parquet, columns=["scientific_name"]))
        if pollinators_parquet.exists():
            stats["n_pollinators"] = len(pd.read_parquet(pollinators_parquet, columns=["animal_name"]))
        if interactions_parquet.exists():
            stats["n_interactions"] = len(pd.read_parquet(interactions_parquet, columns=["interactionTypeId"]))
        if manifest_parquet.exists():
            stats["n_embeddings"] = len(pd.read_parquet(manifest_parquet, columns=["gbif_occurrence_id"]))
    except Exception as e:
        console.print(f"[yellow]stats fill-in failed: {e}[/]")

    ops: list[CommitOperationAdd] = []

    def add(local: Path, remote: str) -> None:
        if not local.exists():
            console.print(f"[yellow]skipping missing {local}[/]")
            return
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))

    add(plants_parquet, "species/plants_california_native.parquet")
    add(pollinators_parquet, "species/pollinators_california_flying.parquet")
    add(interactions_parquet, "interactions/globi_ca_plant_pollinator.parquet")
    add(manifest_parquet, "manifests/image_manifest.parquet")

    # Provenance JSONLs (skip the big embedding shards' provenance — those go via the upload stage).
    if provenance_dir.exists():
        for f in sorted(provenance_dir.glob("*.jsonl")):
            ops.append(CommitOperationAdd(path_in_repo=f"provenance/{f.name}", path_or_fileobj=str(f)))

    if provenance_md.exists():
        ops.append(CommitOperationAdd(path_in_repo="PROVENANCE.md", path_or_fileobj=str(provenance_md)))

    # Write README from template + push.
    readme_text = _render_dataset_card(stats)
    tmp = Path("/tmp/_cfp_README.md")
    tmp.write_text(readme_text)
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(tmp)))

    if not ops:
        raise typer.Exit("nothing to upload")
    msg = f"meta: publish species lists, interactions, manifest, provenance, README ({datetime.now(timezone.utc).date().isoformat()})"
    console.print(f"committing {len(ops)} files to {repo} ({repo_type})…")
    info = api.create_commit(repo_id=repo, repo_type=repo_type, operations=ops, commit_message=msg)
    console.print(f"[bold green]done[/] {info}")
