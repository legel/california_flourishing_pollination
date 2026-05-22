"""Hugging Face dataset publishing.

The streaming pipeline's `upload` stage handles per-embedding-shard pushes
([[cfp.pipeline.upload]]). This subpackage handles the ONE-SHOT publishing
of the dataset's metadata layer:

  - README.md (dataset card) generated from a Jinja template
  - PROVENANCE.md copied verbatim
  - plant + pollinator species lists (parquet)
  - GloBi-filtered interactions (parquet)
  - per-stage provenance JSONL (so every query is auditable)
  - the manifest of all embeddings

Repo: deepearth/california-flourishing-pollination
"""
