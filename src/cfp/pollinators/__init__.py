"""Flying-pollinator cross-check.

Given GloBi-derived candidate pollinators of CA-native plants ([[cfp.globi]]),
gate by:

  (a) flight ability  — curated rule table at the order/family level
  (b) CA observation existence — GBIF iNaturalist Research-grade dataset

Output: ``data/processed/pollinators_california_flying.parquet``.
"""
