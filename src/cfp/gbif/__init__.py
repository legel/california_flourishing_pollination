"""GBIF / iNaturalist image-manifest construction.

Build a complete manifest of iNaturalist Research-grade observation photos for
the union of (CA-native plants ∪ CA flying pollinators), filtered to
``country=US`` + ``stateProvince=California``. Each row is an *image*, not an
observation — multi-photo observations contribute one row per photo.

Output: ``data/processed/image_manifest.parquet`` (see schema in build_manifest.py).
"""
