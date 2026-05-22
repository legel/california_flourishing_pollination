"""California Flourishing & Pollination — Phase 1 pipeline package.

Subpackages:
    cnps         — California-native plant species list (iNat scheme #12, GBIF, CNPS RPI)
    globi        — GloBi interactions ingestion + plant×pollinator filtering
    pollinators  — flight-ability cross-check, CA-observation cross-check
    gbif         — GBIF / iNaturalist image manifest construction
    phenovision  — Lance's PhenoVision Python wrapper (vendored at vendor/phenovision)
    dinov3       — DINOv3 spatial-embedding extractor (ViT-B/16 validation, ViT-L/16 prod)
    pipeline     — async streaming GBIF → GPU → HF orchestrator
    hf           — Hugging Face dataset publishing
"""

__version__ = "0.1.0"
