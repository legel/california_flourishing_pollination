"""GloBi (Global Biotic Interactions) ingestion + pollination filter.

The bulk interpreted TSV is downloaded once, filtered locally with DuckDB,
and joined against the CA-native plant list ([[cfp.cnps]]) to produce a
candidate-pollinator table.

Concept DOI: 10.5281/zenodo.3950589 (Poelen et al.) — version DOI captured at
fetch time and recorded in provenance/.
"""
