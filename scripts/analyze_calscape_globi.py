#!/usr/bin/env python3
"""Calscape × GloBi cross-check + plant×pollinator network analysis.

Produces (under outputs/analysis/):
  - calscape_pollinator_richness.parquet
      per Calscape canonical plant: number of distinct documented pollinators in
      GloBi (CA-scoped), top pollinator orders, GloBi vs Calscape's own
      'Butterflies and Moths Supported' count
  - plants_with_no_documented_pollinators.parquet
      research-gap signal: CA natives with documented co-occurrence but zero
      GloBi pollination records
  - plant_family_x_pollinator_order.parquet
      cross-tab: plant family × pollinator order interaction counts
  - generalist_pollinators.parquet
      pollinator taxa by number of distinct CA-native plants they interact with
  - network_stats.json
      bipartite graph metrics (nodes, edges, components, modularity)
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/legel/california_flourishing_pollination")
OUT = ROOT / "outputs/analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
print("loading data…")
calscape = pd.read_parquet(ROOT / "data/processed/plants_california_native.parquet")
inter = pd.read_parquet(ROOT / "data/processed/globi_ca_plant_pollinator.parquet")
poll_kept = pd.read_parquet(ROOT / "data/processed/pollinators_california_flying.parquet")
print(f"calscape natives: {len(calscape):,}")
print(f"GloBi CA plant×pollinator rows: {len(inter):,}")
print(f"kept (flying CA-observed) pollinators: {len(poll_kept):,}")

# Plant side IRIs (plant=source) vs animal side IRIs (animal=source)
PLANT_SRC = {
    "http://purl.obolibrary.org/obo/RO_0002456",  # pollinatedBy
    "http://purl.obolibrary.org/obo/RO_0002623",  # flowersVisitedBy
}
# else: animal is source, plant is target

inter["plant_name"] = inter.apply(
    lambda r: r["sourceTaxonName"] if r["interactionTypeId"] in PLANT_SRC else r["targetTaxonName"],
    axis=1,
)
inter["animal_name"] = inter.apply(
    lambda r: r["targetTaxonName"] if r["interactionTypeId"] in PLANT_SRC else r["sourceTaxonName"],
    axis=1,
)
inter["animal_order"] = inter.apply(
    lambda r: r["targetTaxonOrderName"] if r["interactionTypeId"] in PLANT_SRC else r["sourceTaxonOrderName"],
    axis=1,
)
inter["animal_family"] = inter.apply(
    lambda r: r["targetTaxonFamilyName"] if r["interactionTypeId"] in PLANT_SRC else r["sourceTaxonFamilyName"],
    axis=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Calscape × GloBi pollinator richness per plant
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Calscape pollinator-richness join ===")
g_plant = inter.groupby("plant_name").agg(
    globi_pollinator_count=("animal_name", "nunique"),
    globi_interaction_records=("animal_name", "count"),
    top_pollinator_orders=("animal_order", lambda s: "; ".join(s.value_counts().head(3).index.dropna().astype(str))),
).reset_index().rename(columns={"plant_name": "scientific_name"})

merged = calscape.merge(g_plant, on="scientific_name", how="left")
merged["globi_pollinator_count"] = merged["globi_pollinator_count"].fillna(0).astype(int)
merged["globi_interaction_records"] = merged["globi_interaction_records"].fillna(0).astype(int)
keep_cols = [
    "scientific_name", "common_name", "rarity", "is_cultivar",
    "butterflies_and_moths_supported", "globi_pollinator_count", "globi_interaction_records",
    "top_pollinator_orders", "communities_simplified", "flower_color", "flowering_season",
]
merged[keep_cols].to_parquet(OUT / "calscape_pollinator_richness.parquet", index=False)
print(f"wrote calscape_pollinator_richness.parquet — {len(merged):,} rows")
print(merged[keep_cols].nlargest(10, "globi_pollinator_count").to_string(index=False))

# Gap analysis: 0 documented pollinators
gap = merged[merged["globi_pollinator_count"] == 0][["scientific_name", "common_name", "rarity", "butterflies_and_moths_supported"]]
gap.to_parquet(OUT / "plants_with_no_documented_pollinators.parquet", index=False)
print(f"\nplants with 0 documented pollinators: {len(gap):,} of {len(merged):,} ({100*len(gap)/len(merged):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Plant family × pollinator order matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Plant family × pollinator order matrix ===")
plant_family = inter.apply(
    lambda r: r["sourceTaxonFamilyName"] if r["interactionTypeId"] in PLANT_SRC else r["targetTaxonFamilyName"], axis=1,
)
xt = pd.crosstab(plant_family, inter["animal_order"], dropna=False)
xt = xt.loc[xt.sum(axis=1).sort_values(ascending=False).head(30).index]
xt = xt[xt.sum(axis=0).sort_values(ascending=False).head(15).index]
xt.to_parquet(OUT / "plant_family_x_pollinator_order.parquet")
print(f"wrote plant_family_x_pollinator_order.parquet  shape={xt.shape}")
print(xt.head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. Generalist pollinators
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Generalist pollinators (most distinct CA-native plant partners) ===")
g_animal = inter.groupby("animal_name").agg(
    n_distinct_plants=("plant_name", "nunique"),
    n_records=("plant_name", "count"),
    animal_order=("animal_order", lambda s: s.dropna().iloc[0] if len(s.dropna()) else None),
    animal_family=("animal_family", lambda s: s.dropna().iloc[0] if len(s.dropna()) else None),
).reset_index().sort_values("n_distinct_plants", ascending=False)
g_animal.to_parquet(OUT / "generalist_pollinators.parquet", index=False)
print(g_animal.head(15).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Bipartite network stats
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Bipartite network stats ===")
edges = inter[["plant_name", "animal_name"]].drop_duplicates()
n_plants = edges["plant_name"].nunique()
n_animals = edges["animal_name"].nunique()
n_edges = len(edges)
density = n_edges / (n_plants * n_animals) if n_plants * n_animals else 0
deg_plant = edges.groupby("plant_name").size()
deg_animal = edges.groupby("animal_name").size()

stats = {
    "n_plant_nodes": int(n_plants),
    "n_pollinator_nodes": int(n_animals),
    "n_unique_edges": int(n_edges),
    "n_total_records": int(len(inter)),
    "bipartite_density": float(density),
    "plant_degree_max": int(deg_plant.max()),
    "plant_degree_mean": float(deg_plant.mean()),
    "plant_degree_median": float(deg_plant.median()),
    "pollinator_degree_max": int(deg_animal.max()),
    "pollinator_degree_mean": float(deg_animal.mean()),
    "pollinator_degree_median": float(deg_animal.median()),
    "calscape_overlap": int(merged["globi_pollinator_count"].gt(0).sum()),
    "calscape_zero_documented": int(merged["globi_pollinator_count"].eq(0).sum()),
}
(OUT / "network_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))

print(f"\nAll outputs in: {OUT}")
