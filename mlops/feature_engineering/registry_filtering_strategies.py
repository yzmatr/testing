# mlops/feature_engineering/registry_filtering_strategies.py

from typing import Callable, Dict
import pandas as pd

FILTERING_STRATEGY_REGISTRY: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}

def register_filtering_strategy(name: str, description: str = ""):
    """Decorator to register a general filtering function."""
    def decorator(func: Callable[[pd.DataFrame], pd.DataFrame]):
        func.description = description
        FILTERING_STRATEGY_REGISTRY[name] = func
        return func
    return decorator

# ---------------------------------------------------------
# Define built-in strategies
# ---------------------------------------------------------

@register_filtering_strategy(
    name="single-species-only",
    description="Only include recordings with only ONE species present."
)
def filter_single_species(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["is_multi_species"]]

@register_filtering_strategy(
    name="multi-species-only",
    description="Only include recordings with greater than one species present."
)
def filter_multi_species(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_multi_species"]]

@register_filtering_strategy(
    name="other-species-only",
    description="Only include recordings of 'other species' (i.e. those not included in the target species list)."
)
def filter_other_species(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["includes_target_species"]]

@register_filtering_strategy(
    name="rare-species-only",
    description="Only include recordings with species occurring fewer than 100 times."
)
def filter_rare_species(df: pd.DataFrame) -> pd.DataFrame:
    species_counts = df["species_names"].explode().value_counts()
    rare_species = set(species_counts[species_counts < 100].index)
    return df[df["species_names"].apply(lambda lst: any(s in rare_species for s in lst))]

@register_filtering_strategy(
    name="multi-2-species-top3-targets",
    description="Multi-species recordings with exactly 2 species and at least one target species, where the anchored label is among the top 3 most frequent."
)
def filter_multispecies_top3_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Filter to multi-species recordings with 2 species and at least one target species
    df = df[
        (df["is_multi_species"]) &
        (df["species_count"] == 2) &
        (df["includes_target_species"])
    ].copy()

    # Step 2: Drop rows with missing class label
    df = df[df["class_label_single"].notnull()].copy()

    # Step 3: Compute top 3 target class labels by frequency
    top_targets = (
        df["class_label_single"]
        .value_counts()
        .nlargest(3)
        .index
        .tolist()
    )

    # Step 4: Keep only rows where the anchored class is in top targets
    return df[df["class_label_single"].isin(top_targets)]

@register_filtering_strategy(
    name="multi-3-species-top3-targets",
    description="Multi-species recordings with exactly 3 species and at least one target species, where the anchored label is among the top 3 most frequent."
)
def filter_multispecies_3species_top3_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Filter to multi-species recordings with 2 species and at least one target species
    df = df[
        (df["is_multi_species"]) &
        (df["species_count"] == 3) &
        (df["includes_target_species"])
    ].copy()

    # Step 2: Drop rows with missing class label
    df = df[df["class_label_single"].notnull()].copy()

    # Step 3: Compute top 3 target class labels by frequency
    top_targets = (
        df["class_label_single"]
        .value_counts()
        .nlargest(3)
        .index
        .tolist()
    )

    # Step 4: Keep only rows where the anchored class is in top targets
    return df[df["class_label_single"].isin(top_targets)]