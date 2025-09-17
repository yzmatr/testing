# mlops/feature_engineering/registry_anchoring_strategies.py

from typing import Callable, Dict
import pandas as pd

ANCHORING_STRATEGY_REGISTRY: Dict[str, Callable] = {}

def register_anchoring_strategy(name: str, description: str = ""):
    def decorator(func: Callable):
        func.description = description
        ANCHORING_STRATEGY_REGISTRY[name] = func
        return func
    return decorator

@register_anchoring_strategy("first", "Use the first species in the list")
def anchor_first(row, species_frequency, target_species_set):
    return row["species_names"][0] if row["species_names"] else None

@register_anchoring_strategy("first-target", "Use the first target species in the list")
def anchor_first_target(row, species_frequency, target_species_set):
    return next((s for s in row["species_names"] if s in target_species_set), None)

@register_anchoring_strategy("most-frequent", "Use globally most frequent species")
def anchor_most_frequent(row, species_frequency, target_species_set):
    return max(row["species_names"], key=lambda s: species_frequency.get(s, 0), default=None)

@register_anchoring_strategy("most-frequent-target", "Most frequent target species")
def anchor_most_frequent_target(row, species_frequency, target_species_set):
    targets = [s for s in row["species_names"] if s in target_species_set]
    return max(targets, key=lambda s: species_frequency.get(s, 0), default=None)