import os
import ast
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple

# Clean any columns that are list of strings
def is_list_string(s):
    return isinstance(s, str) and s.strip().startswith('[') and s.strip().endswith(']')

# Calculate a percentage between two numbers
def calc_percentage(numerator, denominator):
    percentage = numerator / denominator * 100
    percentage = round(percentage, 1)
    return percentage

def read_data(project_root: str):
    # Load the data
    df_raw = pd.read_csv(os.path.join(project_root, "data", "raw", "20250317.csv"), low_memory=False)
    
    # Convert the columns that are lists encoded as strings, back to actual strings
    for col in df_raw.columns:
        if df_raw[col].dropna().astype(str).head(10).apply(is_list_string).mean() >= 0.5:
            try:
                df_raw[col] = df_raw[col].apply(lambda x: ast.literal_eval(x) if is_list_string(x) else x)
            except Exception as e:
                print(f"Error converting column {col}: {e}")
    # Return the data
    return df_raw

def filter_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, int]]]:
    # Track count after each filter
    filters = []
    filters.append(("raw_data count", len(df)))
    df = df[df['validated_frog_names'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    filters.append(("validated_frog_names has at least 1 frog", len(df)))
    df = df[df['validated_status'] == 'published']
    filters.append(("validated_status == published", len(df)))
    df = df[df['call_audio'].notnull() & (df['call_audio'].str.strip() != "")]
    filters.append(("call_audio is not null or empty", len(df)))
    df = df[df['is_duplicate'] != True]
    filters.append(("is_duplicate != True", len(df)))
    df = df[df['inappropriate_content'] != True]
    filters.append(("inappropriate_content != True", len(df)))
    df = df[df['has_people_activity'] != True]
    filters.append(("has_people_activity != True", len(df)))
    df = df[df['quality_call'] != False]
    filters.append(("quality_call != False", len(df)))
    return df, filters

def preprocess_data(df: pd.DataFrame, target_species: list[str]):
    # Add some new columns
    df['is_multi_species'] = df['validated_frog_ids'].apply(lambda x: len(x) > 1)
    df['species_count'] = df['validated_frog_ids'].apply(lambda x: len(x))
    df['first_species_name'] = df['validated_frog_names'].apply(lambda x: x[0])
    df['first_species_id'] = df['validated_frog_ids'].apply(lambda x: x[0])
    df['sorted_species_names'] = df['validated_frog_names'].apply(lambda x: sorted(x))
    df['sorted_species_ids'] = df['validated_frog_ids'].apply(lambda x: sorted(x))
    df['includes_target_species'] = df['validated_frog_names'].apply(lambda species_list: any(name in target_species for name in species_list))
    df['capture_time'] = pd.to_datetime(df['capture_time'], errors='coerce', utc=False)
    # Return the fields needed for the dashboard
    df = df[[
        'id',
        'first_species_name',
        'first_species_id',
        'sorted_species_names', 
        'sorted_species_ids',
        'includes_target_species',
        'species_count', 
        'is_multi_species', 
        'state',
        'capture_time'
    ]]
    return df

def get_data(project_root: str, target_species: list[str]):
    # Load the data
    df_raw = read_data(project_root)
    
    # Apply the baseline filters
    df = df_raw.copy()
    df, filters = filter_data(df)
    df = preprocess_data(df, target_species)
    
    # Filter single species data
    df_single_species = df[~df['is_multi_species']]
    df_single_target = df_single_species[df_single_species['includes_target_species']]
    df_single_other = df_single_species[~df_single_species['includes_target_species']]

    # Filter to multi species data
    df_multi_species = df[df['is_multi_species']]
    df_multi_target = df_multi_species[df_multi_species['includes_target_species']]
    df_multi_other = df_multi_species[~df_multi_species['includes_target_species']]

    species_occurrences = df['sorted_species_names'].explode().value_counts()
    target_occurrences = species_occurrences[species_occurrences.index.isin(target_species)]

    # Generate a summary dictionary
    summary = {
        'raw': {
            'filter_process': filters,
            'count_total': len(df_raw),
            'num_species': df_raw['validated_frog_names'].explode().nunique(),
        },
        'filtered': {
            'count_total': len(df),
            'percent_filtered:raw': round(len(df) / len(df_raw) * 100),
            'num_species': df['sorted_species_names'].explode().nunique(),
            'count_species_occurrences': species_occurrences.sum(),
            'count_target_occurrences': target_occurrences.sum(),
            'percent_target:species': round(target_occurrences.sum() / species_occurrences.sum() * 100)
        },
        'single': {
            # Calculate counts
            'count_total': len(df_single_species),
            'count_target': len(df_single_target),
            'count_other': len(df_single_other),
            # Calculate percentages
            'percent_single:total': round(len(df_single_species) / len(df) * 100),
            'percent_target:total': round(len(df_single_target) / len(df) * 100),
            'percent_other:total': round(len(df_single_other) / len(df) * 100),
            'percent_target:single': round(len(df_single_target) / len(df_single_species) * 100),
            'percent_other:single': round(len(df_single_other) / len(df_single_species) * 100),
            # Calculate the number of species
            'num_species': df_single_species['first_species_name'].nunique(),
            'num_target_species': df_single_target['first_species_name'].nunique(),
            'num_other_species': df_single_other['first_species_name'].nunique()
        },
        'multi': {
            # Calculate counts
            'count_total': len(df_multi_species),
            'count_target': len(df_multi_target),
            'count_other': len(df_multi_other),
            # Calculate percentages
            'percent_multi:total': round(len(df_multi_species) / len(df) * 100),
            'percent_target:total': round(len(df_multi_target) / len(df) * 100),
            'percent_other:total': round(len(df_multi_other) / len(df) * 100),
            'percent_target:multi': round(len(df_multi_target) / len(df_multi_species) * 100),
            'percent_other:multi': round(len(df_multi_other) / len(df_multi_species) * 100),
            # Calculate the number of species
            'num_species': df_multi_species['sorted_species_names'].explode().nunique(),
            'num_target_species': df_multi_target['sorted_species_names'].explode().nunique(),
            'num_other_species': df_multi_other['sorted_species_names'].explode().nunique()
        },
    }

    # Return the data
    return df, summary

@st.cache_data(show_spinner="Loading data...")
def get_app_data(project_root: str, target_species: list[str]):
    """
    Note: This function is identical to the get data function, except
    we use this to cache_data in app, but we can still call the above
    function if we need it in other settings.
    """
    df, df_summary = get_data(project_root, target_species)
    return df, df_summary


def get_occurrence_distribution(df: pd.DataFrame, target_species: List[str] = None):
    # Get the counts of the data
    df_species_counts = df['sorted_species_names'].explode().value_counts()
    total_occurrences = df_species_counts.sum()

    if target_species:
        df_species_counts = df_species_counts[df_species_counts.index.isin(target_species)]

    # Get the percentage of the total for each species
    df_species_counts = df_species_counts.reset_index()
    df_species_counts.columns = ['Species', 'Count']
    df_species_counts['Percentage'] = round(df_species_counts['Count'] / total_occurrences * 100)

    return df_species_counts

def get_target_distribution_by_state(df: pd.DataFrame, target_species: List[str]):
    rows = []
    df_target_species = df[df['includes_target_species']]

    for species in target_species:
        df_filtered = df_target_species[
            df_target_species['sorted_species_names'].apply(lambda lst: species in lst)
        ]
        
        counts = df_filtered['state'].value_counts()
        
        df_counts = counts.reset_index()
        df_counts.columns = ['State', 'Count']
        df_counts['Species'] = species
        
        rows.append(df_counts)

    df_species_by_state = pd.concat(rows, ignore_index=True)

    return df_species_by_state
    
def get_combination_pairs(df: pd.DataFrame, target_species: List[str]):
    # Count the number of species per recording
    df_species = df.copy()

    df_two_species = df_species[df_species['species_count'] == 2]

    combinations_data = []
        
    for idx, species_list in enumerate(df_two_species['sorted_species_names']):
        combo = tuple(sorted(species_list))
        target_species_in_combo = set(species_list) & set(target_species)
        combination_type = 'target_target' if len(target_species_in_combo) == 2 else 'target_other' if len(target_species_in_combo) == 1 else 'other_other'
        
        combinations_data.append({
            'combination': combo,
            'species_1': combo[0],
            'species_2': combo[1],
            'target_count': len(target_species_in_combo),
            'target_species': sorted(list(target_species_in_combo)),
            'has_target': len(target_species_in_combo) > 0,
            'combination_type': combination_type
        })

    df_pairs = pd.DataFrame(combinations_data)
    return df_pairs
    

