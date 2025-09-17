# mlops/feature_engineering/data_sampler.py

import os
import ast
import pandas as pd
from sklearn.utils import resample
from typing import Literal, Optional, Dict, Callable, List, Literal
from dateutil import parser
# Utilities
import mlflow
import sys
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.reporting.output_generator import OutputGenerator
from mlops.utils.config import FrogDataSamplerConfig
from mlops.utils.config import ModellingSamplingStrategyConfig
from mlops.feature_engineering.registry_filtering_strategies import FILTERING_STRATEGY_REGISTRY

class FrogDataSampler:
    def __init__(self, config: FrogDataSamplerConfig):
        # A logger to be used throughout the class instead of print statements
        self.config = config
        self.random_state = config.random_state
        self.logger = setup_logger(self.__class__.__name__)
    
    ###################################################################################
    # SAMPLING THE DATA
    ###################################################################################

    def sample_target_species(
        self,
        df: pd.DataFrame,
        strategy: Literal['downsample', 'upsample', 'none'],
        max_samples_per_class: Optional[int],
    ) -> pd.DataFrame:
        """
        Samples target species using anchored class labels.

        Args:
            df: DataFrame with 'class_label_single' column.
            strategy: 'downsample', 'upsample', or 'none'.
            max_samples_per_class: Max per target class (optional).

        Returns:
            Sampled DataFrame containing only the target species recordings.
        """

        # ----------------------------------------------------------------------------
        # Filter to include only rows where:
        # - a single anchored label was assigned (`notnull`)
        # - the label is not 0 (which is reserved for 'Other')
        # ----------------------------------------------------------------------------
        df_target = df[df["species_type"] == 'Target Only'].copy()

        # ----------------------------------------------------------------------------
        # Group by class_label_single (i.e., anchored target class)
        # Each group corresponds to one frog species
        # ----------------------------------------------------------------------------
        grouped = df_target.groupby("class_label_single")

        sampled_groups = []

        # ----------------------------------------------------------------------------
        # Apply the selected sampling strategy to each class group
        # ----------------------------------------------------------------------------
        for label, group in grouped:
            n = len(group)

            if strategy == 'downsample':
                # Sample at most max_samples_per_class (or the group size if smaller)
                n_samples = min(n, max_samples_per_class or n)
                sampled = group.sample(n=n_samples, random_state=self.random_state)

            elif strategy == 'upsample':
                # Sample with replacement up to max_samples_per_class (or at least the group size)
                n_samples = max(n, max_samples_per_class or n)
                sampled = resample(group, replace=True, n_samples=n_samples, random_state=self.random_state)

            else:  # strategy == 'none'
                # No sampling; just keep all the examples for this label
                sampled = group

            sampled_groups.append(sampled)

        # ----------------------------------------------------------------------------
        # Combine all sampled class groups into a single DataFrame and shuffle rows
        # ----------------------------------------------------------------------------
        df_sampled_target = pd.concat(sampled_groups).reset_index(drop=True)

        return df_sampled_target

    def sample_other_species(
        self,
        df: pd.DataFrame,
        strategy: Literal['random', 'stratify'],
        sample_size: Optional[int],
        exclude_below_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Samples the 'Other' class species using specified strategy.

        Args:
            df: Full dataset with 'class_label_single' and 'species_names' columns.
            strategy: Sampling method: 'random' or 'stratify'.
            sample_size: Total number of 'Other' samples to draw.
            exclude_below_count: Exclude recordings where all species are below this count.

        Returns:
            A DataFrame of sampled 'Other' class data.
        """

        # ----------------------------------------------------------------------------
        # Step 1: Filter to just the "Other" class (label == 0)
        # ----------------------------------------------------------------------------
        df_other = df[df["species_type"] == "Other"].copy()

        if df_other.empty or sample_size == 0:
            return pd.DataFrame(columns=df.columns)

        # ----------------------------------------------------------------------------
        # Step 2: Optionally exclude rare species (below `exclude_below_count`)
        # A row is excluded if *none* of its species meet the count threshold
        # ----------------------------------------------------------------------------

        if exclude_below_count:
            species_counts = df_other['species_names'].explode().value_counts()
            valid_species = set(species_counts[species_counts >= exclude_below_count].index)

            def has_valid_species(species_list):
                return any(name in valid_species for name in species_list)

            df_other = df_other[df_other['species_names'].apply(has_valid_species)].copy()

        if df_other.empty:
            return pd.DataFrame(columns=df.columns)

        # ----------------------------------------------------------------------------
        # Step 3: Apply the sampling strategy
        # ----------------------------------------------------------------------------
        if strategy == 'stratify':
            # Convert species list to tuple for hashing
            df_other['species_names_tuple'] = df_other['species_names'].apply(tuple)
            grouped = df_other.groupby('species_names_tuple')

            if grouped.ngroups == 0 or df_other.empty:
                self.logger.warning("⚠️ No valid groups found for stratified sampling. Falling back to random sampling.")
                strategy = 'random'  # fallthrough to random
            else:
                per_group = max(1, sample_size // grouped.ngroups)
                sampled_groups = []

                for _, group in grouped:
                    if len(group) == 0:
                        continue
                    n_sample = min(len(group), per_group)
                    sampled = group.sample(n=n_sample, random_state=self.random_state)
                    sampled_groups.append(sampled)

                stratified = pd.concat(sampled_groups).reset_index(drop=True)

                # Phase 2: Top-up
                remaining_n = sample_size - len(stratified)
                remaining_pool = df_other[~df_other['id'].isin(stratified['id'])]

                top_up = pd.DataFrame()
                if remaining_n > 0 and not remaining_pool.empty:
                    top_up = remaining_pool.sample(n=min(remaining_n, len(remaining_pool)), random_state=self.random_state)

                sampled = pd.concat([stratified, top_up]).reset_index(drop=True)
                df_other.drop(columns='species_names_tuple', errors='ignore', inplace=True)

        else:
            # Fallback or explicit 'random' strategy
            sampled = df_other.sample(n=min(sample_size, len(df_other)), random_state=self.random_state)

        return sampled

    def sample_modelling_dataset(
        self,
        df_data: pd.DataFrame,
        modelling_strategy: ModellingSamplingStrategyConfig
    ) -> pd.DataFrame:
        """
        Generates the modelling dataset by sampling from target and optional 'Other' class
        based on the configured ModellingSamplingStrategyConfig.
        """
        # ------------------------------------------------------------------------
        # Load cleaned dataset (if not already loaded)
        # ------------------------------------------------------------------------
        df = df_data.copy()

        # ------------------------------------------------------------------------
        # Apply filtering strategy
        # ------------------------------------------------------------------------
        INCLUDE_OTHER_CLASS = modelling_strategy.include_other_species
        SPECIES_DATA_TO_USE = modelling_strategy.species_data_to_use

        if SPECIES_DATA_TO_USE in FILTERING_STRATEGY_REGISTRY:
            filtering_strategy_fn = FILTERING_STRATEGY_REGISTRY[SPECIES_DATA_TO_USE]
            if filtering_strategy_fn:
                df = filtering_strategy_fn(df)
                self.logger.info(f"✅ Filtered data to {SPECIES_DATA_TO_USE} → {len(df)} rows")     

        # ------------------------------------------------------------------------
        # STORE OF DATAFRAMES THAT HAVE BEEN SAMPLED
        # ------------------------------------------------------------------------
        dfs_sampled = []
        
        # ------------------------------------------------------------------------
        # Sample target classes
        # ------------------------------------------------------------------------
        df_sampled_target = self.sample_target_species(
            df=df,
            strategy=modelling_strategy.target_species_sampling_strategy,
            max_samples_per_class=modelling_strategy.target_species_max_samples_per_class,
        )

        dfs_sampled.append(df_sampled_target)

        # ------------------------------------------------------------------------
        # Optionally sample 'Other' class
        # ------------------------------------------------------------------------
        if INCLUDE_OTHER_CLASS:
            sample_size = round(modelling_strategy.target_species_max_samples_per_class * modelling_strategy.other_species_boost_factor)
            df_sampled_other = self.sample_other_species(
                df=df,
                strategy=modelling_strategy.other_species_sampling_strategy,
                sample_size=sample_size,
                exclude_below_count=modelling_strategy.exclude_other_species_below_count,
            )
            dfs_sampled.append(df_sampled_other)
            self.logger.info(f"🧩 Included {len(df_sampled_other)} 'Other' class samples")
        else:
            self.logger.info("🚫 Skipping 'Other' class from modelling dataset")

        # ------------------------------------------------------------------------
        # Combine and shuffle modelling dataset
        # ------------------------------------------------------------------------
        # Before concatenating, filter out empty DataFrames from the list:
        dfs = [df for df in dfs_sampled if not df.empty]
        df_modelling = pd.concat(dfs).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # View the distribution of data that will be used for modelling
        OutputGenerator.grouped_distribution_report(
            title="Species Distribution (Modelling Dataset)",
            df=df_modelling,
            group_by=["class_label_single", "class_name_single"],
            count_col_name="Total Samples", # Name of the count column
            sort_by="class_label_single",
            print_output=True
        )

        self.logger.info(f"✅ Final modelling dataset: {len(df_modelling)} samples, {df_modelling['class_label_single'].nunique()} unique labels")

        return df_modelling
    
    def load_saved_modelling_data(self, run_id: str, subset: Optional[Literal["train", "val", "test"]] = None) -> List[int]:
        self.logger.info(f"📦 Loading modelling IDs from MLflow run: {run_id}")

        ARTIFACT_FILE_PATH = "modelling_ids.csv"
        
        # Download the artifact locally
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ARTIFACT_FILE_PATH)

        # Load the CSV and filter to test split
        df_ids = pd.read_csv(local_path)

        id_list = df_ids['id'].tolist()

        if subset:
            id_list = df_ids[df_ids["split"] == subset]["id"].tolist()

        self.logger.info(f"✅ Loaded {len(id_list)} {subset if subset else 'modelling'} IDs from artifact → {local_path}")

        return id_list

    
    def sample_test_data(
        self,
        df: pd.DataFrame,
        run_id: str
    ) -> pd.DataFrame:
        """
        Loads the test split IDs from the modelling run artifacts and returns
        the corresponding subset of rows from the provided full dataset.

        Args:
            df (pd.DataFrame): Full cleaned dataset (with all IDs).
            run_id (str): MLflow run ID where modelling_ids.csv was logged.
            artifact_path (str): Relative path to the saved IDs file (default: modelling_ids.csv).

        Returns:
            pd.DataFrame: Subset of `df` containing only the test split.
        """

        # Return filtered test subset
        test_ids = self.load_saved_modelling_data(run_id, subset="test")
        df_test = df[df["id"].isin(test_ids)].copy()
        self.logger.info(f"🔍 Matched {len(df_test)} test samples from full dataset")

        return df_test

    def sample_hold_out_data(
        self,
        run_id: str,
        df_cleaned: pd.DataFrame,
        filtering_strategy_fn: Callable[[pd.DataFrame], pd.DataFrame] = None, 
        max_samples_per_class: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Samples hold-out data from the cleaned dataset that was not used in the modelling dataset
        associated with a previous MLflow run.

        Args:
            df_cleaned (pd.DataFrame): Full cleaned dataset (all candidate recordings).
            run_id (str): MLflow run ID where `modelling_ids.csv` was logged.
            filtering_strategy_fn (Callable): Optional function to further filter the data.
            max_samples_per_class (int, optional): Cap number of samples per class label.

        Returns:
            pd.DataFrame: Filtered hold-out dataset.
        """

        # ------------------------------------------------------------------------
        # Retrieve modelling and cleaned datasets
        # ------------------------------------------------------------------------
        random_state = self.random_state
        
        # Return filtered test subset
        id_list = self.load_saved_modelling_data(run_id)

        # Exclude used modelling IDs from full dataset
        df_remaining = df_cleaned[~df_cleaned["id"].isin(id_list)].copy()
        self.logger.info(f"🧹 Excluded {len(id_list)} modelling samples → {len(df_remaining)} rows remaining")

        # ------------------------------------------------------------------------
        # Apply filter strategy
        # - Either dispatch to a named function via the registry
        # - Or use a user-supplied custom filtering function
        # ------------------------------------------------------------------------
        if filtering_strategy_fn:
            df_remaining = filtering_strategy_fn(df_remaining)
            self.logger.info(f"✅ Applied custom holdout strategy → {len(df_remaining)} rows remaining.")

        # ------------------------------------------------------------------------
        # Optionally cap the number of samples per class
        # - This uses 'class_label_single' as the grouping column
        # ------------------------------------------------------------------------
        if max_samples_per_class is not None and max_samples_per_class > 0:
            if "class_label_single" not in df_remaining.columns:
                raise ValueError("Missing 'class_label_single' column for class-capped sampling.")

            def cap_group(g):
                return g.sample(n=min(len(g), max_samples_per_class), random_state=random_state)

            df_remaining = df_remaining.copy()
            df_remaining["class_label_single_cp"] = df_remaining["class_label_single"]

            df_remaining = (
                df_remaining
                .groupby("class_label_single_cp", group_keys=False)
                .apply(cap_group, include_groups=False)
                .reset_index(drop=True)
            )

            self.logger.info(f"🔍 Capped to {max_samples_per_class} per class → {len(df_remaining)} rows")
        
        # View the distribution of data that will be used for modelling
        OutputGenerator.grouped_distribution_report(
            title="Species Distribution (Holdout Dataset)",
            df=df_remaining,
            group_by=["class_label_single", "class_name_single"],
            count_col_name="Total Samples", # Name of the count column
            sort_by="class_label_single",
            print_output=True
        )

        self.logger.info(f"✅ Final modelling dataset: {len(df_remaining)} samples, {df_remaining['class_label_single'].nunique()} unique labels")

        return df_remaining