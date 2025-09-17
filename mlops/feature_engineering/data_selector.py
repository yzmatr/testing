# mlops/feature_engineering/data_selector.py

import os
import ast
import sys
import pandas as pd
from typing import Optional, Dict, Callable, Tuple
from dateutil import parser
# Utilities
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.utils.config import FrogDataSelectorConfig
from mlops.reporting.output_generator import OutputGenerator

class FrogDataSelector:
    def __init__(self, config: FrogDataSelectorConfig):
        self.config = config
        
        # Define the paths for saving
        self.csv_path = config.csv_path

        # Different datasets used to store data along the processing journey
        self.df_raw = None              # This is the original dataset
        self.df_cleaned = None          # This is the filtered and cleaned data that can be sampled from

        # Final list of the target species: This is resolved based on the configuration
        self.target_species = []

        # Mapping of unique species in the dataset to contiguous integer class labels and vice versa
        self.species_to_class_labels = {}
        self.class_labels_to_species = {}

        # Resolve the target species
        self._resolve_target_species()

        # A store of the amount of data remaining after each step in the initial data filtering
        self.filter_log = []

        # A logger to be used throughout the class instead of print statements
        self.logger = setup_logger(self.__class__.__name__)

        # Final dataset column registry
        # Whenever a new column is transformed from the original data, it is appended to the registry
        self.column_registry = {
            'id': "Unique recording identifier",
            'state': "Australian state or territory",
            'call_audio': 'URL to download the call audio recording',
        }
    
    ###################################################################################
    # UTILITIES
    ###################################################################################

    def _add_to_column_registry(self, col_name: str, col_desc: str):
        self.column_registry[col_name] = col_desc

    def log_filter_step(self, description: str, df: pd.DataFrame):
        count = len(df)
        self.filter_log.append({"step": description, "count": count})
        self.logger.info(f"🔍 {description}: {count} rows remaining")
    
    def _resolve_target_species(self) -> list[str]:
        """
        Resolves and stores the final target species mapping
        """

        species_mapping = self.config.target_species_definition.target_species_mapping
        self.class_labels_to_species = species_mapping
        self.species_to_class_labels = {name: i for i, name in enumerate(species_mapping.values())}
        self.target_species = list(species_mapping.values())

        return self.class_labels_to_species
        
    ###################################################################################
    # CHECKS AND BALANCES
    ###################################################################################
    def load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"❌ File not found at: {self.csv_path}")
        # Load the data from the CSV file
        self.logger.info(f"✅ Found CSV at: {self.csv_path}.")
        self.df_raw = pd.read_csv(self.csv_path, low_memory=False)
        self.logger.info(f"✅ Loaded the dataframe with {len(self.df_raw)} records.")
        return self.df_raw
    
    def check_required_columns(self) -> pd.DataFrame:
        if self.df_raw is None:
            raise FileNotFoundError(f"❌ No dataframe loaded or dataframe is empty.")
        required_cols = [
            'id', 
            'call_audio', 
            'quality_call', 
            'state', 
            'capture_time',
            'validated_frog_ids',
            'validated_frog_names', 
            'filters_readable', 
            'is_duplicate',
            'inappropriate_content', 
            'has_people_activity',
            'validator_note_public', 
            'validator_note_private', 
            'validated_status'
        ]
        missing = [col for col in required_cols if col not in self.df_raw.columns]
        if missing:
            raise ValueError(f"❌ Missing columns in CSV: {missing}")
        self.df_raw = self.df_raw[required_cols]
        return self.df_raw
    
    def check_list_columns(self) -> pd.DataFrame:
        def is_list_string(s):
            return isinstance(s, str) and s.strip().startswith('[') and s.strip().endswith(']')
        for col in self.df_raw.columns:
            if self.df_raw[col].dropna().astype(str).head(10).apply(is_list_string).mean() >= 0.5:
                try:
                    self.df_raw[col] = self.df_raw[col].apply(lambda x: ast.literal_eval(x) if is_list_string(x) else x)
                    self.logger.info(f"🔄 Converted column to list: {col}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not convert column {col}: {e}")
        return self.df_raw
   
    ###################################################################################
    # PROCESS THE DATA
    ###################################################################################

    def filter_data(self, df) -> pd.DataFrame:
        # Get the configuration for filtering the initial data
        filtering_config = self.config.filtering_criteria

        self.log_filter_step("Initial dataset", df)
        
        # 1. Remove any data that is not published
        df = df[df['validated_status'] == 'published']
        self.log_filter_step("Remove all not published audio recordings.", df)

        # 2. Keep recordings that have an actual audio recording asosociated with it
        df = df[df['call_audio'].notnull() & (df['call_audio'].str.strip() != "")]
        self.log_filter_step("Removed missing / empty audio calls", df)

        # 3. Keep only recordings that have at least one frog labelled
        df = df[df['validated_frog_names'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        self.log_filter_step("Removed unlabelled frog calls", df)

        # 4. Do not allow duplicate recordings
        if not filtering_config.allow_duplicates:
            df = df[df['is_duplicate'] != True]
            self.log_filter_step("Removed duplicate frog calls", df)
        
        # 5. Do not allow inappropriate content
        if not filtering_config.allow_inappropriate:
            df = df[df['inappropriate_content'] != True]
            self.log_filter_step("Removed frog calls with inappropriate content", df)
        
        # 6. Do not allow allow people activity
        if not filtering_config.allow_people_activity:
            df = df[df['has_people_activity'] != True]
            self.log_filter_step("Removed frog calls with people activity", df)
        
        # 7. Remove poor quality samples
        if filtering_config.avoid_poor_quality:
            df = df[df['quality_call'] != False]
            self.log_filter_step("Removed poor quality frog calls", df)
        
        return df
    
    def transform_data(self, df: pd.DataFrame):
        # Log the beginning of the process
        self.logger.info(f"📝 Generating new columns to assist in data filtering...")

        # Clean up the initial data
        df['capture_time'] = df['capture_time'].apply(lambda x: parser.parse(x) if pd.notnull(x) else None)
        df['validated_frog_names'] = df['validated_frog_names'].apply(sorted)
        df['validated_frog_ids'] = df['validated_frog_ids'].apply(sorted)

        # ----------------------------------------------------------------------------
        # EXTRACT TEMPORAL FEATURES
        # ----------------------------------------------------------------------------
        
        # Extract the year from the recording capture time
        df['year'] = df["capture_time"].apply(lambda x: x.year if pd.notnull(x) else None)
        self._add_to_column_registry("year", "Year of the recording (int)")
        
        # Extract the month from the recording capture time
        df['month'] = df["capture_time"].apply(lambda x: x.month if pd.notnull(x) else None)
        self._add_to_column_registry("year", "Month of the recording (int: 1-12)")

        df['hour'] = df["capture_time"].apply(lambda x: x.hour if pd.notnull(x) else None)
        self._add_to_column_registry("hour", "Hour of the recording (local time: int 0-23)")
        
        # ----------------------------------------------------------------------------
        # EXTRACT SPECIES DATA
        # ----------------------------------------------------------------------------

        # Is there more than 1 species in the recording?
        df['is_multi_species'] = df['validated_frog_ids'].apply(lambda x: len(x) > 1)
        self._add_to_column_registry("is_multi_species", "True if more than one species is present")

        # How many species are there in the recording?
        df['species_count'] = df['validated_frog_ids'].apply(len)
        self._add_to_column_registry('species_count', "Total number of species in the recording")

        # Does the list of species include one of the target species?
        df['includes_target_species'] = df['validated_frog_names'].apply(lambda species: any(name in self.target_species for name in species))
        self._add_to_column_registry('includes_target_species', "True if any target species is in the recording")

        # How many target species exists in the list of species?
        df['target_species_count'] = df['validated_frog_names'].apply(lambda names: sum(1 for name in names if name in self.target_species))
        self._add_to_column_registry('target_species_count', "Number of target species in the recording")
        
        # What is the proportion of target species relative to all species in the list?
        df['target_species_ratio'] = df.apply(
            lambda row: row['target_species_count'] / row['species_count'] if row['species_count'] > 0 else 0.0,
            axis=1
        )
        self._add_to_column_registry('target_species_ratio', "Proportion of species that are target species")

        # Whether the list of species contains only target species, mixed (target + other), or only other
        df['species_type'] = df.apply(
            lambda row: (
                'Other' if row['target_species_count'] == 0 else
                'Target Only' if row['target_species_count'] == row['species_count'] else
                'Mixed'
            ),
            axis=1
        )
        self._add_to_column_registry('species_type', 'Does the label include the target species? "Target Only", "Mixed", "Other"')

        # Log the end of the process
        self.logger.info(f"✅ Created the necessary new columns.")
        return df
    
    def generate_label(self, df: pd.DataFrame, anchor_fn: Optional[Callable] = None) -> pd.DataFrame:
        """
        Generate class label annotations based on the global species-to-class-ID mapping:
        
        - class_labels: list[int], class IDs for all species in each recording
        - class_label_single: int | None, valid only if single species and 'Target Only' or 'Other'
        - class_label_vector: multi-hot vector, representing only target species (ordered by self.target_species)
        """
        if not self.species_to_class_labels or not self.target_species:
            raise RuntimeError("❌ Class mapping or target species not initialized. Did you forget to call _resolve_target_species()?")
        
        # ----------------------------------------------------------------------------
        # BASELINE LABEL COLUMNS
        # ----------------------------------------------------------------------------
        
        # Rename the validated_frog_names to species_names
        df['species_names'] = df['validated_frog_names']
        self._add_to_column_registry('species_names', "List of species names (list of strings)")

        # Rename the validated_frog_ids to species_ids
        df['species_ids'] = df['validated_frog_ids']
        self._add_to_column_registry('species_ids', "List of species IDs (list of integers)")

        # Map all species names to class labels
        df['class_labels'] = df['species_names'].apply(
            lambda names: [self.species_to_class_labels[name] for name in names if name in self.species_to_class_labels]
        )
        self._add_to_column_registry('class_labels', 'List of class labels (Based on class mapping)')

        # ----------------------------------------------------------------------------
        # EXTRACTING A SINGLE CLASS LABEL FROM THE LIST OF SPECIES
        # ----------------------------------------------------------------------------
        # Given that each recording can have many species in a list, we need to select
        # a single species from the list that will be used as our ground truth single
        # class label, even when there are multiple species present. To do this, we
        # have the concept of an anchoring function. If you do not want a species
        # labelled for multi-species recordings, simply keep the anchoring function
        # as None
        # ----------------------------------------------------------------------------
        target_species_set = set(self.target_species)
        species_flat = df['validated_frog_names'].explode()
        species_frequency = species_flat.value_counts().to_dict()

        def select_species_from_list(row) -> Tuple[Optional[str], Optional[int]]:
            species_list = row['species_names']

            # Select species from list
            if not species_list:
                selected_species = None
            elif len(species_list) == 1:
                selected_species = species_list[0]
            elif len(species_list) > 1 and anchor_fn:
                selected_species = anchor_fn(row, species_frequency, target_species_set)
            else:
                selected_species = None

            # Final label decision
            if selected_species in target_species_set:
                class_label = self.species_to_class_labels[selected_species]
            elif "Other" in self.config.target_species_definition.target_species_mapping.values():
                class_label = self.species_to_class_labels.get("Other", 0)
            else:
                class_label = None

            return selected_species, class_label

        df[["species_name_single", "class_label_single"]] = df.apply(
            select_species_from_list, axis=1, result_type="expand"
        )
        self._add_to_column_registry('class_label_single', 'Extracted single label from the class labels list (based on anchoring strategy)')
        self._add_to_column_registry('species_name_single', 'The original species name selected from the row before mapping to class label')

        df['class_name_single'] = df['class_label_single'].apply(lambda class_label: self.class_labels_to_species.get(class_label) if class_label is not None else None)
        self._add_to_column_registry('class_name_single', 'The species name corresponding to the class label that was generated')
        
        # ----------------------------------------------------------------------------
        # MULTI-HOT-ENCODED LABEL OF LENGTH TARGET SPECIES
        # ----------------------------------------------------------------------------
        # This has been created for situations where the experiment setup has changed
        # and rather than choosing to select a single species from the list, you can
        # specify that all the species are present in the recording through the multi-hot
        # encoded vector.
        # ----------------------------------------------------------------------------
        target_index = {name: i for i, name in enumerate(self.target_species)}
        vector_size = len(self.target_species)

        def make_label_vector(names: list[str]) -> list[int]:
            vec = [0] * vector_size
            for name in names:
                if name in target_index:
                    vec[target_index[name]] = 1
            return vec

        df['class_label_vector'] = df['species_names'].apply(make_label_vector)
        self._add_to_column_registry('class_label_vector', 'A multi-hot encoded vector representing the length target_species')

        self.logger.info("✅ Generated class_labels, class_label_single, and class_label_vector.")
        return df
    
    def get_cleaned_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        When returning the cleaned data, only return the dataframe with the columns that
        have been added to the registry. Most of the other columns are not needed for the
        purposes of modelling.
        """
        # Define the final cleaned data
        required_cols = list(self.column_registry.keys())
        df = df[required_cols]
        self.df_cleaned = df
        
        # Return the dataframe
        return df
    
    ###################################################################################
    # LOAD CLEANED DATA - THIS DATA IS USED TO SAMPLE FROM
    ###################################################################################
    
    def load_data(self, label_anchor_fn: Optional[Callable] = None) -> tuple[pd.DataFrame, Dict[int,str]]:
        # Load the CSV file
        self.load_csv()
        # Check that the required columns exist
        self.check_required_columns()
        # Make sure any columns that are actual lists but presented as strings, convert to lists
        self.check_list_columns()
        # If all these checks passed, the raw data should be ready for use
        if self.df_raw is None:
            raise ValueError(f"❌ There was an issue loading the raw data.")
        # Get a copy of the raw data and apply a series of transformations
        df = self.df_raw.copy()
        # # First, build the correct list of target species and their corresponding class label mappings
        # self._resolve_target_species()
        # Apply the filters and return the data
        df = self.filter_data(df)
        # Create new columns of data from the existing ones
        df = self.transform_data(df)
        # Generate the label to be used for machine learning tasks
        df = self.generate_label(df, anchor_fn=label_anchor_fn)
        # Save the cleaned data state
        df = self.get_cleaned_data(df)
        # Display a summary of the dataset
        OutputGenerator.dataset_description(
            column_descriptions=self.column_registry,
            print_output=True
        )
        # Display a summary of the data distribution
        OutputGenerator.grouped_distribution_report(
            title=f"Species Distribution (Cleaned Dataset)",
            df=df,
            group_by=["class_label_single", "class_name_single"],
            count_col_name="Total Samples", # Name of the count column
            sort_by="class_label_single",
            print_output=True
        )
        return df, self.class_labels_to_species