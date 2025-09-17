# mlops/utils/pipeline.py
from dacite import from_dict
from typing import Dict, Any
from dataclasses import replace, is_dataclass, asdict
from pathlib import Path
import mlflow
from mlflow.exceptions import MlflowException
import yaml
from tabulate import tabulate
from dataclasses import dataclass
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")

from mlops.utils.config import (
    FrogDataSelectorConfig, 
    SelectorFilteringCriteria, 
    SelectorTargetSpeciesDefinition,
    FrogDataSamplerConfig,
    FrogAudioDownloaderConfig,
    ModellingConfig,
    ModellingSamplingStrategyConfig,
    ModellingSplitDataConfig,
    ModellingTrainingConfig,
    AudioLoaderConfig,
    AudioFilterConfig,
    AudioChunkerConfig,
    BirdNETExtractorConfig,
    LogMelExtractorConfig,
    MFCCExtractorConfig,
    FeatureNormalizerConfig,
    PreprocessorOrchestratorConfig,
    PreprocessorConfig,
    PipelineConfig,
    FeatureDataSchema,
    EvaluationConfig,
    MLFlowConfig,
    DatabricksDataConfig

)

from mlops.feature_engineering.data_selector import FrogDataSelector
from mlops.feature_engineering.data_downloader import FrogAudioDownloader
from mlops.feature_engineering.data_sampler import FrogDataSampler
from mlops.feature_engineering.preprocessor_orchestrator import PreprocessorOrchestrator
from mlops.feature_engineering.preprocessor_audio_databricks import FrogEmbedDatabricks
from mlops.training.tf_model_trainer import TFModelTrainer
from mlops.evaluation.tf_model_evaluator import TFModelEvaluator


def flatten_config_dict(d, parent_key='', sep='.') -> dict:
    """
    Recursively flattens a nested dictionary using dot notation.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def validate_dot_key_exists(config: Any, dot_key: str) -> bool:
    """
    Validates that a dot-notated key exists in a nested dataclass.
    
    Raises:
        AttributeError if the dot key is invalid.
    """
    parts = dot_key.split(".")
    current = config

    for part in parts:
        if not is_dataclass(current):
            raise AttributeError(f"'{type(current).__name__}' is not a dataclass when accessing '{part}' in '{dot_key}'")
        if not hasattr(current, part):
            raise AttributeError(f"'{type(current).__name__}' has no attribute '{part}' (from key: '{dot_key}')")
        current = getattr(current, part)

    return True

def _deep_replace(obj, dot_key: str, value: Any):
    """
    Recursively replace an attribute in a nested dataclass using dot notation.
    """
    parts = dot_key.split(".")
    if len(parts) == 1:
        return replace(obj, **{parts[0]: value})
    else:
        attr = getattr(obj, parts[0])
        updated_attr = _deep_replace(attr, ".".join(parts[1:]), value)
        return replace(obj, **{parts[0]: updated_attr})

# Print config summary
def print_config_summary_table(config):
    flat_config = flatten_config_dict(asdict(config))
    config_summary_table = tabulate(flat_config.items(), headers=["Config Key", "Value"], tablefmt="psql")

    print(f"\n{'='*70}")
    print("📋 Pipeline Configuration Summary:")
    print(f"{'='*70}\n")
    print(config_summary_table)

def generate_pipeline_config(
    experiment: dict,
    run_id: str,
    overrides: Dict[str, Any] = None,
    force_save: bool = False,
    random_seed: int = 42,
    audio_format: str = "wav",
    sampling_rate: int = 48000
) -> PipelineConfig:
    """
    Returns a fully populated default PipelineConfig based on experiment metadata.
    """
    overrides = overrides or {}
    CONFIG_ARTIFACT_PATH = "config/pipeline_config.yaml"
    if run_id:
        try:
            if not force_save:
                # Check for the existig config artifact
                print(f"🔍 Checking for {CONFIG_ARTIFACT_PATH} in run_id={run_id}...")

                # Download the YAML file
                local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=CONFIG_ARTIFACT_PATH)
                print(f"✅ Found existing config artifact — loading from: {local_path}")

                # Load and parse YAML
                with open(local_path, "r") as f:
                    config_dict = yaml.safe_load(f)

                config = from_dict(data_class=PipelineConfig, data=config_dict)
                
                print_config_summary_table(config)
                
                return config
            else:
                print(f"♻️ Override enabled — will regenerate and replace config for run_id={run_id}")
        except (MlflowException, FileNotFoundError, OSError):
            print(f"📭 No existing config found in run {run_id} — will generate a new one.")

    RANDOM_SEED = random_seed               # Random seed used across the experiment
    TARGET_AUDIO_FORMAT = audio_format      # Specify the audio format of the recordings
    SAMPLING_RATE = sampling_rate           # Set the default sampling rate

    # Extract Paths
    IS_DATABRICKS = experiment["is_databricks"]
    FROGID_CSV_PATH = str(experiment['frogid_csv_path'])             # Path to FrogID CSV file
    AUDIO_FILES_PATH = str(experiment['audio_files_path'])           # Path to the downloaded audio file recordings
    PROCESSOR_OUTPUT_PATH = str(experiment['processing_path'])       # Path for intermediary processing files

    config = PipelineConfig(
        selector_config=FrogDataSelectorConfig(
            csv_path=FROGID_CSV_PATH,
            filtering_criteria=SelectorFilteringCriteria(
                allow_duplicates=False,
                allow_inappropriate=False,
                allow_people_activity=False,
                avoid_poor_quality=True
            ),
            target_species_definition=SelectorTargetSpeciesDefinition(
                target_species_mapping={
                    0: "Other",
                    1: "Rhinella marina",
                    2: "Crinia signifera",
                    3: "Limnodynastes peronii",
                    4: "Litoria moorei",
                    5: "Litoria fallax",
                    6: "Limnodynastes tasmaniensis",
                    7: "Crinia parinsignifera",
                    8: "Limnodynastes dumerilii",
                    9: "Litoria caerulea",
                    10: "Litoria ewingii",
                    11: "Litoria verreauxii",
                    12: "Litoria rubella",
                    13: "Crinia glauerti",
                    14: "Litoria peronii",
                    15: "Litoria ridibunda",
                }
            )
        ),
        sampler_config=FrogDataSamplerConfig(
            random_state=RANDOM_SEED
        ),
        downloader_config=FrogAudioDownloaderConfig(
            is_databricks=IS_DATABRICKS,
            audio_dir=AUDIO_FILES_PATH,
            audio_format=TARGET_AUDIO_FORMAT,
            max_workers=5,
            max_retries=3,
            timeout=30
        ),
        preprocessor_config=PreprocessorConfig(
            orchestrator_config=PreprocessorOrchestratorConfig(
                preprocessor_chain=['audio_loader', 'feature_extractor_birdnet', 'databricks'],
                mode="parallel",
                max_workers=8,
                suppress_output=True,
                recording_id_column="id",
                class_label_column="class_label_single",
                species_name_column="species_name_single",
                audio_ext=TARGET_AUDIO_FORMAT,
                audio_dir=AUDIO_FILES_PATH,
                processor_output_dir=PROCESSOR_OUTPUT_PATH,
                embeddings_save_format="dbx-table"
            ),
            # Used to load in the downloaded recordings into waveform data
            audio_loader_config=AudioLoaderConfig(
                sr=SAMPLING_RATE
            ),
            # Used to apply filters on the loaded waveform (e.g. noisereduce)
            audio_filter_config=AudioFilterConfig(
                sr=SAMPLING_RATE,
                filter_type="both",
                low_freq=200,
                high_freq=16000,
                order=3,
                attenuation=0.25,
                filter_std=1.5
            ),
            audio_chunker_config=AudioChunkerConfig(
                sr=SAMPLING_RATE,
                # Define the size of a window (seconds). This is used to calculate number of chunks
                window_duration=30.0,
                # Only use first 30s of audio
                max_duration=30.0,
                # How to slide across the window
                hop_duration=None, 
                # Option to cap at a set number of chunks 
                max_chunks=None
            ),
            # Log-Mel Feature Extractor Config
            logmel_extractor_config = LogMelExtractorConfig(
                sr=SAMPLING_RATE, 
                n_mels=128, 
                hop_length=512
            ),
            # MFCC Feature Extractor Config
            mfcc_extractor_config = MFCCExtractorConfig(
                sr=SAMPLING_RATE, 
                n_mfcc=20, 
                hop_length=512
            ),
            # BirdNet Feature Extractor config
            birdnet_extractor_config = BirdNETExtractorConfig(
                sr=SAMPLING_RATE,
                segment_duration=3.0,
                output_mode="averaging",
                suppress_output=True
            ),
            # Responsible for normalizing the input features
            feature_normalizer_config = FeatureNormalizerConfig(
                method="db_clip"
            ),
            ####UPDATE: add in the new config here for the databricks functions
            #responsible for the storing and retrieving of data from databricks table
            embeddings_databricks_config = DatabricksDataConfig(
                table = "aus_museum_dbx_dev.frogid_ml.dev_embed_table",
                overlap_duration = 0.0,
                embeddings = 'BirdNet',
                embedding_strategy = 'averaging',
                window_duration = 3.0)
        ),
        modelling_data_strategy=ModellingSamplingStrategyConfig(
            species_data_to_use="single-species-only",
            target_species_sampling_strategy="downsample",
            target_species_max_samples_per_class=1300,
            include_other_species=True,
            other_species_sampling_strategy="stratify",
            other_species_boost_factor=3.0,
            exclude_other_species_below_count=None
        ),
        feature_data_schema=FeatureDataSchema(
            id_col= "id",                       # Column in Dataframe with the recording ids
            feature_col="features",             # Column in DataFrame with feature vectors
            label_col="class_label",            # Column in DataFrame with integer labels
            batch_size=32,                      # Batch size to take in the data
            expand_dims=False,                  # Add channel dim for CNNs
        ),
        modelling_config=ModellingConfig(
            split_data_config=ModellingSplitDataConfig(
                val_size= 0.15,
                test_size= 0.15,
                stratify=True,
                random_state=42
            ),
            training_config=ModellingTrainingConfig(
                epochs=50,
                use_early_stopping=True,
                use_model_checkpoint=True,
                auto_load_best_weights_after_training=True,
                patience=5,
                restore_best_weights=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            )
        ),
        evaluation_config = EvaluationConfig(
            pooling_strategy= 'mean',
            k = 3,
            inference_audio_dir="/Volumes/storage/frogid/wav_files"
        ),
        mlflow_config=MLFlowConfig(
            model_wrapper_path=r'mlops/inference/pyfunc_models/audio_pyfunc_model.py',
            log_model_wrapper=False
        )
    )

    if overrides:
        print("📝 Overriding the base config file")
        for dot_key, value in overrides.items():
            validate_dot_key_exists(config, dot_key)
            config = _deep_replace(config, dot_key, value)

    if run_id:
        mlflow.log_dict(
            run_id=run_id,
            dictionary=asdict(config),
            artifact_file=CONFIG_ARTIFACT_PATH
        )

    print_config_summary_table(config)

    return config

def get_pipeline_config(
    run_id: str = None,
    model_name: str = None,
    model_alias: str = None,
    model_version: str = None,
) -> PipelineConfig:
    """
    Load PipelineConfig from registered model or experiment run.
    
    Args:
        run_id: MLflow run ID to load from experiment
        model_name: Registered model name 
        model_alias: Model alias (e.g., "champion")
        model_version: Model version number (alternative to alias)
    
    Priority: registered model -> experiment run -> default config
    """
    from mlflow.tracking import MlflowClient
    
    config_path = "config/pipeline_config.yaml"
    
    # Strategy 1: Load from registered model
    if model_name:
        try:
            print(f"🔍 Loading config from registered model: {model_name}")
            client = MlflowClient()
            
            # Determine artifact URI based on version/alias
            if model_version:
                print(f"   Using specified version: {model_version}")
                artifact_uri = f"models:/{model_name}/{model_version}/{config_path}"
            elif model_alias:
                print(f"   Using alias: {model_alias}")
                artifact_uri = f"models:/{model_name}@{model_alias}/{config_path}"
            else:
                print("   No version/alias specified, using latest version")
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    raise Exception(f"No versions found for model {model_name}")
                latest_version = max(versions, key=lambda x: int(x.version)).version
                print(f"   Using latest version: {latest_version}")
                artifact_uri = f"models:/{model_name}/{latest_version}/{config_path}"
            
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
            print("   ✅ Downloaded from model registry")
            
            with open(local_path, "r") as f:
                config = from_dict(data_class=PipelineConfig, data=yaml.safe_load(f))
            print_config_summary_table(config)
            return config
            
        except Exception as e:
            print(f"❌ Failed to load from registered model: {e}")
            raise e
    
    # Strategy 2: Load from experiment run
    if run_id:
        try:
            print(f"🔍 Loading config from run: {run_id}")
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=config_path)
            print("   ✅ Downloaded from experiment run")
            
            with open(local_path, "r") as f:
                config = from_dict(data_class=PipelineConfig, data=yaml.safe_load(f))
            print_config_summary_table(config)
            return config
            
        except Exception as e:
            print(f"❌ Failed to load from run: {e}")
            raise e
    
    # No valid source provided
    raise ValueError("Must provide either 'model_name' or 'run_id' to load pipeline config")


@dataclass
class PipelineComponents:
    data_selector: FrogDataSelector
    data_downloader: FrogAudioDownloader
    data_sampler: FrogDataSampler
    data_preprocessor: PreprocessorOrchestrator
    data_databricks: FrogEmbedDatabricks
    model_trainer: TFModelTrainer
    model_evaluator: TFModelEvaluator

def instantiate_pipeline(config: PipelineConfig) -> PipelineComponents:
    return PipelineComponents(
        data_selector=FrogDataSelector(config.selector_config),
        data_downloader=FrogAudioDownloader(config.downloader_config),
        data_sampler=FrogDataSampler(config.sampler_config),
        data_preprocessor=PreprocessorOrchestrator(config.preprocessor_config),
        data_databricks = FrogEmbedDatabricks(config.preprocessor_config.embeddings_databricks_config),
        model_trainer=TFModelTrainer(
            data_schema=config.feature_data_schema, 
            modelling_config=config.modelling_config,
            mlflow_config=config.mlflow_config
        ),
        model_evaluator=TFModelEvaluator(
            data_schema=config.feature_data_schema,
            evaluation_config=config.evaluation_config
        )
    )
