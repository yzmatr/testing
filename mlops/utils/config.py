# mlops/utils/config.py
from typing import Literal, Optional, List, Dict
from dataclasses import dataclass, field
from tabulate import tabulate

@dataclass
class SelectorFilteringCriteria:
    allow_duplicates: bool = False
    allow_inappropriate: bool = False
    allow_people_activity: bool = False
    avoid_poor_quality: bool = True

@dataclass
class SelectorTargetSpeciesDefinition:
    target_species_mapping: Optional[Dict[int, str]] = None         # User-defined core species

@dataclass
class FrogDataSelectorConfig:
    # Required input paths
    csv_path: str = "20250317.csv"
    # Initial data filtering criteria
    filtering_criteria: SelectorFilteringCriteria = field(default_factory=SelectorFilteringCriteria)
    # Define what should be included in the target species
    target_species_definition: SelectorTargetSpeciesDefinition = field(default_factory=SelectorTargetSpeciesDefinition)

@dataclass
class FrogDataSamplerConfig:
    random_state: int = 42

@dataclass
class FrogAudioDownloaderConfig:
    audio_dir: str
    is_databricks: bool = False
    audio_format: Optional[Literal["wav", "aac"]] = "wav"
    max_workers: int = 5
    max_retries: int = 3
    timeout: int = 30

@dataclass
class AudioLoaderConfig:
    """
    Configuration for AudioLoader.
    
    Attributes:
        sr (int): Target sample rate to which all audio should be resampled.
    """
    sr: int = 48000

@dataclass
class AudioFilterConfig:
    """
    Configuration for audio filtering.

    Attributes:
        filter_type: Which filtering strategy to apply.
        sr: Sample rate of the audio.
        low_freq: Lower cutoff for bandpass filter (Hz).
        high_freq: Upper cutoff for bandpass filter (Hz).
        order: Order of the Butterworth filter.
        attenuation: Amount of noise reduction (0 to 1).
        filter_std: Std threshold for stationary noise reduction.
    """
    filter_type: Literal["none", "band-pass", "noisereduce", "both"] = "none"
    sr: int = 48000
    low_freq: float = 200.0
    high_freq: float = 16000.0
    order: int = 3
    attenuation: float = 0.25
    filter_std: float = 1.5

@dataclass
class AudioChunkerConfig:
    """
    Configuration for chunking audio waveforms.

    Attributes:
        sr (int): Sampling rate in Hz.
        window_duration (float): Duration of each chunk in seconds.
        hop_duration (Optional[float]): Step between chunks in seconds.
            If None, only a single chunk is returned.
        max_duration (Optional[float]): Maximum total duration (in seconds) to consider.
        max_chunks (Optional[int]): Maximum number of chunks to return.
    """
    sr: int = 48000
    window_duration: float = 3.0
    hop_duration: Optional[float] = None
    max_duration: Optional[float] = None
    max_chunks: Optional[int] = None

@dataclass
class BirdNETExtractorConfig:
    """
    Configuration for BirdNET embedding extraction.
    - sr: Required sampling rate for BirdNET (must be 48000 Hz).
    - segment_duration: Segment size used internally by BirdNET (default 3.0s).
    - output_mode: 'averaging' to return mean embedding, 'stack' for per-segment embeddings.
    - suppress_output: Whether to suppress stdout/stderr from BirdNET (recommended).
    """
    sr: int = 48000
    segment_duration: float = 3.0
    output_mode: Literal["averaging", "stack"] = "averaging"
    suppress_output: bool = False

@dataclass
class DatabricksDataConfig:
    """
    #Configuration for databricks usage.
    #- 
    """
    table: str = "aus_museum_dbx_dev.frogid_ml.dev_embed_table"
    overlap_duration: float = 0.0
    embeddings: str = 'BirdNet'
    embedding_strategy: str = 'averaging'
    window_duration: float = 3.0
 

@dataclass
class LogMelExtractorConfig:
    """
    Configuration for log-mel spectrogram extraction.

    Attributes:
        sr (int): Sampling rate used during feature extraction.
        n_mels (int): Number of mel bands.
        hop_length (int): Hop length in samples.
    """
    sr: int = 48000
    n_mels: int = 128
    hop_length: int = 512

@dataclass
class MFCCExtractorConfig:
    """
    Configuration for MFCC extraction.

    Attributes:
        sr (int): Sampling rate used during feature extraction.
        n_mfcc (int): Number of MFCC coefficients to extract.
        hop_length (int): Hop length in samples.
    """
    sr: int
    n_mfcc: int = 13
    hop_length: int = 512

@dataclass
class FeatureNormalizerConfig:
    method: Literal["minmax", "db_clip"] = "db_clip"

@dataclass
class PreprocessorOrchestratorConfig:
    preprocessor_chain: List[Literal[
        "audio_loader",
        "audio_filter",
        "audio_chunker",
        "feature_extractor_birdnet",
        "feature_extractor_logmel",
        "feature_extractor_mfcc",
        "feature_normalizer",
        "databricks"
        ####UPDATE: add here what the databricks is going to be called
    ]]
    mode: Literal["sequential", "parallel"] = "sequential"
    max_workers: int = 8
    suppress_output: bool = False
    recording_id_column: str = "id"
    class_label_column: str = "class_label_single"
    species_name_column: str = "species_name_single"
    audio_ext: str = "wav"
    audio_dir: str = "audio_files"
    processor_output_dir: str = "processed"
    embeddings_save_format: Optional[Literal["npz", "pkl", "parquet", "dbx-table"]] = "npz"    

@dataclass
class PreprocessorConfig:
    orchestrator_config: PreprocessorOrchestratorConfig
    # Optional modules in the chain
    audio_loader_config: Optional[AudioLoaderConfig] = None
    audio_filter_config: Optional[AudioFilterConfig] = None
    audio_chunker_config: Optional[AudioChunkerConfig] = None
    # Optional feature extractor
    birdnet_extractor_config: Optional[BirdNETExtractorConfig] = None
    logmel_extractor_config: Optional[LogMelExtractorConfig] = None
    mfcc_extractor_config: Optional[MFCCExtractorConfig] = None
    # Optional normalizer
    feature_normalizer_config: Optional[FeatureNormalizerConfig] = None
    #optional databricks config
    embeddings_databricks_config: Optional[DatabricksDataConfig] = None

@dataclass
class ModellingSamplingStrategyConfig:
    # Define the type of species to use
    species_data_to_use: Optional[Literal['single-species-only', 'multi-species-only']] = 'single-species-only'
    # Define target species sampling criteria
    target_species_sampling_strategy: Optional[Literal['downsample', 'upsample']] = 'downsample'
    target_species_max_samples_per_class: Optional[int] = 1300
    # Define other species sampling criteria
    include_other_species: bool = True
    other_species_sampling_strategy: Optional[Literal['random', 'stratify']] = 'random'
    other_species_boost_factor: Optional[float] = 1.0
    exclude_other_species_below_count: Optional[int] = None

@dataclass
class FeatureDataSchema:
    id_col: str = "id"                      # Column in Dataframe with the recording ids
    feature_col: str = "features"           # Column in DataFrame with feature vectors
    label_col: str = "class_label"          # Column in DataFrame with integer labels
    batch_size: int = 32                    # Batch size to take in the data
    expand_dims: bool = False               # Add channel dim for CNNs

@dataclass
class ModellingSplitDataConfig:
    val_size: float = 0.15
    test_size: float = 0.15
    stratify: bool = True
    random_state: int = 42
    
@dataclass
class ModellingTrainingConfig:
    epochs: int = 50
    use_early_stopping: bool = True
    use_model_checkpoint: bool = True
    auto_load_best_weights_after_training: bool = True
    patience: int = 5
    restore_best_weights: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    verbose: int = 1

@dataclass
class ModellingConfig:
    split_data_config: ModellingSplitDataConfig = field(default_factory=ModellingSplitDataConfig)
    training_config: ModellingTrainingConfig = field(default_factory=ModellingTrainingConfig)

@dataclass
class EvaluationConfig:
    pooling_strategy: Optional[Literal['mean', 'max', 'voting','topk', 'softmax']] = 'mean'
    k: int = 3
    inference_audio_dir: str = "/Volumes/storage/frogid/wav_files"

@dataclass
class MLFlowConfig:
    model_wrapper_path: str = None
    log_model_wrapper: bool = False

@dataclass
class PipelineConfig:
    selector_config: FrogDataSelectorConfig
    sampler_config: FrogDataSamplerConfig
    downloader_config: FrogAudioDownloaderConfig
    preprocessor_config: PreprocessorConfig
    modelling_data_strategy: ModellingSamplingStrategyConfig
    feature_data_schema: FeatureDataSchema
    modelling_config: ModellingConfig
    evaluation_config: EvaluationConfig
    mlflow_config: MLFlowConfig
    

