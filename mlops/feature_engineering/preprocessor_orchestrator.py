import sys
import numpy as np
import pickle
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, StringType, ArrayType, FloatType
from tqdm import tqdm
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from typing import Callable, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from mlops.utils.logging_utils import suppress_output, setup_logger
from mlops.reporting.output_generator import OutputGenerator
from mlops.utils.config import PreprocessorConfig, BirdNETExtractorConfig, MFCCExtractorConfig, LogMelExtractorConfig
from mlops.feature_engineering.preprocessor_audio_loader import AudioLoader
from mlops.feature_engineering.preprocessor_audio_filter import AudioFilter
from mlops.feature_engineering.preprocessor_audio_chunker import AudioChunker
from mlops.feature_engineering.preprocessor_extractor_birdnet import BirdNETExtractor
from mlops.feature_engineering.preprocessor_extractor_mfcc import MFCCExtractor
from mlops.feature_engineering.preprocessor_extractor_logmel import LogMelExtractor
from mlops.feature_engineering.preprocessor_feature_normalizer import FeatureNormalizer
####Update: add the databricks function
from mlops.feature_engineering.preprocessor_audio_databricks import FrogEmbedDatabricks
#this will have the function for storage ^^


class PreprocessorOrchestrator:
    def __init__(self, config: PreprocessorConfig):

        #Can we add some extra values for the config?
        add_param = config.birdnet_extractor_config.output_mode
        self.embedding_strategy = add_param
        add_param = config.birdnet_extractor_config.segment_duration
        self.window_duration = add_param
        # General config for the orchestrator
        self.config = config.orchestrator_config
        # Setup general logging
        self.logger = setup_logger(name = self.__class__.__name__)

        

        # Setup components for the preprocessor
        preprocessor_chain = config.orchestrator_config.preprocessor_chain

        # Dynamically instantiate components based on config
        self.loader = AudioLoader(config.audio_loader_config) if "audio_loader" in preprocessor_chain else None
        self.audio_filter = AudioFilter(config.audio_filter_config) if "audio_filter" in preprocessor_chain else None
        self.chunker = AudioChunker(config.audio_chunker_config) if "audio_chunker" in preprocessor_chain else None
        self.databricks = FrogEmbedDatabricks(config.embeddings_databricks_config) if "databricks" in preprocessor_chain else None

        # Only one extractor should be used
        self.extractor_config = None
        self.extractor = None
        if "feature_extractor_birdnet" in preprocessor_chain:
            self.extractor_config = config.birdnet_extractor_config
            self.extractor = BirdNETExtractor(self.extractor_config)
            self.embedding_strategy = self.extractor_config.output_mode
        elif "feature_extractor_logmel" in preprocessor_chain:
            self.extractor_config = config.logmel_extractor_config
            self.extractor = LogMelExtractor(self.extractor_config)
        elif "feature_extractor_mfcc" in preprocessor_chain:
            self.extractor_config = config.mfcc_extractor_config
            self.extractor = MFCCExtractor(self.extractor_config)

        self.normalizer = FeatureNormalizer(config.feature_normalizer_config) if "feature_normalizer" in preprocessor_chain else None

        # Save embeddings directory
        self.processor_output_dir = self.config.processor_output_dir
        self.embeddings_output_dir = self._generate_cache_folder_name(extractor_config=self.extractor_config, chunker_config=config.audio_chunker_config)

        self.failed_ids: List[str] = []
        self.skipped_cached_count: int = 0
 
    ###################################################################################
    # CACHING OF THE DATA
    ###################################################################################
    def _generate_cache_folder_name(self, extractor_config, chunker_config) -> str:
        parts = ["embeddings"]
        if extractor_config:
            if isinstance(extractor_config, BirdNETExtractorConfig):
                parts.append("birdnet")
                parts.append(extractor_config.output_mode)
                parts.append(f"segment-{int(extractor_config.segment_duration)}s")

            elif isinstance(extractor_config, LogMelExtractorConfig):
                parts.append("logmel")
                parts.append(f"mels-{extractor_config.n_mels}")
                parts.append(f"hop-{extractor_config.hop_length}")

            elif isinstance(extractor_config, MFCCExtractorConfig):
                parts.append("mfcc")
                parts.append(f"coeffs-{extractor_config.n_mfcc}")
                parts.append(f"hop-{extractor_config.hop_length}")

        if chunker_config:
            parts.append(f"window-{int(chunker_config.window_duration)}s")
            if chunker_config.hop_duration:
                parts.append(f"hop-{int(chunker_config.hop_duration)}s")
            if chunker_config.max_duration:
                parts.append(f"max-{int(chunker_config.max_duration)}s")

        # Create the folder
        if len(parts) > 0:
            folder_name = "_".join(parts)
            embedding_path = os.path.join(self.processor_output_dir, folder_name)
            if not os.path.exists(folder_name):
                os.makedirs(embedding_path, exist_ok=True)
                self.logger.info(f'✅ Created {embedding_path} to store embeddings.')
            return embedding_path
        else:
            return self.processor_output_dir

    def _is_cached(self, audio_id: str, chunk_index: int) -> bool:
        """
        Check if the embedding for a given audio chunk is already cached.
        """
        if not self.embeddings_output_dir:
            return False

        filename = f"{audio_id}_chunk{chunk_index}.{self.config.embeddings_save_format}"
        path = os.path.join(self.embeddings_output_dir, filename)
        return os.path.exists(path)
    
    def _save_embedding(self, result: dict):
        if not self.embeddings_output_dir:
            return

        os.makedirs(self.embeddings_output_dir, exist_ok=True)

        file_base = f"{result['id']}_chunk{result['chunk_index']}"
        path = os.path.join(self.embeddings_output_dir, f"{file_base}.{self.config.embeddings_save_format}")

        if self.config.embeddings_save_format == "npz":
            np.savez_compressed(path, features=result["features"])
        elif self.config.embeddings_save_format == "pkl":
            with open(path, "wb") as f:
                pickle.dump(result, f)
        elif self.config.embeddings_save_format == "parquet":
            pd.DataFrame([result]).to_parquet(path, index=False)
        elif self.config.embeddings_save_format == "dbx-table":
            self.logger.info(f"Databricks table requested for storing embeddings - done at full dataset state")
        else:
            raise ValueError(f"Unsupported embeddings_save_format: {self.config.embeddings_save_format}")
        
    def _load_cached_embedding(self, audio_id: str, chunk_index: int) -> Optional[dict]:
        """
        Load a previously cached embedding if it exists.
        """
        if not self.embeddings_output_dir:
            return None

        filename = f"{audio_id}_chunk{chunk_index}.{self.config.embeddings_save_format}"
        path = os.path.join(self.embeddings_output_dir, filename)

        if not os.path.exists(path):
            return None

        try:
            if self.config.embeddings_save_format == "npz":
                data = np.load(path)
                features = data["features"]
            elif self.config.embeddings_save_format == "pkl":
                with open(path, "rb") as f:
                    return pickle.load(f)
            elif self.config.embeddings_save_format == "parquet":
                df = pd.read_parquet(path)
                return df.iloc[0].to_dict()
            else:
                raise ValueError(f"Unsupported embeddings_save_format: {self.config.embeddings_save_format}")

            return {
                "id": audio_id,
                "chunk_index": chunk_index,
                "features": features,
                "class_label": None,         # Will override below if needed
                "species_name": None
            }

        except Exception as e:
            self.logger.info(f"❌ Failed to load cached embedding {path}: {e}")
            return None

    ###################################################################################
    # PROCESSING OF THE DATA
    ###################################################################################
    def process_row(self, row: pd.Series) -> list[dict]:
        audio_id = row[self.config.recording_id_column]
        class_label = row.get(self.config.class_label_column, None)
        species_name = row.get(self.config.species_name_column, None)
        audio_path = os.path.join(self.config.audio_dir, f"{audio_id}.{self.config.audio_ext}")

        if not os.path.exists(audio_path):
            self.logger.info(f"⚠️ Skipping missing file: {audio_path}")
            self.failed_ids.append(audio_id)
            return []

        try:
            # Step 1: Load
            waveform = self.loader.load(audio_path)

            # Step 2: Filter (optional)
            if self.audio_filter is not None:
                waveform = self.audio_filter.apply(waveform)

            # Step 3: Chunk (optional)
            chunks = self.chunker.chunk(waveform) if self.chunker else [waveform]

            # Step 4: Extract + Normalize
            results = []
            for idx, chunk in enumerate(chunks):

                if self._is_cached(audio_id, idx):
                    cached_result = self._load_cached_embedding(audio_id, idx)
                    if cached_result:
                        # Ensure metadata is populated for the row
                        cached_result["class_label"] = class_label
                        cached_result["species_name"] = species_name
                        results.append(cached_result)
                        self.skipped_cached_count += 1
                        continue

                features = chunk
                if self.extractor is not None:
                    features = self.extractor.extract(chunk)
                if self.normalizer is not None:
                    features = self.normalizer.normalize(features)

                # Unique case: birdnet extractor already chunks internally (only valid if chunker is None)
                if self.chunker is None and isinstance(features, np.ndarray) and features.ndim == 2:
                    for sub_idx, vec in enumerate(features):
                        result = {
                            "id": audio_id,
                            "chunk_index": sub_idx,
                            "features": vec,
                            "class_label": class_label,
                            "species_name": species_name
                        }
                        results.append(result)
                        self._save_embedding(result)
                else:
                    result = {
                        "id": audio_id,
                        "chunk_index": idx,
                        "features": features,
                        "class_label": class_label,
                        "species_name": species_name
                    }
                    results.append(result)
                    self._save_embedding(result)

            return results

        except Exception as e:
            self.logger.info(f"❌ Error processing {audio_id}: {e}")
            self.failed_ids.append(audio_id)
            return []

    def process_audio_for_inference(self, waveform: np.ndarray) -> np.ndarray:
        """
        Process audio waveform for inference (similar to process_row but for audio binary input).
        
        Args:
            waveform (np.ndarray): Audio waveform data (already loaded)
            
        Returns:
            np.ndarray: Feature matrix with shape (n_chunks, n_features) ready for model prediction
        """
        try:
            self.logger.info("🎵 Processing audio waveform for inference")
            
            # Step 1: Filter (optional) - waveform is already loaded
            processed_waveform = waveform
            if self.audio_filter is not None:
                processed_waveform = self.audio_filter.apply(waveform)
                self.logger.debug("   ✅ Applied audio filtering")

            # Step 2: Chunk (optional)
            chunks = self.chunker.chunk(processed_waveform) if self.chunker else [processed_waveform]
            self.logger.info(f"   📊 Created {len(chunks)} chunks for processing")

            # Step 3: Extract + Normalize features
            feature_vectors = []
            for idx, chunk in enumerate(chunks):
                features = chunk
                
                # Feature extraction
                if self.extractor is not None:
                    features = self.extractor.extract(chunk)
                    self.logger.debug(f"   🔬 Extracted features for chunk {idx+1}: shape={features.shape}")
                
                # Feature normalization
                if self.normalizer is not None:
                    features = self.normalizer.normalize(features)
                    self.logger.debug(f"   📏 Normalized features for chunk {idx+1}")

                # Handle special case: BirdNET extractor chunks internally
                if self.chunker is None and isinstance(features, np.ndarray) and features.ndim == 2:
                    # BirdNET extractor returns multiple feature vectors (one per internal segment)
                    for sub_idx, vec in enumerate(features):
                        feature_vectors.append(vec)
                        self.logger.debug(f"   ✅ Processed internal chunk {sub_idx+1}: features shape={vec.shape}")
                else:
                    # Standard case: one feature vector per chunk
                    feature_vectors.append(features)
                    self.logger.debug(f"   ✅ Processed chunk {idx+1}: features shape={features.shape}")

            if not feature_vectors:
                raise ValueError("No features extracted from audio")
            
            # Convert to numpy array
            features_array = np.array(feature_vectors)
            self.logger.info(f"   🎯 Successfully processed {len(feature_vectors)} feature chunks for inference")
            self.logger.info(f"   📈 Final feature matrix shape: {features_array.shape} (chunks x features)")
            
            return features_array

        except Exception as e:
            self.logger.error(f"❌ Error processing audio for inference: {e}")
            raise


    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        #can we pull the required information from the database? - also need to check if running local or not
        #check there first

        #if the data is not there then we will go and get the data?

        # Reset state before a new run
        self.failed_ids = []
        self.skipped_cached_count = 0
        if self.config.suppress_output:
            with suppress_output(suppress_stdout=True, suppress_stderr=False):
                df_results = self._run(df)
        else:
            df_results = self._run(df)
        
        #If storing the embeddings into the table this needs to be done at the completion of all the embeddings creation
        # if done at the audio id level it adds a minute to each 
        if self.config.embeddings_save_format == "dbx-table" and df_results.empty == False:
            self.logger.info('Storing embeddings on databricks')
            stored = self.databricks.store_on_databricks(df_results)
            if stored == False:
                self.logger.info('Embeddings not stored from this run.')
            else:
                self.logger.info('Embeddings stored on databricks. Full or additional rows.')
        elif df_results.empty:
            self.logger.info('No embeddings created from this run. Possible missing file or eroneaous file.')
            
        OutputGenerator.preprocessing_summary(
            total_input=len(df),
            failed_count=len(self.failed_ids),
            skipped_cached_count=self.skipped_cached_count,
            final_output=len(df_results),
            print_output=True
        )

        return df_results


    ###################################################################################
    # SINGLE VERSUS MULTI-THREADING SETUP
    ###################################################################################
    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.mode == "sequential":
            return self._run_sequential(df)
        else:
            return self._run_parallel(df)

    def _run_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        all_results = []
        for _, row in tqdm(df.iterrows(), file=sys.__stdout__, total=len(df), desc="🔁 Sequential Extraction"):
            all_results.extend(self.process_row(row))
        return pd.DataFrame(all_results)

    def _run_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        all_results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self.process_row, row): idx for idx, row in df.iterrows()}
            for future in tqdm(as_completed(futures), file=sys.__stdout__, total=len(futures), desc="🚀 Parallel Extraction"):
                result = future.result()
                all_results.extend(result)
        return pd.DataFrame(all_results)