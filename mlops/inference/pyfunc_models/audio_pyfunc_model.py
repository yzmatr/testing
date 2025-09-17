"""
FrogID Audio Inference PyFunc Model

This module provides a simple MLflow PyFunc wrapper for FrogID audio inference.
It encapsulates the complete audio processing pipeline for species classification.
"""

from typing import Any, Dict, List

import tensorflow as tf
import yaml
from dacite import from_dict
from mlflow.models import set_model
from mlflow.pyfunc import PythonModel
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext#().notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.config import PipelineConfig
from mlops.utils.pipeline import instantiate_pipeline 


class FrogIDAudioInferencePyFunc(PythonModel):
    """
    MLflow PyFunc model for FrogID audio species classification.

    Simple wrapper that delegates to existing pipeline components.
    """

    def load_context(self, context):
        """Load the model and pipeline from MLflow context."""

        # Get bundled artifacts from context
        pipeline_config_path = context.artifacts.get("pipeline_config")
        model_path = context.artifacts.get("model")

        print(f"pipeline_config_path: {pipeline_config_path}")
        print(f"model_path: {model_path}")

        if not pipeline_config_path or not model_path:
            raise ValueError(
                "Required artifacts 'pipeline_config' and 'model' not found in context"
            )

        # The artifacts are already local files bundled with the model
        with open(pipeline_config_path, "r") as f:
            config = from_dict(data_class=PipelineConfig, data=yaml.safe_load(f))
        print(f"config loaded: {config}")

        self.pipeline = instantiate_pipeline(config)

        # Load trained model from the bundled artifact
        self.model = tf.keras.models.load_model(model_path)
        print(f"model loaded: {self.model}")

        # Get species mappings
        self.class_labels_to_species = (
            self.pipeline.data_selector.class_labels_to_species
        )
        self.pooling_strategy = (
            self.pipeline.model_evaluator.evaluation_config.pooling_strategy
        )
        self.k = self.pipeline.model_evaluator.evaluation_config.k

        print("FrogID Audio PyFunc model loaded successfully!")

    def predict(self, context, model_input: List[str]) -> List[Dict[str, Any]]:
        """
        Generate species predictions for audio files.

        Args:
            model_input: Audio file path(s) - str, list, or DataFrame

        Returns:
            Prediction results from tf_model_evaluator.predict()
        """
        # Initialize results list to store predictions for each audio file
        results = []
        
        # Process each audio file in the input list
        for audio_file in model_input:
            # Convert S3 paths to local paths if needed
            audio_file = self.process_path(audio_file)
            
            # Load audio waveform from file
            waveform = self.pipeline.data_preprocessor.loader.load(audio_file)
            
            # Extract features from the audio waveform for model inference
            features = self.pipeline.data_preprocessor.process_audio_for_inference(
                waveform
            )

            # Generate species predictions using the trained model
            # This returns top-k species predictions with confidence scores
            result = self.pipeline.model_evaluator.predict(
                model=self.model,
                features=features,
                class_label_to_species_mapping=self.class_labels_to_species,
                pooling_strategy=self.pooling_strategy,
                k=self.k,
            )

            # Add original file path to result for traceability and debugging
            result["file_path"] = audio_file
            results.append(result)

        # Return single result or list
        return results[0] if len(results) == 1 else results

    def get_pipeline(self):
        """Get direct access to the underlying pipeline for advanced operations."""
        return self.pipeline

    def process_path(self, path: str) -> str:
        """Process a single audio file path."""
        if path.startswith("s3://"):
            # Extract the filename from the S3 path
            filename = path.split("/")[-1]
            # Use the configurable inference audio directory from pipeline config
            inference_audio_dir = (
                self.pipeline.model_evaluator.evaluation_config.inference_audio_dir
            )
            return f"{inference_audio_dir}/{filename}"
        return path


# Specify which class definition represents the model instance
set_model(FrogIDAudioInferencePyFunc())
