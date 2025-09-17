# mlops/feature_engineering/preprocessor_feature_normalizer.py
import numpy as np
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.config import FeatureNormalizerConfig

class FeatureNormalizer:
    """
    Normalizer class for applying audio feature normalization techniques.

    Supports:
    - 'db_clip': for decibel-scaled features (e.g., log-mel)
    - 'minmax': for general feature ranges (e.g., raw waveform, MFCCs)
    """
    def __init__(self, config: FeatureNormalizerConfig):
        self.config = config

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize an input feature array using the specified method.
        
        Args:
            arr (np.ndarray): Input feature array (e.g., mel spectrogram, MFCCs)

        Returns:
            np.ndarray: Normalized array in [0, 1] range (shape unchanged)
        """
        if arr.size == 0:
            raise ValueError("Input array is empty. Cannot normalize.")

        if self.config.method == "db_clip":
            return np.clip(arr, -80, 0) / 80.0
        elif self.config.method == "minmax":
            min_, max_ = np.nanmin(arr), np.nanmax(arr)
            if np.isclose(min_, max_):
                return np.zeros_like(arr)  # constant input
            return (arr - min_) / (max_ - min_ + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {self.config.method}")