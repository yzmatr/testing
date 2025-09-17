# mlops/feature_engineering/preprocessor_extractor_logmel.py

import numpy as np
import librosa
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.utils.config import LogMelExtractorConfig

class LogMelExtractor:
    """
    -------------------------------------------------------------
    LogMelExtractor
    -------------------------------------------------------------
    Extracts log-mel spectrogram features from a waveform using librosa.

    Output shape: [n_mels x time_frames]
    """

    def __init__(self, config: LogMelExtractorConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

    def extract(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the log-mel spectrogram for the given waveform.

        Args:
            y (np.ndarray): Input waveform (1D)

        Returns:
            np.ndarray: Log-mel spectrogram [n_mels x time_frames]
        """
        try:
            # Compute the mel spectrogram (power scale)
            S = librosa.feature.melspectrogram(
                y=y,
                sr=self.config.sr,
                n_mels=self.config.n_mels,
                hop_length=self.config.hop_length
            )

            # Convert to log scale (dB)
            log_mel = librosa.power_to_db(S, ref=np.max)

            self.logger.debug(f"✅ Extracted log-mel spectrogram with shape: {log_mel.shape}")
            return log_mel
        except Exception as e:
            self.logger.error(f"❌ Failed to extract log-mel spectrogram: {e}")
            raise