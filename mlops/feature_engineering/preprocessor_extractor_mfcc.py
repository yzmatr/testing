# mlops/feature_engineering/preprocessor_extractor_mfcc.py

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
from mlops.utils.config import MFCCExtractorConfig

class MFCCExtractor:
    """
    -------------------------------------------------------------
    MFCCExtractor
    -------------------------------------------------------------
    Extracts Mel-Frequency Cepstral Coefficients from a waveform using librosa.

    Output shape: [n_mfcc x time_frames]
    """

    def __init__(self, config: MFCCExtractorConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

    def extract(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the MFCCs for the given waveform.

        Args:
            y (np.ndarray): Input waveform (1D)
            sr (int): Sampling rate in Hz

        Returns:
            np.ndarray: MFCC features [n_mfcc x time_frames]
        """
        sr = self.config.sr

        try:
            # Compute MFCCs from the input waveform
            mfccs = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.config.n_mfcc,
                hop_length=self.config.hop_length
            )

            self.logger.debug(f"✅ Extracted MFCCs with shape: {mfccs.shape}")
            return mfccs
        except Exception as e:
            self.logger.error(f"❌ Failed to extract MFCCs: {e}")
            raise
