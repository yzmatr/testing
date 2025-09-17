# mlops/feature_engineering/preprocessor_audio_filter.py

import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.utils.config import AudioFilterConfig

class AudioFilter:
    """
    ----------------------------------------------------------------
    AudioFilter
    ----------------------------------------------------------------
    Applies optional preprocessing filters to audio waveforms:
    - Pass-band Butterworth filter
    - Stationary spectral noise reduction
    - Combination of both
    """

    def __init__(self, config: AudioFilterConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

    def apply(self, y: np.ndarray) -> np.ndarray:
        """
        Apply the configured filtering pipeline to the waveform.

        Args:
            y: Input waveform as 1D NumPy array.

        Returns:
            Filtered waveform as 1D NumPy array.
        """
        waveform = None
        if self.config.filter_type == "none":
            self.logger.debug("⚙️ No filtering applied (filter_type='none').")
            waveform = y
        elif self.config.filter_type == "band-pass":
            self.logger.info("🔊 Applying band-pass filter...")
            waveform = self._apply_bandpass(y)
        elif self.config.filter_type == "noisereduce":
            self.logger.info("🔇 Applying stationary noise reduction...")
            waveform = self._apply_noisereduce(y)
        elif self.config.filter_type == "both":
            self.logger.info("🎛️ Applying bandpass + noise reduction...")
            y = self._apply_bandpass(y)
            waveform = self._apply_noisereduce(y)

        if waveform is None:
            raise ValueError(f"Unknown filter_type: {self.config.filter_type}")
        
        self.logger.info(f"✅ Filtered waveform shape: {waveform.shape}")
        return waveform
        
    def _design_bandpass_filter(self):
        """
        Create Butterworth filter coefficients based on config.
        Automatically falls back to lowpass if high_freq exceeds Nyquist.
        """
        nyquist = 0.5 * self.config.sr
        low = self.config.low_freq / nyquist
        high = self.config.high_freq / nyquist

        if high > 1:
            self.logger.warning("⚠️ high_freq > Nyquist — using lowpass filter.")
            return butter(self.config.order, low, btype="low")
        return butter(self.config.order, [low, high], btype="band")
    
    
    def _apply_bandpass(self, y: np.ndarray) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter.

        Args:
            y: Input waveform.

        Returns:
            Bandpass filtered waveform.
        """
        try:
            b, a = self._design_bandpass_filter()
            filtered = lfilter(b, a, y)
            self.logger.debug(f"✅ Bandpass filter applied: {self.config.low_freq}–{self.config.high_freq} Hz")
            return filtered
        except Exception as e:
            self.logger.warning(f"⚠️ Bandpass filter failed: {e}")
            return y

    def _apply_noisereduce(self, y: np.ndarray) -> np.ndarray:
        """
        Apply stationary spectral noise reduction using noisereduce package.

        Args:
            y: Input waveform.

        Returns:
            Denoised waveform.
        """
        try:
            reduced = nr.reduce_noise(
                y=y,
                sr=self.config.sr,
                prop_decrease=self.config.attenuation,
                stationary=True,
                n_std_thresh_stationary=self.config.filter_std
            )
            self.logger.debug("✅ Noise reduction applied.")
            return reduced
        except Exception as e:
            self.logger.warning(f"⚠️ Noise reduction failed: {e}")
            return y
