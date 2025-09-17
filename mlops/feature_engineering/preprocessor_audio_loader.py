# pipeline/audio/audio_loader.py

import os
import warnings
import numpy as np
import librosa
import soundfile as sf
import sys
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.utils.config import AudioLoaderConfig
import io, requests, urllib.parse

class AudioLoader:
    """
    ---------------------------------------------------------------------------
    AudioLoader
    ---------------------------------------------------------------------------
    Loads audio from disk and resamples it to a consistent target sampling rate.

    Attempts to load with `librosa`, falls back to `soundfile` on failure.
    Converts stereo to mono if needed.

    Example:
        config = AudioLoaderConfig(sr=48000)
        loader = AudioLoader(config)
        y = loader.load("path/to/audio.aac")
    """
    
    def __init__(self, config: AudioLoaderConfig):
        """
        Args:
            config (AudioLoaderConfig): Configuration for audio loading.
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

    def load(self, path: str) -> np.ndarray:
        HOST  = 
        TOKEN = 
        """
        Load an audio file and resample it to the target sample rate.

        Args:
            path (str): Path to the audio file.

        Returns:
            np.ndarray: Loaded and resampled waveform (mono).

        Raises:
            FileNotFoundError: If file does not exist.
            RuntimeError: If loading fails with both librosa and soundfile.
        """
        url = f"{HOST}/api/2.0/fs/files{urllib.parse.quote(path, safe='/')}"
        try:
            resp = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30)
            if resp.status_code == 404:
                self.logger.error(f"❌ File not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")
            resp.raise_for_status()  # other HTTP errors -> raise
        finally:
            try:
                resp.close()
            except Exception:
                pass

        # First try with librosa
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                url = f"{HOST}/api/2.0/fs/files{urllib.parse.quote(path)}"
                r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30)
                r.raise_for_status()
                # librosa can read from a file-like object
                buf = io.BytesIO(r.content)
                y, sr = librosa.load(buf, sr=None)
                self.logger.debug(f"📥 Loaded audio using librosa: {path} (sr={sr})")
        except Exception as e:
            self.logger.warning(f"⚠️ librosa failed for {path}: {e}")
            # Try fallback with soundfile
            try:
                url = f"{HOST}/api/2.0/fs/files{urllib.parse.quote(path)}"
                r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30)
                r.raise_for_status()
                # librosa can read from a file-like object
                buf = io.BytesIO(r.content)
                y, sr = sf.read(buf,dtype="float32")
                self.logger.debug(f"📥 Loaded audio using soundfile: {path} (sr={sr})")
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)  # Convert stereo to mono
                    self.logger.debug(f"🎧 Converted stereo to mono: {path}")
            except Exception as sf_err:
                self.logger.error(f"❌ soundfile also failed for {path}: {sf_err}")
                raise RuntimeError(f"Failed to load audio: {sf_err}")

        # Resample if needed
        if sr != self.config.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.config.sr)
            self.logger.debug(f"🎚️ Resampled {path} from {sr} → {self.config.sr} Hz")

        self.logger.info(f"✅ Loaded waveform of length={len(y)/self.config.sr:.2f}s & shape={y.shape} from {path}")

        return y
