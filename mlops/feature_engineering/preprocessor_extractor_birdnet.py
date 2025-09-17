# mlops/feature_engineering/preprocessor_extractor_birdnet.py

import os
import io
import librosa
import numpy as np
import tempfile
import soundfile as sf
import contextlib
import sys
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.utils.config import BirdNETExtractorConfig

class BirdNETExtractor:
    """
    ----------------------------------------------------------------------
    BirdNETExtractor
    ----------------------------------------------------------------------
    Extracts 1024-dim BirdNET embeddings from waveform using birdnetlib.
    """

    def __init__(self, config: BirdNETExtractorConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        import threading
        self._local = threading.local()

    def _get_analyzer(self):
        """Thread-safe BirdNET Analyzer loader with optional stdout suppression."""
        if not hasattr(self._local, 'analyzer'):
            try:
                from birdnetlib.analyzer import Analyzer

                # Optionally suppress verbose output
                context_mgr = (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO())
                ) if self.config.suppress_output else contextlib.nullcontext()

                with context_mgr[0], context_mgr[1]:
                    analyzer = Analyzer()
                    if hasattr(analyzer, 'interpreter') and analyzer.interpreter:
                        analyzer.interpreter.allocate_tensors()

                self._local.analyzer = analyzer
                self.logger.debug("✅ BirdNET analyzer initialized")

            except ImportError:
                raise ImportError("Install birdnetlib with: pip install birdnetlib")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize BirdNET analyzer: {e}")
                raise

        return self._local.analyzer

    def _create_recording_with_embeddings(self, analyzer, temp_path: str):
        """Create BirdNET recording and extract embeddings, with optional suppression."""
        from birdnetlib import Recording

        recording = Recording(analyzer, temp_path)

        if not hasattr(recording, 'extract_embeddings'):
            raise RuntimeError("BirdNET Recording object missing extract_embeddings()")

        # Suppress during embedding extraction
        context_mgr = (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO())
        ) if self.config.suppress_output else contextlib.nullcontext()

        with context_mgr[0], context_mgr[1]:
            recording.extract_embeddings()

        if not recording.embeddings:
            raise ValueError("No BirdNET embeddings returned")

        return recording

    def extract(self, y: np.ndarray) -> np.ndarray:
        """
        Extract BirdNET features from a waveform.

        Args:
            y (np.ndarray): Raw audio (typically 30s at 48kHz)

        Returns:
            np.ndarray: Either (1024,) or (N, 1024) depending on output_mode
        """
        sr = self.config.sr

        try:
            # Resample if needed
            if sr != self.config.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.config.sr)
                sr = self.config.sr

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            sf.write(temp_path, y, sr)
            analyzer = self._get_analyzer()
            recording = self._create_recording_with_embeddings(analyzer, temp_path)

            embedding_vectors = [
                np.array(seg['embeddings']) for seg in recording.embeddings if 'embeddings' in seg
            ]

            if not embedding_vectors:
                raise ValueError("No valid BirdNET embedding vectors extracted.")
            
            if self.config.output_mode == "averaging":
                result = np.mean(embedding_vectors, axis=0).astype(np.float32)
                self.logger.info(f"✅ Returned mean embedding: {result.shape}")
                return result

            elif self.config.output_mode == "stack":
                result = np.stack(embedding_vectors).astype(np.float32)
                self.logger.info(f"✅ Returned stacked embeddings: {result.shape}")
                return result

            else:
                raise ValueError(f"Unsupported output_mode: {self.config.output_mode}")

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
