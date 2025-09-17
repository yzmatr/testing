# mlops/feature_engineering/preprocessor_audio_chunker.py

import numpy as np
from typing import List
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.utils.config import AudioChunkerConfig

class AudioChunker:
    """
    ----------------------------------------------------------------
    AudioChunker
    ----------------------------------------------------------------
    Splits a waveform into fixed-duration chunks.

    - Single-chunk mode if hop_duration is None
    - Sliding mode with optional max_duration and max_chunks limits
    """

    def __init__(self, config: AudioChunkerConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        self.chunk_samples = int(config.window_duration * config.sr)
        self.hop_samples = int(config.hop_duration * config.sr) if config.hop_duration else None

    def chunk(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Chunk the waveform based on the configuration.

        Args:
            y: 1D waveform as NumPy array.

        Returns:
            List of waveform chunks (each of equal length).
        """
        chunks = []
        if self.hop_samples is None:
            self.logger.debug("🧱 Single-chunk mode (hop_duration=None)")
            chunks = [self._make_single_chunk(y)]
        else:
            chunks = self._make_sliding_chunks(y)
        
        self.logger.info(f"✅ Created {len(chunks)} chunks. Each shape: {chunks[0].shape}")
        return chunks

    def _make_single_chunk(self, y: np.ndarray) -> np.ndarray:
        """
        Pad or trim the input to return a single chunk.
        """
        original_len = len(y)
        if original_len < self.chunk_samples:
            self.logger.debug(f"🧩 Padding waveform from {original_len} → {self.chunk_samples} samples")
            return np.pad(y, (0, self.chunk_samples - original_len), mode="constant")
        elif original_len > self.chunk_samples:
            self.logger.debug(f"✂️ Truncating waveform from {original_len} → {self.chunk_samples} samples")
            return y[:self.chunk_samples]
        else:
            return y

    def _make_sliding_chunks(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Generate fixed-size chunks using sliding windows.

        Applies padding to the final chunk and obeys max_duration / max_chunks if provided.
        """
        chunks = []
        n_total = len(y)

        # Limit total samples if max_duration is set
        max_samples = min(n_total, int(self.config.max_duration * self.config.sr)) if self.config.max_duration else n_total
        num_chunks_possible = max(1, (max_samples - self.chunk_samples + self.hop_samples) // self.hop_samples + 1)

        self.logger.debug(f"🧮 Total possible chunks: {num_chunks_possible}")

        for i, start in enumerate(range(0, max_samples, self.hop_samples)):
            if self.config.max_chunks is not None and i >= self.config.max_chunks:
                self.logger.debug(f"⛔ Reached max_chunks: {self.config.max_chunks}")
                break

            end = start + self.chunk_samples
            chunk = y[start:end]

            if len(chunk) < self.chunk_samples:
                self.logger.debug(f"🧩 Padding final chunk: {len(chunk)} → {self.chunk_samples}")
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode="constant")

            chunks.append(chunk)

            if end >= max_samples:
                break

        self.logger.info(f"✅ Created {len(chunks)} chunks from {n_total} samples")
        return chunks