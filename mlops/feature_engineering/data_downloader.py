# mlops/feature_engineering/data_downloader.py

import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
from typing import Optional, List, Dict
from pydub import AudioSegment
from pathlib import Path
import sys
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.reporting.output_generator import OutputGenerator
from mlops.utils.config import FrogAudioDownloaderConfig

class FrogAudioDownloader:
    def __init__(self, config: FrogAudioDownloaderConfig):
        self.config = config
        self.audio_dir = Path(config.audio_dir)

        self.session = self._create_session()
        self.logger = setup_logger(self.__class__.__name__)
        
        # Ensure pydub is configured and PATH includes common locations
        self._ensure_ffmpeg_available()

    ###################################################################################
    # UTILITIES / HELPERS
    ###################################################################################
    
    # Simple configuration to ensure pydub can find ffmpeg tools
    def _configure_pydub_simple(self):
        """Simple configuration for pydub to find ffmpeg tools."""
        # Ensure PATH includes homebrew bin
        current_path = os.environ.get("PATH", "")
        homebrew_path = "/opt/homebrew/bin"
        if homebrew_path not in current_path:
            os.environ["PATH"] = f"{homebrew_path}:{current_path}"
        
        # Set explicit paths if they exist
        ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
        ffprobe_path = "/opt/homebrew/bin/ffprobe"
        
        if os.path.exists(ffmpeg_path):
            AudioSegment.converter = ffmpeg_path
        if os.path.exists(ffprobe_path):
            AudioSegment.ffprobe = ffprobe_path
    
    def _ensure_ffmpeg_available(self):
        """Ensure ffmpeg/ffprobe are available and configure pydub."""
        # Add common paths to PATH if not already there
        common_paths = ["/opt/homebrew/bin", "/usr/local/bin"]
        current_path = os.environ.get("PATH", "")
        
        for path in common_paths:
            if path not in current_path:
                os.environ["PATH"] = f"{path}:{current_path}"
        
        # Configure pydub
        self._configure_pydub_simple()
        
        # Verify configuration worked
        if not hasattr(AudioSegment, 'converter') or not AudioSegment.converter:
            self.logger.warning("⚠️ ffmpeg not found - audio conversion may fail")
        if not hasattr(AudioSegment, 'ffprobe') or not AudioSegment.ffprobe:
            self.logger.warning("⚠️ ffprobe not found - audio conversion may fail")

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_download_filepath(self, audio_id: str) -> str:
        return os.path.join(self.audio_dir, f"{audio_id}.aac")

    def _get_output_filepath(self, audio_id: str) -> str:
        ext = self.config.audio_format or "aac"  # fallback to aac if None
        return os.path.join(self.audio_dir, f"{audio_id}.{ext}")

    def _file_exists(self, audio_id: str) -> bool:
        return os.path.exists(self._get_output_filepath(audio_id))

    ###################################################################################
    # CHECK AUDIO DIRECTORY
    ###################################################################################
    def check_missing_files(self, df_to_download: pd.DataFrame) -> pd.DataFrame:
        # Fail safe - if the audio directory does not exist
        if not self.audio_dir.exists():
            raise FileExistsError("❌ The audio directory does not exist. Exiting...")
        # If the audio directory does exist, extract a list of all the files that are already there
        self.logger.info(f"✅ Audio Directory Exists: {self.audio_dir}")
        # Extract the current list of files
        current_files = list(self.audio_dir.glob(f"*.{self.config.audio_format}"))
        self.logger.info(f"✅ The directory has {len(current_files)} files already downloaded.")
        # Extract only the STEM IDs (e.g., 'abc123' from 'abc123.wav')
        existing_ids = set()
        for f in current_files:
            if f.stem.isdigit():
                existing_ids.add(int(f.stem))
            else:
                print(f"Audio file name is not a valid ID: {f.stem}")
        #existing_ids = {int(f.stem) for f in current_files}
        # Identify missing entries
        df_missing = df_to_download[~df_to_download['id'].isin(existing_ids)]
        self.logger.info(f"🔍 {len(df_missing)} files are missing and need to be downloaded.")
        return df_missing
    
    ###################################################################################
    # DOWNLOAD LOGIC
    ###################################################################################
    
    def _download_file_to_path(self, url: str, audio_id: str) -> None:
        download_path = self._get_download_filepath(audio_id)
        output_path = self._get_output_filepath(audio_id)

        response = self.session.get(url, stream=True, timeout=self.config.timeout)
        response.raise_for_status()

        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Handle conversion or rename
        if self.config.audio_format == "wav":
            
            # Ensure pydub is configured in this worker thread
            self._configure_pydub_simple()
            
            try:
                audio = AudioSegment.from_file(download_path, format="aac")
                audio.export(output_path, format="wav")
                if not os.path.exists(output_path):
                    raise Exception(f"WAV conversion failed for {audio_id}")
                # Clean up the temporary AAC file after successful conversion
                os.remove(download_path)
            except Exception as e:
                # If conversion fails, clean up and re-raise
                if os.path.exists(download_path):
                    os.remove(download_path)
                raise Exception(f"Audio conversion failed: {e}")
        elif self.config.audio_format == "aac":
            os.rename(download_path, output_path)
        else:
            # audio_format is None — leave file as-is
            output_path = download_path

    def _attempt_download_row(self, row: Dict) -> Optional[Dict]:
        audio_url = row['call_audio']
        audio_id = row['id']

        if self._file_exists(audio_id):
            return None  # Already downloaded

        try:
            self._download_file_to_path(audio_url, audio_id)
            if not self._file_exists(audio_id):
                raise Exception("Output file missing or corrupted")
            return None

        except Exception as e:
            self.logger.error(f"❌ Failed to download {audio_url} (ID: {audio_id}): {e}")
            try:
                os.remove(self._get_output_filepath(audio_id))
            except:
                pass
            return {'id': audio_id, 'url': audio_url, 'error': str(e)}

    def _download_rows(self, rows_to_download: pd.DataFrame):
        failed = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._attempt_download_row, row.to_dict())
                for _, row in rows_to_download.iterrows()
            ]
            with tqdm(total=len(rows_to_download), desc="Downloading audio files") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        failed.append(result)
                    else:
                        pbar.update(1)
        return failed

    def download_files(self, df: pd.DataFrame) -> pd.DataFrame:
        IS_DATABRICKS = self.config.is_databricks

        # In all environments, the incoming dataframe must have a valid "id" column
        if 'id' not in df.columns:
            raise ValueError("DataFrame must contain a valid 'id' column.")
        
        # Check the Audio Directory path and return any files that are missing
        df_missing_from_directory = self.check_missing_files(df_to_download=df)
        df_skipped = df[~df['id'].isin(df_missing_from_directory['id'])]

        # Placeholder for any downloads that failed
        failed_downloads = []
        required_count = len(df)
        available_count = len(df_skipped)
        missing_from_directory_count = len(df_missing_from_directory)

        # Handle missing files
        if missing_from_directory_count > 0:
            if IS_DATABRICKS:
                self.logger.info(f"‼️ There are {missing_from_directory_count} missing files on databricks. Due to environment limitations, we cannot download these and are skipping them.")
            else:
                # Before you can download the files, make sure the call_audio column exists
                if 'call_audio' not in df.columns:
                    error_message = "❌ DataFrame must contain a valid 'call_audio' column."
                    self.logger.info(error_message)
                    raise ValueError(error_message)
                self.logger.info(f"📝 Downloading the {missing_from_directory_count} files you are missing.")
                failed_downloads = self._download_rows(df_missing_from_directory)
        else:
           self.logger.info(f"✅ You have all the files you need.")
            
        failed_count = len(failed_downloads)
        downloaded_count = missing_from_directory_count - failed_count

        # Since we are not downloading new files in the databricks environment, missing would be
        # anything that is not stored in the volume, while in local, missing would be anything that
        # failed to download
        missing_ids_after_download = list(df_missing_from_directory['id']) if IS_DATABRICKS else failed_downloads
        # Update the missing count
        missing_after_download_count = len(missing_ids_after_download)

        # Calculate the number of files actually available for modelling
        ready_for_use_count = required_count - missing_after_download_count

        # Determine whether the full data is available
        has_complete_data = required_count == ready_for_use_count

        # ----------------------------------------------------------------
        # Step 3: Reporting:
        # ----------------------------------------------------------------

        OutputGenerator.downloader_results(
            required_count=required_count,
            available_count=available_count,
            missing_from_directory_count=missing_from_directory_count,
            downloaded_count=downloaded_count,
            failed_to_download_count=failed_count,
            missing_after_download_count=missing_after_download_count,
            ready_for_use_count=ready_for_use_count,
            has_complete_data=has_complete_data,
            print_output=True
        )

        if missing_ids_after_download:
            self.logger.warning(f"⚠️ Removing {len(missing_ids_after_download)} samples with missing audio from the dataframe.")
            df = df[~df["id"].isin(missing_ids_after_download)].reset_index(drop=True)

        return df