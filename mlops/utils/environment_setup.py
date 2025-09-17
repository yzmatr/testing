# mlops/utils/environment_setup.py

import yaml
from pathlib import Path
import mlflow
from typing import Dict, List, Optional, Tuple
from tabulate import tabulate
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.gpu_config import setup_gpu
from pathlib import Path
from datetime import datetime
from mlops.utils.config import PipelineConfig
from mlops.utils.mlflow_utils import MLFlowLogger

def create_required_directories(required_directories: List[str]):
    # Create any missing directories
    for directory in required_directories:
        try:
            if not directory.exists():
                # Locally, create all directories
                directory.mkdir(parents=True, exist_ok=True)
                print(f"📝 Created new directory: {directory}")
            else:
                # Directory exists
                print(f"✅ Directory already exists: {directory}")
        except Exception as e:
            print(f"❌ Directory access error: {directory} - {e}")
            raise

def print_section_header(title: str):
    print(f"\n{'='*70}")
    print(title.upper())
    print(f"{'='*70}\n")

def print_table(dictionary: Dict[str, str]):
    table = [(col, desc) for col, desc in dictionary.items()]
    directories_report = tabulate(table, headers=["Name","Value"], tablefmt="psql")
    print(directories_report)

def start_experiment(
    experiment_name: str,
    root_dir: str,
    is_databricks: bool,
    current_user: str
) -> Dict[str, str]:
    """
    Generate the project path based on the current working directory.
    """
    #---------------------------------------------------------------------------------------
    # CORE DATA DIRECTORIES
    #---------------------------------------------------------------------------------------
    print_section_header("Experiment Directories")
    # Define the directory paths where the raw data resides
    DIR_DATA = Path("/Volumes/aus_museum_dbx_dev/frogid_ml/frogid_files") if is_databricks else root_dir / "data"
    # Define the directory where input data (e.g. CSV file of the frog recordings, resides)
    DIR_INPUT_DATA = DIR_DATA / "input"
    # Define the directory path where the actual frog recording audio files reside
    DIR_AUDIO_FILES = Path("/Volumes/storage/frogid/wav_files") if is_databricks else DIR_DATA / "audio_files"
    # Define the directory path where intermediary files during processing reside
    DIR_PROCESSED = Path("/tmp/frogid_processed") if is_databricks else DIR_DATA / "processed"
    # Define the CSV path for the FrogID files
    FROGID_CSV_PATH = DIR_INPUT_DATA / "20250317.csv"
    # Define the required directories to create
    required_directories = [DIR_PROCESSED]
    # Expand the directories to create if running locally
    if not is_databricks:
        required_directories += [DIR_INPUT_DATA, DIR_AUDIO_FILES]
    # Create the required directories based on the environment
    create_required_directories(required_directories)

    #---------------------------------------------------------------------------------------
    # Setup MLFlow Experiment
    #---------------------------------------------------------------------------------------
    print_section_header("Setup ML Flow")

    # Define the tracking database
    tracking_db = root_dir / "mlops" / "mlflow" / "tracking.db"
    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = "databricks" if is_databricks else f"sqlite:///{tracking_db}"
    mlflow.set_tracking_uri(tracking_uri)

    # Setup the experiment
    experiment_name = f"/Users/{current_user}/{experiment_name}" if is_databricks else experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        print(f"🆕 Created new experiment: {experiment_name}")

    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"🌐 MLFlow Experiment initialized: {experiment_name} with ID {experiment_id}")
    print(f"🌐 Using MLflow tracking at {tracking_uri}")

    # Display a message for local users around how they can open the local MLflow UI
    if not is_databricks:
        print(f"\n🔗 To use MLflow UI, you can start a server manually with:")
        print(f"   mlflow ui --backend-store-uri sqlite:///{root_dir / 'mlops/mlflow/tracking.db'} --port 5001")
        print(f"   Then open: http://localhost:5001\n")
    
    # Display a message for local  users about their GPU setup
    if not is_databricks:
        print_section_header("LOCAL ENVIRONMENT - GPU CONFIGURATION")
        setup_gpu(
            enable_memory_growth=True,
            log_device_placement=False      # Set to True for debugging
        )
    
    #---------------------------------------------------------------------------------------
    # REQUIRED PROJECT PATHS
    #---------------------------------------------------------------------------------------
    print_section_header("Experiment Setup")

    results = {
        "is_databricks": is_databricks,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        'frogid_csv_path': FROGID_CSV_PATH,
        "input_data_path": DIR_INPUT_DATA,
        "audio_files_path": DIR_AUDIO_FILES,
        "processing_path": DIR_PROCESSED
    }

    # Display the essential directories to the user
    print('The following variables are accessible via the dictionary:\n')
    print_table(results)

    # Return the relevant results
    return results

def start_mlflow_run(run_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Starts an MLFlow Run
    Args:
        run_id (Optional[str]): If provided, resume an existing run
    Returns:
        Tuple[str, str]: (run_name, run_id)
    """
    print_section_header("Setup MLFlow Run")

    active_run = mlflow.active_run()
    if active_run:
        mlflow.end_run()
        print(f"🧹 Ended active run: '{active_run.info.run_name}' ({active_run.info.run_id})")

    if run_id:
        # Resume the existing run
        active_run = mlflow.start_run(run_id=run_id)
        run_name = active_run.info.run_name
        print(f"🔁 Resumed existing run: '{run_name}' ({run_id})")
    else:
        # Start a new run
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        active_run = mlflow.start_run(run_name=run_name)
        run_id = active_run.info.run_id
        print(f"🏃 Started new run: '{run_name}' ({run_id})")

    print(f"📁 Artifact URI: {mlflow.get_artifact_uri()}")

    return run_name, run_id

def setup_mlflow(root_dir: str, is_databricks: bool):
    # Define the tracking database
    tracking_db = root_dir / "mlops" / "mlflow" / "tracking.db"
    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = "databricks" if is_databricks else f"sqlite:///{tracking_db}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🌐 Using MLflow tracking at {tracking_uri}")

    # Display a message for local users around how they can open the local MLflow UI
    if not is_databricks:
        print(f"\n🔗 To use MLflow UI, you can start a server manually with:")
        print(f"   mlflow ui --backend-store-uri sqlite:///{root_dir / 'mlops/mlflow/tracking.db'} --port 5001")
        print(f"   Then open: http://localhost:5001\n")
    
    # Display a message for local  users about their GPU setup
    if not is_databricks:
        print_section_header("LOCAL ENVIRONMENT - GPU CONFIGURATION")
        setup_gpu(
            enable_memory_growth=True,
            log_device_placement=False      # Set to True for debugging
        )
