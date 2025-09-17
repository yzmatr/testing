import os
import io
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union
from dataclasses import asdict, is_dataclass
from matplotlib.figure import Figure
import sys
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from matplotlib.figure import Figure
import pickle
import tempfile
import tensorflow as tf
import shutil

DataArtifactType = Union[pd.DataFrame, dict, list, object, Figure]

class MLFlowLogger:
    def __init__(self, output_dir: str = ""):
        self.logger = setup_logger(self.__class__.__name__)
        self.output_dir = output_dir

    ###################################################################################
    # INTERNAL UTILITIES / HELPERS
    ###################################################################################

    def _log_text(self, text: str, artifact_name: str, output_dir: Optional[str] = None):
        output_dir = output_dir or self.output_dir
        filename = f'{artifact_name}.txt'
        artifact_file = os.path.join(output_dir, filename)
        mlflow.log_text(
            text=text, 
            artifact_file=artifact_file
        )
        self.logger.info(f"📤 Logged text: {artifact_file}")

    def _log_json(self, data: Union[dict, list], artifact_name: str, output_dir: Optional[str] = None):
        output_dir = output_dir or self.output_dir
        filename = f"{artifact_name}.json"
        artifact_file = os.path.join(output_dir, filename)
        mlflow.log_dict(data, artifact_file=artifact_file)
        self.logger.info(f"📤 Saved: {artifact_file}")

    def _log_df_to_csv(self, df: pd.DataFrame, artifact_name: str, output_dir: Optional[str] = None):
        output_dir = output_dir or self.output_dir
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        filename = f'{artifact_name}.csv'
        artifact_file = os.path.join(output_dir, filename)
        mlflow.log_text(text=buffer.getvalue(), artifact_file=artifact_file)
        self.logger.info(f"📤 Logged CSV DataFrame: {artifact_file}")

    def _log_figure(self, fig: Figure, artifact_name: str, output_dir: Optional[str] = None, dpi: int = 150):
        output_dir = output_dir or self.output_dir
        filename = f'{artifact_name}.png'
        artifact_file = os.path.join(output_dir, filename)

        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format="png", dpi=dpi, bbox_inches="tight")
        image_buffer.seek(0)

        mlflow.log_image(image=image_buffer, artifact_file=artifact_file)
        self.logger.info(f"🖼️ Logged figure: {artifact_file}")
        plt.close(fig)
    
    def _log_matplotlib_figure(self, fig: Figure, artifact_name: str, output_dir: Optional[str] = None):
        """
        Logs a matplotlib Figure object to MLflow using log_figure.
        """
        output_dir = output_dir or self.output_dir
        filename = f"{artifact_name}.png"
        artifact_file = os.path.join(output_dir, filename)

        mlflow.log_figure(fig, artifact_file)
        self.logger.info(f"🖼️ Logged matplotlib figure: {artifact_file}")
        plt.close(fig)
    
    def _log_dataclass(self, obj, artifact_name: str, output_dir: Optional[str] = None):
        """
        Logs a dataclass as a YAML or JSON artifact to MLflow under a specified subdirectory.

        Args:
            obj: The dataclass instance to log.
            artifact_name: The name of the artifact (filename without extension).
            output_dir: Subdirectory within MLflow artifacts (default: 'config').

        Raises:
            TypeError: If `obj` is not a dataclass.
        """
        output_dir = output_dir or self.output_dir
        if not is_dataclass(obj):
            raise TypeError(f"Expected dataclass instance, got {type(obj)}")

        artifact_file = os.path.join(output_dir, f"{artifact_name}.yaml")

        # Convert dataclass to dictionary and log to MLflow
        mlflow.log_dict(
            dictionary=asdict(obj),
            artifact_file=artifact_file
        )

        self.logger.info(f"📤 Logged dataclass config to MLflow: {artifact_file}")
    
    def _log_pickle(self, obj, artifact_name: str, output_dir: Optional[str] = None):
        output_dir = output_dir or self.output_dir
        filename = f"{artifact_name}.pkl"

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            pickle.dump(obj, temp_file)
            temp_file.flush()

            # Copy to new file with the correct name
            final_path = os.path.join(tempfile.gettempdir(), filename)
            shutil.copy(temp_file.name, final_path)

            mlflow.log_artifact(final_path, artifact_path=output_dir)
            self.logger.info(f"📦 Logged pickle object: {os.path.join(output_dir, filename)}")

    def _log_model(self, model: tf.keras.Model, output_dir: Optional[str] = None, name: Optional[str] = None, sample_input: Optional[tf.Tensor] = None):
        """
        Logs a TensorFlow/Keras model to MLflow.
        
        Args:
            model (tf.keras.Model): The trained model to log.
            output_dir (str): The artifact path under which to log the model.
            name (str): The name of the model.
            sample_input (tf.Tensor): The input example to use for signature inference.
        """
        output_dir = output_dir or self.output_dir
        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"Expected tf.keras.Model, got {type(model)}")
        sample_predictions = model.predict(sample_input)
        signature = mlflow.models.infer_signature(sample_input, sample_predictions)
        self.logger.info(f"📤 Infered signature from the input sample: {signature}")

        mlflow.keras.log_model(model, artifact_path=output_dir, registered_model_name = name, signature=signature)
        self.logger.info(f"📦 Logged Keras model to MLflow under: {output_dir}")

    def _log_model_wrapper(self, python_wrapper_model, output_dir: Optional[str] = None, name: Optional[str] = None, artifacts: Optional[Dict[str, str]] = None, project_root: Optional[str] = None):
        """Log PyFunc wrapper model with automatic artifact and code path detection."""
        from pathlib import Path
        
        # Get artifacts from current run
        current_run = mlflow.active_run()
        if current_run and not artifacts:
            local_config = mlflow.artifacts.download_artifacts(
                run_id=current_run.info.run_id, 
                artifact_path="config/pipeline_config.yaml"
            )
            local_model = mlflow.artifacts.download_artifacts(
                run_id=current_run.info.run_id,
                artifact_path="data/model.keras"
            )
            artifacts = {"pipeline_config": local_config, "model": local_model}
            self.logger.info(f"🔍 Downloaded artifacts: config={local_config}, model={local_model}")
        
        # Find project paths
        if project_root:
            root_path = Path(project_root)
        else:
            # Auto-detect from current directory (notebook-friendly)
            current_dir = Path.cwd()
            for candidate in [current_dir, current_dir.parent, current_dir.parent.parent]:
                if (candidate / 'mlops').exists():
                    root_path = candidate
                    break
            else:
                raise FileNotFoundError(f"Could not find mlops directory from {current_dir}")
        
        mlops_path = str(root_path /  'mlops')
        requirements_path = root_path / "serving_requirements.txt"
        mlops_path_utils = root_path /  'mlops/utils'
        mlops_path_fg = root_path /  'mlops/feature_engineering'
        mlops_path_t = root_path /  'mlops/training'
        
        # Handle python_wrapper_model path resolution
        if python_wrapper_model is None:
            # Default wrapper model path
            wrapper_model_path = root_path / "mlops" / "inference" / "pyfunc_models" / "audio_pyfunc_model.py"
        elif Path(python_wrapper_model).is_absolute():
            # Already absolute path
            wrapper_model_path = Path(python_wrapper_model)
        else:
            # Relative path - resolve from project root
            wrapper_model_path = root_path / python_wrapper_model
        
        self.logger.info(f"📁 Using code path: {mlops_path}")
        self.logger.info(f"🐍 Using wrapper model: {wrapper_model_path}")
        
        mlflow.pyfunc.log_model(
            artifact_path=output_dir or self.output_dir,
            python_model=str(wrapper_model_path),
            registered_model_name=name,
            artifacts=artifacts,
            code_path=[mlops_path],
            #[str(mlops_path_utils),str(mlops_path_fg),str(mlops_path_t)],
            pip_requirements=str(requirements_path) if requirements_path.exists() else None,
            infer_code_paths=False,
            input_example = ["/Volumes/storage/frogid/wav_files/10000.wav"]
        )
        self.logger.info(f"📦 Logged model wrapper to MLflow under: {output_dir}")


    def _log_artifact(self, file_path: str, artifact_name: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Logs a single artifact file to MLflow.

        Args:
            file_path (str): Path to the file on disk to log.
            artifact_name (str, optional): If provided, the file will be renamed in the artifact directory.
            output_dir (str, optional): MLflow artifact directory path to log under.
        """
        output_dir = output_dir or self.output_dir

        if artifact_name:
            # Copy to temp file with the correct name before logging
            tmp_path = os.path.join(tempfile.gettempdir(), artifact_name)
            shutil.copy(file_path, tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path=output_dir)
            self.logger.info(f"📦 Logged artifact as: {os.path.join(output_dir, artifact_name)}")
        else:
            mlflow.log_artifact(file_path, artifact_path=output_dir)
            self.logger.info(f"📦 Logged artifact: {file_path} → {output_dir}")
            

    ###################################################################################
    # PUBLIC LOGGING INTERFACE
    ###################################################################################

    def log_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)
        self.logger.info(f"📊 Logged metrics: {', '.join(metrics.keys())}")

    def log_params(self, params: Dict[str, Union[str, float, int]]):
        mlflow.log_params(params)
        self.logger.info(f"⚙️ Logged params: {', '.join(params.keys())}")