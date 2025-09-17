# mlops/modelling/tf_model_trainer.py
import os
import mlflow
import tensorflow as tf
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.utils.mlflow_utils import MLFlowLogger
from mlops.utils.config import FeatureDataSchema, ModellingConfig, MLFlowConfig
from mlops.reporting.output_generator import OutputGenerator
from mlops.utils.data_utils import convert_to_tf_dataset_with_ids, split_data

class TFModelTrainer:
    def __init__(self, data_schema: FeatureDataSchema, modelling_config: ModellingConfig, mlflow_config: MLFlowConfig):
        # Setup configs
        self.data_schema = data_schema
        self.split_data_config = modelling_config.split_data_config
        self.training_config = modelling_config.training_config
        # Setup datasets
        self.train_ds: Optional[tf.data.Dataset] = None
        self.val_ds: Optional[tf.data.Dataset] = None
        self.test_ds: Optional[tf.data.Dataset] = None
        # Setup model
        self.input_shape = None
        self.num_classes = None
        self.model: Optional[tf.keras.Model] = None
        # Setup logging
        self.logger = setup_logger(name=self.__class__.__name__)
        # Setup the MLFlow instance
        ARTIFACT_SUBDIRECTORY = ""
        self.model_wrapper_path = mlflow_config.model_wrapper_path
        self.log_model_wrapper = mlflow_config.log_model_wrapper

        self.mlflow = MLFlowLogger(output_dir=ARTIFACT_SUBDIRECTORY)

    ###################################################################################
    # CHECKS AND BALANCES
    ###################################################################################
    def _check_required_columns_exist(self, id_col, feature_col, label_col, df: pd.DataFrame):
        # Check the required columns are in the dataframe
        required_cols = { id_col, feature_col, label_col }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"❌ Missing required columns in df_features: {sorted(missing_cols)}")
        
    def _check_required_state_to_build_model(self):
        """
        Raise errors if required steps needed to build the model are not present
        """
        if self.num_classes is None:
            raise RuntimeError("The number of classes in the training data is not defined.")
        
        if self.input_shape is None:
            raise RuntimeError("Model input shape is not defined.")
        
        self.logger.info("✅ Build model prerequisites validated.")

    def _check_required_state_to_train_model(self):
        """
        Raise errors if key steps were skipped before calling train_model.
        """

        # Check that the model exists
        if self.model is None:
            raise RuntimeError("Model has not been built. Call `build_model()` before training.")

        # Check that the training data is ready
        if self.train_ds is None or self.val_ds is None:
            raise RuntimeError("Datasets not prepared. Call `prepare_datasets()` first.")

        # Check that there is a valid number of epochs
        if self.training_config.epochs <= 0:
            raise ValueError("ModelTrainingConfig.epochs must be a positive integer.")

        self.logger.info("✅ Training prerequisites validated.")
    
    ###################################################################################
    # DATASET PREPARATION
    ###################################################################################

    def _prepare_data(self, df_features: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, tf.data.Dataset]]:
        """
        Generate ID splits (train/val/test), ensuring group-aware and optionally stratified sampling.
        """
        # Required columns
        id_col = self.data_schema.id_col
        feature_col = self.data_schema.feature_col
        label_col = self.data_schema.label_col

        self._check_required_columns_exist(id_col, feature_col, label_col, df_features)

        # Split the data
        df_train, df_val, df_test = split_data(
            df_data=df_features,
            id_col=id_col,
            label_col=label_col,
            val_size=self.split_data_config.val_size,
            test_size=self.split_data_config.test_size,
            random_state=self.split_data_config.random_state,
            stratify=self.split_data_config.stratify
        )

        # Store the data frames
        dfs = {
            "train": df_train,
            "val": df_val,
            "test": df_test
        }

        # Save combined ID + split label
        df_all_ids = pd.concat([
            df_train[[id_col]].assign(split="train"),
            df_val[[id_col]].assign(split="val"),
            df_test[[id_col]].assign(split="test"),
        ], axis=0).reset_index(drop=True)

        # Save the ids used for modelling with their split name
        self.mlflow._log_df_to_csv(df_all_ids, artifact_name="modelling_ids")

        # Generate tensorflow datasets
        for name, df_subset in dfs.items():
            # Convert to TensorFlow dataset, shuffling only the training dataset
            shuffle = True if name == "train" else False
            # Exclude IDs from training/validation datasets to avoid XLA compilation issues with string tensors
            include_ids = False if name in ["train", "val"] else True
            tf_dataset = convert_to_tf_dataset_with_ids(
                features=df_subset[feature_col],
                labels=df_subset[label_col],
                ids=df_subset[id_col],
                batch_size=self.data_schema.batch_size,
                shuffle=shuffle,
                include_ids=include_ids
            )
            # Dynamically assign to self.train_ds, self.val_ds, self.test_ds
            setattr(self, f"{name}_ds", tf_dataset)
        
        # Determine the input shape of the data
        # Training dataset has structure (X, y) without IDs to avoid XLA issues
        X_sample, _ = next(iter(self.train_ds))
        self.input_shape = X_sample.shape[1:]

        # Determine the number of classes for the output layer based on the training data
        self.num_classes = dfs['train'][self.data_schema.label_col].nunique()
        self.logger.info(f"📊 Inferred {self.num_classes} unique classes from training data.")

        tf_data = {
            "train": self.train_ds,
            "val": self.val_ds,
            "test": self.test_ds
        }

        # Return the data for the splits
        return dfs, tf_data
    
    ###################################################################################
    # BUILD MODEL USING THE MODEL REGISTRY (CALLABLE FUNCTION)
    ###################################################################################
    def _build_model(self, model_fn: callable) -> tf.keras.Model:
        """
        Build and compile the model using a user-provided function.

        Args:
            model_fn: A function that takes (input_shape, num_classes) and returns a compiled tf.keras.Model.

        Returns:
            A compiled tf.keras.Model instance.
        """
        self._check_required_state_to_build_model()

        self.model = model_fn(self.input_shape, self.num_classes)

        # Show a model summary and save the summary
        OutputGenerator.model_summary_text(
            model=self.model,
            model_fn_desc=model_fn.description,
            mlflow_logger=self.mlflow,
            artifact_name="model_summary",
            print_output=True
        )

        # Return the model
        return self.model

    ###################################################################################
    # TRAIN THE MODEL
    ###################################################################################
    def _train_model(self,python_wrapper_model:Optional = None,name:Optional[str] = None) -> tf.keras.callbacks.History:

        # Perform essential checks before proceeding with the training script
        self._check_required_state_to_train_model()

        # 2. Set up callbacks
        callbacks = []

        # Use a temporary directory for all intermediate files (e.g. checkpoints)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            checkpoint_path = os.path.join(tmp_dir, "best.weights.h5")

            # --------------------------------------------------------------
            # Callbacks: checkpoint + early stopping
            # --------------------------------------------------------------
            if self.training_config.use_model_checkpoint:
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor=self.training_config.monitor,
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ))

            if self.training_config.use_early_stopping:
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor=self.training_config.monitor,
                    patience=self.training_config.patience,
                    restore_best_weights=self.training_config.restore_best_weights,
                    verbose=1
                ))

            # --------------------------------------------------------------
            # Train the model
            # --------------------------------------------------------------
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=self.training_config.epochs,
                callbacks=callbacks,
                verbose=self.training_config.verbose
            )

            self.logger.info("✅ Model training complete.")

            # --------------------------------------------------------------
            # Load best model if required
            # --------------------------------------------------------------
            if self.training_config.use_model_checkpoint and self.training_config.auto_load_best_weights_after_training:
                if os.path.exists(checkpoint_path):
                    self.logger.info("🔍 Updating the model with the best model weights from the saved checkpoint.")
                    self.model.load_weights(checkpoint_path)
                    self.logger.info("✅ Loaded best model weights from checkpoint.")
                    self.mlflow._log_artifact(str(checkpoint_path), artifact_name="best.weights.h5")
                    self.logger.info("📦 Logged best model weights to MLflow → training/best.weights.h5")
                else:
                    self.logger.info("⚠️ You selected to auto_load_best_weights_after_training, but the path to the checkpoint could not be found.")
            
            # Generate the training curves plot and save the plots
            OutputGenerator.training_curves_plot(
                history=history,
                artifact_name="training_curves",
                mlflow_logger=self.mlflow,
                print_output=True
            )

            # From the training data, retrieve a single example with batch dimension
            # Training dataset has structure (X, y) without IDs
            X_batch, _ = next(iter(self.train_ds))
            input_example = X_batch[:1].numpy()
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                np.save(f, input_example)
                f.flush()
                self.mlflow._log_artifact(f.name, artifact_name="input_example.npy")
            
            # Save the model and its history with the sample input example
            self.mlflow._log_pickle(history.history, artifact_name="training_history")
            self.mlflow._log_model(self.model, name = name, sample_input = input_example)
            
            # End the current run to ensure artifacts are committed
            current_run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
            self.mlflow.logger.info(f"🔄 Ended run {current_run_id} after model logging")
            
            # Reactivate the run for wrapper logging
            mlflow.start_run(run_id=current_run_id)
            self.mlflow.logger.info(f"🔄 Reactivated run {current_run_id} for wrapper logging")
            
            # Conditionally log model wrapper based on configuration
            if self.log_model_wrapper:
                # Try to get ROOT_DIR from the calling environment (notebook)
                import sys
                project_root = None
                if 'ROOT_DIR' in globals():
                    project_root = str(globals()['ROOT_DIR'])
                elif hasattr(sys.modules.get('__main__', None), 'ROOT_DIR'):
                    project_root = str(sys.modules['__main__'].ROOT_DIR)
                
                self.mlflow._log_model_wrapper(
                    python_wrapper_model=python_wrapper_model or self.model_wrapper_path, 
                    output_dir="wrapper",
                    name=f"{name}_wrapper",
                    project_root=project_root
                )

            # Log training metrics to MLflow
            training_metrics = {f"final_{k}": v[-1] for k, v in history.history.items()}
            self.mlflow.log_metrics(metrics=training_metrics)
        
            return history
    
    ###################################################################################
    # SCRIPT RUNNER - UTILITY ENTRY POINT
    ###################################################################################

    def train(
        self,
        df: pd.DataFrame,
        model_fn: callable,
        python_wrapper_model: Optional = None,
        name: Optional[str] = None
    ) -> tf.keras.Model:
        self.logger.info("🏋️‍♀️ Training a new model ...")
        # Prepare the data splits
        self._prepare_data(df)
        # Build the model
        self._build_model(model_fn)
        # Train the model
        self._train_model(python_wrapper_model, name)

        return self.model


            
        
            
    
    