# Core packages
# Core packages
import os
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from datetime import datetime
import mlflow
# ML Libraries
from datetime import datetime
import mlflow
# ML Libraries
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
# Utilities
from mlops.utils.logging_utils import setup_logger
from mlops.utils.mlflow_utils import MLFlowLogger
from mlops.reporting.output_generator import OutputGenerator
from mlops.utils.data_utils import convert_to_tf_dataset_with_ids
from mlops.feature_engineering.preprocessor_orchestrator import PreprocessorOrchestrator
from mlops.utils.config import FeatureDataSchema, EvaluationConfig

class TFModelEvaluator:
    def __init__(self, data_schema: FeatureDataSchema, evaluation_config: EvaluationConfig):
        # Define the data schema
        self.data_schema = data_schema
        # Define Evaluation config
        self.evaluation_config = evaluation_config
        # Setup logging
        self.logger = setup_logger(name=self.__class__.__name__)
        # Setup the MLFlow instance
        self.artifact_path = "evaluation"
        self.mlflow = MLFlowLogger(output_dir=self.artifact_path)

    ###################################################################################
    # EVALUATE THE MODEL
    ###################################################################################
    def _predict_and_aggregate(
        self,
        model: tf.keras.Model,
        chunk_features: np.ndarray,
        recording_ids: List[str],
        chunk_labels: Optional[np.ndarray] = None,
        pooling_strategy: str = "mean",
        k: Optional[int] = 3
    ) -> pd.DataFrame:
        """
        Generic prediction and aggregation method for both labeled and unlabeled data.
        
        Args:
            model: Trained model
            chunk_features: Feature array (n_chunks, n_features)  
            recording_ids: Recording ID for each chunk
            chunk_labels: True labels for each chunk (optional, for evaluation)
            pooling_strategy: "mean", "max", or "voting"
            
        Returns:
            DataFrame with recording-level predictions and probabilities
        """
        if model is None:
            raise RuntimeError("Model not found. Train or load a model first.")

        self.logger.info(f"🔮 Predicting {len(chunk_features)} chunks across {len(set(recording_ids))} recordings")

        # ----------------------------------------
        # Step 1: Predict on all chunks
        # ----------------------------------------
        y_probs = model.predict(chunk_features, batch_size=self.data_schema.batch_size, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)
        

        # ----------------------------------------
        # Step 2: Construct chunk-level DataFrame
        # ----------------------------------------
        chunk_data = {
            "recording_id": recording_ids,
            "y_pred": y_pred,
            **{f"prob_{i}": y_probs[:, i] for i in range(y_probs.shape[1])}
        }
        
        # Add labels if provided (for evaluation)
        if chunk_labels is not None:
            chunk_data["y_true"] = chunk_labels
            
        df_chunks = pd.DataFrame(chunk_data)

        # ----------------------------------------
        # Step 3: Aggregate by recording ID
        # ----------------------------------------
        
        def aggregate_predictions(group: pd.DataFrame) -> pd.Series:
            """Aggregate chunk predictions to recording level"""
            chunk_probs = group[[col for col in group.columns if col.startswith("prob_")]].values
            chunk_preds = group["y_pred"].values
            
            # Apply aggregation strategy
            if pooling_strategy == "mean":
                final_probs = chunk_probs.mean(axis=0)
                final_pred = np.argmax(final_probs)
                confidence = final_probs[final_pred]
                
            elif pooling_strategy == "max":
                # Take chunk with highest confidence
                chunk_confidences = np.max(chunk_probs, axis=1)
                best_chunk_idx = np.argmax(chunk_confidences)
                final_probs = chunk_probs[best_chunk_idx]
                final_pred = chunk_preds[best_chunk_idx]
                confidence = chunk_confidences[best_chunk_idx]
                
            elif pooling_strategy == "voting":
                # Majority vote
                from collections import Counter
                votes = Counter(chunk_preds)
                final_pred = votes.most_common(1)[0][0]
                final_probs = chunk_probs.mean(axis=0)
                confidence = final_probs[final_pred]

            elif pooling_strategy == "topk":
                # Use top-k chunks based on confidence
                chunk_confidences = np.max(chunk_probs, axis=1)
                top_k_indices = np.argsort(chunk_confidences)[-k:]
                top_k_probs = chunk_probs[top_k_indices]
                final_probs = top_k_probs.mean(axis=0)
                final_pred = np.argmax(final_probs)
                confidence = final_probs[final_pred]
                
            elif pooling_strategy == "softmax":
                # Apply softmax weighting based on chunk confidences
                chunk_confidences = np.max(chunk_probs, axis=1)
                weights = np.exp(chunk_confidences) / np.sum(np.exp(chunk_confidences))
                final_probs = np.average(chunk_probs, axis=0, weights=weights)
                final_pred = np.argmax(final_probs)
                confidence = final_probs[final_pred]
                
            else:
                raise ValueError(f"Unknown aggregation method: {pooling_strategy}")
            
            result = {
                "y_pred": final_pred,
                "confidence": confidence,
                "num_chunks": len(group),
                **{f"prob_{i}": prob for i, prob in enumerate(final_probs)}
            }
            
            # Handle labels for evaluation
            if "y_true" in group.columns:
                true_label = group["y_true"].iloc[0]  # All chunks from same recording have same label
                # For evaluation, use the aggregation strategy that was historically used
                if pooling_strategy == "mean":
                    # Use original logic: if true label appears in predictions, use it; otherwise use first prediction
                    final_pred_for_eval = true_label if true_label in chunk_preds else chunk_preds[0]
                else:
                    # For other methods, use the aggregated prediction
                    final_pred_for_eval = final_pred
                    
                result.update({
                    "y_true": true_label,
                    "y_pred": final_pred_for_eval  # Override for evaluation consistency
                })
            
            return pd.Series(result)

        # Aggregate by recording ID
        df_recordings = (
            df_chunks.groupby("recording_id", group_keys=False)
            .apply(aggregate_predictions)
            .reset_index()
            .sort_values("recording_id")
            .reset_index(drop=True)
        )
        
        self.logger.info(f"✅ Aggregated to {len(df_recordings)} recording-level predictions")
        
        return df_recordings

    def predict(
        self,
        model: tf.keras.Model,
        features: np.ndarray,
        pooling_strategy: str,
        k: Optional[int],
        class_label_to_species_mapping: Dict[int, str],
        recording_ids: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,

        confidence_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Unified prediction method for both inference and evaluation.
        
        Args:
            model: Trained model
            features: Feature array (n_chunks, n_features)
            class_label_to_species_mapping: Class to species mapping
            recording_ids: Recording ID for each chunk (optional, creates dummy IDs if None)
            labels: True labels for each chunk (optional, for evaluation)
            pooling_strategy: How to aggregate chunks ("mean", "max", "voting")
            confidence_threshold: Minimum confidence threshold (for inference)
            
        Returns:
            Dictionary with:
            - For inference: {"predicted_species", "confidence", "prob_dict", etc.}
            - For evaluation: {"recording_predictions", "y_true", "y_pred", etc.}
        """
        pooling_strategy = pooling_strategy or self.evaluation_config.pooling_strategy
        k = k or self.evaluation_config.k

        if model is None:
            raise RuntimeError("Model not found. Train or load a model first.")
            
        if features is None or len(features) == 0:
            raise ValueError("Features array is empty or None")

        # Create recording IDs if not provided
        if recording_ids is None:
            if labels is not None:
                # For evaluation without explicit IDs, create dummy IDs
                unique_labels = len(set(labels)) if len(labels) > 0 else 1
                recording_ids = [f"record_{i}" for i in range(len(features))]
                self.logger.info(f"🔍 Evaluation mode: {len(features)} chunks, estimated {unique_labels} unique recordings")
            else:
                # For inference, treat as single recording
                recording_ids = ["inference"] * len(features)
                self.logger.info(f"🔮 Inference mode: {len(features)} chunks from single recording")
        else:
            is_evaluation = labels is not None
            mode = "Evaluation" if is_evaluation else "Inference"
            self.logger.info(f"🔍 {mode} mode: {len(features)} chunks across {len(set(recording_ids))} recordings")

        # ----------------------------------------
        # Use the unified prediction and aggregation method
        # ----------------------------------------
        df_recordings = self._predict_and_aggregate(
            model=model,
            chunk_features=features,
            recording_ids=recording_ids,
            chunk_labels=labels,
            pooling_strategy=pooling_strategy,
            k=k
        )

        # ----------------------------------------
        # Format results based on whether we have labels (evaluation vs inference)
        # ----------------------------------------
        if labels is not None:
            # EVALUATION MODE: Return structured evaluation results
            recording_y_true = df_recordings["y_true"].astype(int).values
            recording_y_pred = df_recordings["y_pred"].astype(int).values
            
            prob_cols = [col for col in df_recordings.columns if col.startswith("prob_")]
            recording_y_probs = df_recordings[prob_cols].values
            
            recording_y_true_binarized = label_binarize(
                recording_y_true, 
                classes=sorted(class_label_to_species_mapping.keys())
            )

            # Add species names for readability
            df_enriched = df_recordings.copy()
            df_enriched["predicted_species"] = df_enriched["y_pred"].map(class_label_to_species_mapping)
            df_enriched["true_species"] = df_enriched["y_true"].map(class_label_to_species_mapping)
            
            return {
                "recording_predictions": df_enriched,
                "y_true": recording_y_true,
                "y_pred": recording_y_pred, 
                "y_probs": recording_y_probs,
                "y_true_binarized": recording_y_true_binarized,
                "pooling_strategy": pooling_strategy,
                "num_recordings": len(df_recordings)
            }
        
        else:
            # INFERENCE MODE: Return inference-style results
            if len(df_recordings) == 1:
                # Single recording inference
                row = df_recordings.iloc[0]
                final_class = int(row["y_pred"])
                confidence = float(row["confidence"])
                num_chunks = int(row["num_chunks"])
                
                species_name = class_label_to_species_mapping.get(final_class, f"Unknown_{final_class}")
                
                if confidence >= confidence_threshold:
                    prob_dict = {}
                    prob_cols = [col for col in row.index if col.startswith("prob_")]
                    for i, col in enumerate(prob_cols):
                        species = class_label_to_species_mapping.get(i, f"class_{i}")
                        prob_dict[species] = float(row[col])
                    
                    result = {
                        "predicted_species": species_name,
                        "predicted_class": final_class,
                        "confidence": confidence,
                        "num_chunks": num_chunks,
                        "pooling_strategy": pooling_strategy,
                        "prob_dict": prob_dict
                    }
                    
                    self.logger.info(f"🎯 Prediction: {species_name} (confidence: {confidence:.3f})")
                else:
                    result = {
                        "predicted_species": "LOW_CONFIDENCE",
                        "predicted_class": -1,
                        "confidence": confidence,
                        "num_chunks": num_chunks,
                        "pooling_strategy": pooling_strategy,
                    }
                    self.logger.warning(f"⚠️ Low confidence prediction: {confidence:.3f} < {confidence_threshold}")
                
                return result
            
            else:
                # Multiple recordings inference - return DataFrame
                results = []
                for _, row in df_recordings.iterrows():
                    recording_id = row["recording_id"]
                    final_class = int(row["y_pred"])
                    confidence = float(row["confidence"])
                    num_chunks = int(row["num_chunks"])
                    species_name = class_label_to_species_mapping.get(final_class, f"Unknown_{final_class}")
                    
                    result_item = {
                        "recording_id": recording_id,
                        "predicted_species": species_name,
                        "predicted_class": final_class,
                        "confidence": confidence,
                        "num_chunks": num_chunks,
                        "pooling_strategy": pooling_strategy,
                    }
                    results.append(result_item)
                
                return {"predictions": results, "num_recordings": len(results)}


    
    ###################################################################################
    # RESULTS / REPORT
    ###################################################################################
    def calculate_results(
        self,
        y_true: np.ndarray,
        y_true_binarized: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        class_label_to_species_mapping: Dict[int, str]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        
        # --------------------------------------------------------------------
        # Global Metrics (across all species, macro/weighted averages)
        # --------------------------------------------------------------------
        global_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

        # --------------------------------------------------------------------
        # Per-Species Metrics
        # --------------------------------------------------------------------
        per_species_metrics = {}
        class_indices = sorted(class_label_to_species_mapping.keys())

        # Metrics based on y_true/y_pred (not probabilities)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_indices, zero_division=0
        )

        for i, class_idx in enumerate(class_indices):
            species = class_label_to_species_mapping[class_idx]
            per_species_metrics[species] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": float(support[i])
            }

        # --------------------------------------------------------------------
        # AUC Metrics (macro and per-species)
        # --------------------------------------------------------------------
        try:
            global_metrics["roc_auc_macro"] = roc_auc_score(
                y_true_binarized, y_probs, average="macro", multi_class="ovr"
            )
            global_metrics["pr_auc_macro"] = average_precision_score(
                y_true_binarized, y_probs, average="macro"
            )

            # Map class_idx → column index in y_probs and y_true_binarized
            # The arrays are indexed from 0 to n_classes-1, so we need to map accordingly
            n_classes = y_probs.shape[1]
            expected_classes = list(range(n_classes))
            
            # Create a mapping from class_idx to column index
            # This ensures we only access valid column indices
            class_idx_to_column = {}
            for class_idx in class_indices:
                if class_idx in expected_classes:
                    class_idx_to_column[class_idx] = class_idx
                else:
                    self.logger.warning(f"⚠️ Class index {class_idx} not found in model output (expected 0-{n_classes-1})")

            for class_idx in class_indices:
                if class_idx not in class_idx_to_column:
                    continue
                    
                species = class_label_to_species_mapping[class_idx]
                i = class_idx_to_column[class_idx]

                try:
                    per_species_metrics[species]["roc_auc"] = roc_auc_score(
                        y_true_binarized[:, i], y_probs[:, i]
                    )
                    per_species_metrics[species]["pr_auc"] = average_precision_score(
                        y_true_binarized[:, i], y_probs[:, i]
                    )
                except ValueError:
                    per_species_metrics[species]["roc_auc"] = None
                    per_species_metrics[species]["pr_auc"] = None

        except ValueError:
            self.logger.warning("⚠️ Skipping macro AUC logging due to shape mismatch or invalid labels.")

        return global_metrics, per_species_metrics


    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_true_bin: np.ndarray,
        y_probs: np.ndarray,
        class_label_to_species_mapping: Dict[int, str],
        dataset_name: str,
        mlflow_logger: Optional[MLFlowLogger] = None
    ):
        """
        Generates evaluation plots and metrics for a given dataset.
        """
        # Infer present classes from predictions
        present_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

        OutputGenerator.classification_report_table(
            y_true, y_pred, class_label_to_species_mapping,
            mlflow_logger=mlflow_logger,
            artifact_name="classification_report",
            present_classes=present_classes,
            print_output=True
        )

        OutputGenerator.plot_roc_auc_curves(
            y_true_bin, y_probs, class_label_to_species_mapping,
            title=f"ROC AUC per Species: {dataset_name}",
            mlflow_logger=mlflow_logger,
            artifact_name="roc_auc",
            print_output=True
        )

        OutputGenerator.plot_precision_recall_curves(
            y_true_bin, y_probs, class_label_to_species_mapping,
            title=f"PR Curve per Species: {dataset_name}",
            mlflow_logger=mlflow_logger,
            artifact_name="precision_recall_curve",
            print_output=True
        )

        OutputGenerator.plot_confusion_matrix(
            y_true, y_pred, class_label_to_species_mapping,
            normalize=False,
            title=f"Confusion Matrix {dataset_name}",
            mlflow_logger=mlflow_logger,
            artifact_name="confusion_matrix_raw",
            print_output=True
        )

        OutputGenerator.plot_confusion_matrix(
            y_true, y_pred, class_label_to_species_mapping,
            normalize=True,
            title=f"Confusion Matrix Normalized {dataset_name}",
            mlflow_logger=mlflow_logger,
            artifact_name="confusion_matrix_normalized",
            print_output=True
        )

    ###################################################################################
    # EVALUATE
    ###################################################################################

    def load_model_from_mlflow(self, run_id: str) -> tf.keras.Model:
        try:
            model_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="data/model.keras"
            )
            print(f"📥 Downloaded model from Databricks to local path: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

            model = tf.keras.models.load_model(model_path)
            return model

        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model from run_id='{run_id}': {str(e)}")
            
    def load_model_from_mlflow_registry(self, model_name: str, model_version_alias: str|int) -> tf.keras.Model:
        # if model_version_alias is a number, use version, otherwise use alias
        if model_version_alias.isdigit():
            model_uri = f"models:/{model_name}/{model_version_alias}"
        else:
            model_uri = f"models:/{model_name}@{model_version_alias}"
        model = mlflow.keras.load_model(model_uri)
        return model

    def evaluate(
        self,
        run_id: str,
        df_features: pd.DataFrame,
        class_label_to_species_mapping: Dict[int, str],
        dir_name_to_store_results: Optional[str] = None,
    ):
        """
        Evaluate model performance on labeled dataset.
        
        Args:
            run_id: MLflow run ID to load model from
            df_features: DataFrame with features, labels, and IDs
            class_label_to_species_mapping: Class to species mapping
            dir_name_to_store_results: Directory name for storing results
        """
        # ----------------------------------------
        # Step 1: Load the trained model from MLflow
        # ----------------------------------------
        model = self.load_model_from_mlflow(run_id)
        pooling_strategy = self.evaluation_config.pooling_strategy
        k = self.evaluation_config.k
        
        # Log model information for debugging
        self.logger.info(f"🔍 Model loaded successfully")
        self.logger.info(f"🔍 Model output shape: {model.output_shape}")
        if hasattr(model, 'layers') and len(model.layers) > 0:
            last_layer = model.layers[-1]
            self.logger.info(f"🔍 Last layer: {last_layer.__class__.__name__}")
            if hasattr(last_layer, 'units'):
                self.logger.info(f"🔍 Last layer units: {last_layer.units}")

        # ----------------------------------------
        # Step 2: Extract data from DataFrame  
        # ----------------------------------------
        # Convert features to numpy array
        features_array = np.array(df_features[self.data_schema.feature_col].tolist())
        labels = df_features[self.data_schema.label_col].values
        recording_ids = df_features[self.data_schema.id_col].values.astype(str)
        
        self.logger.info(f"📊 Evaluating {len(features_array)} chunks from {len(set(recording_ids))} recordings")
        
        # ----------------------------------------
        # Step 3: Run predictions using unified method
        # ----------------------------------------
        results = self.predict(
            model=model,
            features=features_array,
            class_label_to_species_mapping=class_label_to_species_mapping,
            recording_ids=recording_ids,
            labels=labels,
            pooling_strategy=pooling_strategy,
            k=k
        )
        
        # Extract arrays for metrics calculation
        y_true = results["y_true"]
        y_true_binarized = results["y_true_binarized"] 
        y_pred = results["y_pred"]
        y_probs = results["y_probs"]
        
        # ----------------------------------------
        # Step 4: Calculate and log metrics
        # ----------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        EVALUATION_NAME = f"{dir_name_to_store_results}_{timestamp}"
        MLFLOW_OUTPUT_PATH = os.path.join(self.artifact_path, EVALUATION_NAME)
        mlflow_logger = MLFlowLogger(output_dir=MLFLOW_OUTPUT_PATH)

        macro_results, per_species_results = self.calculate_results(
            y_true, y_true_binarized, y_pred, y_probs, class_label_to_species_mapping
        )

        mlflow_logger.log_metrics(macro_results)

        self.logger.info(f"✅ {EVALUATION_NAME}: Evaluation complete for {results['num_recordings']} recordings using {pooling_strategy} aggregation")

        # ----------------------------------------
        # Step 5: Generate evaluation report
        # ----------------------------------------
        self.generate_report(
            y_true=y_true,
            y_true_bin=y_true_binarized,
            y_pred=y_pred,
            y_probs=y_probs,
            class_label_to_species_mapping=class_label_to_species_mapping,
            dataset_name=EVALUATION_NAME,
            mlflow_logger=mlflow_logger
        )

        return y_true, y_true_binarized, y_pred, y_probs, macro_results, per_species_results