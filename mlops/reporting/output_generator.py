# mlops/reporting/output_generator.py

from typing import Optional, Union, List, Tuple, Dict
from tabulate import tabulate
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.mlflow_utils import MLFlowLogger
import pandas as pd
import io
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from keras.utils import model_to_dot
import tensorflow as tf
import numpy as np
from IPython.display import display
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.utils.multiclass import unique_labels

class OutputGenerator:
    @staticmethod
    def _get_table_header(title: str) -> str:
        """
        Create a standardized table header for reports.
        """
        print("")
        bar = '=' * 70
        return f"{bar}\n📊 {title.upper()}\n{bar}\n"
    
    @staticmethod
    def dataset_description(
        column_descriptions: dict,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: str = "dataset_description",
        print_output: bool = True
    ) -> str:
        """
        Generate a textual report describing dataset columns.
        """
        header = OutputGenerator._get_table_header("Dataset Description")
        table = [(col, desc) for col, desc in column_descriptions.items()]
        body = tabulate(table, headers=["Column Name","Description"], tablefmt="psql")
        report = header + body + '\n'

        if print_output:
            print(report)

        if mlflow_logger:
            mlflow_logger._log_text(text=report, artifact_name=artifact_name)

        return report
    
    @staticmethod
    def grouped_distribution_report(
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        count_col_name: str = "total_count",
        title: str = "Grouped Distribution",
        sort_by: Optional[Union[str, List[str]]] = None,
        include_total_row: bool = True,
        print_output: bool = True,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Enhanced group summary with columns for total, single-species, and multi-species counts.

        Args:
            df: Input dataframe
            group_by: Column or list of columns to group by
            count_col_name: Name of the column for total count
            title: Title for the printed/logged table
            sort_by: Optional columns to sort by
            include_total_row: Whether to append a final TOTAL row
            print_output: Whether to print the result
            mlflow_logger: Optional MLFlow logger
            artifact_name: Optional name for MLflow text artifact

        Returns:
            - Summary DataFrame
            - Pretty-formatted table string
        """
        group_by = [group_by] if isinstance(group_by, str) else group_by

        # Safeguard for empty input
        if df.empty:
            warning = f"⚠️ Skipping report: input DataFrame is empty."
            if print_output:
                print(warning)

            if mlflow_logger and artifact_name:
                mlflow_logger._log_text(warning, artifact_name=artifact_name)

            return pd.DataFrame(), warning

        # Group by group_by + is_multi_species and count
        # Group by and count
        grouped = (
            df.groupby(group_by + ['is_multi_species'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        # Always create the expected columns, even if missing
        grouped["single_species"] = grouped.get(False, 0)
        grouped["multi_species"] = grouped.get(True, 0)

        # Drop raw boolean columns if they exist (optional cleanup)
        grouped = grouped.drop(columns=[col for col in [False, True] if col in grouped.columns])

        # Compute total count
        grouped[count_col_name] = grouped["single_species"] + grouped["multi_species"]

        # Optional sort
        if sort_by:
            grouped = grouped.sort_values(by=sort_by)

        # ---------------------------------------------------------
        # Compute number of unique species per group
        # ---------------------------------------------------------
        if "species_names" in df.columns:
            exploded_species = df[group_by + ['species_names']].explode('species_names')
            unique_species_counts = (
                exploded_species
                .groupby(group_by)['species_names']
                .nunique()
                .reset_index()
                .rename(columns={'species_names': 'unique_species'})
            )

            # Merge back into grouped summary
            grouped = pd.merge(grouped, unique_species_counts, on=group_by, how='left')
            
            # Reorder columns: move `unique_species` just after group_by
            grouped_columns = grouped.columns.tolist()
            group_by = [group_by] if isinstance(group_by, str) else group_by
            new_order = (
                group_by
                + ['unique_species']
                + [col for col in grouped_columns if col not in group_by + ['unique_species']]
            )
            grouped = grouped[new_order]

        else:
            grouped['unique_species'] = None

        # Add TOTAL row if required
        if include_total_row:
            total_row = {col: '' for col in grouped.columns}
            total_row[group_by[0]] = 'TOTAL'
            total_row["single_species"] = grouped["single_species"].sum()
            total_row["multi_species"] = grouped["multi_species"].sum()
            total_row[count_col_name] = grouped[count_col_name].sum()
            total_row["unique_species"] = (
                df["species_names"].explode().nunique() if "species_names" in df.columns else ''
            )
            grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)

        # Format output
        header = OutputGenerator._get_table_header(title)
        body = tabulate(grouped, headers="keys", tablefmt="psql", showindex=False)
        report = header + body + "\n"

        if print_output:
            print(report)

        if mlflow_logger and artifact_name:
            mlflow_logger._log_text(report, artifact_name=artifact_name)

        return grouped, report

    @staticmethod
    def downloader_results(
        required_count: int,
        available_count: int,
        missing_from_directory_count: int,
        downloaded_count: int,
        failed_to_download_count: int,
        missing_after_download_count: int,
        ready_for_use_count: int,
        has_complete_data: bool,
        # Logging
        print_output: bool = True,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: Optional[str] = None,
    ) -> str:
        """
        Generate a textual report for audio download results.
        """
        title = f"Audio Download Summary"
        header = OutputGenerator._get_table_header(title)
        
        table = [
            ("‼️ Required recordings", required_count),
            ("⚠️ Available recordings", available_count),
            ("🔍 Missing from dir", missing_from_directory_count),
            ("✅ Downloaded", downloaded_count),
            ("❌ Failed to download", failed_to_download_count),
            ("⚠️ Missing after download", missing_after_download_count),
            ("🧪 Ready for modelling", ready_for_use_count),
            ("📦 Complete?", "✅ YES" if has_complete_data else "❌ NO"),
        ]
        
        body = tabulate(table, headers=["Description", "Count"], tablefmt="psql")
        report = header + body + '\n'

        if print_output:
            print(report)

        if mlflow_logger and artifact_name:
            mlflow_logger._log_text(report, artifact_name=artifact_name)

        return report
    
    @staticmethod
    def preprocessing_summary(
        total_input: int,
        failed_count: int,
        skipped_cached_count: int,
        final_output: int,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: Optional[str] = "preprocessing_summary",
        print_output: bool = True
    ) -> str:
        """
        Generate a textual report summarizing the preprocessing step.
        """
        title = "Preprocessing Summary"
        header = OutputGenerator._get_table_header(title)

        table = [
            ("🔢 Input rows", total_input),
            ("❌ Failed files", failed_count),
            ("🧊 Skipped (cached)", skipped_cached_count),
            ("✅ Output rows", final_output)
        ]

        body = tabulate(table, headers=["Description", "Count"], tablefmt="psql")
        report = header + body + '\n'

        if print_output:
            print(report)

        if mlflow_logger:
            mlflow_logger._log_text(text=report, artifact_name=artifact_name)

        return report
    
    @staticmethod
    def model_summary_text(
        model: tf.keras.Model,
        model_fn_desc: Optional[str] = None,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: str = "model_summary",
        print_output: bool = True
    ) -> str:
        """
        Generate and optionally log the text summary of a model.

        Args:
            model: tf.keras.Model instance.
            mlflow_logger: Optional logger to log summary to MLflow.
            artifact_name: MLflow artifact name.
            print_output: Whether to print summary to console.

        Returns:
            The model summary as a string.
        """
        buffer = io.StringIO()
        model.summary(print_fn=lambda line: buffer.write(line + "\n"))
        summary_text = buffer.getvalue()

        header = OutputGenerator._get_table_header("Model Summary")
        full_report = header + summary_text

        if model_fn_desc:
            full_report = full_report + model_fn_desc

        if print_output:
            print(full_report)

        if mlflow_logger:
            mlflow_logger._log_text(text=full_report, artifact_name=artifact_name)

        return full_report

    @staticmethod
    def model_architecture_plot(
        model: tf.keras.Model,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: str = "model_architecture",
        print_output: bool = True
    ) -> Optional[PILImage.Image]:
        """
        Generate and optionally log a model architecture diagram.

        Args:
            model: tf.keras.Model instance.
            mlflow_logger: Optional logger to log plot to MLflow.
            artifact_name: MLflow artifact name.
            print_output: Whether to display the image inline.

        Returns:
            The PIL Image object of the architecture plot.
        """
        try:
            dot = model_to_dot(
                model,
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
                dpi=96
            )
            png_bytes = dot.create(format="png", prog="dot")
            img = PILImage.open(io.BytesIO(png_bytes))

            if mlflow_logger:
                mlflow_logger._log_figure(img, artifact_name=artifact_name)

            if print_output:
                arr = np.array(img)
                plt.imshow(arr)
                plt.axis('off')
                plt.show()

            return img

        except Exception as e:
            print(f"⚠️ Failed to generate model architecture diagram: {e}")
            return None
        
    @staticmethod
    def training_curves_plot(
        history: tf.keras.callbacks.History,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: str = "training_curves",
        print_output: bool = True
    ) -> None:
        """
        Plot training and validation loss/accuracy curves from a History object.

        Args:
            history: Keras training History object.
            mlflow_logger: Optional MLFlowLogger instance.
            artifact_name: Name of the logged image in MLflow.
            print_output: Whether to display the plot inline.
        """
        history_dict = history.history
        epochs = range(1, len(history_dict["loss"]) + 1)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Loss
        axs[0].plot(epochs, history_dict["loss"], label="Train Loss")
        axs[0].plot(epochs, history_dict.get("val_loss", []), label="Val Loss")
        axs[0].set_title("Loss over Epochs")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        # Plot Accuracy (if available)
        acc_keys = [k for k in history_dict if "accuracy" in k and not k.startswith("val_")]
        if acc_keys:
            train_acc_key = acc_keys[0]
            val_acc_key = f"val_{train_acc_key}" if f"val_{train_acc_key}" in history_dict else None

            axs[1].plot(epochs, history_dict[train_acc_key], label="Train Accuracy")
            if val_acc_key:
                axs[1].plot(epochs, history_dict[val_acc_key], label="Val Accuracy")
            axs[1].set_title("Accuracy over Epochs")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy")
            axs[1].legend()
        else:
            axs[1].axis("off")
            axs[1].set_title("No accuracy metrics found")

        plt.tight_layout()

        # Log to MLflow
        if mlflow_logger:
            mlflow_logger._log_matplotlib_figure(fig, artifact_name=artifact_name)

        if print_output:
            display(fig) 
        else:
            plt.close(fig)

    @staticmethod
    def evaluation_dataset_description(
        dataset_name: str,
        dataset_desc: str,
        # Logging
        print_output: bool = True,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: Optional[str] = None,
    ) -> str:
        """
        Generate a textual report for audio download results.
        """
        title = f"Evaluation Data Summary"
        header = OutputGenerator._get_table_header(title)
        
        table = [
            ("🔍 Dataset Name", dataset_name),
            ("📝 Dataset Desc", dataset_desc)
        ]
        
        body = tabulate(table, headers=["Context", "Description"], tablefmt="psql")
        report = '\n' + header + body + '\n'

        if print_output:
            print(report)

        if mlflow_logger and artifact_name:
            mlflow_logger._log_text(report, artifact_name=artifact_name)

        return report
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_label_to_species_mapping: Optional[Dict[int, str]] = None,
        normalize: bool = True,
        title: str = "Confusion Matrix",
        print_output: bool = True,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: str = "confusion_matrix"
    ) -> plt.Figure:
        """
        Plot and optionally log a confusion matrix heatmap.

        Args:
            y_true: Ground truth class labels (1D array)
            y_pred: Predicted class labels (1D array)
            class_label_to_species_mapping: Optional mapping of class labels to species names
            normalize: If True, normalize the matrix row-wise
            title: Plot title
            print_output: Whether to display inline
            mlflow_logger: Optional MLFlowLogger instance
            artifact_name: Artifact name for MLflow logging

        Returns:
            Matplotlib Figure object
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        # Prepare axis labels
        if class_label_to_species_mapping:
            labels = [class_label_to_species_mapping.get(i, f"Class {i}") for i in sorted(set(y_true) | set(y_pred))]
        else:
            labels = [f"Class {i}" for i in range(cm.shape[0])]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, cbar=True, ax=ax)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Log to MLflow
        if mlflow_logger:
            mlflow_logger._log_matplotlib_figure(fig, artifact_name=artifact_name)

        if print_output:
            display(fig)
        else:
            plt.close(fig)

        return fig
    
    @staticmethod
    def classification_report_table(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_label_to_species_mapping: Optional[Dict[int, str]] = None,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: str = "classification_report",
        print_output: bool = True,
        present_classes: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate a classification report as a pandas DataFrame and log to MLflow.
        """
        from sklearn.metrics import classification_report

        # Optional species name mapping
        if class_label_to_species_mapping and present_classes:
            target_names = [class_label_to_species_mapping[i] for i in present_classes]
        else:
            target_names = None

        # Generate report dict
        report_dict = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=present_classes,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )

        # Convert to DataFrame and round
        report_df = pd.DataFrame(report_dict).transpose().round(3)

        # Optionally rename index from class IDs to species names
        if class_label_to_species_mapping:
            def map_index(idx):
                try:
                    return class_label_to_species_mapping.get(int(idx), idx)
                except (ValueError, TypeError):
                    return idx
            report_df.index = [map_index(idx) for idx in report_df.index]

        # Log to MLflow
        if mlflow_logger:
            mlflow_logger._log_df_to_csv(
                df=report_df.reset_index(),
                artifact_name=artifact_name
            )

        if print_output:
            display(report_df)

        return report_df

    @staticmethod
    def filter_unseen_classes(
        y_true_bin: np.ndarray,
        y_probs: np.ndarray,
        class_label_to_species_mapping: Optional[Dict[int, str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, str]], List[int]]:
        """
        Filters out classes not present in y_true_bin (i.e., support=0).
        """
        present_class_indices = np.where(y_true_bin.sum(axis=0) > 0)[0].tolist()

        y_true_bin_filtered = y_true_bin[:, present_class_indices]
        y_probs_filtered = y_probs[:, present_class_indices]

        if class_label_to_species_mapping:
            filtered_mapping = {
                new_idx: class_label_to_species_mapping[old_idx]
                for new_idx, old_idx in enumerate(present_class_indices)
                if old_idx in class_label_to_species_mapping
            }
        else:
            filtered_mapping = None

        return y_true_bin_filtered, y_probs_filtered, filtered_mapping, present_class_indices
    
    @staticmethod
    def plot_roc_auc_curves(
        y_true_bin: np.ndarray,
        y_probs: np.ndarray,
        class_label_to_species_mapping: Optional[Dict[int, str]] = None,
        title: str = "ROC AUC Curves",
        print_output: bool = True,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: str = "roc_auc_curves"
    ) -> plt.Figure:
        """
        Plot ROC AUC curves for binary or multi-class classification, showing only classes
        that are present in the ground truth (non-zero support).

        Args:
            y_true_bin: Binarized ground truth labels (n_samples, n_classes)
            y_probs: Predicted probabilities (n_samples, n_classes)
            class_label_to_species_mapping: Optional mapping of class index to species name
            title: Plot title
            print_output: Whether to display the figure inline
            mlflow_logger: Optional MLFlowLogger instance to log the figure
            artifact_name: Artifact name for MLflow logging

        Returns:
            Matplotlib Figure object
        """
        # --------------------------------------------
        # Filter out unseen classes (with zero support)
        # --------------------------------------------
        y_true_bin, y_probs, class_label_to_species_mapping, present_classes = OutputGenerator.filter_unseen_classes(
            y_true_bin, y_probs, class_label_to_species_mapping
        )

        # --------------------------------------------
        # Compute ROC curves and AUCs
        # --------------------------------------------
        fpr, tpr, roc_auc = {}, {}, {}
        n_classes = y_true_bin.shape[1]

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # --------------------------------------------
        # Plot the curves
        # --------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(n_classes):
            label = class_label_to_species_mapping.get(i, f"Class {i}") if class_label_to_species_mapping else f"Class {i}"
            ax.plot(fpr[i], tpr[i], lw=2, label=f"{label} (AUC = {roc_auc[i]:.2f})")

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)

        # --------------------------------------------
        # Log and display
        # --------------------------------------------
        if mlflow_logger:
            mlflow_logger._log_matplotlib_figure(fig, artifact_name=artifact_name)

        if print_output:
            display(fig)
        else:
            plt.close(fig)

        return fig
    
    @staticmethod
    def plot_precision_recall_curves(
        y_true_bin: np.ndarray,
        y_probs: np.ndarray,
        class_label_to_species_mapping: Optional[Dict[int, str]] = None,
        title: str = "Precision-Recall Curves",
        print_output: bool = True,
        mlflow_logger: Optional[MLFlowLogger] = None,
        artifact_name: str = "precision_recall_curve"
    ) -> plt.Figure:
        """
        Plot precision-recall curves for binary or multi-class classification,
        showing only classes that are present in the ground truth (non-zero support).

        Args:
            y_true_bin: Binarized ground truth labels (n_samples, n_classes)
            y_probs: Predicted probabilities (n_samples, n_classes)
            class_label_to_species_mapping: Optional mapping of class index to species name
            title: Plot title
            print_output: Whether to display the figure inline
            mlflow_logger: Optional MLFlowLogger instance to log the figure
            artifact_name: Artifact name for MLflow logging

        Returns:
            Matplotlib Figure object
        """
        # --------------------------------------------
        # Filter out unseen classes (with zero support)
        # --------------------------------------------
        y_true_bin, y_probs, class_label_to_species_mapping, present_classes = OutputGenerator.filter_unseen_classes(
            y_true_bin, y_probs, class_label_to_species_mapping
        )

        # --------------------------------------------
        # Compute PR curves
        # --------------------------------------------
        precision, recall, pr_auc = {}, {}, {}
        n_classes = y_true_bin.shape[1]

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

        # --------------------------------------------
        # Plot the curves
        # --------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(n_classes):
            label = class_label_to_species_mapping.get(i, f"Class {i}") if class_label_to_species_mapping else f"Class {i}"
            ax.plot(recall[i], precision[i], lw=2, label=f"{label} (AUC = {pr_auc[i]:.2f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True)

        # --------------------------------------------
        # Log and display
        # --------------------------------------------
        if mlflow_logger:
            mlflow_logger._log_matplotlib_figure(fig, artifact_name=artifact_name)

        if print_output:
            display(fig)
        else:
            plt.close(fig)

        return fig