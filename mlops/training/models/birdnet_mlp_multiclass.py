# mlops/modelling/birdnet_mlp_multiclass.py
import sys
import os
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
import tensorflow as tf
from mlops.training.tf_model_registry import register_model

@register_model(
    name="birdnet_mlp_multiclass",
    description="""
    A simple 2-layer MLP for BirdNET 1024-dimensional embeddings.
    - Architecture: Dense(128, relu) → Dropout(0.3) → Dense(num_classes, softmax)
    - Loss: sparse_categorical_crossentropy
    - Metrics: accuracy, sparse_categorical_accuracy, top-3 accuracy
    """
)

def birdnet_mlp_multiclass(input_shape, num_classes) -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        ]
    )
    return model
