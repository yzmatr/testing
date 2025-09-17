# mlops/utils/data_utils.py

import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Literal

def convert_to_tf_dataset_with_ids(
    features: pd.Series,
    labels: pd.Series,
    ids: pd.Series,
    batch_size: int = 32,
    expand_dims: bool = False,
    shuffle: bool = False,
    include_ids: bool = True
) -> tf.data.Dataset:
    """
    Convert features, labels, and recording IDs into a tf.data.Dataset.

    Each sample in the dataset will be a tuple: ((X, id), y) if include_ids=True, or (X, y) if include_ids=False.

    Args:
        features: Series of feature arrays (e.g., BirdNET or log-mel).
        labels: Series of integer class labels.
        ids: Series of string IDs corresponding to each feature.
        batch_size: Batch size for dataset.
        expand_dims: Whether to add a channel dimension (for CNNs).
        shuffle: Whether to shuffle the dataset.
        include_ids: Whether to include string IDs in the dataset. Set to False for training to avoid XLA compilation issues.

    Returns:
        tf.data.Dataset object with structure ((X, id), y) if include_ids=True, or (X, y) if include_ids=False.
    """
    # 1. Stack feature vectors
    X = np.stack(features.values)

    # 2. Optionally add a channel dimension for CNNs
    if expand_dims and X.ndim == 3:
        X = np.expand_dims(X, axis=-1)

    # 3. Prepare labels and ids
    y = labels.astype(np.int32).values
    
    # 4. Create tensors
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)

    # 5. Create dataset with or without IDs to avoid XLA compilation issues
    if include_ids:
        id_arr = ids.astype(str).values
        id_tensor = tf.convert_to_tensor(id_arr, dtype=tf.string)
        # Combine tensors: ((X, id), y)
        dataset = tf.data.Dataset.from_tensor_slices(((X_tensor, id_tensor), y_tensor))
    else:
        # Simple structure for training: (X, y) - avoids XLA string tensor issues
        dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))

    # 6. Shuffle and batch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features), reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def split_data(
    df_data: pd.DataFrame,
    id_col: str,
    label_col: str,
    test_size: int,
    val_size: int,
    stratify: bool = True,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_cols = { id_col, label_col }
    missing_cols = required_cols - set(df_data.columns)
    if missing_cols:
        raise ValueError(f"❌ Missing required columns in df_features: {sorted(missing_cols)}")

    # Prepare dataframe
    df = df_data.copy()
    df[id_col] = df_data[id_col].astype(str)

    # Step 1: Extract unique IDs
    unique_ids = df[id_col].unique()
    n_total = len(unique_ids)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val
    print(f"🔢 Splitting {n_total} IDs → Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Step 2: Build stratify labels (id -> class)
    df_id_to_label = df.drop_duplicates(id_col)[[id_col, label_col]]
    id_to_label = df_id_to_label.set_index(id_col)
    id_to_label = id_to_label[label_col]

    def _get_labels_for_stratification(ids, split_name, min_required):
        if stratify:
            labels = id_to_label.loc[ids]
            num_classes = labels.nunique()
            if len(ids) < num_classes or min_required < num_classes:
                print(f"⚠️ Not enough samples ({len(ids)}) for stratification across {num_classes} classes in {split_name} split. Disabling stratification.")
                return None
            return labels
        return None

    # Step 3: Split into train+val and test
    stratify_test = _get_labels_for_stratification(unique_ids, "test", n_test)
    trainval_ids, test_ids = train_test_split(
        unique_ids,
        test_size=n_test,
        stratify=stratify_test,
        random_state=random_state
    )

    # Step 4: Split trainval into train and val
    stratify_val = _get_labels_for_stratification(trainval_ids, "val", n_val)
    try:
        train_ids, val_ids = train_test_split(
            trainval_ids,
            test_size=n_val,
            stratify=stratify_val,
            random_state=random_state
        )
    except ValueError as e:
        print(f"❌ Stratified val split failed: {e}. Falling back to random split.")
        train_ids, val_ids = train_test_split(
            trainval_ids,
            test_size=n_val,
            random_state=random_state
        )

    # Define splits
    ids_per_split = {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(),
        "test": test_ids.tolist()
    }

    # Store for the data frame splits
    dfs = {}

    for name, ids in ids_per_split.items():
        # Get the corresponding subset
        df_subset = df[df[id_col].isin(ids)].copy()
        dfs[name] = df_subset
    
    return dfs["train"], dfs['val'], dfs['test']
        
def get_data_used_for_modelling(csv_path_to_modelling_ids: str, df_cleaned: pd.DataFrame, subset: Optional[Literal["train", "val", "test"]] = None) -> pd.DataFrame:
    """
    Return the subset of df_cleaned that was used for modelling (train/val/test) if provided,
    otherwise returns all used data.
    """
    df_ids = pd.read_csv(csv_path_to_modelling_ids)

    if subset:
        df_ids = df_ids[df_ids['split'] == subset]

    used_ids = set(df_ids['id'])
    df_modelling_subset = df_cleaned[df_cleaned['id'].isin(used_ids)].copy()

    print(f"✅ Returning {'all' if subset is None else subset} modelling data → {len(df_modelling_subset)} rows.")
    return df_modelling_subset


