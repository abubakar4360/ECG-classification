import os
from typing import Tuple
import pandas as pd
import numpy as np
from subprocess import run
import gc

def load_data(data_dir: str, dataset_files: list, disease_list: list, normal_diseases: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses ECG data from CSV files.

    Args:
        data_dir (str): Directory containing the dataset files.
        dataset_files (list): List of dataset file names.
        disease_list (list): List of diseases to consider in the dataset.
        normal_diseases (list): List of diseases considered normal.

    Returns:
        tuple: A tuple containing the features (X) and targets (y) as numpy arrays.
    """
    all_features = []
    all_targets = []

    i = 1
    for file in dataset_files:
        file_data = _load_file(os.path.join(data_dir, file), disease_list, normal_diseases)

        if file_data is None:
            continue

        features = file_data.iloc[:, 2:].values
        targets = file_data.iloc[:, 1].values

        all_features.append(features)
        all_targets.append(targets)

        del file_data
        gc.collect()
        run(["sync"])
        run(["sudo", "-S", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])

        print(f'File {i} loaded')
        i += 1

    X = np.concatenate(all_features)
    y = np.concatenate(all_targets)

    return X, y

def _load_file(file: str, disease_list: list, normal_diseases: list) -> pd.DataFrame:
    """
    Loads a single dataset file and processes it.

    Args:
        file (str): File path to load.
        disease_list (list): List of diseases to consider in the dataset.
        normal_diseases (list): List of diseases considered normal.

    Returns:
        DataFrame: Processed DataFrame containing loaded data.
    """
    file_data = pd.read_csv(file)
    # file_data = file_data[file_data['disease'].isin(disease_list)]

    if file_data.empty:
        print(f"Skipping file {file} because it does not contain any relevant diseases.")
        return None

    file_data['disease'] = file_data['disease'].apply(lambda x: 0 if x in normal_diseases else 1)
    return file_data
