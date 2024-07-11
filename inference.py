from typing import Tuple, List
import numpy as np
import pandas as pd
import yaml
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score


def load_config(config_filepath: str = "config/config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_filepath (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_test_data(filepath: str) -> pd.DataFrame:
    """
    Load the test dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing the test data.

    Returns:
        pd.DataFrame: Test data.
    """
    return pd.read_csv(filepath, header=None)


def preprocess_data(test_data: pd.DataFrame,
                    normal_diseases: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the test data for model prediction.

    Args:
        test_data (pd.DataFrame): Raw test data.
        normal_diseases (List[str]): List of normal disease labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed features and labels.
    """
    labels = test_data.iloc[1:, 1].apply(
        lambda x: 0 if x in normal_diseases else 1).values
    features = test_data.iloc[1:, 2:].values

    return features, labels


def evaluate_model(model_filepath: str, xt: np.ndarray,
                   yt: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Load the trained model, make predictions, and evaluate performance.

    Args:
        model_filepath (str): Path to the trained model file.
        xt (np.ndarray): Preprocessed features.
        yt (np.ndarray): True labels.

    Returns:
        Tuple[np.ndarray, float]: Confusion matrix and accuracy.
    """
    model = load_model(model_filepath)
    predictions = model.predict(xt)
    predictions = (predictions > 0.5).astype(int)

    cm = confusion_matrix(yt, predictions)
    acc = accuracy_score(yt, predictions)

    return cm, acc


def main() -> None:
    """
    Main function to load config, preprocess data, and evaluate the model.
    """
    config = load_config()

    test_data = load_test_data(config['test_data_filepath'])
    xt, yt = preprocess_data(test_data, config['normal_diseases'])

    cm, acc = evaluate_model(config['model_filepath'], xt, yt)

    print("Confusion Matrix:\n", cm)
    print("Test Accuracy:", acc * 100)


if __name__ == "__main__":
    main()
