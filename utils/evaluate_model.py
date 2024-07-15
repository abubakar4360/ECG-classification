import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tensorflow.keras.models import Sequential

def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluates the model on the test data.

    Args:
        model (Sequential): Compiled Keras Sequential model.
        X_test (ndarray): Test features.
        y_test (ndarray): Test targets.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy for this fold:", accuracy * 100)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

    _print_metrics(metrics)
    _plot_confusion_matrix(metrics['confusion_matrix'])

    return metrics

def _print_metrics(metrics: dict) -> None:
    """
    Prints evaluation metrics.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    print(f"Binary Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"F1 Score (Binary): {metrics['f1']:.2f}")
    print(f"Precision (Binary): {metrics['precision']:.2f}")
    print(f"Recall (Binary): {metrics['recall']:.2f}")
    print(metrics['confusion_matrix'])
    print(metrics['classification_report'])

def _plot_confusion_matrix(cm: np.ndarray) -> None:
    """
    Plots the confusion matrix.

    Args:
        cm (ndarray

): The confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
