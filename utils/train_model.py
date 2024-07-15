import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, History
from tensorflow.keras.models import Sequential
import numpy as np
from typing import Tuple

def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, epochs: int, bs: int) -> History:
    """
    Trains the neural network model.

    Args:
        model (Sequential): Compiled Keras Sequential model.
        X_train (ndarray): Training features.
        y_train (ndarray): Training targets.
        epochs (int): Number of epochs to train.
        bs (int): Batch size.

    Returns:
        History: Keras History object containing training metrics.
    """
    wandb.init(project="ecg-classifier")

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    model_checkpoint = ModelCheckpoint(filepath="checkpoints/best.keras", monitor='val_loss', mode='min', save_best_only=True)
    model_checkpoint_acc = ModelCheckpoint(filepath="checkpoints/best_acc.keras", monitor='val_accuracy', mode='max', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_split=0.2,
                        callbacks=[reduce_lr, model_checkpoint, model_checkpoint_acc, WandbMetricsLogger(log_freq=5)])

    model.save('checkpoints/weights.h5')

    return history
