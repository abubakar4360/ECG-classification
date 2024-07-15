import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

def plot_training_history(history: History) -> None:
    """
    Plots the training history.

    Args:
        history (History): Keras History object containing training metrics.
    """
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
