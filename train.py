import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D, BatchNormalization, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score
import os
import gc
from subprocess import run
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

class ECGClassifier:
    def __init__(self, data_dir, disease_list, normal_diseases, dataset_files):
        self.data_dir = data_dir
        self.disease_list = disease_list
        self.normal_diseases = normal_diseases
        self.dataset_files = dataset_files
        self.model = None

    def load_data(self):
        all_features = []
        all_targets = []

        i = 1
        for file in self.dataset_files:
            file_data = self._load_file(file)

            # Skip processing if file_data is None
            if file_data is None:
                continue

            # data = self._normalization(file_data)
            features = data.iloc[:, 2:].values
            targets = data.iloc[:, 1].values

            all_features.append(features)
            all_targets.append(targets)

            del file_data
            # del data
            gc.collect()
            run(["sync"])
            run(["sudo", "-S", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])

            print(f'File {i} loaded')
            i += 1

        X = np.concatenate(all_features)
        y = np.concatenate(all_targets)

        return X, y

    def _load_file(self, file):
        file_data = pd.read_csv(os.path.join(self.data_dir, file))
        # file_data = file_data[file_data['disease'].isin(self.disease_list)]

        # Check if the filtered DataFrame is empty
        if file_data.empty:
            print(f"Skipping file {file} because it does not contain any relevant diseases.")
            return None

        file_data['disease'] = file_data['disease'].apply(lambda x: 0 if x in self.normal_diseases else 1)
        return file_data

    def _normalization(self, file_data):
        signal = file_data.iloc[:, 2:].values
        features_normalized = (signal - np.min(signal, axis=0)) / (np.max(signal, axis=0) - np.min(signal, axis=0))
        file_data.iloc[:, 2:] = features_normalized
        return file_data

    def preprocess_data(self, X, y):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X, y

    def build_model(self, input_shape):
        model = Sequential()

        model.add(Conv1D(32, 11, padding='same', input_shape=input_shape))
        model.add(Conv1D(32, 11, padding='same'))
        model.add(MaxPooling1D(pool_size=10))

        model.add(Conv1D(64, 9, padding='same'))
        model.add(Conv1D(64, 9, padding='same'))
        model.add(MaxPooling1D(pool_size=7))

        model.add(Conv1D(128, 7, padding='same'))
        model.add(Conv1D(128, 7, padding='same'))
        model.add(MaxPooling1D(pool_size=5))

        model.add(Conv1D(256, 5, padding='same'))
        model.add(Conv1D(256, 5, padding='same'))
        model.add(MaxPooling1D(pool_size=3))

        model.add(Conv1D(512, 3, padding='same'))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(LSTM(512))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
        self.model = model

    def train_model(self, X_train, y_train, epochs, bs):
        wandb.init(project="ecg-classifier")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

        model_checkpoint = ModelCheckpoint(filepath="weights/best.keras", monitor='val_loss', mode='min',
                                           save_best_only=True)
        model_checkpoint_acc = ModelCheckpoint(filepath="weights/best_acc.keras", monitor='val_accuracy', mode='max',
                                           save_best_only=True)

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_split=0.2,
                                 callbacks=[reduce_lr, model_checkpoint, model_checkpoint_acc,
                                            WandbMetricsLogger(log_freq=5)])


        self.model.save('weights/weights.h5')
        return history

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print("Test Accuracy for this fold:", accuracy * 100)

        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

        self._print_metrics(metrics)
        self._plot_confusion_matrix(metrics['confusion_matrix'])
        return metrics

    def _print_metrics(self, metrics):
        print(f"Binary Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"F1 Score (Binary): {metrics['f1']:.2f}")
        print(f"Precision (Binary): {metrics['precision']:.2f}")
        print(f"Recall (Binary): {metrics['recall']:.2f}")
        print(f"AUC-ROC: {metrics['roc_auc']:.2f}")
        print(metrics['confusion_matrix'])
        print(metrics['classification_report'])

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Normal', 'Abnormal'],
                    yticklabels=['Normal', 'Abnormal'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_training_history(self, history):
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


if __name__ == '__main__':
    with open('config/config.yaml', 'r') as con:
        conf = yaml.safe_load(con)

    DIRECTORY = conf['directory']
    dataset_files = conf['dataset_files']
    diseases_to_select = ["NO SIGNIFICANT ABNORMALITIES DETECTED", "TACHYCARDIA", "ST T CHANGES", "Q WAVE",
                          "T WAVE CHANGE", "T INVERSION", "RECAPTURE", "ST ELEVATION", "RBBB", "LOW VOLTAGE",
                          "SINUS RHYTHM", "WITHIN NORMAL LIMITS", "HEART RATE NORMAL"]
    normal_diseases = ["NO SIGNIFICANT ABNORMALITIES DETECTED", "SINUS RHYTHM", "WITHIN NORMAL LIMITS", "HEART RATE NORMAL"]


    ecg_classifier = ECGClassifier(DIRECTORY, diseases_to_select, normal_diseases, dataset_files)

    print('Loading data...')
    X, y = ecg_classifier.load_data()
    print('Data loaded and preprocessed.')

    print('Processing data...')
    X, y = ecg_classifier.preprocess_data(X, y)
    print('Data processed.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

    print('Building model...')
    ecg_classifier.build_model(input_shape=(X_train.shape[1], 1))
    print('Model built.')

    print('Training model...')
    epochs = conf['epochs']
    bs = conf['batch_size']
    history = ecg_classifier.train_model(X_train, y_train, epochs, bs)
    print('Model trained.')

    print('Evaluating model...')
    ecg_classifier.evaluate_model(X_test, y_test)

    print('Plotting training history...')
    ecg_classifier.plot_training_history(history)
