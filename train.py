import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from utils.load_data import load_data
from utils.model_architecture import build_model
from utils.train_model import train_model
from utils.evaluate_model import evaluate_model
from utils.plot_metrics import plot_training_history

# Load config file
with open('src/config/config.yaml', 'r') as con:
    conf = yaml.safe_load(con)

DIRECTORY = conf['directory']
DATASET_FILES = conf['dataset_files']
DISEASES_TO_SELECT = conf['diseases_to_select']
NORMAL_DISEASES = conf['normal_diseases']

print('Loading training data...')
X, y = load_data(DIRECTORY, DATASET_FILES, DISEASES_TO_SELECT, NORMAL_DISEASES)

print('Reshape data for model input...')
X.reshape(X.shape[0], X.shape[1], 1)

print('Splitting the data...')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

print('Building the model...')
model = build_model(input_shape=(X_train.shape[1], 1))

print('Training the model...')
epochs = conf['epochs']
bs = conf['batch_size']
history = train_model(model, X_train, y_train, epochs, bs)

print('Evaluating the model...')
metrics = evaluate_model(model, X_test, y_test)

print('Plotting the training history...')
plot_training_history(history)

