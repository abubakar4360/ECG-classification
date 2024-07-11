import numpy as np
import pandas as pd
import yaml
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model

from decode_signal.main import get_signal

app = Flask(__name__)
CORS(app)

# Load configurations
with open('config/config.yaml', 'r') as conf:
    config = yaml.safe_load(conf)
MODEL_PATH = config['MODEL_PATH']


def normalize_min_max(data: np.ndarray) -> np.ndarray:
    """
    Normalize data using Min-Max Normalization.

    Args:
        data (np.ndarray): Input data to be normalized.

    Returns:
        np.ndarray: Normalized data.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


@app.route('/')
def index():
    """
    Render the index.html template for the main page.

    Returns:
        str: Rendered HTML content.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict ECG signal using a specified model.

    Returns:
        dict: JSON response containing the prediction result or error message.
    """
    try:
        # Extract inputs from POST request
        filename = request.form['filename']
        token = request.form['token']
        model_option = request.form['model']

        # Mapping of model options to model files
        model_files = {
            "model1": "model1.h5",
            "model2": "model2.keras",
            "model3": "model3.h5"
        }

        # Get the corresponding model file
        model_name = model_files.get(model_option)

        # Load the specified model
        model_path = f'{MODEL_PATH}/{model_name}'
        model = load_model(model_path)

        # Get the ECG signal data
        ecg_signal = get_signal(filename, token)

        # Handle short signal length case
        if isinstance(ecg_signal, str) and ecg_signal == 'Signal length less than 48000':
            result = "Warning: Cannot process signals with length less than 4800."
        else:
            # Convert ECG signal to numpy array and reshape
            ecg_signal = np.array(ecg_signal).reshape(1, -1)

            # Normalize data for model1 and model2
            if model_option in ["model1", "model2"]:
                ecg_signal = normalize_min_max(ecg_signal)

            # Convert to DataFrame and save as temporary CSV
            df = pd.DataFrame(ecg_signal)
            temp_file = 'temp_ecg_signal.csv'
            df.to_csv(temp_file, index=False)

            # Read back the CSV data for prediction
            ecg_data = pd.read_csv(temp_file)
            predictions = model.predict(ecg_data.values)
            predictions = (predictions > 0.5).astype(int)
            result = 'abnormal' if predictions[0] == 1 else 'normal'

    except Exception as e:
        if str(e).startswith('local variable'):
            result = "SAS-Token expired. Please obtain a new token and try again."
        else:
            result = f"Error processing the request: {str(e)}"

    return jsonify(result=result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
