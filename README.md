# ECG Classification

Welcome to the ECG Classification Project! This project aims to provide an efficient and accurate system for classifying Electrocardiogram (ECG) signals. ECG signals are critical in diagnosing various cardiac conditions, and automated classification can significantly aid medical professionals by providing quick and reliable results.

The project involves the following key components:

1. **Signal Acquisition and Processing:** Extracting raw ECG signals from source files and preprocessing them for analysis.
2. **Model Training and Deployment:** Using machine learning and deep learning techniques to train models that can classify ECG signals into different categories (normal, abnormal).
3. **Web Application:** A Flask-based web application that allows users to upload ECG signal files and receive classification results in real-time.

## Key Features

1. **Automated Signal Downloading:** Download raw ECG signal data from specified URLs using a provided SAS token and process them into a structured format (CSV).
2. **Normalization and Preprocessing:** Normalize the ECG signals using Min-Max normalization and other preprocessing techniques to prepare them for model inference. 
3. **Multiple Model Support:** Support for multiple trained models (e.g., model1, model2, model3) to allow for flexibility and comparison. 
4. **User-Friendly Interface:** A simple and intuitive web interface for users to upload ECG files and view classification results. 
5. **Concurrency for Efficiency:** Utilize concurrent processing to speed up the downloading and preprocessing of large datasets.


## Repository Structure.
```
wellnest-ecg-ai    
    config
        └── config.yaml                     # Configuration file
    data_loader
        └── download_azure_storage.py       # Script for downloading and processing ECG data
    decode_signal
        └── decode_signal.py                # Signal decoding script
        └── main.py                
    EDA
        └── data_analysis.py                # Data analysis script
    templates
        └── index.html                      # HTML template for the web application
    .gitignore                              # Git ignore file
    README.md                               # Project README file
    app.py                                  # Flask API
    inference.py                            # Inference script for predictions
    requirements.txt                        # List of dependencies
    train.py                                # Script for training models

```


## Installation Guide

### 1. Clone the Repo
First, clone the repository using the following command:
```bash
git clone https://github.com/arjavrdave/wellnest-ecg-ai.git
```
Then, change to the project directory:
```bash
cd wellnest-ecg-ai
```

### 2. Create conda environment
Create a new Conda environment with the following command:
```bash
conda create -n <env-name>
```

Replace ```<env-name>``` with the name of your choice for the environment.

### 3. Activate conda environment
Activate the newly created Conda environment:
```bash
conda activate <env-name>
```
### 4. Install Dependencies
#### For CPU users:
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
#### For GPU users:
Install the required dependencies using pip:
```bash
pip install -r requirements_gpu.txt
```
### 5. Run Flask API
Download all the model weights from the provided [link](https://drive.google.com/drive/folders/1nXUydCMQ3cdfmOkWx3EsyvtJEm-cM9yH?usp=sharing), place them in a directory of your choice, and update the ``config.yaml`` file with the path to this directory. Update the following line in ``config.yaml``:
```angular2html
MODEL_PATH: Path to directory where model weights are downloaded.
```

Now, you can start the API by running:
```bash
python app.py
```

## Train the model

If you want to train your own model, here is complete step-by-step guide:

### 1. Download data

First, create a directory named ``data``. Next, open the ``config.yaml`` file from the ``config`` directory and adjust the configurations according to your needs:
```bash
excel_file: 'Your Excel file name, e.g., ECG-AI.xlsx'

sheet_name: 'Sheet name within the Excel file, e.g., All'

sasToken: 'Updated SAS token'

base_url: 'No need to change this'

output_csv: 'Output file path, e.g., data/ecg_data_1.csv'

start_index: 'Index to start downloading data from; its default value must be kept 2'

end_index: 'End index for downloading files'
```

Now, to download the data, run:
```bash
python data_loader/download_azure_storage.py
```

It is recommended to:
1. Download a maximum of 5000 files at a time, such as from `start_index = 2` to `end_index = 5002`. Then, take the next 5000 files from 5002 to 10002, and so on. This approach prevents potential issues when extracting data from the source URL.
2. Create multiple CSV files and store the data. For example, first 5000 data will be in ``ecg_data_1.csv``, next 5000 data in ``ecg_data_2.csv`` and so.
3. Ensure you create a data directory where you will store these multiple CSV files.

The data directory structure should look like this:
``` 
data/
├── ecg_data_1.csv
├── ecg_data_2.csv
├── ecg_data_3.csv
...
```


### 2. Training

Once the data is downloaded in the ``data`` directory, you can modify the training configuration. Open the ``config.yaml`` and update it with your settings:
```bash
directory: 'Directory where data is stored, e.g., data'

dataset_files: 'List of file names from the data directory, e.g., ["ecg_data_1.csv", "ecg_data_2.csv"]'  # If you have only one file, just add that file to the list

epochs: 'Number of epochs e.g, 20'

batch_size: 'Batch size e.g, 4'
```

Now, run the training script:
```bash
python train.py
```

After the script is executed, the weights will be saved in the ``weights`` directory.


