import time
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from urllib.error import HTTPError
import pandas as pd
import yaml
from decode_signal.decode_signal import processFile

# Get the absolute path to the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, '..', 'config', 'config.yaml')

# Load configuration
with open(config_file_path, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

EXCEL_FILE = config['excel_file']
SHEET_NAME = config['sheet_name']
SAS_TOKEN = config['sasToken']
BASE_URL = config['base_url']
OUTPUT_CSV = config['output_csv']
START_INDEX = config['start_index']
END_INDEX = config['end_index']


def read_filenames(excel_file: str, sheet_name: str, start_index: int, end_index: str) -> List[Tuple[str, str]]:
    """
    Reads filenames and diseases from the Excel file.

    Args:
        excel_file (str): Path to the Excel file.
        sheet_name (str): Name of the sheet within the Excel file.
        start_index (int): Index of the first row in the Excel file.
        end_index (int): Index of the last row in the Excel file.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing filenames and diseases.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    filenames = df.iloc[start_index:end_index, 1].tolist()
    diseases = df.iloc[start_index:end_index, 0].tolist()
    return list(zip(filenames, diseases))


def download_file(file_info: Tuple[str, str]) -> Tuple[str, List, str]:
    """
    Downloads a single file and processes its content.

    Args:
        file_info (Tuple[str, str]): A tuple containing the filename and associated disease.

    Returns:
        Tuple[str, Optional[List[float]], str]: A tuple containing the filename, processed data, and disease.
    """
    file_name, disease = file_info
    source_url = BASE_URL + file_name + "-original" + SAS_TOKEN
    data = None

    try:
        response = urllib.request.urlopen(source_url)
    except HTTPError as err:
        if err.code == 404:
            source_url = BASE_URL + file_name + SAS_TOKEN
            response = urllib.request.urlopen(source_url)

    try:
        originalBytes = response.read()
        data = processFile(originalBytes)
    except Exception as e:
        print(f'Error {e} downloading file {file_name}')

    return file_name, data, disease


def save_to_csv(
        data_list: Tuple[str, List[float], str], output_csv: str) -> None:
    """
       Saves the downloaded and processed data to a CSV file.

       Args:
           data_list (List[Tuple[str, Optional[List[float]], str]]): List of tuples containing filename, data, and disease.
           output_csv (str): Path to the output CSV file.
    """
    data_list_filtered = [(file_name, data, disease)
                          for file_name, data, disease in data_list if data is not None]
    filenames, data, diseases = zip(*data_list_filtered)
    df = pd.DataFrame({'filename': filenames, 'disease': diseases})

    if data:
        df_data = pd.DataFrame(data)
        df_concatenated = pd.concat([df, df_data], axis=1)
        df_concatenated.to_csv(output_csv, index=False)
    else:
        print("No valid data to save.")


def main() -> None:
    """
    Main function to coordinate the downloading and saving process.
    """
    file_info_list = read_filenames(
        EXCEL_FILE, SHEET_NAME, START_INDEX, END_INDEX)
    data_list = []

    # Adjust max_workers based on your system capability
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {
            executor.submit(
                download_file,
                file_info): file_info for file_info in file_info_list}

        for future in as_completed(futures):
            file_info = futures[future]
            try:
                file_name, data, disease = future.result()
                data_list.append((file_name, data, disease))
                print(f"Downloaded: {file_name}")
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")

    save_to_csv(data_list, OUTPUT_CSV)
    print(f"All files have been downloaded and saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
