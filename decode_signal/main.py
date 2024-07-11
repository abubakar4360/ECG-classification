import urllib.request
from urllib.error import HTTPError
from decode_signal.decode_signal import processFile


def get_signal(filename, token):
    fileName = filename
    sasToken = token

    # Create a source url to download data
    source_url = "https://wellnestprodstorage.blob.core.windows.net/ecg-recordings/" + fileName + "-original" + sasToken

    try:
        response = urllib.request.urlopen(source_url)
    except HTTPError as err:
        if err.code == 404:
            source_url = "https://wellnestprodstorage.blob.core.windows.net/ecg-recordings/" + fileName + sasToken
            response = urllib.request.urlopen(source_url)

    originalBytes = response.read()
    ecg_signal = processFile(originalBytes)

    if ecg_signal is None:
        return 'Signal length less than 48000'

    return ecg_signal
