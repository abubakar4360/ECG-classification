from io import BytesIO
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def dataFromMSB(msb,  data):
    temp1 = (msb & 240) >> 4
    value = data + (256 * temp1)
    return (value - 2048) / 124.10


def dataFromLSB(lsb,  data):
    temp1 = lsb & 15
    value = data + (256 * temp1)
    return (value - 2048) / 124.10


def byteFromValue(data):
    value = int((data * 124.10) + 2048)
    for smallByte in range(16):
        bigByte = value - 256 * smallByte
        if bigByte >= 0 and bigByte <= 255:
            break
    bigByte = min(bigByte, 255)
    bigByte = max(bigByte, 0)
    return (bigByte, smallByte)


def processFile(blob):
    dataset = list()
    stream = BytesIO(blob)
    while True:
        data = stream.read(16)
        if not data:
            break
        dataset.append(data)

    chartsData = list()
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())
    chartsData.append(list())

    i = 0
    for reading in dataset:
        if i % 1 == 0:
            chartsData[0].append(dataFromMSB(reading[11], reading[2]))  # L1
            chartsData[1].append(dataFromLSB(reading[11], reading[3]))  # L2

            chartsData[2].append(dataFromMSB(reading[12], reading[4]))  # L3
            chartsData[3].append(dataFromLSB(reading[12], reading[5]))  # L4

            chartsData[4].append(dataFromMSB(reading[13], reading[6]))  # L5
            chartsData[5].append(dataFromLSB(reading[13], reading[7]))  # L6

            chartsData[6].append(dataFromMSB(reading[14], reading[8]))  # L7
            chartsData[7].append(dataFromLSB(reading[14], reading[9]))  # L8

            chartsData[8].append(dataFromMSB(reading[15], reading[10]))  # L9

            chartsData[9].append((chartsData[0][-1] - chartsData[2][-1]) / 2)
            chartsData[10].append((chartsData[0][-1] + chartsData[1][-1]) / -2)
            chartsData[11].append((chartsData[1][-1] + chartsData[2][-1]) / 2)
        i = i + 1

    # Trim data to specific threshold and then concatenate all channels
    signal_threshold = 4000
    filtered_and_trimmed_chartsData = []

    for channel_data in chartsData:
        # Apply notch filter
        b, a = signal.iirnotch(50, 10, 500)
        channel_data = signal.filtfilt(b, a, channel_data)

        # Apply Savitzky-Golay filter
        channel_data = scipy.signal.savgol_filter(channel_data, 19, 3)

        # Trim to specific threshold
        filtered_and_trimmed_chartsData.append(channel_data[:signal_threshold])

    concatenated_data = []
    for channel_data in filtered_and_trimmed_chartsData:
        concatenated_data.extend(channel_data)

    if len(concatenated_data) == 48000:
        return concatenated_data
    else:
        print('Signal length less than 48000')
        return None
