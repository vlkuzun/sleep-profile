import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime

def load_eeg_from_pickle(file_path):
    df = pd.read_pickle(file_path)
    return df['EEG1'].values, df.index

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def calculate_power(signal, fs, window_size):
    power = []
    window_samples = window_size * fs
    for start in range(0, len(signal), window_samples):
        end = start + window_samples
        segment = signal[start:end]
        segment_power = np.sum(segment**2) / len(segment)
        power.append(segment_power)
    return power

def plot_power(power, window_size, start_time):
    # Calculate time points excluding last point
    time = pd.date_range(start=start_time, periods=len(power)-1, freq=f'{window_size}S')
    power = power[:-1]  # Remove last point
    
    plt.figure(figsize=(16, 0.75))
    plt.plot(time, power, color='black')
    plt.axis('off')
    plt.show()

def process_eeg(pickle_file, start_time, range_start, range_end, lowcut, highcut):
    fs = 512  # Sampling frequency

    # Load EEG data
    eeg_signal, timestamps = load_eeg_from_pickle(pickle_file)
    
    # Create time index
    timestamps = pd.date_range(start=start_time, periods=len(eeg_signal), freq=f'{1/fs}S')
    
    # Select time range
    mask = (timestamps >= range_start) & (timestamps <= range_end)
    eeg_selected = eeg_signal[mask]
    
    # Apply bandpass filter
    filtered_signal = bandpass_filter(eeg_selected, lowcut, highcut, fs)
    
    # Calculate power every 10 seconds
    power = calculate_power(filtered_signal, fs, 5)
    
    # Plot power
    plot_power(power, 5, range_start)

# Example usage
process_eeg(
    '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-016_ses-02_recording-01.pkl',
    '2024-11-29 13:29:18',
    '2024-11-30 11:30:00',
    '2024-11-30 13:00:00',
    1, 4  # Lowcut and highcut frequencies for bandpass filter
)