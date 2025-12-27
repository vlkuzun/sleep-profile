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

def plot_power_ratio(power, window_size, start_time):
    # Calculate time points excluding last point
    time = pd.date_range(start=start_time, periods=len(power)-1, freq=f'{window_size}S')
    power = power[:-1]  # Remove last point
    
    plt.figure(figsize=(16, 0.75))
    plt.plot(time, power, color='black')
    plt.axis('off')
    plt.show()

def process_eeg_ratio(pickle_file, start_time, range_start, range_end, lowcut1, highcut1, lowcut2, highcut2):
    fs = 512  # Sampling frequency

    # Load EEG data
    eeg_signal, timestamps = load_eeg_from_pickle(pickle_file)
    
    # Create time index
    timestamps = pd.date_range(start=start_time, periods=len(eeg_signal), freq=f'{1/fs}S')
    
    # Select time range
    mask = (timestamps >= range_start) & (timestamps <= range_end)
    eeg_selected = eeg_signal[mask]
    
    # Apply bandpass filters for both frequency ranges
    filtered_signal1 = bandpass_filter(eeg_selected, lowcut1, highcut1, fs)
    filtered_signal2 = bandpass_filter(eeg_selected, lowcut2, highcut2, fs)
    
    # Calculate power every 10 seconds for both filtered signals
    power1 = calculate_power(filtered_signal1, fs, 10)
    power2 = calculate_power(filtered_signal2, fs, 10)
    
    # Calculate the ratio of power between the two frequency ranges
    power_ratio = np.array(power1) / np.array(power2)
    
    # Plot power ratio
    plot_power_ratio(power_ratio, 10, range_start)

# Example usage
process_eeg_ratio(
    '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-016_ses-02_recording-01.pkl',
    '2024-11-29 13:29:18',
    '2024-11-30 11:30:00',
    '2024-11-30 13:00:00',
    5, 10,  # Lowcut and highcut frequencies for the first bandpass filter
    2, 15  # Lowcut and highcut frequencies for the second bandpass filter
)