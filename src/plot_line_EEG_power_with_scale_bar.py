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
    # Calculate actual time points, excluding last point
    time = pd.date_range(start=start_time, periods=len(power)-1, freq=f'{window_size}S')
    power = power[:-1]  # Remove last point
    
    fig, ax = plt.subplots(figsize=(16, 1))
    
    # Plot main power data with vertical offset
    y_offset = 0.9 * (max(power) - min(power))
    ax.plot(time, power + y_offset, color='black')
    
    # Calculate scale bar position and size
    y_range = max(power) - min(power)
    y_pos = min(power) + 0.85 * y_range
    
    # Calculate correct time points for 500s scale bar
    x_end = time[-1]
    x_start = x_end - pd.Timedelta(seconds=500)
    
    # Draw scale bar with actual time span
    ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=2)
    
    # Add "500s" text
    text_x_pos = x_start + pd.Timedelta(seconds=250)
    text_y_pos = y_pos - 0.25 * y_range
    ax.text(text_x_pos, text_y_pos, '500s', 
            horizontalalignment='center', fontsize=14)
    
    ax.axis('off')
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
    
    # Calculate power with 2-second windows to match scale
    power = calculate_power(filtered_signal, fs, 10)  # Changed from 10 to 2 seconds
    
    # Plot power with 2-second window size
    plot_power(power, 10, range_start)

# Example usage
process_eeg(
    '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-016_ses-02_recording-01.pkl',
    '2024-11-29 13:29:18',
    '2024-11-30 11:30:00',
    '2024-11-30 13:00:00',
    40, 100  # Lowcut and highcut frequencies for bandpass filter
)