import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
import pandas as pd

# Set global style for publication
plt.rcParams.update({
    'font.family': 'Arial',         # Use Arial (or Helvetica as fallback)
    'font.size': 10,                # General font size
    'axes.labelsize': 12,           # Axis label size
    'axes.titlesize': 12,           # Title size
    'xtick.labelsize': 10,          # X tick label size
    'ytick.labelsize': 10,          # Y tick label size
    'legend.fontsize': 10,          # Legend text size
    'figure.dpi': 300,              # High-res output
    'savefig.dpi': 600,             # High-res when saving
    'axes.linewidth': 1,            # Thinner axis borders
    'pdf.fonttype': 42,             # Embed fonts properly in PDFs
    'ps.fonttype': 42
})

class EEGRecording:
    def __init__(self, eeg_data, recording_start_time, fs=512):
        self.data = eeg_data
        self.start_time = datetime.strptime(recording_start_time, "%Y-%m-%d %H:%M:%S")
        self.fs = fs
        
    def get_segment(self, segment_start_time, duration_mins):
        start_delta = datetime.strptime(segment_start_time, "%Y-%m-%d %H:%M:%S") - self.start_time
        start_idx = int(start_delta.total_seconds() * self.fs)
        n_samples = int(duration_mins * 60 * self.fs)
        return self.data[start_idx:start_idx + n_samples]

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_power(signal, fs, window_size):
    """Calculate power in non-overlapping windows.
    Each value represents the power in a window_size-second segment:
    First value = 0 to window_size seconds
    Second value = window_size to 2*window_size seconds
    etc.
    """
    power = []
    window_samples = window_size * fs
    
    # Make windows explicit
    total_windows = len(signal) // window_samples
    for i in range(total_windows):
        start = i * window_samples
        end = start + window_samples
        segment = signal[start:end]
        if len(segment) == window_samples:  # Only process complete windows
            segment_power = np.mean(segment**2)  # Changed from np.sum to np.mean for better scaling
            power.append(segment_power)
    
    return np.array(power)

def combined_plot(pickle_path, recording_start_time, segment_start_time, duration_mins, 
                 lowcut=1, highcut=4, ratio_lowcut1=5, ratio_highcut1=10,
                 ratio_lowcut2=2, ratio_highcut2=15, save_path=None):  # Added save_path parameter
    # Sampling rate definition moved to top
    fs = 512
    
    # Load data
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    recording = EEGRecording(data['EEG1'], recording_start_time)
    emg_data = data['EMG']
    
    # Get segments
    signal_segment = recording.get_segment(segment_start_time, duration_mins)
    start_idx = int((datetime.strptime(segment_start_time, "%Y-%m-%d %H:%M:%S") - 
                    datetime.strptime(recording_start_time, "%Y-%m-%d %H:%M:%S")).total_seconds() * fs)
    end_idx = start_idx + int(duration_mins * 60 * fs)
    emg_segment = emg_data[start_idx:end_idx]
    
    # Create figure with GridSpec spacer between spectrogram and power plots
    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(7, 1,
                          height_ratios=[2, 0.3, 1, 1, 1, 1, 1.2],
                          hspace=0.03)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    ax4 = fig.add_subplot(gs[4], sharex=ax1)
    ax5 = fig.add_subplot(gs[5], sharex=ax1)
    ax6 = fig.add_subplot(gs[6], sharex=ax1)
    
    # Set figure margins while preserving the new gap
    fig.subplots_adjust(left=0.13,
                        right=0.98,
                        top=0.97,
                        bottom=0.05)

    # Spectrogram (top)
    nperseg = 512 * 2  # 2-second windows
    f, t, Sxx = signal.spectrogram(signal_segment, 
                                  fs=fs,
                                  nperseg=nperseg,
                                  noverlap=nperseg//2,
                                  nfft=2048)
    
    # Filter and normalize
    mask = (f >= 1) & (f <= 64)
    power_db = 10 * np.log10(Sxx[mask])
    vmin = np.percentile(power_db, 0.01)  # More extreme minimum
    vmax = np.percentile(power_db, 99.99)  # More extreme maximum
    
    plt.sca(ax1)
    plt.pcolormesh(t, f[mask], power_db,
                   cmap='jet',      # Use viridis colormap for spectrogram
                   shading='gouraud',
                   vmin=vmin,
                   vmax=vmax)
    plt.yscale('log')
    plt.yticks([1, 4, 16, 64], ['1', '4', '16', '64'], fontsize=17)
    ax1.set_ylabel('Frequency (Hz)', fontsize=17)
    ax1.set_ylim(1, 64)
    ax1.yaxis.set_minor_locator(plt.NullLocator())  # Remove minor ticks
    
    # EMG Power plot (middle)
    filtered_emg = bandpass_filter(emg_segment, 30, 250, fs=fs)
    window_size = 5  # Each point represents power in a 5-second window
    emg_power = calculate_power(filtered_emg, fs, window_size)
    
    # Calculate time points to match power windows
    # First point at 2.5s (center of first 5s window)
    total_duration = t[-1] - t[0]
    emg_times = np.arange(window_size/2, total_duration, window_size)[:len(emg_power)]
    
    plt.sca(ax2)
    plt.plot(emg_times, emg_power, color='black', linewidth=1.5)
    
    # EEG Power plot (bottom)
    b, a = butter(2, [lowcut, highcut], btype='band', fs=fs)
    filtered_eeg = filtfilt(b, a, signal_segment)
    eeg_power = calculate_power(filtered_eeg, fs, window_size)
    
    # Use same time points for EEG
    plt.sca(ax3)
    plt.plot(emg_times[:len(eeg_power)], eeg_power, color='black', linewidth=1.5)
    
    # Add EEG Power Ratio plot (bottom)
    # Calculate power ratio using two frequency bands
    filtered_eeg1 = bandpass_filter(signal_segment, ratio_lowcut1, ratio_highcut1, fs=fs)
    filtered_eeg2 = bandpass_filter(signal_segment, ratio_lowcut2, ratio_highcut2, fs=fs)
    
    power1 = calculate_power(filtered_eeg1, fs, window_size)
    power2 = calculate_power(filtered_eeg2, fs, window_size)
    
    # Calculate ratio and use same time points
    power_ratio = np.array(power1) / np.array(power2)
    
    plt.sca(ax4)
    plt.plot(emg_times[:len(power_ratio)], power_ratio, color='black', linewidth=1.5)
    
    # Add EEG Power plot for 9-25Hz (bottom)
    filtered_eeg_925 = bandpass_filter(signal_segment, 9, 25, fs=fs)
    power_925 = calculate_power(filtered_eeg_925, fs, window_size)
    
    plt.sca(ax5)
    plt.plot(emg_times[:len(power_925)], power_925, color='black', linewidth=1.5)
    
    # Add EEG Power plot for 40-100Hz (bottom)
    filtered_eeg_40100 = bandpass_filter(signal_segment, 40, 100, fs=fs)
    power_40100 = calculate_power(filtered_eeg_40100, fs, window_size)
    
    plt.sca(ax6)
    plt.plot(emg_times[:len(power_40100)], power_40100, color='black', linewidth=1.5)
    
    # Clean up and add scale bar
    # Spectrogram cleanup (keep y-ticks)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks([])
    
    # Power plots cleanup (remove all ticks)
    for ax in [ax2, ax3, ax4, ax5, ax6]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add scale bar (500 seconds) to bottom right
    scale_bar_length = 500  # seconds
    x_max = t[-1]
    y_min = ax6.get_ylim()[0]
    
    # Draw scale bar at bottom right
    y_pos = y_min * 0.9  # Position for bar
    ax6.plot([x_max - scale_bar_length, x_max], [y_pos, y_pos], 
             'k-', linewidth=2)
    
    # Add text with larger font and much more space from bar
    text_y_pos = y_min * 0.01  # Changed from 0.3 to 0.1 for much more separation
    ax6.text(x_max - scale_bar_length/2, text_y_pos,
             '500 s', ha='center', va='top', fontsize=17)
    
    # Save figure if path is provided
    if save_path:
        # Save as PNG
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
        # Save as PDF (replace .png extension with .pdf)
        if save_path.endswith('.png'):
            pdf_path = save_path.replace('.png', '.pdf')
        else:
            pdf_path = save_path + '.pdf'
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.show()

# Example usage
combined_plot(
    pickle_path='/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-016_ses-02_recording-01.pkl',
    recording_start_time='2024-11-29 13:29:18',
    segment_start_time='2024-11-30 11:30:00',
    duration_mins=90,
    ratio_lowcut1=5, ratio_highcut1=10,
    ratio_lowcut2=1, ratio_highcut2=4,
    save_path='/Volumes/harris/volkan/sleep-profile/plots/spectrogram_power_emg_eeg/spectrogram_power_emg_eeg_combined_plot_sub-016_ZT2.5-4_ratio_t_d.png'  # Add save path
)