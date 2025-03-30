import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

def plot_eeg_spectrogram(pickle_path, recording_start_time, segment_start_time, duration_mins):
    # Load and initialize
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    recording = EEGRecording(data['EEG1'], recording_start_time)
    
    # Get data segment
    signal_segment = recording.get_segment(segment_start_time, duration_mins)
    
    # Compute spectrogram
    nperseg = 512 * 2  # 2-second windows
    f, t, Sxx = signal.spectrogram(signal_segment, 
                                  fs=512,
                                  nperseg=nperseg,
                                  noverlap=nperseg//2,
                                  nfft=2048)
    
    # Filter and normalize
    mask = (f >= 1) & (f <= 64)
    power_db = 10 * np.log10(Sxx[mask])
    
    # Increase power range for better visibility
    vmin = np.percentile(power_db, 0.1)  # More extreme minimum
    vmax = np.percentile(power_db, 99.9)  # More extreme maximum
    
    # Plot with enhanced power range
    plt.figure(figsize=(16, 2))
    plt.pcolormesh(t, f[mask], power_db,
                   cmap='jet',          # Changed to jet for better contrast
                   shading='gouraud',
                   vmin=vmin,
                   vmax=vmax)
    plt.yscale('log')
    plt.yticks([1, 4, 16, 64], ['1', '4', '16', '64'], fontsize=17)
    
    # Fix spines removal and axis cleanup
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_xticks([])

   # Remove all minor ticks completely
    ax.yaxis.set_minor_locator(plt.NullLocator())

    plt.ylabel('Frequency (Hz)', fontsize=17)
    plt.ylim(1, 64)
    plt.tight_layout()
    plt.show()

    # Function call example
plot_eeg_spectrogram(
    pickle_path='Z:/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-016_ses-02_recording-01.pkl',
    recording_start_time='2024-11-29 13:29:18',  # When recording started
    segment_start_time='2024-11-30 11:30:00',    # Time window of interest
    duration_mins=90                              # Analysis duration
)